

#!/usr/bin/env python3
"""CLI tool for training and evaluating adversarial patches against CLIP.

This script provides a small research-oriented workflow:
- train1: optimize a patch directly against a target text prompt
- train2: refine an existing patch on real images
- evaluate: compare predictions before/after patch application

The implementation is intentionally lightweight so it can be used both locally
and in Google Colab. It relies only on PyTorch, torchvision, PIL and OpenAI CLIP.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import clip  # type: ignore
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm import tqdm


VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_EVAL_LABELS = [
    "a plane",
    "an airplane",
    "a car",
    "a tree",
    "a banana",
    "a bridge",
    "a bottle",
    "a volcano",
    "a lantern",
    "a skyscraper",
    "a mug",
    "a cactus",
    "a table",
    "a camera",
    "a whistle",
]


@dataclass
class ClipContext:
    model: torch.nn.Module
    device: torch.device
    input_size: int
    mean: torch.Tensor
    std: torch.Tensor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_prompt_list(raw: str) -> List[str]:
    prompts = [p.strip() for p in raw.split(";") if p.strip()]
    if not prompts:
        raise ValueError("At least one non-empty prompt is required.")
    return prompts


def list_images(patterns: Sequence[str]) -> List[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        for match in glob.glob(pattern):
            path = Path(match)
            if path.suffix.lower() in VALID_IMAGE_EXTENSIONS and path.is_file():
                paths.append(path)
    unique_paths = sorted({p.resolve() for p in paths})
    if not unique_paths:
        raise FileNotFoundError(f"No images found for patterns: {patterns}")
    return unique_paths


def load_labels(path: str | None) -> List[str]:
    if path is None:
        return DEFAULT_EVAL_LABELS
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Labels file not found: {path}")
    labels = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not labels:
        raise ValueError("Labels file is empty.")
    return labels


def load_clip(model_name: str) -> ClipContext:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load(model_name, device=device, jit=False)
    model.eval()

    # OpenAI CLIP preprocess uses resize/crop to 224 and these normalization stats.
    # We read them directly from the returned transform for robustness.
    input_size = 224
    mean = torch.tensor((0.48145466, 0.4578275, 0.40821073), device=device).view(1, 3, 1, 1)
    std = torch.tensor((0.26862954, 0.26130258, 0.27577711), device=device).view(1, 3, 1, 1)

    # Best effort extraction of CLIP input size from preprocess pipeline.
    for t in getattr(preprocess, "transforms", []):
        size = getattr(t, "size", None)
        if isinstance(size, int):
            input_size = size
        elif isinstance(size, (tuple, list)) and size:
            input_size = int(size[0])

    return ClipContext(model=model, device=device, input_size=input_size, mean=mean, std=std)


def normalize_for_clip(batch: torch.Tensor, ctx: ClipContext) -> torch.Tensor:
    return (batch - ctx.mean) / ctx.std


def encode_text(ctx: ClipContext, prompts: Sequence[str]) -> torch.Tensor:
    tokens = clip.tokenize(list(prompts)).to(ctx.device)
    with torch.no_grad():
        text_features = ctx.model.encode_text(tokens)
        text_features = F.normalize(text_features, dim=-1)
    return text_features


def load_image_tensor(path: Path, size: int, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    tensor = TF.to_tensor(img)
    tensor = TF.resize(tensor, [size, size], interpolation=InterpolationMode.BICUBIC, antialias=True)
    return tensor.to(device)


def save_patch_png(patch: torch.Tensor, output_path: str | Path, upscale: int = 1) -> None:
    patch = patch.detach().cpu().clamp(0, 1)
    if upscale > 1:
        patch = F.interpolate(patch.unsqueeze(0), scale_factor=upscale, mode="nearest").squeeze(0)
    arr = patch.permute(1, 2, 0).numpy()
    arr = (arr * 255).round().astype(np.uint8)
    Image.fromarray(arr).save(output_path)


def save_patch_pt(patch: torch.Tensor, output_path: str | Path) -> None:
    torch.save(patch.detach().cpu(), output_path)


def load_patch(path: str, device: torch.device) -> torch.Tensor:
    patch_path = Path(path)
    if not patch_path.exists():
        raise FileNotFoundError(f"Patch file not found: {path}")

    if patch_path.suffix.lower() == ".pt":
        patch = torch.load(patch_path, map_location=device)
        if isinstance(patch, dict) and "patch" in patch:
            patch = patch["patch"]
        if patch.dim() == 4:
            patch = patch[0]
        if patch.dim() != 3 or patch.shape[0] != 3:
            raise ValueError("Saved patch tensor must have shape [3, H, W].")
        return patch.to(device).float().clamp(0, 1)

    image = Image.open(patch_path).convert("RGB")
    tensor = TF.to_tensor(image).to(device)
    return tensor.float().clamp(0, 1)


def apply_patch(
    image: torch.Tensor,
    patch: torch.Tensor,
    pos: str = "br",
    random_placement: bool = False,
) -> tuple[torch.Tensor, tuple[int, int]]:
    if image.dim() != 3 or patch.dim() != 3:
        raise ValueError("Image and patch must be tensors of shape [3, H, W].")

    patched = image.clone()
    _, h, w = image.shape
    _, ph, pw = patch.shape
    if ph > h or pw > w:
        raise ValueError("Patch must be smaller than or equal to the image size.")

    if random_placement:
        top = random.randint(0, h - ph)
        left = random.randint(0, w - pw)
    else:
        pos = pos.lower()
        if pos == "tl":
            top, left = 0, 0
        elif pos == "tr":
            top, left = 0, w - pw
        elif pos == "bl":
            top, left = h - ph, 0
        elif pos == "br":
            top, left = h - ph, w - pw
        elif pos == "center":
            top, left = (h - ph) // 2, (w - pw) // 2
        else:
            raise ValueError(f"Unsupported position: {pos}")

    patched[:, top : top + ph, left : left + pw] = patch
    return patched, (top, left)


def maybe_augment_patch(patch: torch.Tensor, rotate_deg: float = 0.0, scale_range: tuple[float, float] | None = None) -> torch.Tensor:
    aug = patch
    if scale_range is not None:
        scale = random.uniform(*scale_range)
        size = max(8, int(round(patch.shape[-1] * scale)))
        aug = TF.resize(aug, [size, size], interpolation=InterpolationMode.BICUBIC, antialias=True)
    if rotate_deg > 0:
        angle = random.uniform(-rotate_deg, rotate_deg)
        aug = TF.rotate(aug, angle=angle, interpolation=InterpolationMode.BILINEAR, fill=0.0)
    return aug.clamp(0, 1)


def compute_target_probability(
    images: torch.Tensor,
    ctx: ClipContext,
    text_features: torch.Tensor,
    target_indices: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    image_features = ctx.model.encode_image(normalize_for_clip(images, ctx))
    image_features = F.normalize(image_features, dim=-1)
    logits = 100.0 * image_features @ text_features.T
    probs = logits.softmax(dim=-1)
    target_prob = probs[:, list(target_indices)].sum(dim=-1)
    return target_prob, probs


def train_stage1(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    ctx = load_clip(args.model)
    prompts = parse_prompt_list(args.target)
    text_features = encode_text(ctx, prompts)
    target_indices = list(range(len(prompts)))

    patch = torch.rand(3, args.size, args.size, device=ctx.device, requires_grad=True)
    optimizer = torch.optim.Adam([patch], lr=args.lr)

    best_prob = -1.0
    best_patch = None
    history: list[dict[str, float | int]] = []

    progress = tqdm(range(1, args.steps + 1), desc="Stage1")
    for step in progress:
        optimizer.zero_grad(set_to_none=True)
        batch = patch.unsqueeze(0).clamp(0, 1)
        target_prob, _ = compute_target_probability(batch, ctx, text_features, target_indices)
        loss = -target_prob.mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            patch.clamp_(0, 1)

        prob_value = float(target_prob.mean().item())
        history.append({"step": step, "target_prob": prob_value, "loss": float(loss.item())})
        progress.set_postfix(target_prob=f"{prob_value:.4f}")

        if prob_value > best_prob:
            best_prob = prob_value
            best_patch = patch.detach().clone()

        if args.threshold is not None and prob_value >= args.threshold:
            break

    final_patch = best_patch if best_patch is not None else patch.detach()
    save_outputs(final_patch, args.out, args.out_png, upscale=args.png_upscale)
    save_training_log(args.out_log, history)
    print(json.dumps({"best_target_prob": best_prob, "output": args.out}, indent=2))


def train_stage2(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    ctx = load_clip(args.model)
    prompts = parse_prompt_list(args.target)
    text_features = encode_text(ctx, prompts)
    target_indices = list(range(len(prompts)))

    image_paths = list_images(args.images)
    patch = load_patch(args.patch, ctx.device).detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([patch], lr=args.lr)

    best_prob = -1.0
    best_patch = patch.detach().clone()
    history: list[dict[str, float | int]] = []

    progress = tqdm(range(1, args.steps + 1), desc="Stage2")
    for step in progress:
        batch_images: list[torch.Tensor] = []
        for _ in range(args.batch_size):
            image_path = random.choice(image_paths)
            image = load_image_tensor(image_path, ctx.input_size, ctx.device)
            candidate_patch = maybe_augment_patch(
                patch.clamp(0, 1),
                rotate_deg=args.rotate_deg,
                scale_range=(args.scale_min, args.scale_max) if args.scale_min and args.scale_max else None,
            )
            patched_image, _ = apply_patch(
                image=image,
                patch=candidate_patch,
                pos=args.pos,
                random_placement=args.random_placement,
            )
            batch_images.append(patched_image)

        batch = torch.stack(batch_images, dim=0)
        optimizer.zero_grad(set_to_none=True)
        target_prob, _ = compute_target_probability(batch, ctx, text_features, target_indices)
        loss = -target_prob.mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            patch.clamp_(0, 1)

        prob_value = float(target_prob.mean().item())
        history.append({"step": step, "target_prob": prob_value, "loss": float(loss.item())})
        progress.set_postfix(target_prob=f"{prob_value:.4f}")

        if prob_value > best_prob:
            best_prob = prob_value
            best_patch = patch.detach().clone()

    save_outputs(best_patch, args.out, args.out_png, upscale=args.png_upscale)
    save_training_log(args.out_log, history)
    print(json.dumps({"best_target_prob": best_prob, "output": args.out}, indent=2))


def evaluate(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    ctx = load_clip(args.model)
    labels = load_labels(args.labels)
    text_features = encode_text(ctx, labels)
    image_paths = list_images(args.images)
    patch = load_patch(args.patch, ctx.device)

    rows: list[dict[str, object]] = []

    for image_path in tqdm(image_paths, desc="Evaluate"):
        image = load_image_tensor(image_path, ctx.input_size, ctx.device)
        patched_image, (top, left) = apply_patch(
            image=image,
            patch=patch,
            pos=args.pos,
            random_placement=args.random_placement,
        )

        original_probs = compute_probs_for_labels(image.unsqueeze(0), ctx, text_features)
        patched_probs = compute_probs_for_labels(patched_image.unsqueeze(0), ctx, text_features)

        orig_top_idx = int(original_probs.argmax().item())
        patched_top_idx = int(patched_probs.argmax().item())
        delta = float(patched_probs[patched_top_idx].item() - original_probs[orig_top_idx].item())

        rows.append(
            {
                "image": image_path.name,
                "class_original": labels[orig_top_idx],
                "confidence_original": float(original_probs[orig_top_idx].item()),
                "class_patched": labels[patched_top_idx],
                "confidence_patched": float(patched_probs[patched_top_idx].item()),
                "delta_confidence": delta,
                "patch_top": top,
                "patch_left": left,
                "top5_original": format_topk(original_probs, labels, k=5),
                "top5_patched": format_topk(patched_probs, labels, k=5),
            }
        )

    write_csv(args.out_csv, rows)
    print(json.dumps({"rows": len(rows), "output": args.out_csv}, indent=2))


def compute_probs_for_labels(image_batch: torch.Tensor, ctx: ClipContext, text_features: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        image_features = ctx.model.encode_image(normalize_for_clip(image_batch, ctx))
        image_features = F.normalize(image_features, dim=-1)
        logits = 100.0 * image_features @ text_features.T
        probs = logits.softmax(dim=-1)
    return probs[0]


def format_topk(probs: torch.Tensor, labels: Sequence[str], k: int = 5) -> str:
    values, indices = probs.topk(min(k, len(labels)))
    parts = [f"{labels[i]}::{float(v):.4f}" for v, i in zip(values.tolist(), indices.tolist())]
    return " | ".join(parts)


def write_csv(path: str, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("No evaluation rows to write.")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_training_log(path: str | None, history: Sequence[dict[str, float | int]]) -> None:
    if not path:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "target_prob", "loss"])
        writer.writeheader()
        writer.writerows(history)


def save_outputs(patch: torch.Tensor, out_pt: str, out_png: str | None, upscale: int = 1) -> None:
    pt_path = Path(out_pt)
    pt_path.parent.mkdir(parents=True, exist_ok=True)
    save_patch_pt(patch, pt_path)

    if out_png:
        png_path = Path(out_png)
    else:
        png_path = pt_path.with_suffix(".png")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    save_patch_png(patch, png_path, upscale=upscale)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLIP adversarial patch trainer and evaluator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train1 = subparsers.add_parser("train1", help="Train patch directly against target text")
    train1.add_argument("--target", required=True, help='Semicolon-separated target prompts, e.g. "a plane;an airplane"')
    train1.add_argument("--model", default="ViT-B/32", help="CLIP model name")
    train1.add_argument("--steps", type=int, default=2000)
    train1.add_argument("--lr", type=float, default=0.3)
    train1.add_argument("--size", type=int, default=128)
    train1.add_argument("--threshold", type=float, default=None)
    train1.add_argument("--out", required=True, help="Output .pt path")
    train1.add_argument("--out-png", default=None, help="Optional output PNG path")
    train1.add_argument("--out-log", default=None, help="Optional CSV log path")
    train1.add_argument("--png-upscale", type=int, default=1)
    train1.add_argument("--seed", type=int, default=42)
    train1.set_defaults(func=train_stage1)

    train2 = subparsers.add_parser("train2", help="Refine existing patch on real images")
    train2.add_argument("--target", required=True, help='Semicolon-separated target prompts, e.g. "a plane;an airplane"')
    train2.add_argument("--patch", required=True, help="Input patch path (.pt or image)")
    train2.add_argument("--images", nargs="+", required=True, help="Image glob patterns")
    train2.add_argument("--model", default="ViT-B/32", help="CLIP model name")
    train2.add_argument("--steps", type=int, default=1000)
    train2.add_argument("--lr", type=float, default=0.1)
    train2.add_argument("--batch-size", type=int, default=4)
    train2.add_argument("--pos", default="br", choices=["tl", "tr", "bl", "br", "center"], help="Patch position")
    train2.add_argument("--random-placement", action="store_true")
    train2.add_argument("--rotate-deg", type=float, default=0.0)
    train2.add_argument("--scale-min", type=float, default=1.0)
    train2.add_argument("--scale-max", type=float, default=1.0)
    train2.add_argument("--out", required=True, help="Output .pt path")
    train2.add_argument("--out-png", default=None, help="Optional output PNG path")
    train2.add_argument("--out-log", default=None, help="Optional CSV log path")
    train2.add_argument("--png-upscale", type=int, default=1)
    train2.add_argument("--seed", type=int, default=42)
    train2.set_defaults(func=train_stage2)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate patch on images")
    evaluate_parser.add_argument("--patch", required=True, help="Patch path (.pt or image)")
    evaluate_parser.add_argument("--images", nargs="+", required=True, help="Image glob patterns")
    evaluate_parser.add_argument("--labels", default=None, help="Optional labels file")
    evaluate_parser.add_argument("--model", default="ViT-B/32", help="CLIP model name")
    evaluate_parser.add_argument("--pos", default="br", choices=["tl", "tr", "bl", "br", "center"], help="Patch position")
    evaluate_parser.add_argument("--random-placement", action="store_true")
    evaluate_parser.add_argument("--out-csv", required=True, help="Output CSV path")
    evaluate_parser.add_argument("--seed", type=int, default=42)
    evaluate_parser.set_defaults(func=evaluate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()