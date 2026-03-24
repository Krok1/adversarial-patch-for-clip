#!/usr/bin/env python3
"""Analyze CLIP predictions before and after applying an adversarial patch.

This script is intended as a lightweight research helper for inspecting how an
adversarial patch changes the semantic interpretation of an image according to
CLIP. It can be used both locally and in Google Colab.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Sequence

import clip  # type: ignore
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


VALID_POSITIONS = {"tl", "tr", "bl", "br", "center"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(model_name: str, device: torch.device) -> tuple[torch.nn.Module, int, torch.Tensor, torch.Tensor]:
    model, preprocess = clip.load(model_name, device=device, jit=False)
    model.eval()

    input_size = 224
    for transform in getattr(preprocess, "transforms", []):
        size = getattr(transform, "size", None)
        if isinstance(size, int):
            input_size = size
        elif isinstance(size, (tuple, list)) and size:
            input_size = int(size[0])

    mean = torch.tensor((0.48145466, 0.4578275, 0.40821073), device=device).view(1, 3, 1, 1)
    std = torch.tensor((0.26862954, 0.26130258, 0.27577711), device=device).view(1, 3, 1, 1)
    return model, input_size, mean, std


def normalize_for_clip(batch: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (batch - mean) / std


def load_labels(path: str, device: torch.device) -> tuple[list[str], torch.Tensor]:
    label_path = Path(path)
    if not label_path.exists():
        raise FileNotFoundError(f"Labels file not found: {path}")

    labels = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not labels:
        raise ValueError("Labels file is empty.")

    text_tokens = clip.tokenize(labels).to(device)
    return labels, text_tokens


def load_image(path: str, size: int, device: torch.device) -> torch.Tensor:
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image = Image.open(image_path).convert("RGB")
    tensor = TF.to_tensor(image)
    tensor = TF.resize(tensor, [size, size], interpolation=InterpolationMode.BICUBIC, antialias=True)
    return tensor.unsqueeze(0).to(device)


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
            raise ValueError("Patch tensor must have shape [3, H, W].")
        return patch.float().clamp(0, 1).to(device)

    image = Image.open(patch_path).convert("RGB")
    return TF.to_tensor(image).float().clamp(0, 1).to(device)


def apply_patch(
    image: torch.Tensor,
    patch: torch.Tensor,
    position: str = "br",
    random_placement: bool = False,
) -> tuple[torch.Tensor, tuple[int, int]]:
    if image.dim() != 4:
        raise ValueError("Image tensor must have shape [1, 3, H, W].")

    patched = image.clone()
    _, _, h, w = patched.shape
    _, ph, pw = patch.shape

    if ph > h or pw > w:
        raise ValueError("Patch must be smaller than the image.")

    if random_placement:
        top = random.randint(0, h - ph)
        left = random.randint(0, w - pw)
    else:
        position = position.lower()
        if position not in VALID_POSITIONS:
            raise ValueError(f"Unsupported position: {position}")
        if position == "tl":
            top, left = 0, 0
        elif position == "tr":
            top, left = 0, w - pw
        elif position == "bl":
            top, left = h - ph, 0
        elif position == "br":
            top, left = h - ph, w - pw
        else:
            top, left = (h - ph) // 2, (w - pw) // 2

    patched[:, :, top : top + ph, left : left + pw] = patch.unsqueeze(0)
    return patched, (top, left)


def get_probabilities(
    model: torch.nn.Module,
    image: torch.Tensor,
    text_tokens: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = F.normalize(text_features, dim=-1)

        image_features = model.encode_image(normalize_for_clip(image, mean, std))
        image_features = F.normalize(image_features, dim=-1)

        logits = 100.0 * image_features @ text_features.T
        probs = logits.softmax(dim=-1)

    return probs[0]


def format_topk(probs: torch.Tensor, labels: Sequence[str], k: int = 5) -> str:
    values, indices = probs.topk(min(k, len(labels)))
    rows = []
    for value, index in zip(values.tolist(), indices.tolist()):
        rows.append(f"{labels[index]} ({value:.4f})")
    return " | ".join(rows)


def write_csv(
    path: str,
    image_name: str,
    original_class: str,
    original_conf: float,
    patched_class: str,
    patched_conf: float,
    delta_conf: float,
    top5_original: str,
    top5_patched: str,
    patch_top: int,
    patch_left: int,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "image",
                "class_original",
                "confidence_original",
                "class_patched",
                "confidence_patched",
                "delta_confidence",
                "top5_original",
                "top5_patched",
                "patch_top",
                "patch_left",
            ]
        )
        writer.writerow(
            [
                image_name,
                original_class,
                f"{original_conf:.6f}",
                patched_class,
                f"{patched_conf:.6f}",
                f"{delta_conf:.6f}",
                top5_original,
                top5_patched,
                patch_top,
                patch_left,
            ]
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze CLIP predictions before and after patch application")
    parser.add_argument("--labels", required=True, help="Path to labels file")
    parser.add_argument("--image", required=True, help="Path to a single test image")
    parser.add_argument("--patch", required=True, help="Path to patch (.pt or image)")
    parser.add_argument("--model", default="ViT-B/32", help="CLIP model name")
    parser.add_argument("--position", default="br", choices=sorted(VALID_POSITIONS), help="Patch position")
    parser.add_argument("--random-placement", action="store_true", help="Apply patch at a random location")
    parser.add_argument("--topk", type=int, default=5, help="Number of top predictions to display")
    parser.add_argument("--out-csv", default=None, help="Optional path for CSV export")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, input_size, mean, std = load_model(args.model, device)
    labels, text_tokens = load_labels(args.labels, device)
    image = load_image(args.image, input_size, device)
    patch = load_patch(args.patch, device)

    patched_image, (top, left) = apply_patch(
        image=image,
        patch=patch,
        position=args.position,
        random_placement=args.random_placement,
    )

    original_probs = get_probabilities(model, image, text_tokens, mean, std)
    patched_probs = get_probabilities(model, patched_image, text_tokens, mean, std)

    original_top_idx = int(original_probs.argmax().item())
    patched_top_idx = int(patched_probs.argmax().item())

    original_class = labels[original_top_idx]
    patched_class = labels[patched_top_idx]
    original_conf = float(original_probs[original_top_idx].item())
    patched_conf = float(patched_probs[patched_top_idx].item())
    delta_conf = patched_conf - original_conf

    top5_original = format_topk(original_probs, labels, k=args.topk)
    top5_patched = format_topk(patched_probs, labels, k=args.topk)

    print("\n=== CLIP Patch Analysis ===")
    print(f"Image: {Path(args.image).name}")
    print(f"Patch placement: top={top}, left={left}")
    print(f"Original Top-1: {original_class} ({original_conf:.4f})")
    print(f"Patched  Top-1: {patched_class} ({patched_conf:.4f})")
    print(f"Δ confidence: {delta_conf:.4f}")
    print(f"Top-{args.topk} original: {top5_original}")
    print(f"Top-{args.topk} patched : {top5_patched}")

    if args.out_csv:
        write_csv(
            path=args.out_csv,
            image_name=Path(args.image).name,
            original_class=original_class,
            original_conf=original_conf,
            patched_class=patched_class,
            patched_conf=patched_conf,
            delta_conf=delta_conf,
            top5_original=top5_original,
            top5_patched=top5_patched,
            patch_top=top,
            patch_left=left,
        )
        print(f"CSV written to: {args.out_csv}")


if __name__ == "__main__":
    main()
