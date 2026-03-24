#!/usr/bin/env python3
"""Cross-model analysis for adversarial patches on CLIP-style models.

This script compares how a patch affects predictions across:
- OpenAI CLIP
- OpenCLIP

It is intended for qualitative analysis of semantic shifts at different patch
positions and can be used in Google Colab or a local Python environment.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Sequence

import clip  # type: ignore
import open_clip  # type: ignore
import torch
import torch.nn.functional as F
from PIL import Image


VALID_POSITIONS = {
    "top-left": "top-left",
    "center": "center",
    "bottom-right": "bottom-right",
}


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_images(patterns: Sequence[str]) -> list[tuple[Path, Image.Image]]:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(Path(p) for p in glob.glob(pattern))

    image_paths = sorted({p.resolve() for p in matches if p.is_file()})
    if not image_paths:
        raise FileNotFoundError(f"No images found for patterns: {patterns}")

    return [(path, Image.open(path).convert("RGB")) for path in image_paths]


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
    patch = torch.from_numpy(__import__("numpy").array(image)).permute(2, 0, 1).float() / 255.0
    return patch.clamp(0, 1).to(device)


def load_labels(path: str) -> list[str]:
    labels_path = Path(path)
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {path}")

    labels = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not labels:
        raise ValueError("Labels file is empty.")
    return labels


def apply_patch(image_tensor: torch.Tensor, patch: torch.Tensor, position: str) -> torch.Tensor:
    if image_tensor.dim() != 4:
        raise ValueError("Image tensor must have shape [1, 3, H, W].")

    patched = image_tensor.clone()
    _, _, height, width = patched.shape
    _, patch_h, patch_w = patch.shape

    if patch_h > height or patch_w > width:
        raise ValueError("Patch must be smaller than the image tensor.")

    if position == "top-left":
        left, top = 0, 0
    elif position == "center":
        left, top = (width - patch_w) // 2, (height - patch_h) // 2
    elif position == "bottom-right":
        left, top = width - patch_w, height - patch_h
    else:
        raise ValueError(f"Unsupported position: {position}")

    patched[:, :, top : top + patch_h, left : left + patch_w] = patch.unsqueeze(0)
    return patched


def encode_probs(
    model: torch.nn.Module,
    batch: torch.Tensor,
    text_tokens: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        image_features = model.encode_image(batch)
        image_features = F.normalize(image_features, dim=-1)

        text_features = model.encode_text(text_tokens)
        text_features = F.normalize(text_features, dim=-1)

        logit_scale = model.logit_scale.exp()
        similarities = (image_features @ text_features.T) * logit_scale
        probabilities = similarities.softmax(dim=-1)

    return probabilities


def format_label_probs(probs: torch.Tensor, labels: Sequence[str]) -> list[str]:
    return [f"{label:20s}: {float(prob):.4f}" for label, prob in zip(labels, probs.tolist())]


def analyze_model(
    model_name: str,
    model: torch.nn.Module,
    preprocess,
    tokenizer,
    images: Sequence[tuple[Path, Image.Image]],
    labels: Sequence[str],
    patch: torch.Tensor,
    positions: Sequence[str],
    device: torch.device,
) -> None:
    print(f"\n========== Model analysis: {model_name} ==========")

    text_tokens = tokenizer(list(labels)).to(device)
    image_tensors = [(path, preprocess(image).unsqueeze(0).to(device)) for path, image in images]

    for position in positions:
        print(f"\n--- Patch position: {position} ---")
        patched_batch = torch.cat([apply_patch(img_tensor, patch, position=position) for _, img_tensor in image_tensors], dim=0)
        probabilities = encode_probs(model, patched_batch, text_tokens)

        for index, (path, _) in enumerate(image_tensors):
            print(f"\nImage {index + 1} ({path.name}):")
            for row in format_label_probs(probabilities[index], labels):
                print(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-model patch analysis for CLIP and OpenCLIP")
    parser.add_argument("--images", nargs="+", default=["photo*.jpg"], help="Image glob patterns")
    parser.add_argument("--patch", default="out/adv_patch.pt", help="Path to patch (.pt or image)")
    parser.add_argument("--labels", default="plane_labels.txt", help="Path to labels file")
    parser.add_argument(
        "--positions",
        nargs="+",
        default=["top-left", "center", "bottom-right"],
        choices=sorted(VALID_POSITIONS.keys()),
        help="Patch positions to analyze",
    )
    parser.add_argument("--openai-model", default="ViT-B/32", help="OpenAI CLIP model name")
    parser.add_argument(
        "--openclip-model",
        default="ViT-B-32",
        help="OpenCLIP model architecture name",
    )
    parser.add_argument(
        "--openclip-pretrained",
        default="laion400m_e32",
        help="OpenCLIP pretrained weights identifier",
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    images = load_images(args.images)
    patch = load_patch(args.patch, device)
    labels = load_labels(args.labels)

    openai_model, openai_preprocess = clip.load(args.openai_model, device=device, jit=False)
    openai_model.eval()

    openclip_model, _, openclip_preprocess = open_clip.create_model_and_transforms(
        args.openclip_model,
        pretrained=args.openclip_pretrained,
    )
    openclip_model.to(device)
    openclip_model.eval()

    analyze_model(
        model_name=f"OpenAI CLIP {args.openai_model}",
        model=openai_model,
        preprocess=openai_preprocess,
        tokenizer=clip.tokenize,
        images=images,
        labels=labels,
        patch=patch,
        positions=args.positions,
        device=device,
    )

    analyze_model(
        model_name=f"OpenCLIP {args.openclip_model} ({args.openclip_pretrained})",
        model=openclip_model,
        preprocess=openclip_preprocess,
        tokenizer=open_clip.tokenize,
        images=images,
        labels=labels,
        patch=patch,
        positions=args.positions,
        device=device,
    )


if __name__ == "__main__":
    main()