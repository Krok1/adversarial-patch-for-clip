import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image
import clip
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Використання пристрою:", device)

# === CONFIG ===
IMG_SIZE = 224
PATCH_SIZE = 64
EPOCHS = 200
LR = 0.01

# === LOAD MODELS ===
model32, preprocess32 = clip.load("ViT-B/32", device=device)
model16, preprocess16 = clip.load("ViT-B/16", device=device)
for m in [model32, model16]:
    m.eval()

# === LOAD LABELS ===
with open("plane_labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

text_tokens = clip.tokenize(labels).to(device)

# === LOAD IMAGES ===
image_files = [f for f in os.listdir(".") if f.lower().endswith((".png",".jpg",".jpeg"))]
print("Фото для тренування:", image_files)

images32 = [preprocess32(Image.open(f).convert("RGB")).unsqueeze(0).to(device) for f in image_files]
images16 = [preprocess16(Image.open(f).convert("RGB")).unsqueeze(0).to(device) for f in image_files]

# === LOAD PATCH from Stage3 ===
patch_tensor = torch.load("out/final_patch_stage3.pt").to(device)
patch_tensor = nn.Parameter(patch_tensor.clone())

# === AUGMENTATIONS ===
augment = T.Compose([
    T.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    T.RandomRotation(20),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    T.RandomHorizontalFlip(),
])

# === APPLY PATCH FUNCTION ===
def apply_patch(image, patch, position="random"):
    _, _, H, W = image.shape
    _, _, pH, pW = patch.shape
    patched = image.clone()

    if position == "random":
        top = torch.randint(0, H - pH, (1,)).item()
        left = torch.randint(0, W - pW, (1,)).item()
    else:
        top, left = 0, 0

    patched[:, :, top:top+pH, left:left+pW] = patch
    return patched

# === TRAIN LOOP ===
optimizer = optim.Adam([patch_tensor], lr=LR)

for epoch in range(EPOCHS):
    total_loss = 0

    for img32, img16 in zip(images32, images16):
        optimizer.zero_grad()

        # Аугментації
        pil_img = T.ToPILImage()(img32[0].cpu())
        aug_img = augment(pil_img)
        img_aug32 = preprocess32(aug_img).unsqueeze(0).to(device)
        img_aug16 = preprocess16(aug_img).unsqueeze(0).to(device)

        patched32 = apply_patch(img_aug32, patch_tensor, "random")
        patched16 = apply_patch(img_aug16, patch_tensor, "random")

        img_feat32 = model32.encode_image(patched32)
        img_feat16 = model16.encode_image(patched16)
        text_feats32 = model32.encode_text(text_tokens)
        text_feats16 = model16.encode_text(text_tokens)

        img_feat32 = img_feat32 / img_feat32.norm(dim=-1, keepdim=True)
        img_feat16 = img_feat16 / img_feat16.norm(dim=-1, keepdim=True)
        text_feats32 = text_feats32 / text_feats32.norm(dim=-1, keepdim=True)
        text_feats16 = text_feats16 / text_feats16.norm(dim=-1, keepdim=True)

        logits32 = (img_feat32 @ text_feats32.T)
        logits16 = (img_feat16 @ text_feats16.T)

        target = torch.tensor([0]).to(device)  # "a plane" = клас 0

        loss32 = -logits32[0, target]
        loss16 = -logits16[0, target]
        loss = (loss32 + loss16) / 2

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if epoch % 20 == 0 or epoch == EPOCHS - 1:
        print(f"[Крок {epoch}/{EPOCHS}] Loss = {total_loss:.4f}")

# === SAVE PATCH ===
os.makedirs("out", exist_ok=True)
torch.save(patch_tensor.detach(), "out/final_patch_stage4.pt")

from torchvision.utils import save_image
save_image(patch_tensor.detach().clamp(0,1), "out/final_patch_stage4.png")

print("✅ Патч збережено як out/final_patch_stage4.pt та PNG версію")