# =============================
# Stage 3: Cross-Model Fine-Tuning
# Продовження навчання з готового патчу (Stage2)
# =============================

import torch
import torch.nn.functional as F
import clip
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import os

# -------------------
# 1. Налаштування
# -------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Використання пристрою:", device)

# Шляхи
patch_path = "out/adv_patch.pt"   # <-- твій найкращий патч з Stage2
labels_path = "plane_labels.txt"  # текстові лейбли
images_dir = "."                  # директорія з фото

# -------------------
# 2. Завантаження моделей CLIP
# -------------------
model_names = ["ViT-B/32", "ViT-B/16"]
models = []

for name in model_names:
    m, preprocess = clip.load(name, device=device)
    m.eval()
    models.append((m, preprocess))

# -------------------
# 3. Завантаження патчу зі Stage2
# -------------------
if not os.path.exists(patch_path):
    raise FileNotFoundError(f"Не знайдено {patch_path}, спершу виконай Stage2!")

patch = torch.load(patch_path, map_location=device).to(device)
patch.requires_grad_(True)

print("Початковий патч завантажено з Stage2:", patch.shape)

# -------------------
# 4. Завантаження текстових лейблів
# -------------------
with open(labels_path, "r") as f:
    labels = [line.strip() for line in f.readlines()]

text_tokens = clip.tokenize(labels).to(device)

# -------------------
# 5. Підготовка зображень
# -------------------
image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg") or f.endswith(".png")]
image_files = sorted(image_files)[:10]  # беремо перші 10 фото

print("Фото для тренування:", image_files)

raw_images = [Image.open(os.path.join(images_dir, f)).convert("RGB") for f in image_files]

# -------------------
# 6. Функція для накладання патчу
# -------------------
def apply_patch(img_tensor, patch, position="center"):
    _, H, W = img_tensor.shape
    _, pH, pW = patch.shape

    patched = img_tensor.clone()

    if position == "center":
        y, x = (H - pH) // 2, (W - pW) // 2
    elif position == "top-left":
        y, x = 0, 0
    elif position == "bottom-right":
        y, x = H - pH, W - pW
    else:
        y, x = np.random.randint(0, H - pH), np.random.randint(0, W - pW)

    patched[:, y:y+pH, x:x+pW] = patch
    return patched

# -------------------
# 7. Stage3 Тренування
# -------------------
optimizer = torch.optim.Adam([patch], lr=0.001)
steps = 200  # можна збільшити до 500-1000 для більшого ефекту

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

target_idx = 0  # "a plane" як основна ціль

for step in range(steps):
    optimizer.zero_grad()

    total_loss = 0
    for (model, preprocess) in models:
        batch_images = []
        for img in raw_images:
            img_t = transform(img).to(device)
            img_t = apply_patch(img_t, patch, position="random")
            batch_images.append(img_t.unsqueeze(0))

        batch = torch.cat(batch_images, dim=0)

        img_feats = model.encode_image(batch)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

        text_feats = model.encode_text(text_tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        sims = img_feats @ text_feats.T

        # Ціль: завжди підсилювати ймовірність target_idx (a plane)
        probs = sims.softmax(dim=-1)
        loss = -probs[:, target_idx].mean()
        total_loss += loss

    total_loss.backward()
    optimizer.step()

    # обмежуємо значення у [0,1]
    with torch.no_grad():
        patch.clamp_(0,1)

    if step % 20 == 0 or step == steps-1:
        print(f"[Крок {step}/{steps}] Loss = {total_loss.item():.4f}")

# -------------------
# 8. Збереження результату
# -------------------
os.makedirs("out", exist_ok=True)

torch.save(patch, "out/final_patch_stage3.pt")
print("✅ Патч збережено як out/final_patch_stage3.pt")

# Вивід як PNG для перегляду
to_pil = transforms.ToPILImage()
patch_img = to_pil(patch.detach().cpu())
patch_img.save("out/final_patch_stage3.png")

print("✅ PNG версію збережено як out/final_patch_stage3.png")

plt.imshow(patch_img)
plt.axis("off")
plt.title("Final Patch Stage3")
plt.show()