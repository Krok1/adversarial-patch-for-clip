# Adversarial Patch for CLIP (Privacy Protection)

A research-oriented project implementing adversarial patches to protect visual data from automatic recognition by multimodal AI models such as CLIP.

---

## 📌 Overview

This project presents a prototype system for generating adversarial patches that manipulate the semantic interpretation of images by the CLIP model.

The system was developed as part of an engineering thesis focused on privacy protection against AI-based image recognition.

Main capabilities:

- Generation of adversarial patches
- Targeted manipulation of CLIP classification
- Evaluation on real-world images
- Automated testing pipeline with result logging

---

## ⚙️ Technologies

- Python 3.8+
- PyTorch
- OpenAI CLIP (ViT-B/32)
- Google Colab (GPU environment)

---

## 🧩 Methodology

The adversarial patch is trained using a 3-stage pipeline:

### 🔹 Stage 1 — Patch Optimization
- Patch initialized as a learnable tensor
- Optimized using gradient-based methods (FGSM-like approach)
- Objective: maximize similarity to target text embedding (e.g. "a plane")

### 🔹 Stage 2 — Stabilization (Synthetic Environment)
- Patch applied on a uniform background
- Optimization continued using iterative method (PGD-style)
- Goal: isolate patch influence and stabilize representation

### 🔹 Stage 3 — Real Image Evaluation
- Patch applied to real images
- Random placement across image
- No further training — only evaluation

---

## 📊 Results

Evaluation was performed on a dataset of 21 diverse images.

Key findings:

- ~85% success rate in altering Top-1 prediction
- Average Δ confidence: +0.0469
- Strong semantic shift toward target class ("a plane")

---

## 📈 Metrics

- **Top-1 change** — whether prediction changed
- **Confidence** — similarity score from CLIP
- **Δ confidence** = confidence(patched) - confidence(original)

---

## 🧪 Example

Before patch:
```
a car (0.24)
```

After patch:
```
a plane (0.28)
```

---

## 🚀 Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Run training and evaluation:

```bash
python train_patch_stage1.py
python train_patch_stage2.py
python test_patch.py
```

---

## 📁 Project Structure

```
adversarial-patch-for-clip/
│
├── train_patch_stage1.py
├── train_patch_stage2.py
├── test_patch.py
│
├── utils/
├── data/
├── outputs/
└── examples/
```

---

## 📦 Outputs

- Trained adversarial patches (.pt / .png)
- CSV file with evaluation results
- Visualization examples (before / after)

---

## 🔐 Privacy Context

The system is designed for **defensive purposes** — protecting user data against unauthorized AI-based recognition.

---

## 📜 License

MIT License
# adversarial-patch-for-clip
Adversarial patch system for privacy protection against CLIP image recognition. Implemented in Python and tested in Google Colab with a 3-stage training pipeline.
