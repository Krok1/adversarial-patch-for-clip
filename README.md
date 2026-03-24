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

Train a patch directly against a target class:

```bash
python src/clip_patch_cli.py train1 \
  --target "a plane;an airplane" \
  --out assets/patch_stage1.pt \
  --out-png assets/patch_stage1.png \
  --steps 2000 \
  --lr 0.3 \
  --size 128
```

Refine the patch on real test images:

```bash
python src/clip_patch_cli.py train2 \
  --target "a plane;an airplane" \
  --patch assets/patch_stage1.pt \
  --images "data/samples/*.jpg" \
  --out assets/patch_stage3.pt \
  --out-png assets/patch_stage3.png \
  --steps 1000 \
  --lr 0.1 \
  --random-placement
```

Evaluate the patch on sample images:

```bash
python src/clip_patch_cli.py evaluate \
  --patch assets/patch_stage3.pt \
  --images "data/samples/*.jpg" \
  --labels data/labels/plane_labels.txt \
  --random-placement \
  --out-csv results/results_random.csv
```

Analyze a single image before and after patch application:

```bash
python src/analyze_patch.py \
  --labels data/labels/plane_labels.txt \
  --image data/samples/photo1.jpg \
  --patch assets/patch_stage3.pt \
  --random-placement \
  --out-csv results/patch_analysis_report.csv
```

---

## 📁 Project Structure

```text
adversarial-patch-for-clip/
├── README.md
├── requirements.txt
├── .gitignore
├── assets/
│   ├── patch_stage3.png
│   ├── patch_stage3.pt
│   └── reports/
├── data/
│   ├── labels/
│   │   └── plane_labels.txt
│   └── samples/
├── docs/
│   └── thesis_summary.txt
├── results/
│   ├── patch_analysis_report.csv
│   ├── results.csv
│   ├── results_random.csv
│   └── results.txt
└── src/
    ├── clip_patch_cli.py
    ├── analyze_patch.py
    ├── cool_analyze_patch.py
    ├── train_stage3.py
    └── legacy/
        ├── train_stage4.py
        └── train_stage5_protective.py
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

## 📝 Notes

This project was developed over a longer period of time as part of an engineering thesis.

Due to the iterative nature of the work:
- some parts of the code and structure were created at different stages,
- not all intermediate steps were fully documented,
- minor inconsistencies may be present.

Additionally:
- development notes were originally written in Ukrainian,
- the thesis was prepared in Polish,
- the repository and documentation were later adapted to English.

As a result, you may occasionally encounter mixed naming conventions or minor language inconsistencies.

Despite this, the core functionality, experiments, and results remain fully valid and reproducible.

---

## 📜 License

MIT License
# adversarial-patch-for-clip
Adversarial patch system for privacy protection against CLIP image recognition. Implemented in Python and tested in Google Colab with a 3-stage training pipeline.
