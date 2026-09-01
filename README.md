# Deepfake Detection & Algorithmic Fairness Benchmark (`deepfake-detector`)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![timm](https://img.shields.io/badge/timm-Vision_Backbones-blue.svg)](https://github.com/huggingface/pytorch-image-models)
[![CVPR 2025 AI-Face Fairness](https://img.shields.io/badge/Benchmark-AI--Face_Fairness_(CVPR'25)-brightgreen.svg)](https://github.com/purdue-m2/ai-face-fairnessbench)
[![Monk Scale](https://img.shields.io/badge/Fairness-Monk_Skin_Tone_(1--10)-orange.svg)](https://skintone.google/)

Production-ready PyTorch codebase for training, evaluating, and benchmarking deepfake face detectors across spatial and frequency domains while measuring and mitigating demographic disparities across the **Monk Skin Tone Scale (MST 1–10)** and **Gender**, inspired by the **AI-Face Fairness Benchmark (CVPR 2025)**.

---

## 📌 Features & Architecture

- **Spatial Domain Detector (`models/spatial_detector.py`)**: Fine-tuned CNN/Vision Transformer backbones (`efficientnet_b0`, `efficientnet_b4`, `resnet50`, `xception`, `convnext`) via `timm` with custom classification heads.
- **Frequency Domain Detector (`models/frequency_detector.py`)**: Differentiable 2D Fast Fourier Transform (FFT) log-magnitude spectral analysis and high-pass residual filtering to detect periodic upsampling and generative synthesis artifacts.
- **Dual-Stream Fusion Model (`models/dual_stream_detector.py`)**: Multi-modal gated fusion combining spatial visual textures with spectral Fourier representations.
- **Algorithmic Fairness Suite (`utils/metrics.py`)**:
  - Subgroup False Positive Rates ($\text{FPR}_a$) and True Positive Rates ($\text{TPR}_a$).
  - **Equalized Odds Disparity ($F_{EO}$)** across Monk Skin Tone (1–10), grouped skin tones (Light, Medium, Dark), and Gender.
  - Subgroup AUC, EER, and demographic parity disparity.
- **Fairness Mitigation Loss (`utils/loss.py`)**:
  - Demographic Subgroup Reweighting ($w_g \propto \frac{1}{|G_g|}$).
  - Differentiable Equalized Odds regularization loss ($\mathcal{L}_{\text{fair}} = \mathcal{L}_{\text{BCE}} + \lambda \cdot (\text{Var}(\text{FPR}) + \text{Var}(\text{TPR}))$).
- **Automated Reporting**: Outputs detailed breakdown tables to `./results/evaluation_summary.csv`, generates `./results/fairness_audit_report.md`, and renders diagnostic disparity plots.

---

## 📐 Mathematical Formulation of Fairness Metrics

### Equalized Odds Disparity ($F_{EO}$)
A deepfake detector satisfies Equalized Odds with respect to demographic attribute $A \in \mathcal{A}$ if prediction $\hat{Y}$ is conditionally independent of $A$ given true label $Y$:

$$P(\hat{Y}=1 \mid A=a, Y=y) = P(\hat{Y}=1 \mid A=a', Y=y) \quad \forall y \in \{0, 1\}, \forall a, a' \in \mathcal{A}$$

In practice, we quantify the disparity across demographic groups $a \in \mathcal{A}$:

$$\Delta \text{FPR} = \max_{a \in \mathcal{A}} \text{FPR}_a - \min_{a \in \mathcal{A}} \text{FPR}_a$$

$$\Delta \text{TPR} = \max_{a \in \mathcal{A}} \text{TPR}_a - \min_{a \in \mathcal{A}} \text{TPR}_a$$

$$\mathbf{F_{EO}} = \frac{1}{2} (\Delta \text{FPR} + \Delta \text{TPR})$$

$$\Delta \text{EO}_{\max} = \max(\Delta \text{FPR}, \Delta \text{TPR})$$

### Monk Skin Tone (MST) Categorization
- **Individual Buckets**: Scale integers 1 through 10.
- **Aggregated Groups**:
  - **Light**: MST 1–3
  - **Medium**: MST 4–7
  - **Dark**: MST 8–10

---

## 📂 Repository Structure

```
deepfake-detector/
├── configs/
│   └── default_config.yaml         # Training and model configuration
├── data/
│   ├── generate_mock_data.py       # Generates synthetic face dataset with Monk scale tags
│   ├── create_subset.py            # Balanced demographic subset sampler
│   ├── download_and_create_aiface_subset.py  # AI-Face CVPR 2025 ingestion & 30k sampler
│   └── mock_dataset/               # Synthetic benchmark data directory
├── dataset/                        # AI-Face subset CSVs (generated)
│   ├── train_subset.csv            # 30,000 train samples (AI-Face schema)
│   ├── train_subset_mapped.csv     # 30,000 train samples (project schema)
│   ├── test_subset.csv             # 5,000 test samples (AI-Face schema)
│   └── test_subset_mapped.csv      # 5,000 test samples (project schema)
├── models/
│   ├── __init__.py                 # Model factory (build_model)
│   ├── spatial_detector.py         # EfficientNet / timm spatial detector
│   ├── frequency_detector.py       # 2D FFT & high-pass spectral detector
│   └── dual_stream_detector.py     # Multi-modal spatial + frequency fusion
├── utils/
│   ├── __init__.py
│   ├── dataset_loader.py           # PyTorch Dataset for images + demographic annotations
│   ├── metrics.py                  # FairnessAuditor, Equalized Odds (F_EO), EER
│   └── loss.py                     # Subgroup reweighting & fair regularized loss
├── results/
│   ├── evaluation_summary.csv      # Detailed subgroup metrics table
│   ├── fairness_audit_report.md    # Markdown fairness audit summary
│   └── fairness_disparity_plots.png
├── checkpoints/                    # Saved model checkpoints (.pt)
├── train.py                        # Full training loop with bias mitigation
├── evaluate.py                     # Comprehensive evaluation pipeline
├── inference.py                    # Single-image & batch folder inference
├── benchmark.py                    # Multi-model comparative benchmark
├── requirements.txt                # Project dependencies
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Installation

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Dataset: AI-Face FairnessBench (CVPR 2025)

This project benchmarks on the **[AI-Face FairnessBench](https://github.com/purdue-m2/ai-face-fairnessbench)** dataset — a CVPR 2025 benchmark with 600k+ face images across 37 generative models, annotated with Monk Skin Tone (1–10), Gender, Age, and Intersection groups.

#### Option A — Download Official Data
1. Visit the official repository: https://github.com/purdue-m2/ai-face-fairnessbench
2. Download annotations from Purdue SharePoint (link in repo)
3. Place `train.csv` / `test.csv` in `./dataset/`
4. Run the sampler:

```bash
python data/download_and_create_aiface_subset.py \
  --input_csv dataset/train.csv \
  --output_csv dataset/train_subset_30k.csv \
  --subset_size 30000
```

#### Option B — Use Pre-Generated Structured Subset (included)
A balanced benchmark template (30,000 train / 5,000 test) is already generated at:
- `dataset/train_subset_mapped.csv` — standard project schema
- `dataset/test_subset_mapped.csv` — standard project schema

Use these directly for training (no real images required for CSV testing):
```bash
python train.py --annotations dataset/train_subset_mapped.csv ...
```

### 3. Annotations CSV Format

```csv
image_path,target,gender,skin_tone
images/sample_00001.jpg,0,Female,3
images/sample_00002.jpg,1,Male,8
```

| Column | Values |
|---|---|
| `image_path` | Relative or absolute path to face image |
| `target` | `0` = Real, `1` = Fake/Deepfake |
| `gender` | `Male`, `Female`, `Other` |
| `skin_tone` | Integer `1`–`10` (Monk Skin Tone Scale) |

### 4. Training

#### Train Spatial Detector with Fairness Mitigation:
```bash
python train.py \
  --model_type spatial \
  --backbone efficientnet_b0 \
  --annotations ./data/mock_dataset/annotations.csv \
  --epochs 10 \
  --batch_size 16 \
  --mitigate_bias \
  --loss_type equalized_odds \
  --lambda_fair 0.5
```

#### Train Frequency Detector (2D FFT / High-Pass Filtering):
```bash
python train.py \
  --model_type frequency \
  --backbone resnet34 \
  --annotations ./data/mock_dataset/annotations.csv \
  --epochs 10 \
  --batch_size 16
```

#### Train Dual-Stream Spatial + Frequency Model:
```bash
python train.py \
  --model_type dual_stream \
  --annotations ./data/mock_dataset/annotations.csv \
  --epochs 10 \
  --batch_size 16 \
  --mitigate_bias
```

### 5. Evaluation & Fairness Audit

```bash
python evaluate.py \
  --model_path ./checkpoints/best_model.pt \
  --annotations ./data/mock_dataset/annotations.csv \
  --output_dir ./results \
  --threshold 0.5
```

**Outputs:**
1. `./results/evaluation_summary.csv` — Per-subgroup metrics (Monk Tone 1–10, Gender, Overall)
2. `./results/fairness_audit_report.md` — Full F_EO, ΔFPR, ΔTPR audit
3. `./results/fairness_disparity_plots.png` — Disparity bar charts

### 6. Inference on New Images

```bash
# Single image
python inference.py \
  --model_path checkpoints/best_model.pt \
  --image path/to/face.jpg

# Batch folder
python inference.py \
  --model_path checkpoints/best_model.pt \
  --image_dir path/to/images/ \
  --output_dir inference_results/
```

### 7. Multi-Model Benchmark

Compare all three architectures in one run:

```bash
python benchmark.py \
  --annotations ./data/mock_dataset/annotations.csv \
  --epochs 5 \
  --output_dir ./benchmark_results
```

Generates `benchmark_results/benchmark_report.md` with AUC and F_EO for all models.

---

## ⚡ Virtual Environment Note (Windows)

Always use the `venv` environment (not `.venv`):
```bash
# Activate
.\venv\Scripts\activate

# Or run directly
.\venv\Scripts\python train.py ...
```

---

## 📊 Expected Results (Mock Dataset, 5 Epochs)

| Model | AUC | Accuracy | F_EO↓ |
|---|---|---|---|
| EfficientNet-B0 (Spatial) | ~1.00 | ~93% | ~0.50 |
| ResNet-18 (Frequency) | ~0.95 | ~87% | ~0.45 |
| EfficientNet-B0 (Dual-Stream) | ~1.00 | ~95% | ~0.40 |

> F_EO closer to 0 = more demographically fair model.

---

## 🔬 Citation & References
- **AI-Face Fairness Benchmark (CVPR 2025)**: Lin et al. — Demographic bias and equalized error rate evaluation in facial forensics. https://github.com/purdue-m2/ai-face-fairnessbench
- **Monk Skin Tone (MST) Scale**: Ellis P. Monk, Jr. (Harvard University / Google AI). https://skintone.google/
- **PyTorch Image Models (`timm`)**: Ross Wightman. https://github.com/huggingface/pytorch-image-models
- **Equalized Odds**: Hardt, Price & Srebro, NeurIPS 2016.
