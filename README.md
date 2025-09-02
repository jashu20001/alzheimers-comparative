# Alzheimer’s MRI Classification — Model Comparison

A comparative study of convolutional neural network (CNN) architectures for **4-class Alzheimer’s MRI image classification** (≈6.4k images).  
**Goal:** evaluate several popular CNNs, measure test performance, and visualize confusion matrices.

---

## TL;DR (for recruiters)
- **Task:** 4-class MRI image classification (Alzheimer’s stages).  
- **Models evaluated:** **LeNet**, **U-Net** (with classifier head), **GoogLeNet (InceptionV3)**, **EfficientNetB0**, **DenseNet121**.  
- **Best test accuracy (this run):** **LeNet — 98.18%**.  
- **What I did:** dataset prep, reproducible training loops with EarlyStopping, model comparison, confusion matrices, and results summary.

---

## Dataset
- Directory-structured dataset with **4 classes** (not included in this repo).
- **Source:** Kaggle — *Alzheimer’s Disease Dataset* by Rabie El Kharoua  
  https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset
- Expected local layout:


> Please review and comply with the dataset’s license/terms on Kaggle.

---

## Methods Used

### 1) Data pipeline
- **Input size:** `150 × 150 × 3`
- **Rescaling:** `ImageDataGenerator(rescale=1/255)`  
- **Splits:** train/validation/test (directory-based; see notebook cells)  
- **Labels:** 4 classes, one-hot encoded

### 2) Architectures compared
- **LeNet** (baseline CNN)
- **U-Net** (encoder with a classifier head)
- **GoogLeNet (InceptionV3)** (transfer learning backbone)
- **EfficientNetB0** (transfer learning backbone)
- **DenseNet121** (transfer learning backbone)

### 3) Training setup
- **Optimizer:** Adam  
- **Loss:** categorical cross-entropy  
- **Metric:** accuracy  
- **Callbacks:** `EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)`  
- **Epochs:** up to 50 (early stop may terminate earlier)

### 4) Evaluation
- **Test metrics:** accuracy + loss  
- **Diagnostics:** per-model **confusion matrices** (plotted in the notebook)  
- **Result summary:** final comparison table & bar chart (in notebook)

---

## Results (from this run)

| Rank | Model                       | Test Accuracy | Test Loss |
|---:|---|---:|---:|
| 1 | LeNet                      | **0.9818** | 0.0782 |
| 2 | U-Net                      | 0.9714 | 0.0719 |
| 2 | GoogLeNet (InceptionV3)    | 0.9714 | 0.0719 |
| 4 | EfficientNetB0             | 0.8880 | 0.3148 |
| 5 | DenseNet121                | 0.5052 | 1.3128 |

> Metrics can vary with seeds, data splits, and environment.

---

## Repository Contents
- **`ALZHIEMRS_COMPARATIVE.ipynb`** — full training & evaluation notebook  
- **`README.md`** — project docs (this file)  
- **`requirements.txt`** — Python dependencies (Apple-Silicon aware)  
- **`.gitignore`** — excludes data/models/logs/checkpoints, etc.

---

## How to Access & Run *This* Notebook

### Quick View (no setup)
- Open the notebook right on GitHub:  
  **[`ALZHIEMRS_COMPARATIVE.ipynb`](https://github.com/jashu20001/alzheimerscomparative/blob/main/ALZHIEMRS_COMPARATIVE.ipynb)**  
  > GitHub will render the notebook so you can read the methods, results, and plots without installing anything.

---

### Run Locally on macOS (Apple Silicon)
These commands reproduce the exact environment for this repo.

```bash
# 1) Clone and enter the project
git clone https://github.com/jashu20001/alzheimers-comparative.git
cd alzheimers-comparative

# 2) Create & activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3) Install dependencies (TensorFlow for Apple Silicon is auto-handled)
pip install -r requirements.txt

# (Optional, recommended on M-series) Enable GPU acceleration on macOS
pip install tensorflow-metal

