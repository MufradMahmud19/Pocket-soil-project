# 🌍 Pocket Geologist: Soil Erodibility Prediction from Images

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![Best Backbone](https://img.shields.io/badge/Best%20Backbone-ConvNeXt--Tiny-blueviolet)
![Best Head](https://img.shields.io/badge/Best%20Head-Dual--Pool%20MLP-blueviolet)
![Test Accuracy](https://img.shields.io/badge/Test%20Accuracy-93.50%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**Pocket Geologist** is a deep learning computer vision system built with PyTorch that predicts **soil erodibility** directly from field photographs. It classifies soil samples across **9 distinct soil types**, maps them to physical **USDA K-Factors** ($0.02 \text{ to } 0.69$), and computes both discrete and probability-weighted **Erodibility Risk Scores (`1/10` to `10/10`)**.

Using a two-phase transfer learning pipeline, a **Dual-Pooling MLP** classifier head, dampened class-imbalance sampling, and Test-Time Augmentation (TTA), the best-performing configuration (**ConvNeXt-Tiny + Dual-Pool MLP**) achieves **93.50% test accuracy** (`0.934` weighted F1-score, `0.2475` best validation loss).

---

## ✨ Key Features

* **USDA K-Factor & Risk Mapping:** Translates visual soil classification into physical USDA erodibility K-factors, a 1–10 erodibility score, and a probability-weighted continuous risk score.
* **Dual-Pooling Feature Extraction (`dual_pool_mlp`):** Concatenates **Global Average Pooling** (captures overall soil chromaticity and tone) with **Global Max Pooling** (captures sharp grain boundaries, cinder edges, and surface cracks).
* **Modular Backbones & Classifier Heads:** Supports plug-and-play switching via CLI between 4 pretrained CNN backbones (`efficientnet_b0`, `convnext_tiny`, `resnet50`, `mobilenet_v3_large`) and 3 classifier heads (`dual_pool_mlp`, `cosine`, `mlp`).
* **Two-Phase Fine-Tuning with Frozen `BatchNorm2d`:**
  * **Phase 1 (15 Epochs):** Warms up the custom classifier head (`LR = 1e-3`) while keeping the ImageNet backbone frozen.
  * **Phase 2 (Up to 25 Epochs):** Unfreezes the top 3 backbone blocks using differential learning rates (`1e-4` backbone / `3e-4` head) while keeping `BatchNorm2d` statistics frozen to prevent small-batch drift.
* **Smart Class-Imbalance Handling:** Combines square-root dampened `WeightedRandomSampler` (`power = 0.5`) with mild class weights (`power = 0.20`) and label smoothing (`0.05`) to achieve high recall on rare classes (`Cinder Soil`, `Peat Soil`, `Alluvial_Soil`) without false-positive spikes.
* **Soil-Preserving Augmentations & TTA:** Uses gentle color jittering to preserve diagnostic soil hues (Red vs. Yellow vs. Laterite vs. Black soil), `RandomErasing` for occlusion robustness, and horizontal-flip Test-Time Augmentation (TTA) during evaluation and inference.

---

## 🪨 Soil Erodibility Knowledge Base & Dataset Distribution

Each soil class is mapped to its standard USDA K-factor and scaled to a `1–10` erodibility score using:

$$\text{Score} = \text{round}\left(\frac{K}{0.69} \times 9\right) + 1$$

The dataset contains **1,331 total images** split via stratified sampling into **70% Train (931)**, **15% Validation (200)**, and **15% Test (200)**:

| Soil Type | Total Images | USDA K-Factor | Erodibility Score | Risk Level |
| :--- | :---: | :---: | :---: | :--- |
| **Alluvial_Soil** | 38 | `0.40` | `6 / 10` | Medium-High |
| **Arid_Soil** | 284 | `0.52` | `8 / 10` | High |
| **Black Soil** | 292 | `0.17` | `2 / 10` | Low |
| **Cinder Soil** | 30 | `0.47` | `7 / 10` | High |
| **Laterite Soil** | 249 | `0.15` | `3 / 10` | Low |
| **Mountain_Soil** | 201 | `0.30` | `4 / 10` | Medium |
| **Peat Soil** | 30 | `0.20` | `3 / 10` | Low |
| **Red_Soil** | 109 | `0.35` | `5 / 10` | Medium |
| **Yellow Soil** | 98 | `0.30` | `5 / 10` | Medium |

---

## 📊 Experimental Results — Multi-Backbone & Multi-Head Comparison

All experiments were run on the same stratified 70/15/15 split (931 train / 200 val / 200 test), identical augmentation pipeline, dampened `WeightedRandomSampler`, label smoothing (`0.05`), and TTA-enabled evaluation. Only the backbone and classifier head were changed between runs.

### Summary Leaderboard

| Rank | Backbone | Head | Test Acc | Wtd F1 | Macro F1 | Best Val Acc | Best Val Loss | Total Params | Model Size | Train Time (CPU) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | ---: | :---: | :---: |
| 🥇 | **ConvNeXt-Tiny** | **Dual-Pool MLP** | **93.50%** | **0.934** | **0.855** | 94.50% | 0.2475 | 28.6M | 109.2 MB | 41.0 min |
| 🥈 | EfficientNet-B0 | Dual-Pool MLP | 92.00% | 0.918 | 0.872 | 91.00% | 0.3673 | 5.3M | 20.6 MB | 16.1 min |
| 🥈 | EfficientNet-B0 | Cosine | 92.00% | 0.917 | 0.837 | 91.50% | 0.3429 | 4.7M | 18.1 MB | 23.0 min |
| 4th | MobileNetV3-Large | Dual-Pool MLP | 91.50% | 0.914 | 0.866 | 89.00% | 0.4166 | 4.0M | 15.3 MB | 14.2 min |
| 5th | ResNet-50 | Dual-Pool MLP | 84.00% | 0.843 | 0.732 | 89.50% | 0.4131 | 25.6M | 98.1 MB | 31.4 min |

### Key Takeaways

* **ConvNeXt-Tiny + Dual-Pool MLP is the clear winner** at **93.50%** test accuracy. Its modernized convolutional architecture with larger effective receptive fields excels at capturing soil texture patterns, and it held the highest validation accuracy (94.50%) during training.
* **EfficientNet-B0 is the best efficiency pick** — it matches 92% accuracy with only **5.3M parameters** (5.4× fewer than ConvNeXt) and trains in just **16.1 minutes** on CPU.
* **The Cosine head slightly trails Dual-Pool MLP** on the same EfficientNet-B0 backbone (same 92% accuracy but lower macro F1: 0.837 vs 0.872), confirming that dual-pooling captures complementary texture signals.
* **MobileNetV3-Large** delivers **91.50%** accuracy in only **14.2 minutes** with the smallest model size (**15.3 MB**), making it ideal for edge/mobile deployment.
* **ResNet-50 struggled** at 84% — its Phase 2 fine-tuning diverged (val loss spiked to >1.0 before early stop), and it completely failed on Peat Soil (0% recall). Its large parameter count (25.6M) with no modern training-friendly design made it a poor fit for this small dataset.

---

### Per-Class Accuracy Heatmap (All Configurations)

| Soil Type (support) | ConvNeXt-Tiny + DPM | EffNet-B0 + DPM | EffNet-B0 + Cosine | MobileNetV3 + DPM | ResNet-50 + DPM |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Alluvial_Soil** (6) | 50.0% | 66.7% | 66.7% | 66.7% | 50.0% |
| **Arid_Soil** (43) | **97.7%** | 95.3% | 93.0% | 95.3% | **97.7%** |
| **Black Soil** (44) | **97.7%** | **97.7%** | **97.7%** | 90.9% | 77.3% |
| **Cinder Soil** (5) | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| **Laterite Soil** (37) | **91.9%** | 83.8% | **91.9%** | **94.6%** | 83.8% |
| **Mountain_Soil** (30) | **96.7%** | 93.3% | 93.3% | **96.7%** | 90.0% |
| **Peat Soil** (4) | 50.0% | **50.0%** | 25.0% | **50.0%** | 0.0% |
| **Red_Soil** (16) | **100.0%** | **100.0%** | **100.0%** | 87.5% | 81.2% |
| **Yellow Soil** (15) | 86.7% | **93.3%** | 86.7% | 86.7% | 86.7% |

> **Note:** Alluvial_Soil (6 test images) and Peat Soil (4 test images) have very small test supports, so their per-class accuracies are inherently volatile. All models achieve ≥ 50% on these rare classes except ResNet-50 on Peat Soil.

---

### Detailed Per-Class Metrics — 🥇 Best Model (ConvNeXt-Tiny + Dual-Pool MLP)

| Soil Type | Precision | Recall | F1-Score | Correct / Total | Class Accuracy | Erodibility (Risk) |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Alluvial_Soil** | 0.600 | 0.500 | 0.545 | 3 / 6 | 50.0% | 6/10 (Medium-High) |
| **Arid_Soil** | 0.933 | 0.977 | 0.955 | 42 / 43 | 97.7% | 8/10 (High) |
| **Black Soil** | 0.977 | 0.977 | 0.977 | 43 / 44 | 97.7% | 2/10 (Low) |
| **Cinder Soil** | 0.833 | 1.000 | 0.909 | 5 / 5 | **100.0%** | 7/10 (High) |
| **Laterite Soil** | 0.971 | 0.919 | 0.944 | 34 / 37 | 91.9% | 3/10 (Low) |
| **Mountain_Soil** | 0.906 | 0.967 | 0.935 | 29 / 30 | 96.7% | 4/10 (Medium) |
| **Peat Soil** | 0.500 | 0.500 | 0.500 | 2 / 4 | 50.0% | 3/10 (Low) |
| **Red_Soil** | 1.000 | 1.000 | 1.000 | 16 / 16 | **100.0%** | 5/10 (Medium) |
| **Yellow Soil** | 1.000 | 0.867 | 0.929 | 13 / 15 | 86.7% | 5/10 (Medium) |
| **Macro Avg** | **0.858** | **0.856** | **0.855** | — | — | — |
| **Weighted Avg** | **0.935** | **0.935** | **0.934** | **187 / 200** | **93.50%** | — |

---

### Training Dynamics Summary

| Backbone + Head | P1 Best Val Acc | P1 Best Val Loss | P2 Best Val Acc | P2 Best Val Loss | P2 Stopped At | Converged? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ConvNeXt-Tiny + DPM** | 90.50% | 0.3624 | **94.50%** | **0.2475** | Epoch 18 | ✅ Smooth |
| EfficientNet-B0 + DPM | 90.00% | 0.4379 | 91.00% | 0.3673 | Epoch 19 | ✅ Smooth |
| EfficientNet-B0 + Cosine | 89.00% | 0.4048 | 91.50% | 0.3429 | Epoch 25 (full) | ✅ Smooth |
| MobileNetV3-Large + DPM | 88.00% | 0.4892 | 89.00% | 0.4166 | Epoch 22 | ✅ Smooth |
| ResNet-50 + DPM | 89.50% | 0.4131 | 89.50% | 0.4131* | Epoch 10 | ❌ Diverged |

> *ResNet-50's Phase 2 **failed to improve** over Phase 1 — validation loss climbed from 0.47 → 1.04 across 10 epochs before early stopping. The Phase 1 checkpoint was restored as the best model, meaning fine-tuning was effectively skipped.

---

## 📁 Project Structure

```text
Pocket_soil_project/
├── final_dataset/                  # Dataset directory (organized by soil class)
│   ├── Alluvial_Soil/
│   ├── Arid_Soil/
│   ├── Black Soil/
│   ├── Cinder Soil/
│   ├── Laterite Soil/
│   ├── Mountain_Soil/
│   ├── Peat Soil/
│   ├── Red_Soil/
│   └── Yellow Soil/
├── models/
│   └── soil_erodibility_model.pth  # Saved best model checkpoint + metadata
├── results/
│   ├── convnext_tiny dual_pool_mlp/    # 🥇 Best: 93.50% test accuracy
│   │   ├── result.txt
│   │   ├── confusion_matrix.png
│   │   └── training_history.png
│   ├── efficientnet_b0 + dual_pool_mlp/  # 🥈 92.00% test accuracy
│   │   ├── result.txt
│   │   ├── confusion_matrix.png
│   │   └── training_history.png
│   ├── efficientnet_b0 cosine/           # 🥈 92.00% test accuracy
│   │   ├── result.txt
│   │   ├── confusion_matrix.png
│   │   └── training_history.png
│   ├── mobilenet_v3_large dual_pool_mlp/ # 91.50% test accuracy
│   │   ├── result.txt.txt
│   │   ├── confusion_matrix.png
│   │   └── training_history.png
│   ├── resnet50 dual_pool_mlp/           # 84.00% test accuracy
│   │   ├── result.txt.txt
│   │   ├── confusion_matrix.png
│   │   └── training_history.png
│   ├── classification_report.txt   # Save from CNN architecture that you will run
│   ├── confusion_matrix.png        # Save from CNN architecture that you will run
│   ├── training_history.png        # Save from CNN architecture that you will run
│   └── training_results.json       # Save from CNN architecture that you will run
├── main.py                         # Complete training, evaluation, and CLI prediction script
├── requirements.txt                # Python package dependencies
└── README.md                       # Project documentation
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/pocket-geologist.git
cd pocket-geologist
```

### 2. Create a Virtual Environment & Install Dependencies
```bash
python -m venv venv
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

pip install torch torchvision pillow scikit-learn matplotlib numpy
```

Or install via:
```bash
pip install -r requirements.txt
```

### 3. Configure Dataset Path
In `main.py` (under `# 1. CONFIGURATION`), verify that `BASE_DIR` points to your project root:
```python
BASE_DIR        = Path(r'S:\Projects\Pocket_soil_project')
DATASET_DIR     = BASE_DIR / 'final_dataset'
MODEL_SAVE_PATH = BASE_DIR / 'models' / 'soil_erodibility_model.pth'
RESULTS_DIR     = BASE_DIR / 'results'
```

---

## 🚀 Usage

### 1. Train the Default Model (`efficientnet_b0` + `dual_pool_mlp`)
Runs the complete 8-step pipeline (Dataset scan → Stratified 70/15/15 split → Phase 1 head training → Phase 2 differential fine-tuning → Test evaluation with TTA → Plot generation → Checkpoint saving → Demo prediction):
```bash
python main.py
```

### 2. Train with Alternative Backbones & Classifier Heads
You can test different CNN backbones and classifier heads directly from the command line without editing `main.py`:
```bash
# ConvNeXt-Tiny backbone with Dual-Pool MLP head (BEST — 93.50%)
python main.py convnext_tiny dual_pool_mlp

# EfficientNet-B0 with Cosine Similarity Classifier head
python main.py efficientnet_b0 cosine

# ResNet-50 with Dual-Pool MLP head
python main.py resnet50 dual_pool_mlp

# MobileNetV3-Large for ultra-fast CPU inference
python main.py mobilenet_v3_large dual_pool_mlp
```

### 3. Predict Erodibility on a Single Image
Once trained, run standalone inference on any soil photograph:
```bash
python main.py predict path/to/soil_image.jpg
```

**Sample CLI Output:**
```text
  +============================================================+
  |        POCKET GEOLOGIST  -  Erodibility Prediction         |
  +------------------------------------------------------------+
  |  Image: sample_soil.jpg                                    |
  +------------------------------------------------------------+
  |  Predicted Soil Type : Mountain_Soil                       |
  |  Confidence          :  93.30%                             |
  +------------------------------------------------------------+
  |  Erodibility Score   : 4/10                                |
  |  K-Factor (USDA)     : 0.30                                |
  |  Risk Level          : Medium                              |
  |  Risk Meter          : [MED] [####------]                  |
  +------------------------------------------------------------+
  |  Weighted Score (probability-adjusted): 4.12/10            |
  +------------------------------------------------------------+
  |  Top 3 Predictions:                                        |
  |    1. Mountain_Soil         93.30%   (erodibility: 4/10)   |
  |    2. Arid_Soil              4.50%   (erodibility: 8/10)   |
  |    3. Laterite Soil          1.20%   (erodibility: 3/10)   |
  +============================================================+
```

---

## ⚙️ Hyperparameter Configuration

| Parameter | Default Value | Description |
| :--- | :---: | :--- |
| `MODEL_NAME` | `'efficientnet_b0'` | Pretrained CNN backbone (`efficientnet_b0`, `convnext_tiny`, `resnet50`, `mobilenet_v3_large`) |
| `HEAD_TYPE` | `'dual_pool_mlp'` | Classifier architecture (`dual_pool_mlp`, `cosine`, `mlp`) |
| `IMAGE_SIZE` / `RESIZE_SIZE` | `224` / `256` | Input crop resolution and pre-crop resize dimension |
| `BATCH_SIZE` | `32` | Mini-batch size for training and evaluation |
| `NUM_EPOCHS` | `15` | Phase 1 epochs (frozen backbone, classifier warmup) |
| `FINE_TUNE_EPOCHS` | `25` | Phase 2 max epochs (last 3 backbone blocks unfrozen) |
| `LEARNING_RATE` | `1e-3` | Phase 1 AdamW learning rate |
| `FINE_TUNE_LR_BB` | `1e-4` | Phase 2 learning rate for unfrozen backbone blocks |
| `FINE_TUNE_LR_HD` | `3e-4` | Phase 2 learning rate for classifier head |
| `WEIGHT_DECAY` | `1e-4` | AdamW $L_2$ weight decay regularization |
| `LABEL_SMOOTHING` | `0.05` | Cross-entropy label smoothing to prevent logit overconfidence |
| `SAMPLER_POWER` | `0.5` | Dampening exponent ($1 / N^{0.5}$) for `WeightedRandomSampler` |
| `PATIENCE` | `10` | Early stopping patience based on validation accuracy & loss |
| `USE_TTA` | `True` | Averages original and horizontally flipped predictions at test time |

---

## 📜 License

This project is licensed under the **MIT License**.
