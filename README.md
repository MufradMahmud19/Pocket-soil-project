# 🌍 Pocket Geologist: Soil Erodibility Prediction from Images

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![Backbone](https://img.shields.io/badge/Backbone-EfficientNet--B0-success)
![Test Accuracy](https://img.shields.io/badge/Test%20Accuracy-92.00%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**Pocket Geologist** is a deep learning computer vision system built with PyTorch that predicts **soil erodibility** directly from field photographs. It classifies soil samples across **9 distinct soil types**, maps them to physical **USDA K-Factors** ($0.02 \text{ to } 0.69$), and computes both discrete and probability-weighted **Erodibility Risk Scores (`1/10` to `10/10`)**.

Using a two-phase transfer learning pipeline, a **Dual-Pooling MLP** classifier head, dampened class-imbalance sampling, and Test-Time Augmentation (TTA), the system achieves **92.00% test accuracy** (`0.918` weighted F1-score, `0.3673` validation loss).

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
│   └── soil_erodibility_model.pth  # Saved best model checkpoint + metadata (20.6 MB)
├── results/
│   ├── classification_report.txt   # Per-class Precision, Recall, F1 report
│   ├── confusion_matrix.png        # Saved confusion matrix heatmap
│   ├── training_history.png        # Phase 1 + Phase 2 Loss & Accuracy curves
│   └── training_results.json       # Run metadata and accuracy summary
├── main.py                         # Complete training, evaluation, and CLI prediction script
├── requirements.txt                # Python package dependencies
└── README.md                       # Project documentation
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/pocket-geologist.git](https://github.com/YOUR_USERNAME/pocket-geologist.git)
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

Or create a `requirements.txt` file with:
```text
torch>=2.0.0
torchvision>=0.15.0
Pillow>=9.0.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
numpy>=1.21.0
```
And install via:
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
Runs the complete 8-step pipeline (Dataset scan $\rightarrow$ Stratified 70/15/15 split $\rightarrow$ Phase 1 head training $\rightarrow$ Phase 2 differential fine-tuning $\rightarrow$ Test evaluation with TTA $\rightarrow$ Plot generation $\rightarrow$ Checkpoint saving $\rightarrow$ Demo prediction):
```bash
python main.py
```

### 2. Train with Alternative Backbones & Classifier Heads
You can test different CNN backbones and classifier heads directly from the command line without editing `main.py`:
```bash
# ConvNeXt-Tiny backbone with Dual-Pool MLP head
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

## 📊 Experimental Results

### Training Progression (`efficientnet_b0` + `dual_pool_mlp`)
* **Total Parameters:** `5,329,541` (`1,321,993` trainable in Phase 1; `4,451,877` trainable in Phase 2)
* **Phase 1 Best Validation Accuracy:** `90.00%` (Lowest Val Loss: `0.4379` at Epoch 14)
* **Phase 2 Best Validation Accuracy:** `91.00%` (Lowest Val Loss: `0.3673` at Epoch 9, early stopped at Epoch 19)
* **Final Test Set Accuracy (with TTA):** **`92.00%`** (`184 / 200` test images correct)
* **Total Training Time (CPU):** `16.1 minutes`

### Per-Class Test Set Performance (200 Test Images)

| Soil Type | Precision | Recall | F1-Score | Correct / Total | Class Accuracy | Erodibility (Risk) |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Alluvial_Soil** | 0.800 | 0.667 | 0.727 | 4 / 6 | 66.7% | 6/10 (Medium-High) |
| **Arid_Soil** | 0.932 | 0.953 | 0.943 | 41 / 43 | 95.3% | 8/10 (High) |
| **Black Soil** | 0.977 | 0.977 | 0.977 | 43 / 44 | 97.7% | 2/10 (Low) |
| **Cinder Soil** | 0.833 | 1.000 | 0.909 | 5 / 5 | **100.0%** | 7/10 (High) |
| **Laterite Soil** | 0.912 | 0.838 | 0.873 | 31 / 37 | 83.8% | 3/10 (Low) |
| **Mountain_Soil** | 0.933 | 0.933 | 0.933 | 28 / 30 | 93.3% | 4/10 (Medium) |
| **Peat Soil** | 1.000 | 0.500 | 0.667 | 2 / 4 | 50.0% | 3/10 (Low) |
| **Red_Soil** | 0.800 | 1.000 | 0.889 | 16 / 16 | **100.0%** | 5/10 (Medium) |
| **Yellow Soil** | 0.933 | 0.933 | 0.933 | 14 / 15 | 93.3% | 5/10 (Medium) |
| **Macro Avg** | **0.902** | **0.867** | **0.872** | — | — | — |
| **Weighted Avg** | **0.923** | **0.920** | **0.918** | **184 / 200** | **92.00%** | — |

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
