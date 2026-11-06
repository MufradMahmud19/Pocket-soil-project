# POCKET GEOLOGIST (Soil erodibility prediction from images)

import os
import sys
import json
import math
import random
import time
import numpy as np
from pathlib import Path
from collections import Counter
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


# 1. CONFIGURATION

# Paths
BASE_DIR = Path(r'S:\Projects\Pocket_soil_project')
DATASET_DIR = BASE_DIR / 'final_dataset'
MODEL_SAVE_PATH = BASE_DIR / 'models' / 'soil_erodibility_model.pth'
RESULTS_DIR = BASE_DIR / 'results'

# Model & Classifier Architecture Options:
#   MODEL_NAME : 'efficientnet_b0' | 'convnext_tiny' | 'resnet50' | 'mobilenet_v3_large'
#   HEAD_TYPE  : 'dual_pool_mlp'   | 'cosine'        | 'mlp'
MODEL_NAME = 'efficientnet_b0'
HEAD_TYPE = 'dual_pool_mlp'

# Training hyperparameters
IMAGE_SIZE = 224        # Standard input size
RESIZE_SIZE = 256        # Pre-crop resize for consistent grain scale
BATCH_SIZE = 32
NUM_EPOCHS = 15         # Phase 1: train classifier head only
FINE_TUNE_EPOCHS = 25         # Phase 2: unfreeze backbone + fine-tune
LEARNING_RATE = 1e-3       # Phase 1 LR (head only)
FINE_TUNE_LR_BB = 1e-4       # Phase 2 LR (unfrozen backbone blocks)
FINE_TUNE_LR_HD = 3e-4       # Phase 2 LR (classifier head)
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05       # Prevents overconfident logit spikes
SAMPLER_POWER = 0.5        # 0.5 = sqrt dampening (avoids extreme oversampling)
PATIENCE = 10         # Early stopping patience
USE_TTA = True       # Test-Time Augmentation (orig + hflip)
NUM_WORKERS = 0          # Set 0 for Windows compatibility
RANDOM_SEED = 42

# Device selection
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PIN_MEMORY = torch.cuda.is_available()


# 2. SOIL ERODIBILITY KNOWLEDGE BASE (USDA K-Factor Mapping)

# Each soil type is mapped to its USDA K-factor and an erodibility score
# on a 1-10 scale.  Formula: score = round((k_factor / 0.69) * 9) + 1
# K-factor ranges from 0.02 (very resistant) to 0.69 (very erodible).

SOIL_ERODIBILITY = {
    'Alluvial_Soil': {'k_factor': 0.40, 'score': 6, 'risk': 'Medium-High'},
    'Arid_Soil':     {'k_factor': 0.52, 'score': 8, 'risk': 'High'},
    'Black Soil':    {'k_factor': 0.17, 'score': 2, 'risk': 'Low'},
    'Cinder Soil':   {'k_factor': 0.47, 'score': 7, 'risk': 'High'},
    'Laterite Soil': {'k_factor': 0.15, 'score': 3, 'risk': 'Low'},
    'Mountain_Soil': {'k_factor': 0.30, 'score': 4, 'risk': 'Medium'},
    'Peat Soil':     {'k_factor': 0.20, 'score': 3, 'risk': 'Low'},
    'Red_Soil':      {'k_factor': 0.35, 'score': 5, 'risk': 'Medium'},
    'Yellow Soil':   {'k_factor': 0.30, 'score': 5, 'risk': 'Medium'},
}



# 3. UTILITY FUNCTIONS

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_header(title):
    width = 70
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")



# 4. DATASET & DATA LOADING

class SoilDataset(Dataset):
    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"  [Warning] Could not load {img_path}: {e}")
            image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, label

    @staticmethod
    def scan_dataset(dataset_dir):
        dataset_dir = Path(dataset_dir)
        class_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
        class_names = [d.name for d in class_dirs]
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        image_paths = []
        labels = []

        for class_dir in class_dirs:
            class_idx = class_to_idx[class_dir.name]
            for img_file in sorted(class_dir.iterdir()):
                if img_file.suffix.lower() in SoilDataset.VALID_EXTENSIONS:
                    image_paths.append(str(img_file))
                    labels.append(class_idx)

        return image_paths, labels, class_names, class_to_idx


def get_transforms(is_training=True):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    if is_training:
        return transforms.Compose([
            transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.10, hue=0.01),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.12)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])


def create_weighted_sampler(labels, power=SAMPLER_POWER):
    class_counts = Counter(labels)

    # Dampened weight per class: 1 / (count ^ power)
    class_weights = {
        cls: 1.0 / (count ** power)
        for cls, count in class_counts.items()
    }

    sample_weights = torch.DoubleTensor(
        [class_weights[label] for label in labels]
    )

    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True,)


def compute_mild_class_weights(labels, num_classes, power=0.25):
    class_counts = Counter(labels)
    raw_weights = np.array([
        1.0 / (class_counts[idx] ** power)
        for idx in range(num_classes)
    ], dtype=np.float32)

    norm_weights = raw_weights / raw_weights.mean()
    return torch.FloatTensor(norm_weights).to(DEVICE)

# 5. MODEL & CLASSIFIER DEFINITIONS

class CosineClassifier(nn.Module):
    def __init__(self, in_features, num_classes, init_scale=16.0):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(num_classes, in_features))
        self.scale  = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        return self.scale * F.linear(x_norm, w_norm)


class SoilClassifierNet(nn.Module):
    def __init__(self, num_classes, model_name=MODEL_NAME,
                 head_type=HEAD_TYPE, freeze_backbone=True):
        super().__init__()
        self.model_name = model_name.lower()
        self.head_type  = head_type.lower()

        # 1. Load pretrained feature extractor
        if self.model_name == 'efficientnet_b0':
            base = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
            )
            self.features = base.features
            self.blocks   = list(self.features.children())
            in_features   = 1280

        elif self.model_name == 'convnext_tiny':
            base = models.convnext_tiny(
                weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
            )
            self.features = base.features
            self.blocks   = list(self.features.children())
            in_features   = 768

        elif self.model_name == 'resnet50':
            base = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V2
            )
            self.features = nn.Sequential(
                base.conv1, base.bn1, base.relu, base.maxpool,
                base.layer1, base.layer2, base.layer3, base.layer4,
            )
            self.blocks = [
                nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool),
                base.layer1, base.layer2, base.layer3, base.layer4,
            ]
            in_features = 2048

        elif self.model_name == 'mobilenet_v3_large':
            base = models.mobilenet_v3_large(
                weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
            )
            self.features = base.features
            self.blocks   = list(self.features.children())
            in_features   = 960

        else:
            raise ValueError(f"Unsupported MODEL_NAME: {model_name}")

        # Freeze / unfreeze backbone
        for param in self.features.parameters():
            param.requires_grad = not freeze_backbone

        # 2. Build pooling & classifier head
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        if self.head_type == 'dual_pool_mlp':
            # Dual pooling captures both average color (AvgPool) and
            # sharp grain/cinder/crack edges (MaxPool)
            pool_dim = in_features * 2
            self.classifier = nn.Sequential(
                nn.BatchNorm1d(pool_dim),
                nn.Dropout(p=0.35),
                nn.Linear(pool_dim, 512),
                nn.BatchNorm1d(512),
                nn.SiLU(inplace=True),
                nn.Dropout(p=0.25),
                nn.Linear(512, num_classes),
            )
        elif self.head_type == 'cosine':
            self.classifier = nn.Sequential(
                nn.BatchNorm1d(in_features),
                nn.Dropout(p=0.30),
                nn.Linear(in_features, 512),
                nn.BatchNorm1d(512),
                nn.SiLU(inplace=True),
                CosineClassifier(512, num_classes),
            )
        elif self.head_type == 'mlp':
            self.classifier = nn.Sequential(
                nn.BatchNorm1d(in_features),
                nn.Dropout(p=0.30),
                nn.Linear(in_features, 512),
                nn.BatchNorm1d(512),
                nn.SiLU(inplace=True),
                nn.Dropout(p=0.20),
                nn.Linear(512, num_classes),
            )
        else:
            raise ValueError(f"Unsupported HEAD_TYPE: {head_type}")

    def forward(self, x):
        feat_map = self.features(x)
        avg_f = torch.flatten(self.avg_pool(feat_map), 1)

        if self.head_type == 'dual_pool_mlp':
            max_f = torch.flatten(self.max_pool(feat_map), 1)
            pooled = torch.cat([avg_f, max_f], dim=1)
        else:
            pooled = avg_f

        return self.classifier(pooled)


def build_model(num_classes, model_name=MODEL_NAME, head_type=HEAD_TYPE, freeze_backbone=True):
    model = SoilClassifierNet(num_classes=num_classes, model_name=model_name, head_type=head_type, freeze_backbone=freeze_backbone,)
    return model.to(DEVICE)


def freeze_backbone_batchnorm(model):
    for module in model.features.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False


def unfreeze_backbone(model, num_blocks_to_unfreeze=3):
    for param in model.features.parameters():
        param.requires_grad = False

    total_blocks = len(model.blocks)
    n_unfreeze = min(num_blocks_to_unfreeze, total_blocks)
    start_idx = total_blocks - n_unfreeze

    for i in range(start_idx, total_blocks):
        for param in model.blocks[i].parameters():
            param.requires_grad = True

    # Freeze BatchNorm2d parameters inside unfrozen blocks
    freeze_backbone_batchnorm(model)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Unfroze last {n_unfreeze}/{total_blocks} backbone blocks "
          f"(BatchNorm2d kept frozen)")
    print(f"  Trainable parameters: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.1f}%)")

# 6. TRAINING & VALIDATION

def train_one_epoch(model, loader, criterion, optimizer, keep_bn_frozen=False):
    model.train()
    if keep_bn_frozen:
        freeze_backbone_batchnorm(model)

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping prevents sudden loss spikes during fine-tuning
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total   += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def validate(model, loader, criterion, use_tta=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        if use_tta:
            outputs_flipped = model(torch.flip(images, dims=[3]))
            outputs = 0.5 * (outputs + outputs_flipped)

        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total   += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, 100.0 * correct / total


def train_model(model, train_loader, val_loader, train_criterion, val_criterion,
                optimizer, scheduler, num_epochs, phase_name="Training",
                keep_bn_frozen=False, initial_best_acc=0.0, initial_best_loss=float('inf')):

    best_val_acc  = initial_best_acc
    best_val_loss = initial_best_loss
    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    patience_counter = 0

    history = {'train_loss': [], 'train_acc': [], 'val_loss':   [], 'val_acc':   [],}

    header = (f"  {'Epoch':<8}{'Train Loss':<13}{'Train Acc':<13}"
              f"{'Val Loss':<13}{'Val Acc':<13}{'Status'}")
    print(f"\n{header}")
    print(f"  {'-' * 62}")

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, train_criterion, optimizer,
            keep_bn_frozen=keep_bn_frozen,
        )
        val_loss, val_acc = validate(model, val_loader, val_criterion)

        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Save checkpoint if Val Acc improves, or if Val Acc ties with lower Val Loss
        status = ""
        improved = (val_acc > best_val_acc) or (
            math.isclose(val_acc, best_val_acc, abs_tol=1e-4) and val_loss < best_val_loss
        )

        if improved:
            best_val_acc  = val_acc
            best_val_loss = min(best_val_loss, val_loss)
            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            patience_counter = 0
            status = "<-- Best"
        else:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            patience_counter += 1

        print(f"  {epoch:<8}{train_loss:<13.4f}{train_acc:<13.2f}"
              f"{val_loss:<13.4f}{val_acc:<13.2f}{status}")

        if patience_counter >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(no improvement for {PATIENCE} epochs)")
            break

    # Restore best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"\n  Best validation accuracy ({phase_name}): {best_val_acc:.2f}% "
          f"(Lowest Val Loss: {best_val_loss:.4f})")
    return history, best_val_acc, best_val_loss



# 7. EVALUATION & VISUALIZATION

@torch.no_grad()
def evaluate_model(model, test_loader, class_names, results_dir, use_tta=USE_TTA):
    model.eval()
    all_preds  = []
    all_labels = []

    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        if use_tta:
            # Test-Time Augmentation: average predictions of original + horizontal flip
            outputs_flip = model(torch.flip(images, dims=[3]))
            outputs = 0.5 * (outputs + outputs_flip)

        _, predicted = outputs.max(1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Classification report
    print(f"\n  Classification Report (TTA={'Enabled' if use_tta else 'Disabled'}):")
    print("  " + "-" * 66)
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=3, zero_division=0)
    for line in report.split('\n'):
        print(f"  {line}")

    # Confusion matrix 
    cm = confusion_matrix(all_labels, all_preds)
    _plot_confusion_matrix(cm, class_names, results_dir)

    # Erodibility analysis
    print("\n  Erodibility-Score Analysis (Test Set):")
    print("  " + "-" * 66)
    print(f"  {'Soil Type':<20}{'Correct':<10}{'Total':<10}"
          f"{'Accuracy':<12}{'Erodibility'}")
    print(f"  {'-' * 66}")

    for idx, name in enumerate(class_names):
        mask = (all_labels == idx)
        if mask.sum() > 0:
            correct = (all_preds[mask] == idx).sum()
            total = mask.sum()
            acc = 100.0 * correct / total
            score = SOIL_ERODIBILITY.get(name, {}).get('score', '?')
            risk = SOIL_ERODIBILITY.get(name, {}).get('risk', '?')
            print(f"  {name:<20}{correct:<10}{total:<10}"
                  f"{acc:<12.1f}{score}/10 ({risk})")

    overall_acc = 100.0 * (all_preds == all_labels).sum() / len(all_labels)
    print(f"\n  Overall Test Accuracy: {overall_acc:.2f}%")

    return overall_acc, report


def _plot_confusion_matrix(cm, class_names, results_dir):
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(f'Confusion Matrix - Soil Type Classification ({MODEL_NAME})', fontsize=14)
    plt.colorbar(im, ax=ax)

    short = [n.replace('_Soil', '').replace(' Soil', '').replace('_', ' ') for n in class_names]
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_xticklabels(short, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(ticks)
    ax.set_yticklabels(short, fontsize=10)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black',
                    fontsize=10)

    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    save_path = Path(results_dir) / 'confusion_matrix.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Confusion matrix saved to: {save_path}")


def _plot_training_history(history_p1, history_p2, results_dir):
    train_loss = history_p1['train_loss'] + history_p2['train_loss']
    val_loss   = history_p1['val_loss']   + history_p2['val_loss']
    train_acc  = history_p1['train_acc']  + history_p2['train_acc']
    val_acc    = history_p1['val_acc']    + history_p2['val_acc']
    p1_len = len(history_p1['train_loss'])
    total  = len(train_loss)
    epochs = range(1, total + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(epochs, train_loss, 'b-', label='Train Loss')
    ax1.plot(epochs, val_loss, 'r-', label='Val Loss')
    ax1.axvline(x=p1_len, color='green', linestyle='--', alpha=0.7, label=f'Fine-tune start (epoch {p1_len})')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, train_acc, 'b-', label='Train Acc')
    ax2.plot(epochs, val_acc, 'r-', label='Val Acc')
    ax2.axvline(x=p1_len, color='green', linestyle='--', alpha=0.7, label=f'Fine-tune start (epoch {p1_len})')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training & Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'Pocket Geologist - Training History ({MODEL_NAME} + {HEAD_TYPE})', fontsize=14, y=1.02)
    plt.tight_layout()

    save_path = Path(results_dir) / 'training_history.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to: {save_path}")


# 8. PREDICTION

def predict_erodibility(model, image_path, class_names, use_tta=USE_TTA):
    model.eval()
    transform = get_transforms(is_training=False)

    # Load and preprocess
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Inference (with optional horizontal flip TTA)
    with torch.no_grad():
        outputs = model(input_tensor)
        if use_tta:
            outputs_flip = model(torch.flip(input_tensor, dims=[3]))
            outputs = 0.5 * (outputs + outputs_flip)
        probabilities = F.softmax(outputs, dim=1)[0]

    probs = probabilities.cpu().numpy()

    # Top-3 predictions
    top3_idx = np.argsort(probs)[::-1][:3]
    pred_idx = top3_idx[0]
    pred_class = class_names[pred_idx]
    pred_conf = probs[pred_idx] * 100

    # Look up erodibility for the top prediction
    erod = SOIL_ERODIBILITY.get(pred_class, {'k_factor': 0.30, 'score': 5, 'risk': 'Unknown'})

    # Weighted erodibility: uses all class probabilities for a softer score
    weighted_score = sum(
        probs[i] * SOIL_ERODIBILITY.get(cn, {'score': 5})['score']
        for i, cn in enumerate(class_names)
    )

    # Risk-level indicator
    score = erod['score']
    bar = '#' * score + '-' * (10 - score)
    if score <= 3:
        risk_icon = "[LOW]"
    elif score <= 6:
        risk_icon = "[MED]"
    else:
        risk_icon = "[HIGH]"

    # Display results
    img_name = Path(image_path).name
    print()
    print(f"  +{'=' * 60}+")
    print(f"  |{'POCKET GEOLOGIST  -  Erodibility Prediction':^60}|")
    print(f"  +{'-' * 60}+")
    print(f"  |  Image: {img_name:<50}|")
    print(f"  +{'-' * 60}+")
    print(f"  |  Predicted Soil Type : {pred_class:<35}|")
    print(f"  |  Confidence          : {pred_conf:>6.2f}%{' ' * 28}|")
    print(f"  +{'-' * 60}+")
    print(f"  |  Erodibility Score   : {erod['score']}/10{' ' * 34}|")
    print(f"  |  K-Factor (USDA)     : {erod['k_factor']:.2f}{' ' * 35}|")
    print(f"  |  Risk Level          : {erod['risk']:<35}|")
    print(f"  |  Risk Meter          : {risk_icon} [{bar}]{' ' * (22 - len(risk_icon))}|")
    print(f"  +{'-' * 60}+")
    print(f"  |  Weighted Score (probability-adjusted): "
          f"{weighted_score:.2f}/10{' ' * 12}|")
    print(f"  +{'-' * 60}+")
    print(f"  |  Top 3 Predictions:{' ' * 39}|")
    for rank, idx in enumerate(top3_idx, 1):
        name = class_names[idx]
        conf = probs[idx] * 100
        s = SOIL_ERODIBILITY.get(name, {'score': '?'})['score']
        line = f"    {rank}. {name:<20} {conf:>6.2f}%   (erodibility: {s}/10)"
        print(f"  |{line:<60}|")
    print(f"  +{'=' * 60}+")

    return {
        'predicted_soil_type': pred_class,
        'confidence': round(pred_conf, 2),
        'erodibility_score': erod['score'],
        'k_factor': erod['k_factor'],
        'risk_level': erod['risk'],
        'weighted_score': round(weighted_score, 2),
        'top3': [
            {
                'class': class_names[i],
                'confidence': round(float(probs[i]) * 100, 2),
                'erodibility_score': SOIL_ERODIBILITY.get(
                    class_names[i], {'score': 5}
                )['score'],
            }
            for i in top3_idx
        ],
    }


# 9. MAIN PIPELINE

def main(model_name=MODEL_NAME, head_type=HEAD_TYPE):
    start_time = time.time()
    set_seed(RANDOM_SEED)

    print_header("POCKET GEOLOGIST - Soil Erodibility Prediction System")
    print(f"  Device      : {DEVICE}")
    print(f"  Backbone    : {model_name}")
    print(f"  Head Type   : {head_type}")
    print(f"  Timestamp   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Create output directories
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


    # STEP 1 : Load & split dataset

    print_header("STEP 1: Loading Dataset")

    image_paths, labels, class_names, class_to_idx = \
        SoilDataset.scan_dataset(DATASET_DIR)

    num_classes = len(class_names)
    print(f"Dataset directory : {DATASET_DIR}")
    print(f"Total images : {len(image_paths)}")
    print(f"Number of classes : {num_classes}")

    class_counts = Counter(labels)
    print(f"\nClass distribution:")
    for idx, name in enumerate(class_names):
        count = class_counts[idx]
        erod = SOIL_ERODIBILITY.get(name, {})
        score = erod.get('score', '?')
        bar = '#' * (count // 10)
        print(f"    {name:<20} {count:>4} images  "
              f"(erodibility: {score}/10)  {bar}")

    # Stratified split: 70 / 15 / 15
    print(f"\n  Splitting data (70% train / 15% val / 15% test) ...")

    train_val_paths, test_paths, train_val_labels, test_labels = \
        train_test_split(image_paths, labels, test_size=0.15, random_state=RANDOM_SEED, stratify=labels)
    train_paths, val_paths, train_labels, val_labels = \
        train_test_split(train_val_paths, train_val_labels,
                         test_size=0.176,           # 0.176 of 85% ~ 15%
                         random_state=RANDOM_SEED,
                         stratify=train_val_labels)

    print(f"  Training set   : {len(train_paths):>5} images")
    print(f"  Validation set : {len(val_paths):>5} images")
    print(f"  Test set       : {len(test_paths):>5} images")


    # STEP 2 : Create DataLoaders

    print_header("STEP 2: Creating DataLoaders")
    train_dataset = SoilDataset(train_paths, train_labels, transform=get_transforms(is_training=True))
    val_dataset   = SoilDataset(val_paths, val_labels, transform=get_transforms(is_training=False))
    test_dataset  = SoilDataset(test_paths, test_labels, transform=get_transforms(is_training=False))
    sampler = create_weighted_sampler(train_labels, power=SAMPLER_POWER)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    print(f"  Train batches/epoch : {len(train_loader)}")
    print(f"  Val batches         : {len(val_loader)}")
    print(f"  Test batches        : {len(test_loader)}")
    print(f"  Weighted sampling   : Dampened (power={SAMPLER_POWER}, prevents overfitting)")


    # STEP 3 : Build model

    print_header(f"STEP 3: Building Model ({model_name} + {head_type})")

    model = build_model(num_classes, model_name=model_name, head_type=head_type, freeze_backbone=True,)
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Architecture      : {model_name} (ImageNet pretrained)")
    print(f"  Classifier head   : {head_type}")
    print(f"  Total parameters  : {total_params:>12,}")
    print(f"  Trainable params  : {trainable_params:>12,} "
          f"({100 * trainable_params / total_params:.1f}%)")
    print(f"  Backbone status   : FROZEN (Phase 1)")

    # Mild class weights + label smoothing for training; unweighted for validation
    mild_weights    = compute_mild_class_weights(train_labels, num_classes, power=0.20)
    train_criterion = nn.CrossEntropyLoss(weight=mild_weights, label_smoothing=LABEL_SMOOTHING)
    val_criterion   = nn.CrossEntropyLoss()


    # STEP 4 : Phase 1 — Train classifier head (backbone frozen)

    print_header("STEP 4: Phase 1 - Training Classifier Head (backbone frozen)")
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)

    history_p1, best_p1, best_loss_p1 = train_model(
        model, train_loader, val_loader, train_criterion, val_criterion,
        optimizer, scheduler, NUM_EPOCHS, phase_name="Phase 1",
        keep_bn_frozen=True,
    )


    # STEP 5 : Phase 2 Fine-tune backbone with differential learning rates

    print_header("STEP 5: Phase 2 - Fine-tuning Backbone (differential LR)")

    unfreeze_backbone(model, num_blocks_to_unfreeze=3)

    # Separate learning rates: smaller LR for pretrained backbone, slightly higher for head
    backbone_params = [p for p in model.features.parameters() if p.requires_grad]
    head_params = [p for p in model.classifier.parameters() if p.requires_grad]

    optimizer_ft = optim.AdamW([
        {'params': backbone_params, 'lr': FINE_TUNE_LR_BB},
        {'params': head_params,     'lr': FINE_TUNE_LR_HD},
    ], weight_decay=WEIGHT_DECAY)

    scheduler_ft = optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=FINE_TUNE_EPOCHS, eta_min=1e-6)
    history_p2, best_p2, best_loss_p2 = train_model(model, train_loader, val_loader, train_criterion, val_criterion,
        optimizer_ft, scheduler_ft, FINE_TUNE_EPOCHS, phase_name="Phase 2",
        keep_bn_frozen=True,
        initial_best_acc=best_p1, initial_best_loss=best_loss_p1,
    )


    # STEP 6 : Evaluate on test set

    print_header("STEP 6: Evaluating on Test Set")
    test_acc, report = evaluate_model(model, test_loader, class_names, RESULTS_DIR, use_tta=USE_TTA)

    # Training history plots
    _plot_training_history(history_p1, history_p2, RESULTS_DIR)

    # STEP 7 : Save model & metadata

    print_header("STEP 7: Saving Model")
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_name':       model_name,
        'head_type':        head_type,
        'class_names':      class_names,
        'class_to_idx':     class_to_idx,
        'soil_erodibility': SOIL_ERODIBILITY,
        'image_size':       IMAGE_SIZE,
        'num_classes':      num_classes,
        'test_accuracy':    test_acc,
        'training_config': {
            'batch_size':      BATCH_SIZE,
            'phase1_epochs':   NUM_EPOCHS,
            'phase2_epochs':   FINE_TUNE_EPOCHS,
            'phase1_lr':       LEARNING_RATE,
            'phase2_lr_bb':    FINE_TUNE_LR_BB,
            'phase2_lr_head':  FINE_TUNE_LR_HD,
            'label_smoothing': LABEL_SMOOTHING,
            'sampler_power':   SAMPLER_POWER,
            'seed':            RANDOM_SEED,
        },
    }

    torch.save(checkpoint, MODEL_SAVE_PATH)
    model_mb = MODEL_SAVE_PATH.stat().st_size / (1024 * 1024)
    print(f"  Model saved to : {MODEL_SAVE_PATH}")
    print(f"  File size      : {model_mb:.1f} MB")

    # Save results summary as JSON
    results_summary = {
        'timestamp':           datetime.now().isoformat(),
        'device':              str(DEVICE),
        'model_name':          model_name,
        'head_type':           head_type,
        'dataset_size':        len(image_paths),
        'num_classes':         num_classes,
        'class_names':         class_names,
        'test_accuracy':       test_acc,
        'phase1_best_val_acc': best_p1,
        'phase2_best_val_acc': best_p2,
        'best_val_loss':       best_loss_p2,
    }
    results_json = RESULTS_DIR / 'training_results.json'
    with open(results_json, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"  Results JSON   : {results_json}")

    # Save the text classification report
    report_path = RESULTS_DIR / 'classification_report.txt'
    with open(report_path, 'w') as f:
        f.write(f"Pocket Geologist - Classification Report\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {DEVICE}\n")
        f.write(f"Backbone: {model_name} | Head: {head_type}\n")
        f.write(f"Test Accuracy: {test_acc:.2f}%\n\n")
        f.write(report)
    print(f"  Report saved   : {report_path}")


    # STEP 8 : Demo prediction

    print_header("STEP 8: Demo Prediction (random test image)")
    demo_idx  = random.randint(0, len(test_paths) - 1)
    demo_path = test_paths[demo_idx]
    demo_true = class_names[test_labels[demo_idx]]
    print(f"True label: {demo_true}")
    predict_erodibility(model, demo_path, class_names, use_tta=USE_TTA)


    elapsed = time.time() - start_time
    print_header("TRAINING COMPLETE")
    print(f"  Total time      : {elapsed / 60:.1f} minutes")
    print(f"  Test accuracy   : {test_acc:.2f}%")
    print(f"  Model location  : {MODEL_SAVE_PATH}")
    print(f"  Results folder  : {RESULTS_DIR}")
    print()
    print(f"  To predict erodibility on a new image, run:")
    print(f"    python main.py predict <path_to_soil_image>")
    print()


def predict_mode(image_path):
    if not MODEL_SAVE_PATH.exists():
        print(f"\n[Error] No trained model found at {MODEL_SAVE_PATH}")
        print(f"Run 'python main.py' first to train the model.")
        sys.exit(1)

    image_path = Path(image_path)
    if not image_path.exists():
        print(f"\n[Error] Image not found: {image_path}")
        sys.exit(1)

    print_header("POCKET GEOLOGIST - Single Image Prediction")
    print(f"  Loading model from: {MODEL_SAVE_PATH}")

    # Load checkpoint
    checkpoint  = torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=False)
    class_names = checkpoint['class_names']
    num_classes = checkpoint['num_classes']
    model_name  = checkpoint.get('model_name', MODEL_NAME)
    head_type   = checkpoint.get('head_type', HEAD_TYPE)

    # Rebuild and load model
    model = build_model(num_classes, model_name=model_name, head_type=head_type, freeze_backbone=False,)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded successfully ({model_name} + {head_type}, " f"test accuracy was {checkpoint['test_accuracy']:.2f}%)")

    result = predict_erodibility(model, str(image_path), class_names, use_tta=USE_TTA)
    return result



# ENTRY POINT

if __name__ == '__main__':
    if len(sys.argv) >= 3 and sys.argv[1].lower() == 'predict':
        predict_mode(sys.argv[2])
    elif len(sys.argv) >= 2 and sys.argv[1].lower() in {'efficientnet_b0', 'convnext_tiny', 'resnet50', 'mobilenet_v3_large'}:
        chosen_model = sys.argv[1].lower()
        chosen_head  = sys.argv[2].lower() if len(sys.argv) >= 3 else HEAD_TYPE
        main(model_name=chosen_model, head_type=chosen_head)
    else:
        main()
