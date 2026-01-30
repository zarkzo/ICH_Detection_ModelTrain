# Intracranial Hemorrhage Detection - Training Pipeline

Comprehensive guide to train the intracranial hemorrhage detection model from preprocessing to model training using the [RSNA Intracranial Hemorrhage Detection dataset](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection).

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Pipeline Workflow](#pipeline-workflow)
- [Step-by-Step Guide](#step-by-step-guide)
- [Model Architecture](#model-architecture)
- [Training Configuration](#training-configuration)
- [Results and Evaluation](#results-and-evaluation)

---

## Overview

This project implements a cascade deep learning model for multi-label classification of intracranial hemorrhages from CT scan images. The pipeline processes DICOM files, applies windowing techniques, performs data augmentation, and trains an ensemble model combining EfficientNetV2 and ConvNeXt architectures.

**Hemorrhage Types Detected:**
- Any hemorrhage (binary indicator)
- Epidural
- Intraparenchymal
- Intraventricular
- Subarachnoid
- Subdural

---

## Requirements

### Hardware
- **GPU**: NVIDIA GPU with 16GB+ VRAM (L4, V100, A100 recommended)
- **RAM**: 32GB+ system RAM
- **Storage**: 
  - Original dataset: ~100GB
  - Processed data: ~20GB
  - Training space: 50GB+

### Software Dependencies

```bash
# Core libraries
tensorflow==2.10.1
keras==2.10.0

# Data processing
pandas
numpy
pydicom
pillow
scikit-image
scikit-learn

# Augmentation
albumentations

# Visualization
matplotlib
seaborn
tqdm

# System
opencv-python-headless
```

### Installation

```bash
# Create conda environment
conda create -n ich_detection python=3.9
conda activate ich_detection

# Install dependencies
pip install tensorflow-gpu==2.10.1
pip install pandas numpy pydicom pillow scikit-image scikit-learn
pip install albumentations opencv-python-headless
pip install matplotlib seaborn tqdm
```

---

## Dataset

**Source**: [RSNA Intracranial Hemorrhage Detection](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection)

**Original Structure:**
```
rsna-intracranial-hemorrhage-detection/
├── stage_2_train/           # ~675,000 DICOM files
└── stage_2_train.csv        # Labels file
```

**Labels Format:**
Each image has 6 binary labels in the format: `ImageID_LabelType,Label`

---

## Pipeline Workflow

```
1. CSV Filtering → 2. Data Sampling → 3. Copy Files → 4. Preprocessing
                                                              ↓
6. Model Training ← 5. Data Augmentation ← 4. Train/Val/Test Split
```

---

## Step-by-Step Guide

### Step 1: Filter and Sample CSV Labels

**File**: `CSV_filter.ipynb`

**Purpose**: Balance the dataset by sampling 10,000 images per label class.

**Process**:
1. Read original CSV labels
2. Split ID and label type columns
3. Pivot data (one row per image)
4. Sample 10k images per label
5. Export balanced dataset

**Run**:
```bash
jupyter notebook CSV_filter.ipynb
```

**Input**: 
- `label_RSNA.csv` (original labels)

**Output**: 
- `data_55k.csv` (55,297 balanced samples)

**Key Configuration**:
```python
target_per_label = {label: 10000 for label in label_cols}
```

---

### Step 2: Copy Sampled Files

**File**: `Copy_filter.ipynb`

**Purpose**: Copy only the sampled DICOM files from original dataset to working directory.

**Process**:
1. Read sampled IDs from CSV
2. Multi-threaded file copying from SSD
3. Verify all files copied successfully

**Run**:
```bash
jupyter notebook Copy_filter.ipynb
```

**Input**:
- `data_55k.csv`
- Source: `D:/dataset/rsna-intracranial-hemorrhage-detection/stage_2_train/`

**Output**:
- `raw_data55k/` directory with 55,297 .dcm files

**Key Configuration**:
```python
source_dir = "D:/dataset/rsna-intracranial-hemorrhage-detection/stage_2_train"
target_dir = "raw_data55k"
max_workers = 8  # Adjust based on your system
```

**Note**: Uses ThreadPoolExecutor for parallel copying (~6 minutes with 8 threads).

---

### Step 3: Preprocess DICOM to PNG

**File**: `Prepo.ipynb`

**Purpose**: Convert DICOM files to windowed RGB PNG images.

**Process**:
1. Read DICOM metadata (RescaleIntercept, RescaleSlope)
2. Convert pixel values to Hounsfield Units (HU)
3. Apply three clinical windows:
   - **Blood window**: WL=75, WW=215 (R channel)
   - **Brain window**: WL=40, WW=80 (G channel)
   - **Bone window**: WL=600, WW=2800 (B channel)
4. Resize to 256×256
5. Normalize to [0, 1] range
6. Save as PNG (uint8)

**Run**:
```bash
jupyter notebook Prepo.ipynb
```

**Input**:
- `raw_data55k/*.dcm`

**Output**:
- `raw_png/*.png` (55,297 RGB images)

**Window Explanation**:
- Each window highlights different tissue types
- RGB combination provides comprehensive clinical view
- Handles MONOCHROME1 inversion automatically

**Key Functions**:
```python
def window_wlww_to_01(img, wc, ww, intercept, slope, invert=False):
    # Converts grayscale to windowed [0,1] range
    
def prepare_image(dcm_path, out_dir, target_size=256):
    # Full preprocessing pipeline per image
```

**Performance**: ~6-8 hours for 55k images (single-threaded).

---

### Step 4: Train/Validation/Test Split

**File**: `Split.ipynb`

**Purpose**: Stratified split maintaining class distribution.

**Split Ratios**:
- Training: 80% (44,237 images)
- Validation: 10% (5,530 images)
- Test: 10% (5,530 images)

**Run**:
```bash
jupyter notebook Split.ipynb
```

**Process**:
1. Load labels CSV
2. Stratified split by "any" hemorrhage
3. Verify distribution balance
4. Copy images to separate directories
5. Save split CSV files

**Input**:
- `data_55k.csv`
- `raw_png/*.png`

**Output**:
- `data_train.csv` + `data_train/` directory
- `data_val.csv` + `data_val/` directory
- `data_test.csv` + `data_test/` directory

**Verification**:
```python
# Check distribution across splits
dist_df = pd.DataFrame({
    'Train': train_df[label_cols].sum(),
    'Val': val_df[label_cols].sum(),
    'Test': test_df[label_cols].sum()
})
```

---

### Step 5: Data Augmentation (Training Set Only)

**File**: `Augmen.ipynb`

**Purpose**: Balance minority classes (especially epidural) through augmentation.

**Augmentation Techniques**:
```python
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussianBlur(blur_limit=(3,5), p=0.2),
    A.ElasticTransform(alpha=30, sigma=5, p=0.3)
])
```

**Strategy**:
- Target: 8,000 samples per minority class
- Keep all original images
- Add augmented copies only for minority classes
- New filename format: `ID_{original}_aug_{index}.png`

**Run**:
```bash
jupyter notebook Augmen.ipynb
```

**Input**:
- `data_train.csv`
- `data_train/*.png`

**Output**:
- Updated `data_train.csv` (49,737 samples)
- Augmented images in `data_train/`

**Final Distribution**:
```
any               : 41,737 positive (83.92%)
epidural          :  8,000 positive (16.08%)
intraparenchymal  : 15,130 positive (30.42%)
intraventricular  : 12,010 positive (24.15%)
subarachnoid      : 14,569 positive (29.29%)
subdural          : 15,576 positive (31.32%)
```

**Note**: Only augment training set. Validation/test sets remain unchanged.

---

### Step 6: Model Training

**Files**: 
- `Cascade.ipynb` (original)
- `modif_cascade.ipynb` (modified)
- `Eff2.ipynb` (EfficientNet variant)
- `Conx.ipynb` (ConvNeXt variant)

**Architecture**: Cascade Ensemble

**Stage 1: Binary Classifier (Any Hemorrhage)**
- Input: 256×256×3 RGB images
- Backbone: EfficientNetV2M (frozen for first epochs)
- Output: 1 sigmoid unit (binary classification)
- Purpose: Filter negative cases early

**Stage 2: Multi-label Classifier (5 Subtypes)**
- Input: Same as Stage 1
- Backbone: ConvNeXtBase + EfficientNetV2M
- Output: 5 sigmoid units (multi-label)
- Purpose: Classify hemorrhage subtypes

**Key Features**:
1. **Mixed Precision Training**: FP16 for 2× speedup
2. **XLA Compilation**: 10-20% faster execution
3. **Class Weights**: Handle imbalanced classes
4. **Custom Attention Mechanism**: Focus on hemorrhage regions
5. **Ensemble Learning**: Combines two SOTA architectures

**Run Training**:

```bash
# For Google Colab
jupyter notebook Cascade.ipynb

# Mount Google Drive first
from google.colab import drive
drive.mount('/content/drive')
```

**Training Configuration**:

```python
# Hyperparameters
IMG_SIZE = 256
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 50

# Optimizer
optimizer = Adam(learning_rate=LEARNING_RATE)

# Loss functions
loss = {
    'stage1': 'binary_crossentropy',
    'stage2': 'binary_crossentropy'
}

# Metrics
metrics = ['accuracy', Precision(), Recall(), AUC()]

# Callbacks
callbacks = [
    ModelCheckpoint('best_model.h5', save_best_only=True),
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7)
]
```

**Data Generators**:
```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)
```

**Training Process**:
1. Load preprocessed images
2. Apply class weights
3. Train Stage 1 (binary)
4. Train Stage 2 (multi-label)
5. Fine-tune both stages end-to-end
6. Evaluate on validation set
7. Save best model

**Directory Structure During Training**:
```
/content/
├── data_train/           # Training images
├── data_val/             # Validation images
├── data_test/            # Test images
├── data_train.csv        # Training labels
├── data_val.csv          # Validation labels
├── data_test.csv         # Test labels
└── models/
    ├── stage1_best.h5
    ├── stage2_best.h5
    └── cascade_final.h5
```

**Monitoring**:
- Use TensorBoard for live metrics
- Track loss, accuracy, precision, recall, AUC
- Monitor validation performance
- Watch for overfitting

---

## Model Architecture

### Cascade Architecture Overview

```
Input Image (256×256×3)
         ↓
    [Stage 1: Binary Classifier]
         │
         ├─→ No Hemorrhage (0) → DONE
         │
         └─→ Hemorrhage (1)
                 ↓
    [Stage 2: Multi-label Classifier]
         ↓
    [5 Hemorrhage Subtypes]
    - Epidural
    - Intraparenchymal
    - Intraventricular
    - Subarachnoid
    - Subdural
```

### Stage 1: Binary Classifier

```python
def build_stage1():
    base = EfficientNetV2M(
        include_top=False,
        weights='imagenet',
        input_shape=(256, 256, 3)
    )
    
    # Freeze base initially
    base.trainable = False
    
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='sigmoid', name='any')(x)
    
    return Model(inputs=base.input, outputs=output)
```

### Stage 2: Multi-label Classifier

```python
def build_stage2():
    # Dual backbone
    eff_base = EfficientNetV2M(...)
    conv_base = ConvNeXtBase(...)
    
    # Feature fusion
    eff_features = GlobalAveragePooling2D()(eff_base.output)
    conv_features = GlobalAveragePooling2D()(conv_base.output)
    
    # Attention mechanism
    attention = Attention()([eff_features, conv_features])
    
    # Concatenate
    merged = tf.concat([eff_features, conv_features, attention], axis=-1)
    
    # Classification heads
    x = Dense(1024, activation='relu')(merged)
    x = Dropout(0.4)(x)
    
    outputs = {
        'epidural': Dense(1, activation='sigmoid')(x),
        'intraparenchymal': Dense(1, activation='sigmoid')(x),
        'intraventricular': Dense(1, activation='sigmoid')(x),
        'subarachnoid': Dense(1, activation='sigmoid')(x),
        'subdural': Dense(1, activation='sigmoid')(x)
    }
    
    return Model(inputs=[eff_base.input, conv_base.input], outputs=outputs)
```

### Model Summary

**Parameters**:
- Stage 1: ~54M parameters
- Stage 2: ~135M parameters
- Total: ~189M parameters

**Trainable Parameters** (after fine-tuning):
- Stage 1: ~45M
- Stage 2: ~120M

---

## Training Configuration

### Hardware Setup (Google Colab)

```python
# Enable GPU
# Runtime → Change runtime type → GPU → L4 (recommended)

# Check GPU
import tensorflow as tf
print("GPUs:", tf.config.list_physical_devices('GPU'))

# Mixed precision
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

# XLA compilation
tf.config.optimizer.set_jit(True)

# GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

### Class Weights Computation

```python
# Compute class weights for imbalanced data
def compute_weights(df, label_cols):
    weights = {}
    for label in label_cols:
        classes = df[label].values
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(classes),
            y=classes
        )
        weights[label] = dict(enumerate(class_weights))
    return weights

class_weights = compute_weights(train_df, label_cols)
```

### Learning Rate Schedule

```python
# Initial training (frozen backbone)
initial_lr = 1e-3
optimizer = Adam(learning_rate=initial_lr)

# Fine-tuning (unfrozen backbone)
fine_tune_lr = 1e-5
optimizer = Adam(learning_rate=fine_tune_lr)

# With ReduceLROnPlateau
lr_reducer = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)
```

### Training Phases

**Phase 1: Warmup (10 epochs)**
- Frozen backbone
- Train only classification head
- LR: 1e-3

**Phase 2: Fine-tuning (40 epochs)**
- Unfreeze top layers
- End-to-end training
- LR: 1e-5

**Phase 3: Final tuning (Optional)**
- Unfreeze all layers
- Very low LR: 1e-6
- Monitor validation carefully

---

## Results and Evaluation

### Evaluation Metrics

```python
# Per-class metrics
for label in label_cols:
    y_true = test_df[label]
    y_pred = model.predict(test_images)[label]
    
    # Binary predictions (threshold=0.5)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Metrics
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)
    auc_roc = roc_auc_score(y_true, y_pred)
    
    print(f"{label}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  AUC-ROC: {auc_roc:.4f}")
```

### Confusion Matrix

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

for label in label_cols:
    cm = confusion_matrix(y_true, y_pred_binary)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Negative', 'Positive'])
    disp.plot()
    plt.title(f'Confusion Matrix - {label}')
    plt.savefig(f'cm_{label}.png')
    plt.close()
```

### ROC Curves

```python
from sklearn.metrics import roc_curve, auc

plt.figure(figsize=(10, 8))
for label in label_cols:
    fpr, tpr, _ = roc_curve(test_df[label], predictions[label])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - All Hemorrhage Types')
plt.legend()
plt.savefig('roc_curves.png')
```

### Expected Performance

**Stage 1 (Binary "Any" Detection)**:
- Accuracy: ~95%
- Sensitivity: ~93%
- Specificity: ~96%
- AUC-ROC: ~0.98

**Stage 2 (Subtype Classification)**:
- Average AUC-ROC: ~0.92-0.95
- Epidural: Highest challenge (rare class)
- Subdural: Usually best performance
- Intraparenchymal: Good performance

---

## Troubleshooting

### Common Issues

**1. Out of Memory (OOM)**
```python
# Reduce batch size
BATCH_SIZE = 16  # or 8

# Use gradient accumulation
# Train with smaller batches, accumulate gradients
```

**2. Slow Training**
```python
# Enable mixed precision
mixed_precision.set_global_policy('mixed_float16')

# Use XLA
tf.config.optimizer.set_jit(True)

# Reduce image size
IMG_SIZE = 224  # instead of 256
```

**3. Overfitting**
```python
# Increase dropout
Dropout(0.5)

# Add more augmentation
# Reduce model complexity
# Add L2 regularization
```

**4. Poor Performance on Minority Classes**
```python
# Increase class weights
class_weights['epidural'][1] *= 2

# Oversample minority class
# Use focal loss
```

**5. Validation Loss Increases**
```python
# Reduce learning rate earlier
ReduceLROnPlateau(patience=3)

# Stop training earlier
EarlyStopping(patience=7)

# Check for data leakage
```

---

## File Structure Summary

```
project/
├── README.md                    # This file
├── notebooks/
│   ├── CSV_filter.ipynb        # Step 1: Sample balanced dataset
│   ├── Copy_filter.ipynb       # Step 2: Copy DICOM files
│   ├── Prepo.ipynb             # Step 3: Preprocess to PNG
│   ├── Split.ipynb             # Step 4: Train/val/test split
│   ├── Augmen.ipynb            # Step 5: Data augmentation
│   ├── Cascade.ipynb           # Step 6: Main training
│   ├── modif_cascade.ipynb     # Alternative training
│   ├── Eff2.ipynb              # EfficientNet experiments
│   └── Conx.ipynb              # ConvNeXt experiments
├── data/
│   ├── label_RSNA.csv          # Original labels
│   ├── data_55k.csv            # Sampled dataset
│   ├── data_train.csv          # Training labels
│   ├── data_val.csv            # Validation labels
│   ├── data_test.csv           # Test labels
│   ├── raw_data55k/            # DICOM files
│   ├── raw_png/                # Preprocessed PNG
│   ├── data_train/             # Training images
│   ├── data_val/               # Validation images
│   └── data_test/              # Test images
└── models/
    ├── stage1_best.h5          # Best Stage 1 model
    ├── stage2_best.h5          # Best Stage 2 model
    └── cascade_final.h5        # Final ensemble model
```

---

## Tips for Success

1. **Monitor Training**: Use TensorBoard and watch validation metrics closely
2. **Class Imbalance**: Pay special attention to epidural (rarest class)
3. **Computational Resources**: L4 GPU recommended, minimum 16GB VRAM
4. **Data Quality**: Verify preprocessing output visually before training
5. **Checkpointing**: Save models frequently to avoid losing progress
6. **Reproducibility**: Set random seeds for consistent results
7. **Validation Strategy**: Use stratified split to maintain class distribution
8. **Hyperparameter Tuning**: Start with provided values, adjust based on results

---

## Citation

If you use this pipeline, please cite:

```
RSNA Intracranial Hemorrhage Detection Challenge
https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection
```

---

## License

This project is for educational and research purposes. Please respect the original dataset license and terms.

---

## Contact & Support

For questions or issues:
1. Check the troubleshooting section
2. Review notebook comments
3. Verify data preprocessing steps
4. Ensure hardware requirements are met

---

## Acknowledgments

- RSNA for providing the dataset
- Kaggle for hosting the competition
- TensorFlow and Keras teams
- Albumentations library developers

---

**Last Updated**: January 2026

**Pipeline Version**: 2.0

**Tested On**: 
- Google Colab (L4 GPU)
- Local (RTX 3090)
- Windows 10/11
- Ubuntu 20.04/22.04
