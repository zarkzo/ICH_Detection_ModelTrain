# 🧠 ICH Detection System – Training Pipeline

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.10-red.svg)](https://keras.io/)
[![License](https://img.shields.io/badge/License-Academic-yellow.svg)]()

Deep Learning Based Intracranial Hemorrhage Detection Model Training System
Using Cascade EfficientNetV2 and ConvNeXt Architectures

---

## 📋 Table of Contents

* Overview
* Requirements
* Dataset
* Pipeline Workflow
* Step by Step Guide
* Model Architecture
* Training Configuration
* Results and Evaluation
* Project Structure
* Troubleshooting
* Disclaimer
* Citation
* License
* Acknowledgments

---

## 🎯 Overview

This project focuses on an AI training pipeline for multi label classification of Intracranial Hemorrhage from brain CT scan images.
The system is designed for academic research and model development, not for clinical deployment.

The main model uses a two stage cascade approach.

* The first stage detects the presence of hemorrhage using binary classification
* The second stage classifies five hemorrhage subtypes

Detected hemorrhage types

* Any hemorrhage
* Epidural
* Intraparenchymal
* Intraventricular
* Subarachnoid
* Subdural

---

## 💻 Requirements

### Hardware

* NVIDIA GPU with at least 16 GB VRAM
* System RAM at least 32 GB
* Storage requirements

  * Raw dataset around 300 GB
  * Preprocessed data around 20 GB
  * Training workspace at least 50 GB

Recommended GPUs

* NVIDIA L4
* V100
* A100

### Software

* Python 3.9
* TensorFlow 2.10.1
* Keras 2.10.0
* CUDA and cuDNN compatible with the TensorFlow version

Main libraries

* pandas, numpy
* pydicom, pillow
* scikit image, scikit learn
* albumentations
* opencv python headless
* matplotlib, seaborn, tqdm

---

## 📦 Dataset

The dataset used is the [RSNA Intracranial Hemorrhage Detection Dataset](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection) from Kaggle.

Dataset characteristics

* More than 675 thousand CT scan images
* DICOM format
* Multi label annotations by radiologists
* Six labels per image

Original dataset structure

* stage_2_train contains DICOM files
* stage_2_train.csv contains labels

---

## 🧩 Pipeline Workflow

Training pipeline workflow

1. CSV label filtering and sampling
2. Copying selected DICOM files
3. DICOM preprocessing to RGB PNG
4. Train, validation, and test split
5. Data augmentation for the training set
6. Cascade model training

The pipeline is designed to be modular so each stage can be executed and verified independently.

---

## 🛠️ Step by Step Guide

### Step 1. Label Filtering and Sampling

Notebook
`CSV_filter.ipynb`

Objectives

* Balance the dataset
* Sample approximately 10,000 images per label

Main output

* `data_55k.csv` containing around 55,000 images

---

### Step 2. Copy DICOM Files

Notebook
`Copy_filter.ipynb`

Objectives

* Copy only required DICOM files
* Reduce storage usage and I O overhead

Techniques

* ThreadPoolExecutor
* Multi threaded file copy

Output

* `raw_data55k` directory containing selected DICOM files

---

### Step 3. DICOM Preprocessing

Notebook
`Prepo.ipynb`

Preprocessing steps

* Read DICOM metadata
* Convert pixel values to Hounsfield Units
* Apply three clinical window settings
* Blood window as red channel
* Brain window as green channel
* Bone window as blue channel
* Resize to 256 x 256
* Normalize pixel values to range 0 to 1

Output

* RGB PNG images stored in the `raw_png` directory

---

### Step 4. Dataset Split

Notebook
`Split.ipynb`

Data split configuration

* Training set 80 percent
* Validation set 10 percent
* Test set 10 percent

The split is performed using stratified sampling to preserve class distribution.

---

### Step 5. Data Augmentation

Notebook
`Augmen.ipynb`

Augmentation is applied only to the training set.

Augmentation techniques

* Horizontal and vertical flip
* Mild rotation
* Brightness and contrast adjustment
* Gaussian blur
* Elastic transform

Main focus

* Increasing representation of minority classes such as epidural hemorrhage

---

### Step 6. Model Training

Main notebooks

* `Cascade.ipynb`
* `modif_cascade.ipynb`
* `Eff2.ipynb`
* `Conx.ipynb`

Training stages

* Initial training with frozen backbone
* Fine tuning with unfrozen backbone
* Validation set evaluation
* Best model checkpoint saving

---

## 🧠 Model Architecture

### Cascade Architecture

Stage 1

* EfficientNetV2 as backbone
* Binary classifier for hemorrhage presence detection

Stage 2

* EfficientNetV2 and ConvNeXt as dual backbones
* Feature fusion with attention mechanism
* Multi label classifier for five ICH subtypes

The cascade approach helps reduce false positives and improves focus on positive hemorrhage cases.

---

## ⚙️ Training Configuration

Main parameters

* Image size 256
* Batch size 32
* Epochs 50
* Optimizer Adam

Optimization techniques

* Mixed precision training
* XLA compilation
* Class weighting for data imbalance
* Early stopping
* Learning rate scheduling

Training is conducted in multiple phases using different learning rates.

---

## 📊 Results and Evaluation

Evaluation metrics

* Accuracy
* Precision
* Recall
* F1 score
* AUC ROC

Evaluation is performed per class and overall.

Expected performance

* Any hemorrhage detection AUC around 0.98
* Subtype classification average AUC between 0.92 and 0.95

---

## 📂 Project Structure

```
project
├── notebooks     # All Jupyter notebooks for the pipeline
├── data          # Labels, DICOM files, and preprocessed PNG images
├── models        # Trained model checkpoints
└── README.md     # Pipeline documentation
```

---

## 🔧 Troubleshooting

Common issues

* Out of memory, reduce batch size
* Slow training, enable mixed precision
* Overfitting, increase augmentation and dropout
* Poor minority class performance, adjust class weights

---

## ⚠️ Disclaimer

This pipeline is developed for academic research purposes only.

It is not intended for clinical diagnosis or real world medical use.

The models have not undergone clinical validation and do not have regulatory approval.

---

## 📝 Citation

If you use this pipeline, please cite

RSNA Intracranial Hemorrhage Detection Challenge
Kaggle

---

## 📄 License

This project is licensed for academic and research use only.

---

## 🙏 Acknowledgments

* RSNA for providing the dataset
* Kaggle
* TensorFlow and Keras teams
* Albumentations contributors

---

Last updated January 2026
Pipeline version 2.0
