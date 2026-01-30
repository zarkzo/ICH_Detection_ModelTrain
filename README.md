# 🧠 ICH Detection System - Training Pipeline

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.10-red.svg)](https://keras.io/)
[![License](https://img.shields.io/badge/License-Academic-yellow.svg)]()

Sistem Pelatihan Model Deteksi Perdarahan Intrakranial Berbasis Deep Learning
Menggunakan Arsitektur Cascade EfficientNetV2 dan ConvNeXt

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

---

## 🎯 Overview

Proyek ini berfokus pada pipeline pelatihan model AI untuk klasifikasi multi-label Intracranial Hemorrhage dari citra CT Scan otak. Sistem dirancang untuk kebutuhan riset akademik dan pengembangan model, bukan untuk deployment klinis.

Model utama menggunakan pendekatan cascade dua tahap:

* Tahap pertama mendeteksi keberadaan perdarahan secara biner
* Tahap kedua mengklasifikasikan lima subtipe perdarahan

Jenis perdarahan yang dideteksi:

* Any hemorrhage
* Epidural
* Intraparenchymal
* Intraventricular
* Subarachnoid
* Subdural

---

## 💻 Requirements

### Hardware

* GPU NVIDIA dengan minimal 16 GB VRAM
* RAM sistem minimal 32 GB
* Storage:

  * Dataset mentah sekitar 300 GB
  * Data hasil preprocessing sekitar 20 GB
  * Ruang training minimal 50 GB

GPU yang direkomendasikan:

* NVIDIA L4
* V100
* A100

### Software

* Python 3.9
* TensorFlow 2.10.1
* Keras 2.10.0
* CUDA dan cuDNN sesuai versi TensorFlow

Library utama:

* pandas, numpy
* pydicom, pillow
* scikit-image, scikit-learn
* albumentations
* opencv-python-headless
* matplotlib, seaborn, tqdm

---

## 📦 Dataset

Dataset yang digunakan adalah RSNA Intracranial Hemorrhage Detection Dataset dari Kaggle.

Karakteristik dataset:

* Lebih dari 675 ribu citra CT Scan
* Format DICOM
* Multi-label annotation oleh radiolog
* Enam label untuk setiap citra

Struktur asli dataset:

* stage_2_train berisi file DICOM
* stage_2_train.csv berisi label

---

## 🧩 Pipeline Workflow

Alur kerja pipeline pelatihan:

1. Filtering dan sampling label CSV
2. Penyalinan file DICOM terpilih
3. Preprocessing DICOM ke PNG RGB
4. Pembagian data train, validation, dan test
5. Augmentasi data pada training set
6. Training model cascade

Pipeline ini dirancang modular agar setiap tahap dapat dijalankan dan diverifikasi secara terpisah.

---

## 🛠️ Step by Step Guide

### Step 1. Filter dan Sampling Label

Notebook: CSV_filter.ipynb

Tujuan:

* Menyeimbangkan dataset
* Sampling sekitar 10.000 citra per label

Output utama:

* data_55k.csv dengan sekitar 55 ribu citra

---

### Step 2. Copy File DICOM

Notebook: Copy_filter.ipynb

Tujuan:

* Menyalin hanya file DICOM yang digunakan
* Mengurangi beban storage dan I O

Teknik:

* ThreadPoolExecutor
* Multi-threaded copy

Output:

* Folder raw_data55k berisi file DICOM terpilih

---

### Step 3. Preprocessing DICOM

Notebook: Prepo.ipynb

Tahapan preprocessing:

* Membaca metadata DICOM
* Konversi pixel ke Hounsfield Unit
* Penerapan tiga window klinis
* Blood window sebagai channel merah
* Brain window sebagai channel hijau
* Bone window sebagai channel biru
* Resize ke 256 x 256
* Normalisasi ke rentang 0 sampai 1

Output:

* File PNG RGB di folder raw_png

---

### Step 4. Split Dataset

Notebook: Split.ipynb

Pembagian data:

* Training 80 persen
* Validation 10 persen
* Test 10 persen

Split dilakukan secara stratified untuk menjaga distribusi kelas.

---

### Step 5. Data Augmentation

Notebook: Augmen.ipynb

Augmentasi hanya diterapkan pada training set.

Teknik augmentasi:

* Flip horizontal dan vertikal
* Rotasi ringan
* Perubahan brightness dan contrast
* Gaussian blur
* Elastic transform

Fokus utama augmentasi:

* Menambah representasi kelas minoritas seperti epidural

---

### Step 6. Model Training

Notebook utama:

* Cascade.ipynb
* modif_cascade.ipynb
* Eff2.ipynb
* Conx.ipynb

Tahap training:

* Training awal dengan backbone dibekukan
* Fine tuning dengan backbone dibuka
* Evaluasi pada validation set
* Penyimpanan model terbaik

---

## 🧠 Model Architecture

### Cascade Architecture

Tahap 1:

* EfficientNetV2 sebagai backbone
* Binary classifier untuk mendeteksi adanya perdarahan

Tahap 2:

* EfficientNetV2 dan ConvNeXt sebagai dual backbone
* Feature fusion dan attention mechanism
* Multi-label classifier untuk lima subtipe ICH

Pendekatan cascade membantu mengurangi false positive dan meningkatkan fokus pada citra positif.

---

## ⚙️ Training Configuration

Parameter utama:

* Image size 256
* Batch size 32
* Epoch 50
* Optimizer Adam

Teknik optimasi:

* Mixed precision training
* XLA compilation
* Class weighting untuk data imbalance
* Early stopping dan learning rate scheduling

Training dilakukan dalam beberapa fase dengan learning rate berbeda.

---

## 📊 Results and Evaluation

Evaluasi dilakukan menggunakan:

* Accuracy
* Precision
* Recall
* F1 Score
* AUC ROC

Evaluasi dilakukan per kelas dan secara keseluruhan.

Ekspektasi performa:

* Deteksi any hemorrhage dengan AUC sekitar 0.98
* Subtype classification dengan AUC rata-rata 0.92 sampai 0.95

---

## 📂 Project Structure

project

* notebooks berisi seluruh pipeline Jupyter Notebook
* data berisi label, DICOM, dan PNG hasil preprocessing
* models berisi model hasil training
* README berisi dokumentasi pipeline

---

## 🔧 Troubleshooting

Masalah umum:

* Out of memory, kurangi batch size
* Training lambat, aktifkan mixed precision
* Overfitting, tingkatkan augmentasi dan dropout
* Performa kelas minoritas rendah, sesuaikan class weight

---

## ⚠️ Disclaimer

Pipeline ini dibuat untuk keperluan akademik dan riset.

Tidak ditujukan untuk diagnosis klinis atau penggunaan medis nyata.

Model belum melalui validasi klinis dan tidak memiliki persetujuan regulator.

---

## 📝 Citation

Jika menggunakan pipeline ini, silakan sitasi:

RSNA Intracranial Hemorrhage Detection Challenge
Kaggle

---

## 📄 License

Lisensi hanya untuk penggunaan akademik dan riset.

---

## 🙏 Acknowledgments

* RSNA sebagai penyedia dataset
* Kaggle
* TensorFlow dan Keras
* Albumentations

---

Last updated January 2026
Pipeline version 2.0
