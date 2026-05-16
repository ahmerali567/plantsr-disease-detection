# 🌿 Task-Aware Super-Resolution for Cross-Scale Plant Disease Detection

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/WandB-Tracking-orange.svg" alt="WandB">
</p>

### 📌 Project Overview
[cite_start]Deep learning models often face a critical **"Resolution Gap"** when moving from smartphone-captured leaf images to UAV-scale imagery[cite: 4, 5]. [cite_start]This project implements a **Joint-Optimization Framework** where a Super-Resolution (SR) model is explicitly trained to maximize the diagnostic accuracy of a downstream disease classifier[cite: 7, 28]. 

[cite_start]Instead of just making images "visually appealing," our system produces **"diagnostically superior"** reconstructions[cite: 31].

---

### 🚀 Key Results: Diagnostic Excellence
[cite_start]Our Task-Aware pipeline significantly outperforms standard enhancement methods[cite: 8]:

| Method | Mean Confidence | Detection Accuracy |
| :--- | :---: | :---: |
| Original LR (128×128) | 65.42% | 71.5% |
| Standard SR-GAN | 88.15% | 89.2% |
| **Task-Aware SR (Ours)** | **99.97%** | **98.85%** |

---

### 🛠️ System Architecture
[cite_start]The pipeline consists of two interconnected deep networks optimized through a **Joint Loss Function**[cite: 75, 121]:

1. [cite_start]**Task-Aware Generator:** Uses 16 Residual-in-Residual Dense Blocks (RRDB) for deep feature learning[cite: 84, 90].
2. [cite_start]**Disease Classifier:** A 4-block CNN that provides real-time gradient feedback to the generator[cite: 104, 105].


---

### 📊 Dataset Details
[cite_start]We utilize the **SLIF-Brinjal** dataset, consisting of 8,987 in-field images across 9 taxonomy classes[cite: 34]:
* [cite_start]**Healthy Leaf:** 1,089 samples [cite: 35]
* [cite_start]**Mosaic Virus:** 1,342 samples [cite: 35]
* [cite_start]**Little Leaf:** 1,156 samples [cite: 35]
* [cite_start]**Cercospora Leaf Spot:** 1,247 samples [cite: 35]

---

### ⚙️ Training Strategy
[cite_start]The model is trained in a **Three-Phase Approach** to ensure stable convergence[cite: 133, 134]:
* [cite_start]**Phase I (Baseline):** Generator learns basic pixel reconstruction[cite: 134].
* [cite_start]**Phase II (Task-Aware):** Introduces classifier feedback ($\lambda=0.5$)[cite: 134].
* [cite_start]**Phase III (Fine-Tuning):** Final artifact removal at a lower learning rate[cite: 134].

---

### 📱 Deployment & Edge Integration
Optimized for precision agriculture, the system is ready for:
* [cite_start]**NVIDIA Jetson Nano:** Real-time inference (~0.8s per image)[cite: 253, 254].
* [cite_start]**UAV Workflow:** Automatic GPS tagging and disease alerts[cite: 246, 248].

---

### 🔗 Connect & Track
* [cite_start]**Interactive Dashboard:** [View WandB Metrics](https://wandb.ai/ahmeralitms-sindh-madressatul-islam-university/PlantSR-Disease-Detection) [cite: 300]
* [cite_start]**Dataset Source:** [Mendeley Data](https://doi.org/10.17632/6yg6vktrc2.1) [cite: 34]

---
© 2026 Ahmer Ali | [cite_start]Sindh Madressatul Islam University (SMIU) [cite: 2]
