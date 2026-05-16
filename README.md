# 🌿 Task-Aware Super-Resolution for Plant Disease Detection
### Cross-Scale Disease Diagnostics from UAV to Leaf-Level

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.14+-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red?style=for-the-badge&logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/CUDA-11.8+-green?style=for-the-badge&logo=nvidia" alt="CUDA">
  <img src="https://img.shields.io/badge/WandB-Tracking-orange?style=for-the-badge&logo=weightsandbiases" alt="WandB">
</p>

---

## 🎯 Project Overview
Deep learning models achieve **>95% accuracy** on controlled leaf-level datasets, but their performance drops dramatically when applied to UAV (drone) or field-scale imagery due to:

* [cite_start]❌ **Low spatial resolution** (128×128 pixels or lower) [cite: 12]
* [cite_start]❌ **Motion blur** and atmospheric interference [cite: 13]
* [cite_start]❌ **Domain shift** from lab to field conditions [cite: 14]

[cite_start]This project bridges the **"Resolution Gap"** by implementing a **Task-Aware Super-Resolution (SR) Pipeline** that doesn't just make images sharper—it makes them **diagnostically superior**[cite: 6, 7].

---

## 🔬 The Innovation: Task-Aware Superiority
[cite_start]Unlike conventional SR methods (ESRGAN, Bicubic) that optimize for visual quality, our model is **jointly trained** with a disease classifier to maximize diagnostic accuracy[cite: 7].

> **Result:** 99.97% classification confidence | [cite_start]98.85% detection accuracy [cite: 8]

### 🔑 Quantitative Performance Comparison
| Method | Mean Confidence | Detection Accuracy | Training Loss | IoU |
| :--- | :---: | :---: | :---: | :---: |
| Original Low-Res (128×128) | 65.42% | 71.5% | N/A | 62.4% |
| Standard SR-GAN | 88.15% | 89.2% | 0.0075 | 81.4% |
| **Task-Aware SR (Ours)** | **99.97% ✨** | **98.85% ✨** | **0.0032** | **86.1%** |
[cite_start][cite: 183]

---

## 🏗️ Architecture: The Joint-Optimization Secret
[cite_start]Our system unifies image enhancement and classification into a single end-to-end trainable pipeline[cite: 286].



### 🧠 Loss Function Logic
$$L_{total} = L_{pixel} + \lambda \cdot L_{task}$$
* [cite_start]**$L_{pixel}$ (MSE):** Ensures structural fidelity to ground truth[cite: 123].
* **$L_{task}$ (Cross-Entropy):** Classifier feedback that guides feature reconstruction[cite: 126, 127].
* **$\lambda = 0.5$:** Task weight for balancing sharpness and accuracy[cite: 129].

---

## 📦 Dataset: SLIF-Brinjal
* [cite_start]**Source:** Mendeley Data Repository (Singh, S. et al.)[cite: 34, 293].
* [cite_start]**Total Images:** 8,987 in-field leaf photographs[cite: 34].
* **Classes:** 9 (8 diseases + 1 healthy)[cite: 34].

| Class ID | Disease Name | Sample Count | Key Visual Markers |
| :---: | :--- | :---: | :--- |
| 0 | Cercospora Leaf Spot | 1,247 | Circular brown lesions with yellow halos |
| 1 | Healthy Leaf | 1,089 | Uniform green coloration |
| 2 | Little Leaf | 1,156 | Reduced size, chlorotic patterns |
| 3 | Mosaic Virus | 1,342 | Irregular yellow-green mosaic |
[cite_start][cite: 35]

---

## 🚀 Getting Started

### Installation
```bash
# Clone the repository
git clone [https://github.com/ahmerali567/plantsr-disease-detection.git](https://github.com/ahmerali567/plantsr-disease-detection.git)
cd plantsr-disease-detection

# Install dependencies
pip install -r requirements.txt
🎯 UsageTraining: python src/train_task_aware.py (Approx. 55 hours total )  Testing: python src/test_final_presentation.pyWandB Sync: python src/sync_to_wandb.py📈 Training Strategy: Three-Phase ApproachPhase I: Baseline Reconstruction (Epochs 1-10) - Focus on basic geometry using Pixel Loss.  Phase II: Task-Aware Optimization (Epochs 11-15) - Introduction of classifier feedback ($\lambda=0.5$).  Phase III: Fine-Tuning (Epochs 16-20) - Artifact removal and crystal clarity.  💡 Practical Applications & Impact🚁 UAV Integration: Optimized for NVIDIA Jetson Nano with ~0.8s inference time.  💰 Economic ROI: Early detection can save $15,000-$20,000 per season for a 100-hectare farm.🌍 Sustainable Agriculture: Reduces the need for manual field inspection and allows precise pesticide application .  👤 AuthorAhmer AliSindh Madressatul Islam University (SMIU)📧 Email: ahmeralitms@gmail.com🔗 LinkedIn: linkedin.com/in/ahmerali567🐙 GitHub: @ahmerali567
