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

<img width="1014" height="470" alt="Screenshot 2026-05-07 014804" src="https://github.com/user-attachments/assets/5f581296-c289-4d49-857c-acec06cbccb9" />
<img width="1005" height="467" alt="Screenshot 2026-05-07 014309" src="https://github.com/user-attachments/assets/29890b7d-6c3e-4434-969b-81d493ce9f58" />



* ❌ **Low spatial resolution** (128×128 pixels or lower)
* ❌ **Motion blur** and atmospheric interference
* ❌ **Domain shift** from lab to field conditions

This project bridges the **"Resolution Gap"** by implementing a **Task-Aware Super-Resolution (SR) Pipeline** that doesn't just make images sharper—it makes them **diagnostically superior**.

---

## 🔬 The Innovation
Unlike conventional SR methods (ESRGAN, Bicubic) that optimize for visual quality, our model is **jointly trained** with a disease classifier to maximize diagnostic accuracy.

> **Result:** 99.97% classification confidence | 98.85% detection accuracy

### 🔑 Key Results: Task-Aware Superiority
#### Quantitative Performance Comparison

| Method | Mean Confidence | Detection Accuracy | Training Loss | IoU |
| :--- | :---: | :---: | :---: | :---: |
| Original Low-Res (128×128) | 65.42% | 71.5% | N/A | 62.4% |
| Standard SR-GAN | 88.15% | 89.2% | 0.0075 | 81.4% |
| **Task-Aware SR (Ours)** | **99.97% ✨** | **98.85% ✨** | **0.0032** | **86.1%** |

---

## 📊 Visual Comparison: The Proof is in the Pixels

<div align="center">
  <table>
    <tr>
      <td align="center"><b>Low-Resolution Input</b></td>
      <td align="center"><b>Standard SR</b></td>
      <td align="center"><b>Task-Aware SR (Ours)</b></td>
      <td align="center"><b>Ground Truth</b></td>
    </tr>
    <tr>
      <td align="center">🔴 Blurry (128×128)</td>
      <td align="center">🟡 Smooth (512×512)</td>
      <td align="center">🟢 Diagnostic (512×512)</td>
      <td align="center">✅ Original (512×512)</td>
    </tr>
    <tr>
      <td align="center">Disease markers lost</td>
      <td align="center">Spots smoothed out</td>
      <td align="center">Lesions preserved</td>
      <td align="center">Reference</td>
    </tr>
  </table>
  <p><i>Sample result showing Cercospora Leaf Spot detection with 100% confidence</i></p>
</div>

---


### 🧠 Loss Function Logic
$$L_{total} = L_{pixel} + \lambda \cdot L_{task}$$

* **$L_{pixel}$ (MSE):** Ensures structural fidelity to ground truth.
* **$L_{task}$ (Cross-Entropy):** Classifier feedback that guides feature reconstruction.
* **$\lambda = 0.5$:** Task weight for balancing sharpness and accuracy.

> **Why It Works:** Traditional SR optimizes for PSNR (human perception). Our model optimizes for diagnostic saliency (machine interpretation).

<img width="1442" height="852" alt="Screenshot 2026-05-07 013808" src="https://github.com/user-attachments/assets/f643e30a-550f-421d-bc06-ac7634344928" />


---

## 📦 Dataset: SLIF-Brinjal

* **Source:** Mendeley Data Repository
* **Total Images:** 8,987 in-field leaf photographs
* **Classes:** 9 (8 diseases + 1 healthy)
* **Resolution:** 512×512 to 1024×1024 (original)
* **Conditions:** Natural outdoor lighting, complex backgrounds

### Disease Categories

| Class ID | Disease Name | Sample Count | Key Visual Markers |
| :---: | :--- | :---: | :--- |
| 0 | Cercospora Leaf Spot | 1,247 | Circular brown lesions with yellow halos |
| 1 | Healthy Leaf | 1,089 | Uniform green coloration |
| 2 | Little Leaf | 1,156 | Reduced size, chlorotic patterns |
| 3 | Mosaic Virus | 1,342 | Irregular yellow-green mosaic |
| 4 | Small Leaf | 987 | Stunted growth, vein clearing |
| 5-8 | Other Diseases | 3,166 | Bacterial Wilt, Powdery Mildew, etc. |

---

## 🚀 Getting Started

### Prerequisites
* Python 3.14+
* CUDA 11.8+ *(Optional, for GPU acceleration)*
* 16GB RAM minimum

### Installation


# Clone the repository
git clone [https://github.com/ahmerali567/plantsr-disease-detection.git](https://github.com/ahmerali567/plantsr-disease-detection.git)
cd plantsr-disease-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
🎯 Usage1. Training the Task-Aware ModelBash# Set Python path and start training
export PYTHONPATH=src  # On Windows use: $env:PYTHONPATH = "src"
python src/train_task_aware.py
 
👤 Author
Ahmer Ali
Sindh Madressatul Islam University (SMIU)

📧 Email: ahmeralitms@gmail.com

🔗 LinkedIn: linkedin.com/in/ahmerali567

🐙 GitHub: @ahmerali567
