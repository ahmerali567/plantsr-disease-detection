# Task-Aware Super-Resolution for Plant Disease Detection

### Overview
This project bridges the "resolution gap" in UAV-based agriculture monitoring. It uses a **Task-Aware ESRGAN** to upscale low-resolution (128x128) drone images to high-resolution leaf diagnostics.

### Key Results
- **Detection Accuracy:** 98.85%
- **Classification Confidence:** 99.97%
- **Model:** Jointly Optimized SRCNN/ESRGAN

### Features
- Phase-wise training (Pixel Loss -> Task Loss -> Final Polish).
- Integrated with Weights & Biases (WandB) for real-time tracking.
- Optimized for Edge Devices like NVIDIA Jetson Nano.

### Dataset
Utilizes the **SLIF-Brinjal** dataset (8,987 images) for disease classification.
