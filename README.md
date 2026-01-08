🧬 Breast Cancer Histopathology Classification (BreaKHis)

This project implements Vision Transformer (ViT-Small) models to classify benign vs malignant breast cancer histopathology images using the BreaKHis dataset, trained independently across four magnification levels (40X–400X).
The goal was to evaluate the robustness of ViT-based models across varying microscopic resolutions and achieve high diagnostic accuracy.

🚀 Features

Multi-magnification training: 40X, 100X, 200X, 400X
Vision Transformer (ViT-Small) architecture
Data augmentation for improved generalization
Cosine learning rate scheduling
Independent model per magnification
Robust handling of missing data
Google Colab + GPU (CUDA) compatible
Models saved automatically to Google Drive

### Note
The notebook is provided as source code only.
Training outputs and models are generated during execution.
