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


======================================================================
COMPLETE TRAINING SUMMARY - ALL MAGNIFICATIONS
======================================================================

Training Duration: 3:46:55

Magnification   Accuracy    Status
--------------------------------------------------
40X             99.50%      ✓ EXCELLENT
100X            98.80%      ✓ GOOD
200X            99.50%      ✓ EXCELLENT
400X            99.18%      ✓ EXCELLENT

--------------------------------------------------
Average         99.24%
======================================================================

MODELS SAVED TO GOOGLE DRIVE:
  • best_breakhis_40X.pth
  • best_breakhis_100X.pth
  • best_breakhis_200X.pth
  • best_breakhis_400X.pth

======================================================================
✓ ALL MAGNIFICATIONS TRAINING COMPLETE!
Target (Gella 2024): 99.99% per magnification
Your Results: See table above
======================================================================
