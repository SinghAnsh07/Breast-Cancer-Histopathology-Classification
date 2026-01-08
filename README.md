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

📊 Dataset

BreaKHis: Breast Cancer Histopathological Images

Binary classification:

Benign

Malignant

Separate training and evaluation per magnification

⚙️ Training Setup

Device: CUDA (GPU)

Epochs per magnification: 100

Optimizer: AdamW

Loss Function: Cross-Entropy Loss

Model: ViT-Small

Training duration: ~3 hours 47 minutes

📈 Results Summary
Magnification	Accuracy	Status
40X	99.50%	✅ Excellent
100X	98.80%	✅ Good
200X	99.50%	✅ Excellent
400X	99.18%	✅ Excellent
Average	99.24%	⭐
🧪 Detailed Evaluation Highlights
🔹 40X Magnification

Accuracy: 99.50%

Near-perfect precision & recall for both classes

Confusion Matrix errors: 2 / 399

🔹 100X Magnification

Accuracy: 98.80%

Strong malignant detection (Recall: 1.00)

Minor benign misclassifications

🔹 200X Magnification

Accuracy: 99.50%

Balanced precision and recall

Confusion Matrix errors: 2 / 403

🔹 400X Magnification

Accuracy: 99.18%

Stable performance at highest magnification

Confusion Matrix errors: 3 / 364

💾 Saved Models

All best-performing models were saved automatically:

best_breakhis_40X.pth
best_breakhis_100X.pth
best_breakhis_200X.pth
best_breakhis_400X.pth

📝 Notes

The notebook is provided as source code only.

Training outputs and models are generated during execution.

Full training logs are intentionally excluded to keep the repository clean and readable.

🎯 Conclusion

This project demonstrates that Vision Transformers can achieve highly reliable performance on breast cancer histopathology classification across multiple magnifications, making them a strong candidate for medical imaging applications.
