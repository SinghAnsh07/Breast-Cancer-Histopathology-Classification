## Breast Cancer Histopathology Classification (BreaKHis)

This project trains Vision Transformer (ViT) models to classify
benign vs malignant breast cancer histopathology images using the
BreaKHis dataset across multiple magnifications.

### Features
- Multi-magnification training (40X–400X)
- Vision Transformer (ViT-Small)
- Data augmentation & cosine LR scheduling
- Robust handling of missing magnification data
- Google Colab compatible

### Note
The notebook is provided as source code only.
Training outputs and models are generated during execution.


### RESULT

Device: cuda


######################################################################
# BREAKHIS BREAST CANCER CLASSIFICATION - ALL MAGNIFICATIONS
# Training on: 40X, 100X, 200X, 400X
# Target: 99% accuracy on each magnification
######################################################################

======================================================================
TRAINING: 40X MAGNIFICATION
======================================================================

Loading 40X dataset...
Total images: 1995
  Benign: 625 | Malignant: 1370
  Train batches: 25 | Val batches: 7

Creating ViT-Small model for 40X...
/usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: 
The secret `HF_TOKEN` does not exist in your Colab secrets.
To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
You will be able to reuse this secret in all of your notebooks.
Please note that authentication is recommended but still optional to access public models or datasets.
  warnings.warn(
model.safetensors: 100%
 88.2M/88.2M [00:02<00:00, 45.6MB/s]
Starting training for 100 epochs...

40X Epoch 1/100: 100%|██████████| 25/25 [00:36<00:00,  1.46s/it]
Epoch 1/100 - Loss: 0.5906 | Accuracy: 81.45% ✓ BEST!
40X Epoch 2/100: 100%|██████████| 25/25 [00:28<00:00,  1.14s/it]
Epoch 2/100 - Loss: 0.3391 | Accuracy: 88.22% ✓ BEST!
40X Epoch 3/100: 100%|██████████| 25/25 [00:31<00:00,  1.26s/it]
Epoch 3/100 - Loss: 0.2516 | Accuracy: 91.73% ✓ BEST!
40X Epoch 4/100: 100%|██████████| 25/25 [00:29<00:00,  1.19s/it]
Epoch 4/100 - Loss: 0.2051 | Accuracy: 94.24% ✓ BEST!
40X Epoch 5/100: 100%|██████████| 25/25 [00:29<00:00,  1.19s/it]
Epoch 5/100 - Loss: 0.1594 | Accuracy: 96.49% ✓ BEST!
40X Epoch 6/100: 100%|██████████| 25/25 [00:29<00:00,  1.19s/it]
Epoch 6/100 - Loss: 0.1640 | Accuracy: 94.24%
40X Epoch 7/100: 100%|██████████| 25/25 [00:28<00:00,  1.12s/it]
Epoch 7/100 - Loss: 0.1367 | Accuracy: 95.74%
40X Epoch 8/100: 100%|██████████| 25/25 [00:28<00:00,  1.13s/it]
Epoch 8/100 - Loss: 0.0947 | Accuracy: 97.49% ✓ BEST!
40X Epoch 9/100: 100%|██████████| 25/25 [00:31<00:00,  1.26s/it]
Epoch 9/100 - Loss: 0.0960 | Accuracy: 94.99%
40X Epoch 10/100: 100%|██████████| 25/25 [00:28<00:00,  1.13s/it]
Epoch 10/100 - Loss: 0.0803 | Accuracy: 96.49%
40X Epoch 11/100: 100%|██████████| 25/25 [00:27<00:00,  1.11s/it]
Epoch 11/100 - Loss: 0.0607 | Accuracy: 96.24%
40X Epoch 12/100: 100%|██████████| 25/25 [00:28<00:00,  1.13s/it]
Epoch 12/100 - Loss: 0.1001 | Accuracy: 96.74%
40X Epoch 13/100: 100%|██████████| 25/25 [00:28<00:00,  1.12s/it]
Epoch 13/100 - Loss: 0.0759 | Accuracy: 93.73%
40X Epoch 14/100: 100%|██████████| 25/25 [00:27<00:00,  1.10s/it]
Epoch 14/100 - Loss: 0.0787 | Accuracy: 97.49%
40X Epoch 15/100: 100%|██████████| 25/25 [00:27<00:00,  1.11s/it]
Epoch 15/100 - Loss: 0.0652 | Accuracy: 96.49%
40X Epoch 16/100: 100%|██████████| 25/25 [00:27<00:00,  1.12s/it]
Epoch 16/100 - Loss: 0.0504 | Accuracy: 99.00% ✓ BEST!
40X Epoch 17/100: 100%|██████████| 25/25 [00:31<00:00,  1.25s/it]
Epoch 17/100 - Loss: 0.0644 | Accuracy: 97.99%
40X Epoch 18/100: 100%|██████████| 25/25 [00:28<00:00,  1.13s/it]
Epoch 18/100 - Loss: 0.0343 | Accuracy: 97.24%
40X Epoch 19/100: 100%|██████████| 25/25 [00:28<00:00,  1.12s/it]
Epoch 19/100 - Loss: 0.0268 | Accuracy: 98.75%
40X Epoch 20/100: 100%|██████████| 25/25 [00:28<00:00,  1.13s/it]
Epoch 20/100 - Loss: 0.0401 | Accuracy: 98.75%
40X Epoch 21/100: 100%|██████████| 25/25 [00:27<00:00,  1.10s/it]
Epoch 21/100 - Loss: 0.0385 | Accuracy: 94.49%
40X Epoch 22/100: 100%|██████████| 25/25 [00:28<00:00,  1.12s/it]
Epoch 22/100 - Loss: 0.0515 | Accuracy: 97.49%
40X Epoch 23/100: 100%|██████████| 25/25 [00:28<00:00,  1.13s/it]
Epoch 23/100 - Loss: 0.0217 | Accuracy: 98.75%
40X Epoch 24/100: 100%|██████████| 25/25 [00:27<00:00,  1.12s/it]
Epoch 24/100 - Loss: 0.0264 | Accuracy: 99.00%
40X Epoch 25/100: 100%|██████████| 25/25 [00:29<00:00,  1.18s/it]
Epoch 25/100 - Loss: 0.0163 | Accuracy: 96.99%
40X Epoch 26/100: 100%|██████████| 25/25 [00:27<00:00,  1.11s/it]
Epoch 26/100 - Loss: 0.0600 | Accuracy: 96.49%
40X Epoch 27/100: 100%|██████████| 25/25 [00:28<00:00,  1.12s/it]
Epoch 27/100 - Loss: 0.0514 | Accuracy: 97.24%
40X Epoch 28/100: 100%|██████████| 25/25 [00:27<00:00,  1.11s/it]
Epoch 28/100 - Loss: 0.0492 | Accuracy: 96.74%
40X Epoch 29/100: 100%|██████████| 25/25 [00:27<00:00,  1.11s/it]
Epoch 29/100 - Loss: 0.0148 | Accuracy: 97.99%
40X Epoch 30/100: 100%|██████████| 25/25 [00:28<00:00,  1.13s/it]
Epoch 30/100 - Loss: 0.0195 | Accuracy: 98.75%
40X Epoch 31/100: 100%|██████████| 25/25 [00:27<00:00,  1.11s/it]
Epoch 31/100 - Loss: 0.0759 | Accuracy: 97.49%
40X Epoch 32/100: 100%|██████████| 25/25 [00:28<00:00,  1.12s/it]
Epoch 32/100 - Loss: 0.0144 | Accuracy: 97.24%
40X Epoch 33/100: 100%|██████████| 25/25 [00:30<00:00,  1.21s/it]
Epoch 33/100 - Loss: 0.0502 | Accuracy: 95.99%
40X Epoch 34/100: 100%|██████████| 25/25 [00:28<00:00,  1.13s/it]
Epoch 34/100 - Loss: 0.0237 | Accuracy: 98.75%
40X Epoch 35/100: 100%|██████████| 25/25 [00:28<00:00,  1.13s/it]
Epoch 35/100 - Loss: 0.0212 | Accuracy: 99.00%
40X Epoch 36/100: 100%|██████████| 25/25 [00:28<00:00,  1.12s/it]
Epoch 36/100 - Loss: 0.0149 | Accuracy: 98.50%
40X Epoch 37/100: 100%|██████████| 25/25 [00:28<00:00,  1.13s/it]
Epoch 37/100 - Loss: 0.0066 | Accuracy: 98.50%
40X Epoch 38/100: 100%|██████████| 25/25 [00:27<00:00,  1.11s/it]
Epoch 38/100 - Loss: 0.0043 | Accuracy: 99.00%
40X Epoch 39/100: 100%|██████████| 25/25 [00:28<00:00,  1.13s/it]
Epoch 39/100 - Loss: 0.0088 | Accuracy: 99.00%
40X Epoch 40/100: 100%|██████████| 25/25 [00:27<00:00,  1.12s/it]
Epoch 40/100 - Loss: 0.0203 | Accuracy: 98.50%
40X Epoch 41/100: 100%|██████████| 25/25 [00:28<00:00,  1.12s/it]
Epoch 41/100 - Loss: 0.0070 | Accuracy: 98.75%
40X Epoch 42/100: 100%|██████████| 25/25 [00:27<00:00,  1.12s/it]
Epoch 42/100 - Loss: 0.0074 | Accuracy: 99.00%
40X Epoch 43/100: 100%|██████████| 25/25 [00:28<00:00,  1.13s/it]
Epoch 43/100 - Loss: 0.0057 | Accuracy: 98.50%
40X Epoch 44/100: 100%|██████████| 25/25 [00:28<00:00,  1.13s/it]
Epoch 44/100 - Loss: 0.0011 | Accuracy: 98.75%
40X Epoch 45/100: 100%|██████████| 25/25 [00:28<00:00,  1.12s/it]
Epoch 45/100 - Loss: 0.0002 | Accuracy: 98.75%
40X Epoch 46/100: 100%|██████████| 25/25 [00:28<00:00,  1.13s/it]
Epoch 46/100 - Loss: 0.0001 | Accuracy: 98.75%
40X Epoch 47/100: 100%|██████████| 25/25 [00:27<00:00,  1.11s/it]
Epoch 47/100 - Loss: 0.0002 | Accuracy: 98.75%
40X Epoch 48/100: 100%|██████████| 25/25 [00:28<00:00,  1.13s/it]
Epoch 48/100 - Loss: 0.0005 | Accuracy: 99.25% ✓ BEST!
40X Epoch 49/100: 100%|██████████| 25/25 [00:30<00:00,  1.22s/it]
Epoch 49/100 - Loss: 0.0003 | Accuracy: 99.00%
40X Epoch 50/100: 100%|██████████| 25/25 [00:27<00:00,  1.11s/it]
Epoch 50/100 - Loss: 0.0001 | Accuracy: 99.25%
40X Epoch 51/100: 100%|██████████| 25/25 [00:27<00:00,  1.11s/it]
Epoch 51/100 - Loss: 0.0001 | Accuracy: 99.25%
40X Epoch 52/100: 100%|██████████| 25/25 [00:27<00:00,  1.10s/it]
Epoch 52/100 - Loss: 0.0001 | Accuracy: 99.25%
40X Epoch 53/100: 100%|██████████| 25/25 [00:27<00:00,  1.12s/it]
Epoch 53/100 - Loss: 0.0006 | Accuracy: 99.25%
40X Epoch 54/100: 100%|██████████| 25/25 [00:27<00:00,  1.12s/it]
Epoch 54/100 - Loss: 0.0002 | Accuracy: 99.00%
40X Epoch 55/100: 100%|██████████| 25/25 [00:27<00:00,  1.11s/it]
Epoch 55/100 - Loss: 0.0001 | Accuracy: 98.50%
40X Epoch 56/100: 100%|██████████| 25/25 [00:28<00:00,  1.12s/it]
Epoch 56/100 - Loss: 0.0007 | Accuracy: 98.75%
40X Epoch 57/100: 100%|██████████| 25/25 [00:28<00:00,  1.13s/it]
Epoch 57/100 - Loss: 0.0017 | Accuracy: 99.25%
40X Epoch 58/100: 100%|██████████| 25/25 [00:29<00:00,  1.18s/it]
Epoch 58/100 - Loss: 0.0060 | Accuracy: 99.25%
40X Epoch 59/100: 100%|██████████| 25/25 [00:28<00:00,  1.12s/it]
Epoch 59/100 - Loss: 0.0045 | Accuracy: 98.25%
40X Epoch 60/100: 100%|██████████| 25/25 [00:28<00:00,  1.13s/it]
Epoch 60/100 - Loss: 0.0144 | Accuracy: 97.24%
40X Epoch 61/100: 100%|██████████| 25/25 [00:27<00:00,  1.12s/it]
Epoch 61/100 - Loss: 0.0074 | Accuracy: 98.75%
40X Epoch 62/100: 100%|██████████| 25/25 [00:28<00:00,  1.12s/it]
Epoch 62/100 - Loss: 0.0030 | Accuracy: 98.75%
40X Epoch 63/100: 100%|██████████| 25/25 [00:27<00:00,  1.12s/it]
Epoch 63/100 - Loss: 0.0015 | Accuracy: 99.50% ✓ BEST!
40X Epoch 64/100: 100%|██████████| 25/25 [00:30<00:00,  1.22s/it]
Epoch 64/100 - Loss: 0.0006 | Accuracy: 98.50%
40X Epoch 65/100: 100%|██████████| 25/25 [00:28<00:00,  1.12s/it]
Epoch 65/100 - Loss: 0.0016 | Accuracy: 98.75%
40X Epoch 66/100: 100%|██████████| 25/25 [00:29<00:00,  1.19s/it]
Epoch 66/100 - Loss: 0.0004 | Accuracy: 98.50%
40X Epoch 67/100: 100%|██████████| 25/25 [00:28<00:00,  1.12s/it]
Epoch 67/100 - Loss: 0.0006 | Accuracy: 98.50%
40X Epoch 68/100: 100%|██████████| 25/25 [00:27<00:00,  1.11s/it]
Epoch 68/100 - Loss: 0.0001 | Accuracy: 98.50%
40X Epoch 69/100: 100%|██████████| 25/25 [00:28<00:00,  1.13s/it]
Epoch 69/100 - Loss: 0.0002 | Accuracy: 98.50%
40X Epoch 70/100: 100%|██████████| 25/25 [00:28<00:00,  1.12s/it]
Epoch 70/100 - Loss: 0.0000 | Accuracy: 98.50%
40X Epoch 71/100: 100%|██████████| 25/25 [00:27<00:00,  1.10s/it]
Epoch 71/100 - Loss: 0.0009 | Accuracy: 98.50%
40X Epoch 72/100: 100%|██████████| 25/25 [00:27<00:00,  1.11s/it]
Epoch 72/100 - Loss: 0.0001 | Accuracy: 98.75%
40X Epoch 73/100: 100%|██████████| 25/25 [00:27<00:00,  1.11s/it]
Epoch 73/100 - Loss: 0.0001 | Accuracy: 98.75%
40X Epoch 74/100: 100%|██████████| 25/25 [00:27<00:00,  1.11s/it]
Epoch 74/100 - Loss: 0.0000 | Accuracy: 98.75%
40X Epoch 75/100: 100%|██████████| 25/25 [00:27<00:00,  1.12s/it]
Epoch 75/100 - Loss: 0.0000 | Accuracy: 98.75%
40X Epoch 76/100: 100%|██████████| 25/25 [00:27<00:00,  1.11s/it]
Epoch 76/100 - Loss: 0.0002 | Accuracy: 98.75%
40X Epoch 77/100: 100%|██████████| 25/25 [00:27<00:00,  1.10s/it]
Epoch 77/100 - Loss: 0.0000 | Accuracy: 98.75%
40X Epoch 78/100: 100%|██████████| 25/25 [00:28<00:00,  1.12s/it]
Epoch 78/100 - Loss: 0.0001 | Accuracy: 98.75%
40X Epoch 79/100: 100%|██████████| 25/25 [00:27<00:00,  1.11s/it]
Epoch 79/100 - Loss: 0.0001 | Accuracy: 99.00%
40X Epoch 80/100: 100%|██████████| 25/25 [00:28<00:00,  1.13s/it]
Epoch 80/100 - Loss: 0.0000 | Accuracy: 99.00%
40X Epoch 81/100: 100%|██████████| 25/25 [00:27<00:00,  1.11s/it]
Epoch 81/100 - Loss: 0.0000 | Accuracy: 99.00%
40X Epoch 82/100: 100%|██████████| 25/25 [00:29<00:00,  1.17s/it]
Epoch 82/100 - Loss: 0.0001 | Accuracy: 98.75%
40X Epoch 83/100: 100%|██████████| 25/25 [00:28<00:00,  1.12s/it]
Epoch 83/100 - Loss: 0.0001 | Accuracy: 98.75%
40X Epoch 84/100: 100%|██████████| 25/25 [00:28<00:00,  1.13s/it]
Epoch 84/100 - Loss: 0.0000 | Accuracy: 98.75%
40X Epoch 85/100: 100%|██████████| 25/25 [00:27<00:00,  1.12s/it]
Epoch 85/100 - Loss: 0.0000 | Accuracy: 98.75%
40X Epoch 86/100: 100%|██████████| 25/25 [00:27<00:00,  1.10s/it]
Epoch 86/100 - Loss: 0.0000 | Accuracy: 98.75%
40X Epoch 87/100: 100%|██████████| 25/25 [00:27<00:00,  1.11s/it]
Epoch 87/100 - Loss: 0.0000 | Accuracy: 98.75%
40X Epoch 88/100: 100%|██████████| 25/25 [00:27<00:00,  1.11s/it]
Epoch 88/100 - Loss: 0.0000 | Accuracy: 98.75%
40X Epoch 89/100: 100%|██████████| 25/25 [00:28<00:00,  1.12s/it]
Epoch 89/100 - Loss: 0.0000 | Accuracy: 98.75%
40X Epoch 90/100: 100%|██████████| 25/25 [00:29<00:00,  1.18s/it]
Epoch 90/100 - Loss: 0.0000 | Accuracy: 98.75%
40X Epoch 91/100: 100%|██████████| 25/25 [00:28<00:00,  1.13s/it]
Epoch 91/100 - Loss: 0.0001 | Accuracy: 99.00%
40X Epoch 92/100: 100%|██████████| 25/25 [00:27<00:00,  1.11s/it]
Epoch 92/100 - Loss: 0.0000 | Accuracy: 99.00%
40X Epoch 93/100: 100%|██████████| 25/25 [00:28<00:00,  1.13s/it]
Epoch 93/100 - Loss: 0.0000 | Accuracy: 99.00%
40X Epoch 94/100: 100%|██████████| 25/25 [00:28<00:00,  1.13s/it]
Epoch 94/100 - Loss: 0.0000 | Accuracy: 99.00%
40X Epoch 95/100: 100%|██████████| 25/25 [00:28<00:00,  1.14s/it]
Epoch 95/100 - Loss: 0.0000 | Accuracy: 99.00%
40X Epoch 96/100: 100%|██████████| 25/25 [00:27<00:00,  1.12s/it]
Epoch 96/100 - Loss: 0.0000 | Accuracy: 99.00%
40X Epoch 97/100: 100%|██████████| 25/25 [00:27<00:00,  1.12s/it]
Epoch 97/100 - Loss: 0.0000 | Accuracy: 99.00%
40X Epoch 98/100: 100%|██████████| 25/25 [00:29<00:00,  1.17s/it]
Epoch 98/100 - Loss: 0.0000 | Accuracy: 99.00%
40X Epoch 99/100: 100%|██████████| 25/25 [00:27<00:00,  1.12s/it]
Epoch 99/100 - Loss: 0.0000 | Accuracy: 99.00%
40X Epoch 100/100: 100%|██████████| 25/25 [00:28<00:00,  1.13s/it]
Epoch 100/100 - Loss: 0.0000 | Accuracy: 99.00%

──────────────────────────────────────────────────────────────────────
FINAL EVALUATION - 40X
──────────────────────────────────────────────────────────────────────
              precision    recall  f1-score   support

      Benign     0.9925    0.9925    0.9925       133
   Malignant     0.9962    0.9962    0.9962       266

    accuracy                         0.9950       399
   macro avg     0.9944    0.9944    0.9944       399
weighted avg     0.9950    0.9950    0.9950       399


Confusion Matrix:
                Predicted
              Benign  Malignant
Actual Benign    132         1
    Malignant      1       265

✓ 40X Training Complete!
Best Accuracy: 99.50%
Model saved: /content/drive/MyDrive/best_breakhis_40X.pth

======================================================================
TRAINING: 100X MAGNIFICATION
======================================================================

Loading 100X dataset...
Total images: 2081
  Benign: 644 | Malignant: 1437
  Train batches: 26 | Val batches: 7

Creating ViT-Small model for 100X...
Starting training for 100 epochs...

100X Epoch 1/100: 100%|██████████| 26/26 [00:32<00:00,  1.24s/it]
Epoch 1/100 - Loss: 0.7195 | Accuracy: 80.58% ✓ BEST!
100X Epoch 2/100: 100%|██████████| 26/26 [00:29<00:00,  1.13s/it]
Epoch 2/100 - Loss: 0.4818 | Accuracy: 83.45% ✓ BEST!
100X Epoch 3/100: 100%|██████████| 26/26 [00:31<00:00,  1.20s/it]
Epoch 3/100 - Loss: 0.4191 | Accuracy: 78.90%
100X Epoch 4/100: 100%|██████████| 26/26 [00:29<00:00,  1.13s/it]
Epoch 4/100 - Loss: 0.3485 | Accuracy: 85.61% ✓ BEST!
100X Epoch 5/100: 100%|██████████| 26/26 [00:31<00:00,  1.19s/it]
Epoch 5/100 - Loss: 0.4101 | Accuracy: 87.05% ✓ BEST!
100X Epoch 6/100: 100%|██████████| 26/26 [00:32<00:00,  1.26s/it]
Epoch 6/100 - Loss: 0.3091 | Accuracy: 88.73% ✓ BEST!
100X Epoch 7/100: 100%|██████████| 26/26 [00:31<00:00,  1.22s/it]
Epoch 7/100 - Loss: 0.2789 | Accuracy: 90.65% ✓ BEST!
100X Epoch 8/100: 100%|██████████| 26/26 [00:30<00:00,  1.19s/it]
Epoch 8/100 - Loss: 0.2997 | Accuracy: 89.69%
100X Epoch 9/100: 100%|██████████| 26/26 [00:29<00:00,  1.14s/it]
Epoch 9/100 - Loss: 0.2683 | Accuracy: 87.53%
100X Epoch 10/100: 100%|██████████| 26/26 [00:29<00:00,  1.15s/it]
Epoch 10/100 - Loss: 0.2528 | Accuracy: 88.97%
100X Epoch 11/100: 100%|██████████| 26/26 [00:29<00:00,  1.14s/it]
Epoch 11/100 - Loss: 0.2074 | Accuracy: 90.65%
100X Epoch 12/100: 100%|██████████| 26/26 [00:29<00:00,  1.13s/it]
Epoch 12/100 - Loss: 0.2366 | Accuracy: 82.25%
100X Epoch 13/100: 100%|██████████| 26/26 [00:30<00:00,  1.17s/it]
Epoch 13/100 - Loss: 0.2114 | Accuracy: 91.85% ✓ BEST!
100X Epoch 14/100: 100%|██████████| 26/26 [00:31<00:00,  1.20s/it]
Epoch 14/100 - Loss: 0.1801 | Accuracy: 92.09% ✓ BEST!
100X Epoch 15/100: 100%|██████████| 26/26 [00:32<00:00,  1.23s/it]
Epoch 15/100 - Loss: 0.1676 | Accuracy: 90.65%
100X Epoch 16/100: 100%|██████████| 26/26 [00:29<00:00,  1.12s/it]
Epoch 16/100 - Loss: 0.1697 | Accuracy: 95.20% ✓ BEST!
100X Epoch 17/100: 100%|██████████| 26/26 [00:31<00:00,  1.19s/it]
Epoch 17/100 - Loss: 0.1291 | Accuracy: 95.20%
100X Epoch 18/100: 100%|██████████| 26/26 [00:29<00:00,  1.12s/it]
Epoch 18/100 - Loss: 0.1512 | Accuracy: 91.61%
100X Epoch 19/100: 100%|██████████| 26/26 [00:29<00:00,  1.12s/it]
Epoch 19/100 - Loss: 0.1214 | Accuracy: 96.16% ✓ BEST!
100X Epoch 20/100: 100%|██████████| 26/26 [00:31<00:00,  1.20s/it]
Epoch 20/100 - Loss: 0.0979 | Accuracy: 94.24%
100X Epoch 21/100: 100%|██████████| 26/26 [00:29<00:00,  1.13s/it]
Epoch 21/100 - Loss: 0.0946 | Accuracy: 95.44%
100X Epoch 22/100: 100%|██████████| 26/26 [00:29<00:00,  1.12s/it]
Epoch 22/100 - Loss: 0.1295 | Accuracy: 85.13%
100X Epoch 23/100: 100%|██████████| 26/26 [00:29<00:00,  1.12s/it]
Epoch 23/100 - Loss: 0.1389 | Accuracy: 95.20%
100X Epoch 24/100: 100%|██████████| 26/26 [00:29<00:00,  1.13s/it]
Epoch 24/100 - Loss: 0.0831 | Accuracy: 93.76%
100X Epoch 25/100: 100%|██████████| 26/26 [00:28<00:00,  1.11s/it]
Epoch 25/100 - Loss: 0.1157 | Accuracy: 95.44%
100X Epoch 26/100: 100%|██████████| 26/26 [00:29<00:00,  1.13s/it]
Epoch 26/100 - Loss: 0.0647 | Accuracy: 96.88% ✓ BEST!
100X Epoch 27/100: 100%|██████████| 26/26 [00:30<00:00,  1.17s/it]
Epoch 27/100 - Loss: 0.0587 | Accuracy: 97.36% ✓ BEST!
100X Epoch 28/100: 100%|██████████| 26/26 [00:32<00:00,  1.23s/it]
Epoch 28/100 - Loss: 0.0429 | Accuracy: 97.12%
100X Epoch 29/100: 100%|██████████| 26/26 [00:29<00:00,  1.13s/it]
Epoch 29/100 - Loss: 0.0627 | Accuracy: 95.68%
100X Epoch 30/100: 100%|██████████| 26/26 [00:29<00:00,  1.12s/it]
Epoch 30/100 - Loss: 0.0305 | Accuracy: 96.88%
100X Epoch 31/100: 100%|██████████| 26/26 [00:29<00:00,  1.13s/it]
Epoch 31/100 - Loss: 0.0546 | Accuracy: 96.16%
100X Epoch 32/100: 100%|██████████| 26/26 [00:29<00:00,  1.14s/it]
Epoch 32/100 - Loss: 0.0443 | Accuracy: 96.88%
100X Epoch 33/100: 100%|██████████| 26/26 [00:29<00:00,  1.13s/it]
Epoch 33/100 - Loss: 0.0547 | Accuracy: 97.36%
100X Epoch 34/100: 100%|██████████| 26/26 [00:29<00:00,  1.12s/it]
Epoch 34/100 - Loss: 0.0251 | Accuracy: 96.64%
100X Epoch 35/100: 100%|██████████| 26/26 [00:29<00:00,  1.12s/it]
Epoch 35/100 - Loss: 0.0611 | Accuracy: 95.68%
100X Epoch 36/100: 100%|██████████| 26/26 [00:30<00:00,  1.18s/it]
Epoch 36/100 - Loss: 0.0279 | Accuracy: 98.08% ✓ BEST!
100X Epoch 37/100: 100%|██████████| 26/26 [00:30<00:00,  1.18s/it]
Epoch 37/100 - Loss: 0.0117 | Accuracy: 97.84%
100X Epoch 38/100: 100%|██████████| 26/26 [00:29<00:00,  1.13s/it]
Epoch 38/100 - Loss: 0.0645 | Accuracy: 95.44%
100X Epoch 39/100: 100%|██████████| 26/26 [00:29<00:00,  1.14s/it]
Epoch 39/100 - Loss: 0.0484 | Accuracy: 97.60%
100X Epoch 40/100: 100%|██████████| 26/26 [00:29<00:00,  1.14s/it]
Epoch 40/100 - Loss: 0.0230 | Accuracy: 96.64%
100X Epoch 41/100: 100%|██████████| 26/26 [00:29<00:00,  1.14s/it]
Epoch 41/100 - Loss: 0.0226 | Accuracy: 97.36%
100X Epoch 42/100: 100%|██████████| 26/26 [00:29<00:00,  1.14s/it]
Epoch 42/100 - Loss: 0.0240 | Accuracy: 97.84%
100X Epoch 43/100: 100%|██████████| 26/26 [00:30<00:00,  1.18s/it]
Epoch 43/100 - Loss: 0.0085 | Accuracy: 98.08%
100X Epoch 44/100: 100%|██████████| 26/26 [00:29<00:00,  1.13s/it]
Epoch 44/100 - Loss: 0.0171 | Accuracy: 96.16%
100X Epoch 45/100: 100%|██████████| 26/26 [00:29<00:00,  1.12s/it]
Epoch 45/100 - Loss: 0.0473 | Accuracy: 96.40%
100X Epoch 46/100: 100%|██████████| 26/26 [00:29<00:00,  1.14s/it]
Epoch 46/100 - Loss: 0.0154 | Accuracy: 95.44%
100X Epoch 47/100: 100%|██████████| 26/26 [00:29<00:00,  1.12s/it]
Epoch 47/100 - Loss: 0.0166 | Accuracy: 96.88%
100X Epoch 48/100: 100%|██████████| 26/26 [00:29<00:00,  1.12s/it]
Epoch 48/100 - Loss: 0.0229 | Accuracy: 97.36%
100X Epoch 49/100: 100%|██████████| 26/26 [00:28<00:00,  1.10s/it]
Epoch 49/100 - Loss: 0.0228 | Accuracy: 96.88%
100X Epoch 50/100: 100%|██████████| 26/26 [00:29<00:00,  1.12s/it]
Epoch 50/100 - Loss: 0.0135 | Accuracy: 95.92%
100X Epoch 51/100: 100%|██████████| 26/26 [00:30<00:00,  1.17s/it]
Epoch 51/100 - Loss: 0.0115 | Accuracy: 97.36%
100X Epoch 52/100: 100%|██████████| 26/26 [00:29<00:00,  1.12s/it]
Epoch 52/100 - Loss: 0.0059 | Accuracy: 97.84%
100X Epoch 53/100: 100%|██████████| 26/26 [00:29<00:00,  1.12s/it]
Epoch 53/100 - Loss: 0.0045 | Accuracy: 97.12%
100X Epoch 54/100: 100%|██████████| 26/26 [00:29<00:00,  1.13s/it]
Epoch 54/100 - Loss: 0.0087 | Accuracy: 98.56% ✓ BEST!
100X Epoch 55/100: 100%|██████████| 26/26 [00:31<00:00,  1.22s/it]
Epoch 55/100 - Loss: 0.0448 | Accuracy: 97.12%
100X Epoch 56/100: 100%|██████████| 26/26 [00:29<00:00,  1.13s/it]
Epoch 56/100 - Loss: 0.0252 | Accuracy: 98.32%
100X Epoch 57/100: 100%|██████████| 26/26 [00:29<00:00,  1.12s/it]
Epoch 57/100 - Loss: 0.0041 | Accuracy: 98.80% ✓ BEST!
100X Epoch 58/100: 100%|██████████| 26/26 [00:30<00:00,  1.18s/it]
Epoch 58/100 - Loss: 0.0010 | Accuracy: 98.56%
100X Epoch 59/100: 100%|██████████| 26/26 [00:30<00:00,  1.17s/it]
Epoch 59/100 - Loss: 0.0009 | Accuracy: 98.56%
100X Epoch 60/100: 100%|██████████| 26/26 [00:29<00:00,  1.12s/it]
Epoch 60/100 - Loss: 0.0013 | Accuracy: 98.80%
100X Epoch 61/100: 100%|██████████| 26/26 [00:28<00:00,  1.11s/it]
Epoch 61/100 - Loss: 0.0016 | Accuracy: 98.56%
100X Epoch 62/100: 100%|██████████| 26/26 [00:29<00:00,  1.12s/it]
Epoch 62/100 - Loss: 0.0007 | Accuracy: 98.56%
100X Epoch 63/100: 100%|██████████| 26/26 [00:29<00:00,  1.12s/it]
Epoch 63/100 - Loss: 0.0050 | Accuracy: 97.84%
100X Epoch 64/100: 100%|██████████| 26/26 [00:29<00:00,  1.13s/it]
Epoch 64/100 - Loss: 0.0010 | Accuracy: 98.56%
100X Epoch 65/100: 100%|██████████| 26/26 [00:29<00:00,  1.13s/it]
Epoch 65/100 - Loss: 0.0032 | Accuracy: 98.32%
100X Epoch 66/100: 100%|██████████| 26/26 [00:29<00:00,  1.13s/it]
Epoch 66/100 - Loss: 0.0004 | Accuracy: 98.32%
100X Epoch 67/100: 100%|██████████| 26/26 [00:29<00:00,  1.13s/it]
Epoch 67/100 - Loss: 0.0004 | Accuracy: 98.32%
100X Epoch 68/100: 100%|██████████| 26/26 [00:29<00:00,  1.14s/it]
Epoch 68/100 - Loss: 0.0014 | Accuracy: 97.60%
100X Epoch 69/100: 100%|██████████| 26/26 [00:29<00:00,  1.13s/it]
Epoch 69/100 - Loss: 0.0009 | Accuracy: 98.32%
100X Epoch 70/100: 100%|██████████| 26/26 [00:29<00:00,  1.12s/it]
Epoch 70/100 - Loss: 0.0010 | Accuracy: 98.32%
100X Epoch 71/100: 100%|██████████| 26/26 [00:29<00:00,  1.13s/it]
Epoch 71/100 - Loss: 0.0010 | Accuracy: 98.08%
100X Epoch 72/100: 100%|██████████| 26/26 [00:29<00:00,  1.13s/it]
Epoch 72/100 - Loss: 0.0003 | Accuracy: 98.08%
100X Epoch 73/100: 100%|██████████| 26/26 [00:29<00:00,  1.14s/it]
Epoch 73/100 - Loss: 0.0005 | Accuracy: 97.84%
100X Epoch 74/100: 100%|██████████| 26/26 [00:30<00:00,  1.16s/it]
Epoch 74/100 - Loss: 0.0002 | Accuracy: 98.08%
100X Epoch 75/100: 100%|██████████| 26/26 [00:29<00:00,  1.15s/it]
Epoch 75/100 - Loss: 0.0002 | Accuracy: 98.08%
100X Epoch 76/100: 100%|██████████| 26/26 [00:29<00:00,  1.12s/it]
Epoch 76/100 - Loss: 0.0001 | Accuracy: 98.08%
100X Epoch 77/100: 100%|██████████| 26/26 [00:29<00:00,  1.14s/it]
Epoch 77/100 - Loss: 0.0001 | Accuracy: 98.08%
100X Epoch 78/100: 100%|██████████| 26/26 [00:29<00:00,  1.12s/it]
Epoch 78/100 - Loss: 0.0003 | Accuracy: 98.08%
100X Epoch 79/100: 100%|██████████| 26/26 [00:29<00:00,  1.12s/it]
Epoch 79/100 - Loss: 0.0009 | Accuracy: 98.32%
100X Epoch 80/100: 100%|██████████| 26/26 [00:29<00:00,  1.14s/it]
Epoch 80/100 - Loss: 0.0003 | Accuracy: 98.32%
100X Epoch 81/100: 100%|██████████| 26/26 [00:29<00:00,  1.14s/it]
Epoch 81/100 - Loss: 0.0001 | Accuracy: 98.32%
100X Epoch 82/100: 100%|██████████| 26/26 [00:30<00:00,  1.19s/it]
Epoch 82/100 - Loss: 0.0001 | Accuracy: 98.32%
100X Epoch 83/100: 100%|██████████| 26/26 [00:29<00:00,  1.13s/it]
Epoch 83/100 - Loss: 0.0006 | Accuracy: 98.32%
100X Epoch 84/100: 100%|██████████| 26/26 [00:29<00:00,  1.12s/it]
Epoch 84/100 - Loss: 0.0001 | Accuracy: 98.32%
100X Epoch 85/100: 100%|██████████| 26/26 [00:28<00:00,  1.11s/it]
Epoch 85/100 - Loss: 0.0014 | Accuracy: 98.32%
100X Epoch 86/100: 100%|██████████| 26/26 [00:28<00:00,  1.11s/it]
Epoch 86/100 - Loss: 0.0001 | Accuracy: 98.08%
100X Epoch 87/100: 100%|██████████| 26/26 [00:29<00:00,  1.13s/it]
Epoch 87/100 - Loss: 0.0001 | Accuracy: 98.08%
100X Epoch 88/100: 100%|██████████| 26/26 [00:29<00:00,  1.14s/it]
Epoch 88/100 - Loss: 0.0001 | Accuracy: 98.08%
100X Epoch 89/100: 100%|██████████| 26/26 [00:28<00:00,  1.11s/it]
Epoch 89/100 - Loss: 0.0002 | Accuracy: 98.32%
100X Epoch 90/100: 100%|██████████| 26/26 [00:30<00:00,  1.19s/it]
Epoch 90/100 - Loss: 0.0001 | Accuracy: 98.32%
100X Epoch 91/100: 100%|██████████| 26/26 [00:30<00:00,  1.17s/it]
Epoch 91/100 - Loss: 0.0001 | Accuracy: 98.32%
100X Epoch 92/100: 100%|██████████| 26/26 [00:29<00:00,  1.15s/it]
Epoch 92/100 - Loss: 0.0005 | Accuracy: 98.32%
100X Epoch 93/100: 100%|██████████| 26/26 [00:30<00:00,  1.17s/it]
Epoch 93/100 - Loss: 0.0001 | Accuracy: 98.56%
100X Epoch 94/100: 100%|██████████| 26/26 [00:30<00:00,  1.17s/it]
Epoch 94/100 - Loss: 0.0001 | Accuracy: 98.56%
100X Epoch 95/100: 100%|██████████| 26/26 [00:30<00:00,  1.17s/it]
Epoch 95/100 - Loss: 0.0001 | Accuracy: 98.56%
100X Epoch 96/100: 100%|██████████| 26/26 [00:30<00:00,  1.16s/it]
Epoch 96/100 - Loss: 0.0001 | Accuracy: 98.56%
100X Epoch 97/100: 100%|██████████| 26/26 [00:30<00:00,  1.19s/it]
Epoch 97/100 - Loss: 0.0000 | Accuracy: 98.56%
100X Epoch 98/100: 100%|██████████| 26/26 [00:30<00:00,  1.19s/it]
Epoch 98/100 - Loss: 0.0001 | Accuracy: 98.56%
100X Epoch 99/100: 100%|██████████| 26/26 [00:29<00:00,  1.15s/it]
Epoch 99/100 - Loss: 0.0001 | Accuracy: 98.56%
100X Epoch 100/100: 100%|██████████| 26/26 [00:30<00:00,  1.19s/it]
Epoch 100/100 - Loss: 0.0004 | Accuracy: 98.56%

──────────────────────────────────────────────────────────────────────
FINAL EVALUATION - 100X
──────────────────────────────────────────────────────────────────────
              precision    recall  f1-score   support

      Benign     1.0000    0.9612    0.9802       129
   Malignant     0.9829    1.0000    0.9914       288

    accuracy                         0.9880       417
   macro avg     0.9915    0.9806    0.9858       417
weighted avg     0.9882    0.9880    0.9879       417


Confusion Matrix:
                Predicted
              Benign  Malignant
Actual Benign    124         5
    Malignant      0       288

✓ 100X Training Complete!
Best Accuracy: 98.80%
Model saved: /content/drive/MyDrive/best_breakhis_100X.pth

======================================================================
TRAINING: 200X MAGNIFICATION
======================================================================

Loading 200X dataset...
Total images: 2013
  Benign: 623 | Malignant: 1390
  Train batches: 26 | Val batches: 7

Creating ViT-Small model for 200X...
Starting training for 100 epochs...

200X Epoch 1/100: 100%|██████████| 26/26 [00:31<00:00,  1.20s/it]
Epoch 1/100 - Loss: 0.6544 | Accuracy: 74.44% ✓ BEST!
200X Epoch 2/100: 100%|██████████| 26/26 [00:29<00:00,  1.12s/it]
Epoch 2/100 - Loss: 0.4521 | Accuracy: 86.85% ✓ BEST!
200X Epoch 3/100: 100%|██████████| 26/26 [00:30<00:00,  1.18s/it]
Epoch 3/100 - Loss: 0.3699 | Accuracy: 87.10% ✓ BEST!
200X Epoch 4/100: 100%|██████████| 26/26 [00:31<00:00,  1.19s/it]
Epoch 4/100 - Loss: 0.3121 | Accuracy: 88.59% ✓ BEST!
200X Epoch 5/100: 100%|██████████| 26/26 [00:32<00:00,  1.24s/it]
Epoch 5/100 - Loss: 0.3469 | Accuracy: 89.58% ✓ BEST!
200X Epoch 6/100: 100%|██████████| 26/26 [00:29<00:00,  1.15s/it]
Epoch 6/100 - Loss: 0.2705 | Accuracy: 94.79% ✓ BEST!
200X Epoch 7/100: 100%|██████████| 26/26 [00:30<00:00,  1.17s/it]
Epoch 7/100 - Loss: 0.2609 | Accuracy: 91.32%
200X Epoch 8/100: 100%|██████████| 26/26 [00:28<00:00,  1.09s/it]
Epoch 8/100 - Loss: 0.3061 | Accuracy: 84.37%
200X Epoch 9/100: 100%|██████████| 26/26 [00:28<00:00,  1.09s/it]
Epoch 9/100 - Loss: 0.2514 | Accuracy: 94.54%
200X Epoch 10/100: 100%|██████████| 26/26 [00:28<00:00,  1.10s/it]
Epoch 10/100 - Loss: 0.1961 | Accuracy: 91.07%
200X Epoch 11/100: 100%|██████████| 26/26 [00:28<00:00,  1.10s/it]
Epoch 11/100 - Loss: 0.1764 | Accuracy: 95.04% ✓ BEST!
200X Epoch 12/100: 100%|██████████| 26/26 [00:30<00:00,  1.15s/it]
Epoch 12/100 - Loss: 0.1629 | Accuracy: 93.80%
200X Epoch 13/100: 100%|██████████| 26/26 [00:30<00:00,  1.15s/it]
Epoch 13/100 - Loss: 0.1407 | Accuracy: 97.27% ✓ BEST!
200X Epoch 14/100: 100%|██████████| 26/26 [00:29<00:00,  1.14s/it]
Epoch 14/100 - Loss: 0.1371 | Accuracy: 93.55%
200X Epoch 15/100: 100%|██████████| 26/26 [00:27<00:00,  1.07s/it]
Epoch 15/100 - Loss: 0.1311 | Accuracy: 97.27%
200X Epoch 16/100: 100%|██████████| 26/26 [00:28<00:00,  1.10s/it]
Epoch 16/100 - Loss: 0.1098 | Accuracy: 96.53%
200X Epoch 17/100: 100%|██████████| 26/26 [00:28<00:00,  1.08s/it]
Epoch 17/100 - Loss: 0.1358 | Accuracy: 96.77%
200X Epoch 18/100: 100%|██████████| 26/26 [00:28<00:00,  1.10s/it]
Epoch 18/100 - Loss: 0.1056 | Accuracy: 96.53%
200X Epoch 19/100: 100%|██████████| 26/26 [00:29<00:00,  1.12s/it]
Epoch 19/100 - Loss: 0.0642 | Accuracy: 91.81%
200X Epoch 20/100: 100%|██████████| 26/26 [00:28<00:00,  1.11s/it]
Epoch 20/100 - Loss: 0.1297 | Accuracy: 98.26% ✓ BEST!
200X Epoch 21/100: 100%|██████████| 26/26 [00:31<00:00,  1.22s/it]
Epoch 21/100 - Loss: 0.0740 | Accuracy: 98.51% ✓ BEST!
200X Epoch 22/100: 100%|██████████| 26/26 [00:30<00:00,  1.16s/it]
Epoch 22/100 - Loss: 0.0602 | Accuracy: 97.27%
200X Epoch 23/100: 100%|██████████| 26/26 [00:28<00:00,  1.10s/it]
Epoch 23/100 - Loss: 0.0746 | Accuracy: 95.78%
200X Epoch 24/100: 100%|██████████| 26/26 [00:28<00:00,  1.10s/it]
Epoch 24/100 - Loss: 0.0680 | Accuracy: 93.30%
200X Epoch 25/100: 100%|██████████| 26/26 [00:28<00:00,  1.11s/it]
Epoch 25/100 - Loss: 0.0618 | Accuracy: 98.26%
200X Epoch 26/100: 100%|██████████| 26/26 [00:28<00:00,  1.11s/it]
Epoch 26/100 - Loss: 0.0359 | Accuracy: 98.51%
200X Epoch 27/100: 100%|██████████| 26/26 [00:28<00:00,  1.10s/it]
Epoch 27/100 - Loss: 0.0627 | Accuracy: 98.01%
200X Epoch 28/100: 100%|██████████| 26/26 [00:29<00:00,  1.14s/it]
Epoch 28/100 - Loss: 0.0569 | Accuracy: 96.28%
200X Epoch 29/100: 100%|██████████| 26/26 [00:30<00:00,  1.19s/it]
Epoch 29/100 - Loss: 0.1102 | Accuracy: 94.54%
200X Epoch 30/100: 100%|██████████| 26/26 [00:28<00:00,  1.11s/it]
Epoch 30/100 - Loss: 0.0627 | Accuracy: 97.02%
200X Epoch 31/100: 100%|██████████| 26/26 [00:29<00:00,  1.15s/it]
Epoch 31/100 - Loss: 0.0592 | Accuracy: 95.04%
200X Epoch 32/100: 100%|██████████| 26/26 [00:30<00:00,  1.16s/it]
Epoch 32/100 - Loss: 0.1054 | Accuracy: 98.26%
200X Epoch 33/100: 100%|██████████| 26/26 [00:30<00:00,  1.15s/it]
Epoch 33/100 - Loss: 0.0428 | Accuracy: 98.76% ✓ BEST!
200X Epoch 34/100: 100%|██████████| 26/26 [00:31<00:00,  1.22s/it]
Epoch 34/100 - Loss: 0.0481 | Accuracy: 92.06%
200X Epoch 35/100: 100%|██████████| 26/26 [00:30<00:00,  1.16s/it]
Epoch 35/100 - Loss: 0.1027 | Accuracy: 98.01%
200X Epoch 36/100: 100%|██████████| 26/26 [00:29<00:00,  1.14s/it]
Epoch 36/100 - Loss: 0.0380 | Accuracy: 97.52%
200X Epoch 37/100: 100%|██████████| 26/26 [00:29<00:00,  1.13s/it]
Epoch 37/100 - Loss: 0.0562 | Accuracy: 97.27%
200X Epoch 38/100: 100%|██████████| 26/26 [00:29<00:00,  1.13s/it]
Epoch 38/100 - Loss: 0.0355 | Accuracy: 97.27%
200X Epoch 39/100: 100%|██████████| 26/26 [00:29<00:00,  1.14s/it]
Epoch 39/100 - Loss: 0.0606 | Accuracy: 94.29%
200X Epoch 40/100: 100%|██████████| 26/26 [00:29<00:00,  1.14s/it]
Epoch 40/100 - Loss: 0.1034 | Accuracy: 96.77%
200X Epoch 41/100: 100%|██████████| 26/26 [00:29<00:00,  1.15s/it]
Epoch 41/100 - Loss: 0.0315 | Accuracy: 98.51%
200X Epoch 42/100: 100%|██████████| 26/26 [00:29<00:00,  1.15s/it]
Epoch 42/100 - Loss: 0.0130 | Accuracy: 97.77%
200X Epoch 43/100: 100%|██████████| 26/26 [00:28<00:00,  1.11s/it]
Epoch 43/100 - Loss: 0.0387 | Accuracy: 96.53%
200X Epoch 44/100: 100%|██████████| 26/26 [00:28<00:00,  1.11s/it]
Epoch 44/100 - Loss: 0.0314 | Accuracy: 98.51%
200X Epoch 45/100: 100%|██████████| 26/26 [00:29<00:00,  1.13s/it]
Epoch 45/100 - Loss: 0.0112 | Accuracy: 98.76%
200X Epoch 46/100: 100%|██████████| 26/26 [00:28<00:00,  1.11s/it]
Epoch 46/100 - Loss: 0.0139 | Accuracy: 98.26%
200X Epoch 47/100: 100%|██████████| 26/26 [00:28<00:00,  1.11s/it]
Epoch 47/100 - Loss: 0.0231 | Accuracy: 97.27%
200X Epoch 48/100: 100%|██████████| 26/26 [00:28<00:00,  1.10s/it]
Epoch 48/100 - Loss: 0.0189 | Accuracy: 98.76%
200X Epoch 49/100: 100%|██████████| 26/26 [00:28<00:00,  1.08s/it]
Epoch 49/100 - Loss: 0.0057 | Accuracy: 97.77%
200X Epoch 50/100: 100%|██████████| 26/26 [00:28<00:00,  1.09s/it]
Epoch 50/100 - Loss: 0.0144 | Accuracy: 98.51%
200X Epoch 51/100: 100%|██████████| 26/26 [00:28<00:00,  1.10s/it]
Epoch 51/100 - Loss: 0.0056 | Accuracy: 98.26%
200X Epoch 52/100: 100%|██████████| 26/26 [00:28<00:00,  1.09s/it]
Epoch 52/100 - Loss: 0.0119 | Accuracy: 98.51%
200X Epoch 53/100: 100%|██████████| 26/26 [00:29<00:00,  1.14s/it]
Epoch 53/100 - Loss: 0.0071 | Accuracy: 98.51%
200X Epoch 54/100: 100%|██████████| 26/26 [00:28<00:00,  1.10s/it]
Epoch 54/100 - Loss: 0.0076 | Accuracy: 98.76%
200X Epoch 55/100: 100%|██████████| 26/26 [00:28<00:00,  1.10s/it]
Epoch 55/100 - Loss: 0.0059 | Accuracy: 98.51%
200X Epoch 56/100: 100%|██████████| 26/26 [00:28<00:00,  1.09s/it]
Epoch 56/100 - Loss: 0.0181 | Accuracy: 98.26%
200X Epoch 57/100: 100%|██████████| 26/26 [00:28<00:00,  1.10s/it]
Epoch 57/100 - Loss: 0.0153 | Accuracy: 98.51%
200X Epoch 58/100: 100%|██████████| 26/26 [00:28<00:00,  1.08s/it]
Epoch 58/100 - Loss: 0.0170 | Accuracy: 98.76%
200X Epoch 59/100: 100%|██████████| 26/26 [00:28<00:00,  1.09s/it]
Epoch 59/100 - Loss: 0.0052 | Accuracy: 99.01% ✓ BEST!
200X Epoch 60/100: 100%|██████████| 26/26 [00:30<00:00,  1.18s/it]
Epoch 60/100 - Loss: 0.0089 | Accuracy: 99.01%
200X Epoch 61/100: 100%|██████████| 26/26 [00:29<00:00,  1.15s/it]
Epoch 61/100 - Loss: 0.0029 | Accuracy: 99.01%
200X Epoch 62/100: 100%|██████████| 26/26 [00:28<00:00,  1.08s/it]
Epoch 62/100 - Loss: 0.0008 | Accuracy: 99.26% ✓ BEST!
200X Epoch 63/100: 100%|██████████| 26/26 [00:29<00:00,  1.15s/it]
Epoch 63/100 - Loss: 0.0004 | Accuracy: 99.26%
200X Epoch 64/100: 100%|██████████| 26/26 [00:27<00:00,  1.06s/it]
Epoch 64/100 - Loss: 0.0003 | Accuracy: 99.26%
200X Epoch 65/100: 100%|██████████| 26/26 [00:28<00:00,  1.08s/it]
Epoch 65/100 - Loss: 0.0002 | Accuracy: 99.26%
200X Epoch 66/100: 100%|██████████| 26/26 [00:27<00:00,  1.07s/it]
Epoch 66/100 - Loss: 0.0024 | Accuracy: 99.26%
200X Epoch 67/100: 100%|██████████| 26/26 [00:27<00:00,  1.07s/it]
Epoch 67/100 - Loss: 0.0008 | Accuracy: 99.01%
200X Epoch 68/100: 100%|██████████| 26/26 [00:28<00:00,  1.08s/it]
Epoch 68/100 - Loss: 0.0007 | Accuracy: 99.01%
200X Epoch 69/100: 100%|██████████| 26/26 [00:29<00:00,  1.14s/it]
Epoch 69/100 - Loss: 0.0004 | Accuracy: 99.26%
200X Epoch 70/100: 100%|██████████| 26/26 [00:28<00:00,  1.08s/it]
Epoch 70/100 - Loss: 0.0008 | Accuracy: 99.01%
200X Epoch 71/100: 100%|██████████| 26/26 [00:27<00:00,  1.06s/it]
Epoch 71/100 - Loss: 0.0027 | Accuracy: 99.26%
200X Epoch 72/100: 100%|██████████| 26/26 [00:28<00:00,  1.09s/it]
Epoch 72/100 - Loss: 0.0063 | Accuracy: 99.26%
200X Epoch 73/100: 100%|██████████| 26/26 [00:28<00:00,  1.10s/it]
Epoch 73/100 - Loss: 0.0023 | Accuracy: 99.50% ✓ BEST!
200X Epoch 74/100: 100%|██████████| 26/26 [00:30<00:00,  1.16s/it]
Epoch 74/100 - Loss: 0.0083 | Accuracy: 99.01%
200X Epoch 75/100: 100%|██████████| 26/26 [00:28<00:00,  1.08s/it]
Epoch 75/100 - Loss: 0.0007 | Accuracy: 99.26%
200X Epoch 76/100: 100%|██████████| 26/26 [00:28<00:00,  1.08s/it]
Epoch 76/100 - Loss: 0.0014 | Accuracy: 99.26%
200X Epoch 77/100: 100%|██████████| 26/26 [00:28<00:00,  1.09s/it]
Epoch 77/100 - Loss: 0.0018 | Accuracy: 99.26%
200X Epoch 78/100: 100%|██████████| 26/26 [00:29<00:00,  1.15s/it]
Epoch 78/100 - Loss: 0.0023 | Accuracy: 99.26%
200X Epoch 79/100: 100%|██████████| 26/26 [00:28<00:00,  1.10s/it]
Epoch 79/100 - Loss: 0.0023 | Accuracy: 99.01%
200X Epoch 80/100: 100%|██████████| 26/26 [00:28<00:00,  1.09s/it]
Epoch 80/100 - Loss: 0.0008 | Accuracy: 99.26%
200X Epoch 81/100: 100%|██████████| 26/26 [00:28<00:00,  1.09s/it]
Epoch 81/100 - Loss: 0.0006 | Accuracy: 99.26%
200X Epoch 82/100: 100%|██████████| 26/26 [00:28<00:00,  1.09s/it]
Epoch 82/100 - Loss: 0.0002 | Accuracy: 99.01%
200X Epoch 83/100: 100%|██████████| 26/26 [00:28<00:00,  1.09s/it]
Epoch 83/100 - Loss: 0.0001 | Accuracy: 99.01%
200X Epoch 84/100: 100%|██████████| 26/26 [00:28<00:00,  1.10s/it]
Epoch 84/100 - Loss: 0.0001 | Accuracy: 99.01%
200X Epoch 85/100: 100%|██████████| 26/26 [00:28<00:00,  1.09s/it]
Epoch 85/100 - Loss: 0.0002 | Accuracy: 99.01%
200X Epoch 86/100: 100%|██████████| 26/26 [00:30<00:00,  1.18s/it]
Epoch 86/100 - Loss: 0.0001 | Accuracy: 99.01%
200X Epoch 87/100: 100%|██████████| 26/26 [00:28<00:00,  1.10s/it]
Epoch 87/100 - Loss: 0.0002 | Accuracy: 99.01%
200X Epoch 88/100: 100%|██████████| 26/26 [00:28<00:00,  1.08s/it]
Epoch 88/100 - Loss: 0.0001 | Accuracy: 99.01%
200X Epoch 89/100: 100%|██████████| 26/26 [00:28<00:00,  1.08s/it]
Epoch 89/100 - Loss: 0.0004 | Accuracy: 99.01%
200X Epoch 90/100: 100%|██████████| 26/26 [00:28<00:00,  1.08s/it]
Epoch 90/100 - Loss: 0.0001 | Accuracy: 99.01%
200X Epoch 91/100: 100%|██████████| 26/26 [00:28<00:00,  1.09s/it]
Epoch 91/100 - Loss: 0.0001 | Accuracy: 99.01%
200X Epoch 92/100: 100%|██████████| 26/26 [00:28<00:00,  1.09s/it]
Epoch 92/100 - Loss: 0.0005 | Accuracy: 99.01%
200X Epoch 93/100: 100%|██████████| 26/26 [00:27<00:00,  1.05s/it]
Epoch 93/100 - Loss: 0.0002 | Accuracy: 99.01%
200X Epoch 94/100: 100%|██████████| 26/26 [00:28<00:00,  1.10s/it]
Epoch 94/100 - Loss: 0.0001 | Accuracy: 99.01%
200X Epoch 95/100: 100%|██████████| 26/26 [00:30<00:00,  1.17s/it]
Epoch 95/100 - Loss: 0.0001 | Accuracy: 99.01%
200X Epoch 96/100: 100%|██████████| 26/26 [00:28<00:00,  1.09s/it]
Epoch 96/100 - Loss: 0.0001 | Accuracy: 99.01%
200X Epoch 97/100: 100%|██████████| 26/26 [00:28<00:00,  1.10s/it]
Epoch 97/100 - Loss: 0.0003 | Accuracy: 99.01%
200X Epoch 98/100: 100%|██████████| 26/26 [00:28<00:00,  1.10s/it]
Epoch 98/100 - Loss: 0.0002 | Accuracy: 99.01%
200X Epoch 99/100: 100%|██████████| 26/26 [00:28<00:00,  1.10s/it]
Epoch 99/100 - Loss: 0.0025 | Accuracy: 99.01%
200X Epoch 100/100: 100%|██████████| 26/26 [00:27<00:00,  1.07s/it]
Epoch 100/100 - Loss: 0.0002 | Accuracy: 99.01%

──────────────────────────────────────────────────────────────────────
FINAL EVALUATION - 200X
──────────────────────────────────────────────────────────────────────
              precision    recall  f1-score   support

      Benign     1.0000    0.9846    0.9922       130
   Malignant     0.9927    1.0000    0.9964       273

    accuracy                         0.9950       403
   macro avg     0.9964    0.9923    0.9943       403
weighted avg     0.9951    0.9950    0.9950       403


Confusion Matrix:
                Predicted
              Benign  Malignant
Actual Benign    128         2
    Malignant      0       273

✓ 200X Training Complete!
Best Accuracy: 99.50%
Model saved: /content/drive/MyDrive/best_breakhis_200X.pth

======================================================================
TRAINING: 400X MAGNIFICATION
======================================================================

Loading 400X dataset...
Total images: 1820
  Benign: 588 | Malignant: 1232
  Train batches: 23 | Val batches: 6

Creating ViT-Small model for 400X...
Starting training for 100 epochs...

400X Epoch 1/100: 100%|██████████| 23/23 [00:27<00:00,  1.18s/it]
Epoch 1/100 - Loss: 0.5801 | Accuracy: 86.81% ✓ BEST!
400X Epoch 2/100: 100%|██████████| 23/23 [00:25<00:00,  1.11s/it]
Epoch 2/100 - Loss: 0.3741 | Accuracy: 89.01% ✓ BEST!
400X Epoch 3/100: 100%|██████████| 23/23 [00:27<00:00,  1.18s/it]
Epoch 3/100 - Loss: 0.3644 | Accuracy: 88.46%
400X Epoch 4/100: 100%|██████████| 23/23 [00:26<00:00,  1.16s/it]
Epoch 4/100 - Loss: 0.3193 | Accuracy: 90.11% ✓ BEST!
400X Epoch 5/100: 100%|██████████| 23/23 [00:26<00:00,  1.16s/it]
Epoch 5/100 - Loss: 0.3133 | Accuracy: 88.46%
400X Epoch 6/100: 100%|██████████| 23/23 [00:25<00:00,  1.10s/it]
Epoch 6/100 - Loss: 0.2929 | Accuracy: 87.64%
400X Epoch 7/100: 100%|██████████| 23/23 [00:25<00:00,  1.10s/it]
Epoch 7/100 - Loss: 0.2458 | Accuracy: 90.11%
400X Epoch 8/100: 100%|██████████| 23/23 [00:25<00:00,  1.10s/it]
Epoch 8/100 - Loss: 0.2136 | Accuracy: 91.76% ✓ BEST!
400X Epoch 9/100: 100%|██████████| 23/23 [00:27<00:00,  1.19s/it]
Epoch 9/100 - Loss: 0.2244 | Accuracy: 94.51% ✓ BEST!
400X Epoch 10/100: 100%|██████████| 23/23 [00:26<00:00,  1.17s/it]
Epoch 10/100 - Loss: 0.1976 | Accuracy: 89.29%
400X Epoch 11/100: 100%|██████████| 23/23 [00:25<00:00,  1.09s/it]
Epoch 11/100 - Loss: 0.2096 | Accuracy: 93.68%
400X Epoch 12/100: 100%|██████████| 23/23 [00:25<00:00,  1.10s/it]
Epoch 12/100 - Loss: 0.1584 | Accuracy: 92.58%
400X Epoch 13/100: 100%|██████████| 23/23 [00:26<00:00,  1.15s/it]
Epoch 13/100 - Loss: 0.1421 | Accuracy: 92.86%
400X Epoch 14/100: 100%|██████████| 23/23 [00:24<00:00,  1.08s/it]
Epoch 14/100 - Loss: 0.1640 | Accuracy: 94.23%
400X Epoch 15/100: 100%|██████████| 23/23 [00:25<00:00,  1.10s/it]
Epoch 15/100 - Loss: 0.1480 | Accuracy: 95.88% ✓ BEST!
400X Epoch 16/100: 100%|██████████| 23/23 [00:27<00:00,  1.21s/it]
Epoch 16/100 - Loss: 0.1022 | Accuracy: 93.96%
400X Epoch 17/100: 100%|██████████| 23/23 [00:25<00:00,  1.09s/it]
Epoch 17/100 - Loss: 0.1087 | Accuracy: 94.78%
400X Epoch 18/100: 100%|██████████| 23/23 [00:24<00:00,  1.06s/it]
Epoch 18/100 - Loss: 0.1448 | Accuracy: 93.41%
400X Epoch 19/100: 100%|██████████| 23/23 [00:25<00:00,  1.10s/it]
Epoch 19/100 - Loss: 0.0918 | Accuracy: 95.05%
400X Epoch 20/100: 100%|██████████| 23/23 [00:25<00:00,  1.11s/it]
Epoch 20/100 - Loss: 0.0733 | Accuracy: 96.98% ✓ BEST!
400X Epoch 21/100: 100%|██████████| 23/23 [00:27<00:00,  1.18s/it]
Epoch 21/100 - Loss: 0.0962 | Accuracy: 90.93%
400X Epoch 22/100: 100%|██████████| 23/23 [00:24<00:00,  1.08s/it]
Epoch 22/100 - Loss: 0.0691 | Accuracy: 96.15%
400X Epoch 23/100: 100%|██████████| 23/23 [00:26<00:00,  1.16s/it]
Epoch 23/100 - Loss: 0.0592 | Accuracy: 94.51%
400X Epoch 24/100: 100%|██████████| 23/23 [00:25<00:00,  1.11s/it]
Epoch 24/100 - Loss: 0.0693 | Accuracy: 94.51%
400X Epoch 25/100: 100%|██████████| 23/23 [00:25<00:00,  1.11s/it]
Epoch 25/100 - Loss: 0.0753 | Accuracy: 94.51%
400X Epoch 26/100: 100%|██████████| 23/23 [00:25<00:00,  1.11s/it]
Epoch 26/100 - Loss: 0.0580 | Accuracy: 92.58%
400X Epoch 27/100: 100%|██████████| 23/23 [00:24<00:00,  1.07s/it]
Epoch 27/100 - Loss: 0.0767 | Accuracy: 96.98%
400X Epoch 28/100: 100%|██████████| 23/23 [00:24<00:00,  1.08s/it]
Epoch 28/100 - Loss: 0.0380 | Accuracy: 96.98%
400X Epoch 29/100: 100%|██████████| 23/23 [00:25<00:00,  1.10s/it]
Epoch 29/100 - Loss: 0.0524 | Accuracy: 96.70%
400X Epoch 30/100: 100%|██████████| 23/23 [00:25<00:00,  1.10s/it]
Epoch 30/100 - Loss: 0.0444 | Accuracy: 96.98%
400X Epoch 31/100: 100%|██████████| 23/23 [00:25<00:00,  1.10s/it]
Epoch 31/100 - Loss: 0.0410 | Accuracy: 96.70%
400X Epoch 32/100: 100%|██████████| 23/23 [00:24<00:00,  1.07s/it]
Epoch 32/100 - Loss: 0.0277 | Accuracy: 95.60%
400X Epoch 33/100: 100%|██████████| 23/23 [00:25<00:00,  1.12s/it]
Epoch 33/100 - Loss: 0.0534 | Accuracy: 95.60%
400X Epoch 34/100: 100%|██████████| 23/23 [00:25<00:00,  1.11s/it]
Epoch 34/100 - Loss: 0.0444 | Accuracy: 96.15%
400X Epoch 35/100: 100%|██████████| 23/23 [00:25<00:00,  1.12s/it]
Epoch 35/100 - Loss: 0.0323 | Accuracy: 97.25% ✓ BEST!
400X Epoch 36/100: 100%|██████████| 23/23 [00:26<00:00,  1.17s/it]
Epoch 36/100 - Loss: 0.0217 | Accuracy: 97.25%
400X Epoch 37/100: 100%|██████████| 23/23 [00:24<00:00,  1.06s/it]
Epoch 37/100 - Loss: 0.0171 | Accuracy: 97.25%
400X Epoch 38/100: 100%|██████████| 23/23 [00:25<00:00,  1.09s/it]
Epoch 38/100 - Loss: 0.0583 | Accuracy: 96.43%
400X Epoch 39/100: 100%|██████████| 23/23 [00:25<00:00,  1.13s/it]
Epoch 39/100 - Loss: 0.0479 | Accuracy: 96.70%
400X Epoch 40/100: 100%|██████████| 23/23 [00:25<00:00,  1.10s/it]
Epoch 40/100 - Loss: 0.0234 | Accuracy: 95.88%
400X Epoch 41/100: 100%|██████████| 23/23 [00:25<00:00,  1.10s/it]
Epoch 41/100 - Loss: 0.0311 | Accuracy: 96.70%
400X Epoch 42/100: 100%|██████████| 23/23 [00:24<00:00,  1.06s/it]
Epoch 42/100 - Loss: 0.0362 | Accuracy: 98.35% ✓ BEST!
400X Epoch 43/100: 100%|██████████| 23/23 [00:28<00:00,  1.23s/it]
Epoch 43/100 - Loss: 0.0231 | Accuracy: 95.60%
400X Epoch 44/100: 100%|██████████| 23/23 [00:25<00:00,  1.12s/it]
Epoch 44/100 - Loss: 0.0151 | Accuracy: 98.63% ✓ BEST!
400X Epoch 45/100: 100%|██████████| 23/23 [00:27<00:00,  1.17s/it]
Epoch 45/100 - Loss: 0.0094 | Accuracy: 98.63%
400X Epoch 46/100: 100%|██████████| 23/23 [00:24<00:00,  1.05s/it]
Epoch 46/100 - Loss: 0.0298 | Accuracy: 97.80%
400X Epoch 47/100: 100%|██████████| 23/23 [00:24<00:00,  1.05s/it]
Epoch 47/100 - Loss: 0.0108 | Accuracy: 98.63%
400X Epoch 48/100: 100%|██████████| 23/23 [00:25<00:00,  1.09s/it]
Epoch 48/100 - Loss: 0.0112 | Accuracy: 97.53%
400X Epoch 49/100: 100%|██████████| 23/23 [00:25<00:00,  1.12s/it]
Epoch 49/100 - Loss: 0.0193 | Accuracy: 94.78%
400X Epoch 50/100: 100%|██████████| 23/23 [00:25<00:00,  1.10s/it]
Epoch 50/100 - Loss: 0.0256 | Accuracy: 98.08%
400X Epoch 51/100: 100%|██████████| 23/23 [00:25<00:00,  1.11s/it]
Epoch 51/100 - Loss: 0.0020 | Accuracy: 97.80%
400X Epoch 52/100: 100%|██████████| 23/23 [00:23<00:00,  1.04s/it]
Epoch 52/100 - Loss: 0.0016 | Accuracy: 97.80%
400X Epoch 53/100: 100%|██████████| 23/23 [00:25<00:00,  1.09s/it]
Epoch 53/100 - Loss: 0.0198 | Accuracy: 98.63%
400X Epoch 54/100: 100%|██████████| 23/23 [00:25<00:00,  1.11s/it]
Epoch 54/100 - Loss: 0.0037 | Accuracy: 97.80%
400X Epoch 55/100: 100%|██████████| 23/23 [00:25<00:00,  1.10s/it]
Epoch 55/100 - Loss: 0.0133 | Accuracy: 98.08%
400X Epoch 56/100: 100%|██████████| 23/23 [00:25<00:00,  1.10s/it]
Epoch 56/100 - Loss: 0.0065 | Accuracy: 98.08%
400X Epoch 57/100: 100%|██████████| 23/23 [00:24<00:00,  1.08s/it]
Epoch 57/100 - Loss: 0.0024 | Accuracy: 98.35%
400X Epoch 58/100: 100%|██████████| 23/23 [00:24<00:00,  1.07s/it]
Epoch 58/100 - Loss: 0.0015 | Accuracy: 97.53%
400X Epoch 59/100: 100%|██████████| 23/23 [00:25<00:00,  1.10s/it]
Epoch 59/100 - Loss: 0.0022 | Accuracy: 98.35%
400X Epoch 60/100: 100%|██████████| 23/23 [00:25<00:00,  1.10s/it]
Epoch 60/100 - Loss: 0.0112 | Accuracy: 97.53%
400X Epoch 61/100: 100%|██████████| 23/23 [00:25<00:00,  1.10s/it]
Epoch 61/100 - Loss: 0.0070 | Accuracy: 98.63%
400X Epoch 62/100: 100%|██████████| 23/23 [00:25<00:00,  1.10s/it]
Epoch 62/100 - Loss: 0.0046 | Accuracy: 98.35%
400X Epoch 63/100: 100%|██████████| 23/23 [00:24<00:00,  1.06s/it]
Epoch 63/100 - Loss: 0.0030 | Accuracy: 98.90% ✓ BEST!
400X Epoch 64/100: 100%|██████████| 23/23 [00:28<00:00,  1.24s/it]
Epoch 64/100 - Loss: 0.0007 | Accuracy: 98.08%
400X Epoch 65/100: 100%|██████████| 23/23 [00:25<00:00,  1.11s/it]
Epoch 65/100 - Loss: 0.0024 | Accuracy: 98.90%
400X Epoch 66/100: 100%|██████████| 23/23 [00:24<00:00,  1.07s/it]
Epoch 66/100 - Loss: 0.0010 | Accuracy: 98.90%
400X Epoch 67/100: 100%|██████████| 23/23 [00:24<00:00,  1.05s/it]
Epoch 67/100 - Loss: 0.0030 | Accuracy: 98.08%
400X Epoch 68/100: 100%|██████████| 23/23 [00:25<00:00,  1.10s/it]
Epoch 68/100 - Loss: 0.0022 | Accuracy: 98.35%
400X Epoch 69/100: 100%|██████████| 23/23 [00:25<00:00,  1.11s/it]
Epoch 69/100 - Loss: 0.0038 | Accuracy: 98.08%
400X Epoch 70/100: 100%|██████████| 23/23 [00:25<00:00,  1.10s/it]
Epoch 70/100 - Loss: 0.0004 | Accuracy: 98.63%
400X Epoch 71/100: 100%|██████████| 23/23 [00:25<00:00,  1.09s/it]
Epoch 71/100 - Loss: 0.0006 | Accuracy: 99.18% ✓ BEST!
400X Epoch 72/100: 100%|██████████| 23/23 [00:26<00:00,  1.13s/it]
Epoch 72/100 - Loss: 0.0012 | Accuracy: 98.90%
400X Epoch 73/100: 100%|██████████| 23/23 [00:25<00:00,  1.10s/it]
Epoch 73/100 - Loss: 0.0035 | Accuracy: 98.35%
400X Epoch 74/100: 100%|██████████| 23/23 [00:27<00:00,  1.19s/it]
Epoch 74/100 - Loss: 0.0011 | Accuracy: 98.63%
400X Epoch 75/100: 100%|██████████| 23/23 [00:25<00:00,  1.11s/it]
Epoch 75/100 - Loss: 0.0006 | Accuracy: 98.63%
400X Epoch 76/100: 100%|██████████| 23/23 [00:25<00:00,  1.11s/it]
Epoch 76/100 - Loss: 0.0051 | Accuracy: 98.35%
400X Epoch 77/100: 100%|██████████| 23/23 [00:25<00:00,  1.09s/it]
Epoch 77/100 - Loss: 0.0009 | Accuracy: 98.35%
400X Epoch 78/100: 100%|██████████| 23/23 [00:25<00:00,  1.12s/it]
Epoch 78/100 - Loss: 0.0022 | Accuracy: 98.90%
400X Epoch 79/100: 100%|██████████| 23/23 [00:25<00:00,  1.11s/it]
Epoch 79/100 - Loss: 0.0005 | Accuracy: 98.90%
400X Epoch 80/100: 100%|██████████| 23/23 [00:25<00:00,  1.11s/it]
Epoch 80/100 - Loss: 0.0006 | Accuracy: 98.90%
400X Epoch 81/100: 100%|██████████| 23/23 [00:25<00:00,  1.09s/it]
Epoch 81/100 - Loss: 0.0003 | Accuracy: 98.90%
400X Epoch 82/100: 100%|██████████| 23/23 [00:24<00:00,  1.06s/it]
Epoch 82/100 - Loss: 0.0001 | Accuracy: 98.90%
400X Epoch 83/100: 100%|██████████| 23/23 [00:25<00:00,  1.12s/it]
Epoch 83/100 - Loss: 0.0011 | Accuracy: 98.90%
400X Epoch 84/100: 100%|██████████| 23/23 [00:25<00:00,  1.12s/it]
Epoch 84/100 - Loss: 0.0011 | Accuracy: 98.63%
400X Epoch 85/100: 100%|██████████| 23/23 [00:26<00:00,  1.15s/it]
Epoch 85/100 - Loss: 0.0004 | Accuracy: 98.90%
400X Epoch 86/100: 100%|██████████| 23/23 [00:24<00:00,  1.09s/it]
Epoch 86/100 - Loss: 0.0003 | Accuracy: 99.18%
400X Epoch 87/100: 100%|██████████| 23/23 [00:24<00:00,  1.08s/it]
Epoch 87/100 - Loss: 0.0001 | Accuracy: 99.18%
400X Epoch 88/100: 100%|██████████| 23/23 [00:25<00:00,  1.12s/it]
Epoch 88/100 - Loss: 0.0006 | Accuracy: 99.18%
400X Epoch 89/100: 100%|██████████| 23/23 [00:25<00:00,  1.12s/it]
Epoch 89/100 - Loss: 0.0000 | Accuracy: 99.18%
400X Epoch 90/100: 100%|██████████| 23/23 [00:25<00:00,  1.12s/it]
Epoch 90/100 - Loss: 0.0000 | Accuracy: 99.18%
400X Epoch 91/100: 100%|██████████| 23/23 [00:25<00:00,  1.09s/it]
Epoch 91/100 - Loss: 0.0001 | Accuracy: 99.18%
400X Epoch 92/100: 100%|██████████| 23/23 [00:25<00:00,  1.09s/it]
Epoch 92/100 - Loss: 0.0001 | Accuracy: 99.18%
400X Epoch 93/100: 100%|██████████| 23/23 [00:25<00:00,  1.12s/it]
Epoch 93/100 - Loss: 0.0001 | Accuracy: 99.18%
400X Epoch 94/100: 100%|██████████| 23/23 [00:26<00:00,  1.17s/it]
Epoch 94/100 - Loss: 0.0001 | Accuracy: 99.18%
400X Epoch 95/100: 100%|██████████| 23/23 [00:27<00:00,  1.18s/it]
Epoch 95/100 - Loss: 0.0002 | Accuracy: 98.90%
400X Epoch 96/100: 100%|██████████| 23/23 [00:24<00:00,  1.07s/it]
Epoch 96/100 - Loss: 0.0001 | Accuracy: 98.90%
400X Epoch 97/100: 100%|██████████| 23/23 [00:25<00:00,  1.13s/it]
Epoch 97/100 - Loss: 0.0001 | Accuracy: 98.90%
400X Epoch 98/100: 100%|██████████| 23/23 [00:25<00:00,  1.13s/it]
Epoch 98/100 - Loss: 0.0001 | Accuracy: 98.90%
400X Epoch 99/100: 100%|██████████| 23/23 [00:26<00:00,  1.13s/it]
Epoch 99/100 - Loss: 0.0001 | Accuracy: 98.90%
400X Epoch 100/100: 100%|██████████| 23/23 [00:25<00:00,  1.11s/it]
Epoch 100/100 - Loss: 0.0001 | Accuracy: 98.90%

──────────────────────────────────────────────────────────────────────
FINAL EVALUATION - 400X
──────────────────────────────────────────────────────────────────────
              precision    recall  f1-score   support

      Benign     0.9837    0.9918    0.9878       122
   Malignant     0.9959    0.9917    0.9938       242

    accuracy                         0.9918       364
   macro avg     0.9898    0.9918    0.9908       364
weighted avg     0.9918    0.9918    0.9918       364


Confusion Matrix:
                Predicted
              Benign  Malignant
Actual Benign    121         1
    Malignant      2       240

✓ 400X Training Complete!
Best Accuracy: 99.18%
Model saved: /content/drive/MyDrive/best_breakhis_400X.pth


======================================================================
COMPLETE TRAINING SUMMARY - ALL MAGNIFICATIONS
======================================================================

Training Duration: 3:46:55.305447

Magnification   Accuracy        Status
----------------------------------------------------------------------
40X              99.50%         ✓ EXCELLENT
100X             98.80%         ✓ GOOD
200X             99.50%         ✓ EXCELLENT
400X             99.18%         ✓ EXCELLENT

----------------------------------------------------------------------
Average          99.24%

======================================================================
MODELS SAVED TO GOOGLE DRIVE:
======================================================================
  • best_breakhis_40X.pth
  • best_breakhis_100X.pth
  • best_breakhis_200X.pth
  • best_breakhis_400X.pth

======================================================================
✓ ALL MAGNIFICATIONS TRAINING COMPLETE!
Target (Gella 2024): 99.99% per magnification
Your Results: See table above
======================================================================
