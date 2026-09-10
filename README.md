# 🤖 AI Product Image Classifier

### Explainable AI-Based Product Classification System with Confidence-Aware Prediction and Interactive Analytics

---

## 📌 Project Overview

The **AI Product Image Classifier** is a deep learning-based computer vision system that automatically classifies product images into three categories:

- 👕 **Apparel**
- 💻 **Electronics**
- 🏠 **Home**

The system uses **Transfer Learning with MobileNetV2** and an ImageNet-pretrained convolutional neural network backbone.

Unlike a basic image classifier, this project also provides:

- 🎯 Confidence-aware predictions
- 📊 Top-3 predictions
- 📈 Detailed model evaluation
- 🔥 Grad-CAM explainability
- ❌ Error analysis
- 🧪 Baseline vs fine-tuned comparison
- 💻 Interactive AI prediction interface

---

## 🎯 Key Features

### 🧠 Deep Learning Classification

- MobileNetV2 transfer learning
- ImageNet-pretrained backbone
- 224 × 224 RGB image input
- Three product categories
- Controlled fine-tuning

### 🎯 Confidence-Aware Prediction

Predictions are categorized based on confidence:

| Confidence | Status |
|---|---|
| ≥ 80% | 🟢 High Confidence |
| 60–80% | 🟡 Moderate Confidence |
| < 60% | 🔴 Low Confidence |

The system also displays the **Top-3 predicted classes**.

### 🔍 Explainable AI

The project uses **Grad-CAM (Gradient-weighted Class Activation Mapping)** to visualize the image regions that influenced the model's prediction.

This makes the system more interpretable and reduces the black-box nature of deep learning predictions.

---

## 🏗️ System Architecture

```text
                    Product Image
                          │
                          ▼
                 Image Preprocessing
                 Resize → 224 × 224
                          │
                          ▼
                 MobileNetV2 Backbone
                  ImageNet Pretrained
                          │
                          ▼
                 Global Average Pooling
                          │
                          ▼
                       Dropout
                          │
                          ▼
                  Softmax Classifier
                          │
            ┌─────────────┼─────────────┐
            ▼             ▼             ▼
         Apparel      Electronics      Home
                          │
                          ▼
                Confidence Analysis
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
          Top-3 Results            Grad-CAM
              │                       │
              └───────────┬───────────┘
                          ▼
                  Interactive UI
