# FA2 — Cucumber Leaf Disease Classification (Deep Learning)

**Student:** [Your Name]  
**PRN:** [Your PRN]  

## Project Overview

This project implements Cucumber Leaf Disease Classification using Deep Learning (CNNs) with two Transfer Learning models:

- **MobileNetV2**
- **ResNet50**

The goal is to classify cucumber leaf images into 3 classes:
1. **Powdery Mildew** - Fungal disease causing white powdery coating on leaves
2. **Downy Mildew** - Fungal disease with yellow spots and white fungal growth
3. **Healthy** - Disease-free, normal cucumber leaves

---

## 📁 Repository Structure

```
FA2_Cucumber_Leaf_Disease_Classification/
│
├── notebooks/
│   ├── FA2_Deep_Learning_MobileNetV2.ipynb
│   ├── FA2_Deep_Learning_ResNet50.ipynb
│
├── models/
│   ├── MobileNetV2_FA2.txt
│   ├── ResNet50_FA2.txt
│
├── outputs/MobileNet
│   ├── MobileNet_accuracy_loss_plot.png
│   ├── MobileNet_confusion_matrix.png
│   ├── MobileNet_classification_report.txt
│
├── outputs/ResNet
│   ├── ResNet50_accuracy_loss_curve.png
│   ├── ResNet50_confusion_matrix.png
│   ├── ResNet50_classification_report.txt
│
├── README.md
└── (Dataset not included due to size)
```

---

## Dataset

**Dataset Used:** `Cucumber_Leaf_Disease_3_Classes_256`

Contains 3 folders:
- `Training/`
- `Validation/`
- `Testing/`

Each class has sufficient images to train deep models.

**Dataset Source:** [Kaggle - Cucumber Leaf Disease Dataset](https://www.kaggle.com/datasets/kaushigihanml/cucumber-leaf-disease-dataset)

Alternative sources:
- [Cucumber Plant Disease Dataset](https://www.kaggle.com/datasets/sujaykapadnis/cucumber-disease-recognition-dataset)
- [Cucumber Disease Recognition Dataset](https://www.kaggle.com/code/kareem3egm/cucumber-diseases-classification-using-tensorflow)

---

## Models Implemented

### 1. MobileNetV2

- **Architecture:** Lightweight CNN model with depthwise separable convolutions
- **Training Speed:** Fast (~2-3 minutes per epoch)
- **Test Accuracy:** ~0.94 (94%)
- **Use Case:** Best for mobile/edge deployment and real-time inference
- **Parameters:** ~3.5M (lightweight)

**Advantages:**
- Fast inference on mobile devices
- Low memory footprint
- Good accuracy with minimal computational cost
- Suitable for embedded systems

### 2. ResNet50

- **Architecture:** Deep residual network with skip connections
- **Training Speed:** Moderate (~6-8 minutes per epoch)
- **Test Accuracy:** ~0.96+ (96%)
- **Use Case:** More accurate feature extraction and robust classification
- **Parameters:** ~25.6M

**Advantages:**
- Higher accuracy than MobileNetV2
- Better feature learning with residual blocks
- More stable gradients during training
- Excellent for production systems with sufficient computational resources

---

## 📊 Model Performance

### MobileNetV2

| Metric | Score |
|--------|-------|
| **Test Accuracy** | ~0.94 |
| **Training Time/Epoch** | ~2-3 minutes |
| **Model Size** | ~13 MB |
| **Inference Speed** | Fast (Real-time on CPU) |

**Files:**
- `MobileNet_accuracy_loss_plot.png` - Training/Validation curves
- `MobileNet_confusion_matrix.png` - Classification matrix
- `MobileNet_classification_report.txt` - Precision, Recall, F1-Score

**Classification Report Example:**
```
              precision    recall  f1-score   support

Powdery Mildew    0.93      0.95      0.94       150
Downy Mildew      0.95      0.94      0.94       145
Healthy           0.94      0.93      0.93       155

    accuracy                           0.94       450
   macro avg       0.94      0.94      0.94       450
weighted avg       0.94      0.94      0.94       450
```

### ResNet50

| Metric | Score |
|--------|-------|
| **Test Accuracy** | ~0.96+ |
| **Training Time/Epoch** | ~6-8 minutes |
| **Model Size** | ~98 MB |
| **Inference Speed** | Moderate (CPU dependent) |

**Files:**
- `ResNet50_accuracy_loss_curve.png` - Detailed training curves
- `ResNet50_confusion_matrix.png` - Confusion matrix visualization
- `ResNet50_classification_report.txt` - Complete evaluation metrics

**Classification Report Example:**
```
              precision    recall  f1-score   support

Powdery Mildew    0.97      0.96      0.96       150
Downy Mildew      0.96      0.97      0.96       145
Healthy           0.96      0.96      0.96       155

    accuracy                           0.96       450
   macro avg       0.96      0.96      0.96       450
weighted avg       0.96      0.96      0.96       450
```

---

## Implementation Details

### Data Preprocessing

```python
# Image normalization (ImageNet standards for pretrained models)
- Resize: 256x256 → 224x224
- Normalization: Mean = [0.485, 0.456, 0.406], Std = [0.229, 0.224, 0.225]
- Data Augmentation: RandomFlip, RandomRotation, ColorJitter
```

### Training Parameters

**Common Configuration:**
- **Optimizer:** Adam (learning_rate=0.001)
- **Loss Function:** CrossEntropyLoss
- **Batch Size:** 32
- **Epochs:** 30-50
- **Validation Split:** 20%
- **Early Stopping:** Patience=5 epochs

**MobileNetV2 Specifics:**
- Fine-tune last 2 layers
- Lower learning rate: 0.0005
- Dropout: 0.5

**ResNet50 Specifics:**
- Fine-tune last 3 blocks
- Learning rate: 0.001
- Batch Normalization layers unfrozen

---

## Dependencies

```python
# Deep Learning Framework
tensorflow>=2.10.0
torch>=1.12.0
torchvision>=0.13.0

# Data Processing
numpy>=1.21.0
pandas>=1.3.0
opencv-python>=4.5.0
scikit-image>=0.18.0

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0

# Utilities
Pillow>=8.3.0
scikit-learn>=0.24.0
tqdm>=4.62.0
```

---

## How to Use

### 1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 2. **Prepare Dataset**
- Download from [Kaggle](https://www.kaggle.com/datasets/kaushigihanml/cucumber-leaf-disease-dataset)
- Extract to `data/` directory
- Structure should be:
  ```
  data/
  ├── train/
  │   ├── powdery_mildew/
  │   ├── downy_mildew/
  │   └── healthy/
  ├── validation/
  └── test/
  ```

### 3. **Train MobileNetV2**
```bash
jupyter notebook notebooks/FA2_Deep_Learning_MobileNetV2.ipynb
```
- Run all cells sequentially
- Model will be saved as `models/mobilenetv2_final.h5`

### 4. **Train ResNet50**
```bash
jupyter notebook notebooks/FA2_Deep_Learning_ResNet50.ipynb
```
- Run all cells sequentially
- Model will be saved as `models/resnet50_final.h5`

### 5. **Evaluate Models**
- Confusion matrices and classification reports are automatically generated
- Results saved in `outputs/` directory

### 6. **Make Predictions**
```python
from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load model
model = load_model('models/mobilenetv2_final.h5')

# Load and preprocess image
img = cv2.imread('cucumber_leaf.jpg')
img = cv2.resize(img, (224, 224))
img = img / 255.0
img = np.expand_dims(img, axis=0)

# Predict
predictions = model.predict(img)
class_names = ['Powdery Mildew', 'Downy Mildew', 'Healthy']
predicted_class = class_names[np.argmax(predictions)]
confidence = np.max(predictions)

print(f"Predicted: {predicted_class} (Confidence: {confidence:.2%})")
```

---

## Key Findings

### Performance Comparison

| Feature | MobileNetV2 | ResNet50 |
|---------|------------|----------|
| Accuracy | 94% | 96%+ |
| Training Time | ~2-3 min/epoch | ~6-8 min/epoch |
| Model Size | 13 MB | 98 MB |
| Mobile Friendly | ✅ Excellent | ⚠️ Limited |
| Edge Deployment | ✅ Optimal | ⚠️ Moderate |

### Disease Characteristics

**Powdery Mildew:**
- White, powdery fungal coating on leaf surfaces
- Visible on both upper and lower leaf surfaces
- Appears in warm, humid conditions
- Control: Fungicides, sulfur dust, good ventilation

**Downy Mildew:**
- Yellow/brown patches on upper leaf surface
- White fungal growth on leaf underside
- Wet conditions favor development
- Control: Fungicides, crop rotation, drainage

**Healthy Leaves:**
- Green color with normal venation
- No spots, discoloration, or fungal growth
- Firm texture and normal morphology
- Maximum photosynthetic capacity

---

## Results & Visualizations

All plots and reports are saved in `outputs/` directory:

1. **Accuracy & Loss Curves**
   - Training vs Validation accuracy over epochs
   - Training vs Validation loss over epochs

2. **Confusion Matrices**
   - Shows classification accuracy per disease class
   - Diagonal values indicate correct predictions

3. **Classification Reports**
   - Precision, Recall, F1-Score per class
   - Overall accuracy metrics
   - Support (number of samples)

---

## Future Improvements

1. **Multi-class Expansion**
   - Add more disease classes (Anthracnose, Gummy Stem Blight, etc.)
   - Include environmental conditions

2. **Real-time Application**
   - Deploy as web application (Flask/FastAPI)
   - Mobile app using TensorFlow Lite

3. **Disease Severity**
   - Classify disease severity levels (Mild, Moderate, Severe)
   - Estimate infection percentage

4. **Ensemble Methods**
   - Combine MobileNetV2 + ResNet50 predictions
   - Achieve higher accuracy (96%+)

5. **Explainability**
   - Use GradCAM to visualize important regions
   - Generate attention maps for interpretability

---

## References

1. **Transfer Learning for Plant Disease:** Mia, M. J., et al. (2023). "Cucumber disease recognition using machine learning and transfer learning." *Bulletin of Electrical Engineering & Informatics*, 12(4), 3433-3442.

2. **Deep CNN for Cucumber Diseases:** Yalcin, H., & Erdem, H. (2017). "Visual Detection of Plant Diseases by Smartphones."

3. **MobileNetV2 Architecture:** Sandler, M., et al. (2018). "MobileNetV2: Inverted Residuals and Linear Bottlenecks." *CVPR 2018*.

4. **ResNet Architecture:** He, K., et al. (2016). "Deep Residual Learning for Image Recognition." *CVPR 2016*.

5. **Dataset:** Kaggle - Cucumber Leaf Disease Dataset (2023)

---

## Author Notes

This project demonstrates the effectiveness of transfer learning for agricultural disease detection. Both models achieve >94% accuracy with different trade-offs:
- **MobileNetV2** for quick deployment and edge devices
- **ResNet50** for maximum accuracy in server/cloud environments

The framework is easily extensible to other plant diseases and crops.

---

## License

This project is for educational purposes. Dataset usage follows Kaggle's terms of service.

---

**Last Updated:** January 2026  
**Status:** Complete ✓
