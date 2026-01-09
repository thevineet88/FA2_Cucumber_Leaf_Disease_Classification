# Setup Guide for Cucumber Leaf Disease Classification

## Installation Steps

### 1. Clone the Repository
```bash
git clone <repository-url>
cd FA2_Cucumber_Leaf_Disease_Classification
```

### 2. Create Virtual Environment (Optional but Recommended)
```bash
# Using venv
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Or using conda
conda create -n cucumber_disease python=3.10
conda activate cucumber_disease
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Dataset
1. Visit [Kaggle - Cucumber Leaf Disease Dataset](https://www.kaggle.com/datasets/kaushigihanml/cucumber-leaf-disease-dataset)
2. Download the dataset
3. Extract to `data/` folder in this directory

Expected structure:
```
FA2_Cucumber_Leaf_Disease_Classification/
├── data/
│   ├── train/
│   │   ├── downy_mildew/
│   │   ├── healthy/
│   │   └── powdery_mildew/
│   ├── validation/
│   └── test/
├── notebooks/
├── models/
└── outputs/
```

### 5. Prepare Directory Structure
```bash
mkdir -p models
mkdir -p outputs/MobileNet
mkdir -p outputs/ResNet
mkdir -p data/{train,validation,test}/{Downy\ Mildew,Healthy,Powdery\ Mildew}
```

## Running Notebooks

### Option 1: Jupyter Notebook
```bash
# Start Jupyter server
jupyter notebook

# Navigate to notebooks folder and open:
# - FA2_Deep_Learning_MobileNetV2.ipynb
# - FA2_Deep_Learning_ResNet50.ipynb
```

### Option 2: Google Colab
1. Upload this repository to Google Drive
2. Open in Google Colab
3. Mount Google Drive in first cell:
```python
from google.colab import drive
drive.mount('/content/drive')
```
4. Change paths to point to Drive location
5. Run cells sequentially

### Option 3: Command Line (if using Python scripts)
```bash
python train_mobilenet.py
python train_resnet50.py
```

## Troubleshooting

### CUDA/GPU Issues
If GPU not detected:
```bash
# Install CUDA-compatible TensorFlow
pip install tensorflow[and-cuda]

# Or check GPU availability in Python
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Memory Issues
If running out of memory:
- Reduce BATCH_SIZE in notebooks (32 → 16 or 8)
- Use gradient accumulation
- Run on Google Colab with GPU

### Dataset Download Issues
Alternative ways to get dataset:
- Download directly from Kaggle CLI:
```bash
pip install kaggle
kaggle datasets download kaushigihanml/cucumber-leaf-disease-dataset
unzip cucumber-leaf-disease-dataset.zip -d data/
```

## File Descriptions

- **notebooks/**: Jupyter notebooks for training both models
  - MobileNetV2: Fast, lightweight model (~2-3 min/epoch)
  - ResNet50: Accurate, powerful model (~6-8 min/epoch)

- **models/**: Saved model architectures and weights
  - .h5 files: Trained models
  - .txt files: Model summaries

- **outputs/**: Training results
  - Accuracy/Loss plots
  - Confusion matrices
  - Classification reports

- **requirements.txt**: Python dependencies

## Next Steps

1. ✅ Install dependencies
2. ✅ Download dataset
3. ✅ Prepare directory structure
4. ✅ Run notebooks sequentially
5. ✅ Evaluate results
6. ✅ Deploy model (optional)

## Support

For issues or questions:
- Check notebook comments
- Verify dataset format matches expected structure
- Ensure Python version >= 3.8

## References

- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [TensorFlow Documentation](https://www.tensorflow.org/guide)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
