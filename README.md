
# 🚗 Cars196 Vehicle Classification Project

## 📌 Project Overview

This project focuses on building deep learning models to classify car images from the [Cars196 dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html). The dataset contains **16,185 images** of cars from **196 different classes**, with significant variation in image sizes and vehicle types.

The goal of this project is to evaluate different modeling strategies for **multi-class image classification** and compare their performance.

## 🧠 Objectives

- Classify images into one of 196 car categories using deep learning.
- Explore and compare three different model configurations:
  1. **Transfer Learning with ResNet50**
  2. **Image Retrieval using Embeddings**
  3. **Custom End-to-End CNN**

## 🗂️ Project Structure

```
├── transfer_pytorch_annotated.ipynb     # Transfer learning using ResNet50
├── Image_pytorch.ipynb                  # Image retrieval via ResNet50 embeddings
├── end_to_end_cnn_(1).ipynb             # Custom CNN trained from scratch
├── Final Project DL DS (2).pdf          # Project instructions (Hebrew)
├── README.md                            # Project documentation
```

## 🔧 Model Configurations

### 1️⃣ Transfer Learning (ResNet50)
- **Architecture**: Pre-trained ResNet50 from `torchvision.models`.
- **Modifications**:
  - Feature extractor layers are **frozen** (`requires_grad = False`).
  - The classifier is **replaced** with a custom `nn.Sequential` block:
    - Linear → ReLU → Dropout → Linear
- **Training**:
  - Optimizer: `Adam`
  - Loss: `CrossEntropyLoss`
  - Device: GPU
- **Data Augmentation**:
  - `RandomHorizontalFlip`, `RandomRotation`, `Resize`, `CenterCrop`, `Normalize`

### 2️⃣ Image Retrieval via Embedding Similarity
- **Model**: ResNet50 (same as in transfer learning)
- **Technique**:
  - Embeddings are extracted from the `avgpool` layer.
  - Cosine similarity is computed between embeddings.
  - The system retrieves the most similar images (top-k neighbors).
- **Visualization**: Includes image examples of retrieval results.

### 3️⃣ End-to-End CNN (Custom)
- **Architecture**: Manually built CNN with:
  - 3 convolutional layers
  - Dropout layers
  - Flatten → Fully connected classification head
- **Training**: From scratch, without any pre-trained components.
- **Data Augmentation**:
  - `RandomHorizontalFlip`, `RandomRotation`, `ColorJitter`, `RandomCrop`
- **Evaluation**:
  - Accuracy plots, confusion matrix

## 📁 Dataset Info

- Dataset: [Cars196](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
- Total Images: 16,185
- Classes: 196
- Pre-divided into training and test sets
- Option to modify train/test ratio

## 🧪 Tools & Techniques

- **PyTorch** for model development
- **Data Augmentation** to improve generalization
- **Cosine Similarity** for retrieval
- **Confusion Matrix** and **Accuracy Visualization**

## 🧩 Dependencies

```bash
torch
torchvision
numpy
matplotlib
scikit-learn
Pillow
```

## ✅ Results & Summary

- **Transfer Learning (ResNet50)** delivered strong performance with limited training time.
- **Image Retrieval** demonstrated high-quality similarity matching.
- **End-to-End CNN** required more tuning and experimentation, offering learning flexibility and full control.

## 📌 Future Work

- Experiment with other pre-trained backbones (e.g., EfficientNet)
- Hyperparameter optimization (e.g., Optuna, Ray Tune)
- Deploy as a car recognition web app (Streamlit/Flask + frontend)
- Extend dataset with synthetic data




## 📄 License

This project is for academic use. Please contact the author for other uses.

