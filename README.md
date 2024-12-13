# AI Corrosion Segmentation Pipeline

## Project Overview

This Streamlit application provides an advanced image segmentation tool for detecting corrosion in industrial pipelines using state-of-the-art deep learning models. The project aims to help industrial maintenance teams quickly identify and assess corrosion risks in infrastructure.

## Features

### 1. About Menu

- Comprehensive information about corrosion and image segmentation
- Detailed explanations of:
  - What is corrosion?
  - What is image segmentation?
  - Standardization techniques used
  - Available segmentation models

### 2. Segmentation Image Menu

- Image upload and preprocessing
- Multiple segmentation models:
  - Mobile U-Net
  - FCN8
  - BiSeNetV2

## Supported Segmentation Types

1. Asset Detection
2. Corrosion Detection
3. Corrosion within Asset Detection

## Prerequisites

- Python 3.8+
- Libraries:
  - Streamlit
  - TensorFlow
  - NumPy
  - OpenCV
  - Pillow
- Pre-trained models (included in `./model/` directory)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-repo/corrosion-segmentation.git
cd corrosion-segmentation
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

## Running the Application

```bash
streamlit run app.py
```

### Navigation

The application has two main sections:

#### 1. About Page

- Provides detailed information about:
  - Corrosion definition
  - Image segmentation explanation
  - Standardization used in the project
  - Detailed model descriptions

#### 2. Segmentation Image Page

1. Upload an image (JPG, PNG, JPEG formats)
2. Click "Preprocessing" to prepare the image
3. Select a segmentation model (Mobile U-Net, FCN-8, BiSeNetV2)
4. Choose segmentation type:
   - Predict Asset
   - Predict Corrosion
   - Predict Asset on Corrosion

### Output

- Original image
- Segmented images
- Pixel analysis
- Downloadable segmented images
- Detailed segmentation statistics

## Model Standardization

- Image Resize: 128x128 pixels
- Normalization techniques
- Data augmentation
- Specific colormap for visualization

## Visualization Colors

- Asset: Blue `[0, 0, 255]`
- Corrosion: Red `[255, 0, 0]`
- Corrosion on Asset: White `[255, 255, 255]`
- Corrosion outside Asset: Yellow `[255, 255, 0]`

## Usage Guide

### 1. About Menu

- Explore detailed information about corrosion, segmentation, and models
- Understand the technical background and methodologies

### 2. Segmentation Image Menu

1. Upload an image (JPG, PNG, JPEG)
2. Click "Preprocessing" to prepare the image
3. Select a segmentation model:
   - 🔴 Mobile U-Net
   - 🟢 FCN8
   - 🔵 BiSeNetV2
4. Choose segmentation type:
   - Predict Asset
   - Predict Corrosion
   - Predict Asset on Corrosion

## Model Details

### 1. Mobile U-Net + ResNet-50

- Computationally efficient
- Transfer learning capabilities
- Lightweight and fast

### 2. FCN-8 + ResNet-50

- Accurate segmentation
- Multi-scale image processing
- Scalable architecture

### 3. BiSeNetV2 + ResNet-50

- Real-time processing
- High accuracy
- Memory-efficient
