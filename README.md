# Road Crack (and pothole) Segmentation Project !!!

## Overview

This repository contains the implementation of various models for **road crack segmentation**, including:  
- **Canny Edge Detection**  
- **Watershed Algorithm**  
- **U-Net**  
- **YOLOV11 (You Only Look Once)**  

Each model has been evaluated for its performance in segmenting road cracks, with datasets sourced from Kaggle and Roboflow.

---

## Datasets

1. **Kaggle Dataset**  
   - Used for training and testing the **Canny Edge**, **Watershed**, and **U-Net** models.  
   - Crack [https://www.kaggle.com/datasets/lakshaymiddha/crack-segmentation-dataset ]  

2. **Roboflow Datasets**  
   - Two datasets combined to train the **YOLO model**.  
   - Crack & Pothole [https://universe.roboflow.com/demo-hbwry/try-3-cqnje/dataset/1# ]  
   - Pothole [https://public.roboflow.com/object-detection/pothole]

---

## Setup Instructions

### Prerequisites
- Python 3.8+
- Required libraries: TensorFlow, PyTorch, OpenCV, scikit-learn, NumPy, Pandas, Matplotlib, etc.

### Installation
1. Clone the repository:  
   ```bash
   git clone https://github.com/AndreasKrist/GotongRoyongSegmentation.git 
   cd GotongRoyongSegmentation
   ```

2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage  

### 1. **Canny Edge Detection**  
- **Parameter Tuning**: Use `cannyEdgeSearch.ipynb` to find the best parameters for edge detection.  
- **Interactive App**: Run `cannyEdgeSliderApp.py` for a slider-based app to adjust parameters dynamically.  
  ```bash
  python cannyEdgeSliderApp.py
  ```

### 2. **Watershed Algorithm**  
- **Parameter Tuning**: Use `watershedSearch.ipynb` to identify optimal parameters for segmentation.  
- **Interactive App**: Run `watershedSliderApp.py` to explore parameters interactively.  
  ```bash
  python watershedSliderApp.py
  ```

### 3. **U-Net**  
- **Training and Inference**: The entire implementation is in `unet.ipynb`. Open the notebook to train or test the U-Net model for segmentation tasks.  

### 4. **YOLO**  
- **Hugging Face Model**: For YOLO, refer to the files hosted on [Hugging Face YOLO](https://huggingface.co/spaces/AndreasKrist/GotongRoyongSegmentation/tree/main) for the final model (small.pt).  
- **Demo Application**: Explore the [CRACK & POTHOLE Detector](https://huggingface.co/spaces/AndreasKrist/GotongRoyongSegmentation) in the provided YOLO demo.  

---
## References
1. [Kaggle Dataset - Road Crack Segmentation](https://www.kaggle.com/datasets/lakshaymiddha/crack-segmentation-dataset)
2. [Roboflow Crack & Pothole](https://universe.roboflow.com/demo-hbwry/try-3-cqnje/dataset/1#)  
3. [Roboflow Pothole](https://public.roboflow.com/object-detection/pothole)  
4. [Hugging Face YOLO space](https://huggingface.co/spaces/AndreasKrist/GotongRoyongSegmentation/tree/main)

---

## Participated in this project
- Andreas Mardohar Kristianto - 2602100096 
- Gabriel Enrico - 2602105481 
- Alexander Prajna Felipe - 2602097403
