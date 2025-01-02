# Road Crack (and Pothole) Segmentation Project

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

### YOLO 
#### For YOLO, we suggest to visit for the file used [Hugging Face YOLO space](https://huggingface.co/spaces/AndreasKrist/GotongRoyongSegmentation/tree/main) 
#### YOLO DEMO : [CRACK & POTHOLE Detector](https://huggingface.co/spaces/AndreasKrist/GotongRoyongSegmentation) 


---
## References
1. [Kaggle Dataset - Road Crack Segmentation](https://www.kaggle.com/datasets/lakshaymiddha/crack-segmentation-dataset)
2. [Roboflow Crack & Pothole](https://universe.roboflow.com/demo-hbwry/try-3-cqnje/dataset/1#)  
3. [Roboflow Pothole](https://public.roboflow.com/object-detection/pothole)  
4. [Hugging Face YOLO space](https://huggingface.co/spaces/AndreasKrist/GotongRoyongSegmentation/tree/main)

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to add any missing details, such as specific dataset/model links or your own customization. Let me know if you'd like further refinements!
