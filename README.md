# Road Damage Detection using YOLO11 🛣️

A deep learning-based **Road Damage Detection** system using **YOLO11n** for detecting and classifying different types of road damage from images.

The project is implemented using **Python, Ultralytics YOLO, PyTorch, OpenCV, and Kaggle**.

---

## 📌 Project Overview

Road damage such as cracks and potholes can affect road safety and increase maintenance costs. Manual inspection of roads is time-consuming and difficult to scale.

This project uses an object detection model to automatically identify road damage from road images and classify each detected damage into one of five categories.

The model can detect multiple damage objects in a single image and return their locations using bounding boxes.

---

## 🎯 Objectives

* Automatically detect road damage from images.
* Classify different types of road damage.
* Localize damaged areas using bounding boxes.
* Evaluate the performance of the YOLO11 object detection model.
* Provide a foundation for future road inspection and smart-city applications.

---

## 🧠 Model

The project uses:

**YOLO11n (YOLO11 Nano)**

YOLO11n was selected because it provides a good balance between detection performance and computational efficiency, making it suitable for relatively lightweight deployment.

### Training Configuration

| Parameter            | Value            |
| -------------------- | ---------------- |
| Model                | YOLO11n          |
| Task                 | Object Detection |
| Image Size           | 640 × 640        |
| Epochs               | 50               |
| Batch Size           | 16               |
| GPU                  | NVIDIA Tesla T4  |
| Framework            | Ultralytics      |
| Optimizer            | Auto             |
| Confidence Threshold | 0.25             |

---

## 🗂️ Dataset

The project uses the **RDD2022 (Road Damage Dataset 2022)**.

The dataset is divided into:

* Training set: **26,869 images**
* Validation set: **5,758 images**
* Test set: **5,758 images**

### Damage Classes

The model detects five types of road damage:

1. **Longitudinal Crack**
2. **Transverse Crack**
3. **Alligator Crack**
4. **Other Corruption**
5. **Pothole**

### Training Class Distribution

| Class              | Objects |
| ------------------ | ------: |
| Longitudinal Crack |  18,201 |
| Transverse Crack   |   8,386 |
| Alligator Crack    |   7,527 |
| Other Corruption   |   7,554 |
| Pothole            |   4,628 |

---

## 🔄 Project Pipeline

```text
Road Images
     ↓
Dataset Preparation
     ↓
Data Analysis & Visualization
     ↓
YOLO11n Pretrained Model
     ↓
Model Training
     ↓
Validation
     ↓
Performance Evaluation
     ↓
Road Damage Detection
     ↓
Bounding Boxes + Damage Classes
```

---

## ⚙️ Technologies Used

* **Python**
* **YOLO11**
* **Ultralytics**
* **PyTorch**
* **OpenCV**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **Seaborn**
* **PIL**
* **Kaggle**

---

## 📊 Model Evaluation

The trained model was evaluated on the validation dataset using common object detection metrics:

* Precision
* Recall
* mAP@50
* mAP@50-95

### Validation Results

| Metric    |     Score |
| --------- | --------: |
| Precision | **0.621** |
| Recall    | **0.543** |
| mAP@50    | **0.576** |
| mAP@50-95 | **0.311** |

### Per-Class Performance

| Class              | Precision | Recall | mAP@50 | mAP@50-95 |
| ------------------ | --------: | -----: | -----: | --------: |
| Longitudinal Crack |     0.598 |  0.507 |  0.535 |     0.297 |
| Transverse Crack   |     0.577 |  0.506 |  0.520 |     0.255 |
| Alligator Crack    |     0.669 |  0.607 |  0.650 |     0.349 |
| Other Corruption   |     0.643 |  0.730 |  0.747 |     0.468 |
| Pothole            |     0.616 |  0.366 |  0.428 |     0.187 |

The results show that **Other Corruption** achieved the highest mAP@50, while **Pothole** was the most challenging class for the current model.

---

## 🧪 Detection Example

The trained model can be used to detect road damage in previously unseen road images.

Example output:

```text
Input Image
     ↓
YOLO11n
     ↓
Detected Objects
     ↓
┌─────────────────────────────┐
│ Longitudinal Crack           │
│ Confidence: 0.XX             │
│ Bounding Box: (x1,y1,x2,y2) │
└─────────────────────────────┘
```

The notebook also generates visual predictions with bounding boxes and class labels.

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/road-damage-detection.git
cd road-damage-detection
```

Install the required dependencies:

```bash
pip install ultralytics
pip install numpy pandas matplotlib seaborn opencv-python pillow
```

---

## ▶️ Usage

Load the YOLO11 model:

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
```

Train the model:

```python
results = model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,
    name="road_damage_yolo11"
)
```

Run inference:

```python
results = model("path/to/road/image.jpg", conf=0.25)

for result in results:
    result.show()
```

---

## 📁 Project Structure

```text
road-damage-detection/
│
├── road-damage-detection.ipynb
├── data.yaml
├── README.md
│
├── runs/
│   └── detect/
│       └── road_damage_yolo11/
│           ├── weights/
│           │   ├── best.pt
│           │   └── last.pt
│           ├── results.csv
│           └── results.png
│
└── assets/
    └── detection_examples/
```

> Dataset files and trained weights may be excluded from the repository because of their large file size.

---

## 📈 Future Improvements

Possible improvements include:

* Training for more epochs.
* Hyperparameter optimization.
* Data augmentation.
* Addressing class imbalance.
* Improving pothole detection.
* Comparing YOLO11n with larger YOLO11 models.
* Testing the model on real-world road images.
* Deploying the model as a web or mobile application.
* Integrating GPS/location information for road-damage mapping.
* Building a road maintenance monitoring dashboard.

---

## 🌍 Potential Applications

This system can be extended for:

* Smart city infrastructure monitoring
* Automated road inspection
* Road maintenance planning
* Municipal infrastructure management
* Road safety systems
* AI-powered road condition mapping

---

## 📚 Dataset

**RDD2022 – Road Damage Dataset 2022**

The dataset contains road images collected from multiple countries and annotated with different types of road damage.

Please refer to the original dataset source for licensing and usage information.

---

## 👨‍💻 Project

**Road Damage Detection using YOLO11**

Developed as a Computer Science / Artificial Intelligence project focusing on **Computer Vision and Object Detection**.

---

## ⭐ Acknowledgments

* **Ultralytics** for the YOLO object detection framework.
* **RDD2022** dataset contributors.
* **Kaggle** for providing the computational environment.
* **PyTorch** for the deep learning framework.

---

## 📄 License

This project is intended for educational and research purposes. Please check the licenses of the dataset and third-party libraries before using the project commercially.
