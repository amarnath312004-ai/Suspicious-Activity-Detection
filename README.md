# 🛒 Suspicious Activity Detection using YOLO11 Pose and XGBoost

A computer vision project that detects **suspicious human activities** from videos using **YOLO11 Pose Estimation** and **XGBoost Classification**. The system extracts human body keypoints from each detected person and classifies their behavior as **Normal** or **Suspicious**.

---

## 📌 Project Overview

This project combines deep learning-based human pose estimation with machine learning classification.

### Workflow

```
Input Video
      │
      ▼
YOLO11 Pose Detection
      │
      ▼
17 Human Keypoints
      │
      ▼
Feature Extraction (x,y coordinates)
      │
      ▼
XGBoost Classifier
      │
      ▼
Normal / Suspicious Activity
```

---

## ✨ Features

- Human detection using YOLO11 Pose
- 17 body keypoint extraction
- Pose-based feature generation
- XGBoost activity classification
- Real-time video prediction
- Person tracking
- Bounding box visualization
- Supports Normal and Suspicious activity detection

---

## 🏗️ Project Structure

```
├── main.py                  # Activity detection on video
├── model.py                 # Train XGBoost model
├── dataset.py               # Create labeled dataset
├── Normal.py                # Extract normal activity keypoints
├── Suspicious.py            # Extract suspicious activity keypoints
├── normalvideo.py           # Prepare normal videos
├── suspiciousvideo.py       # Prepare suspicious videos
├── imgshuffle.py            # Organize dataset images
├── yolo11-pose.py           # YOLO11 pose estimation demo
├── trained_model.json       # Trained XGBoost model
├── keypoint.csv             # Extracted keypoint dataset
├── README.md
```

---

## 📖 Methodology

### 1. Video Preprocessing

- Read input video
- Sample frames
- Detect humans

### 2. Pose Estimation

YOLO11 Pose estimates **17 human body keypoints**.

| Index | Body Part |
|-------|-----------|
|0|Nose|
|1|Left Eye|
|2|Right Eye|
|3|Left Ear|
|4|Right Ear|
|5|Left Shoulder|
|6|Right Shoulder|
|7|Left Elbow|
|8|Right Elbow|
|9|Left Wrist|
|10|Right Wrist|
|11|Left Hip|
|12|Right Hip|
|13|Left Knee|
|14|Right Knee|
|15|Left Ankle|
|16|Right Ankle|

Each keypoint contains normalized **(x,y)** coordinates.

Total Features:

```
17 Keypoints × 2 Coordinates = 34 Features
```

---

## 🧠 Model

### Pose Estimation

- YOLO11 Pose

### Classifier

- XGBoost Binary Classifier

Classes

| Label | Value |
|--------|------|
|Suspicious|0|
|Normal|1|

---

## 📂 Dataset Preparation

### Step 1

Extract person crops and pose keypoints from:

- Normal videos
- Suspicious videos

### Step 2

Store:

- Cropped images
- Keypoint coordinates

### Step 3

Generate

```
keypoint.csv
```

### Step 4

Assign labels

```
Normal
Suspicious
```

### Step 5

Train XGBoost

Output:

```
trained_model.json
```

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/suspicious-activity-detection.git

cd suspicious-activity-detection
```

Install dependencies

```bash
pip install ultralytics
pip install xgboost
pip install opencv-python
pip install pandas
pip install numpy
pip install cvzone
pip install scikit-learn
```

---

## ▶️ Usage

### 1. Generate Dataset

```bash
python Normal.py
python Suspicious.py
```

---

### 2. Label Dataset

```bash
python dataset.py
```

---

### 3. Train Model

```bash
python model.py
```

---

### 4. Run Detection

```bash
python main.py
```

---

## 📸 Example Pipeline

```
Video
   │
   ▼
YOLO11 Pose
   │
   ▼
17 Keypoints
   │
   ▼
34 Features
   │
   ▼
XGBoost
   │
   ├── Normal
   └── Suspicious
```

---

## 📊 Technologies Used

- Python
- YOLO11 Pose
- OpenCV
- XGBoost
- Pandas
- NumPy
- Scikit-learn
- CVZone

---

## 📈 Future Improvements

- Multi-person activity recognition
- LSTM/Transformer temporal modeling
- Real-time webcam inference
- Activity confidence score
- Streamlit/Web dashboard
- Edge device deployment
- Multi-class activity detection

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Amalan Richard A C**

Computer Vision | Deep Learning | Machine Learning

If you found this project useful, consider giving it a ⭐ on GitHub!
