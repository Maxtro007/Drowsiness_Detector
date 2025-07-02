# 🛌 Real-Time Drowsiness Detection using YOLOv5 and PyTorch

This project implements a **real-time drowsiness detection system** using a custom-trained YOLOv5 model. The system detects "awake" and "drowsy" states from webcam input and is trained on personal image data. It uses PyTorch, OpenCV, and Ultralytics' YOLOv5 for object detection.

> ⚠️ **Note**: This project uses personal images for training. Model weights (`best.pt` / `last.pt`) are not included in the repo to preserve privacy.

---

## 📦 Features

- 🎥 Real-time webcam detection of `awake` vs `drowsy` states
- 🧠 Model training using YOLOv5
- 🖼️ Custom dataset collection using OpenCV
- 🏷️ Image annotation using `labelImg`
- 📊 Visualization using `matplotlib` and OpenCV

---

## 🔧 Setup Instructions

### 1. Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```

### 2. Clone and Set Up Annotation Tool

```bash
git clone https://github.com/tzutalin/labelImg
pip install pyqt5 lxml --upgrade
cd labelImg
pyrcc5 -o libs/resources.py resources.qrc
```

---

## 📸 Dataset Collection

Use OpenCV to collect webcam images.

---

## ✏️ Annotation with LabelImg

Use `labelImg` to annotate collected images and save in YOLO format.

---

## 📄 Dataset Configuration (`dataset.yaml`)

```yaml
train: ../data/images/train
val: ../data/images/val

nc: 2
names: ['awake', 'drowsy']
```

---

## 🏋️‍♂️ Model Training

```bash
cd yolov5
python train.py --img 320 --batch 16 --epochs 500 --data dataset.yaml --weights yolov5s.pt --workers 2
```

---

## 🔍 Inference (on Image)

```python
import torch
from matplotlib import pyplot as plt
import numpy as np
import os

model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp/weights/best.pt')
img = os.path.join('data', 'images', 'drowsy.xxxxx.jpg')
results = model(img)
plt.imshow(np.squeeze(results.render()))
plt.show()
```

---

## 🎥 Real-Time Webcam Detection

```python
import cv2
import torch
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp/weights/best.pt')
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    results = model(frame)
    cv2.imshow('YOLO', np.squeeze(results.render()))
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---





## 👨‍💻 Author

**Shivam Yadav**  
🧠 Full Stack Developer (MERN) & ML Enthusiast  


---

## 📌 Future Improvements

- [ ] Add drowsiness alarm alert system
- [ ] Integrate facial landmarks for accuracy
- [ ] Host model on a web/mobile interface

---

## 📝 License

This project is licensed for educational and personal research use. Please do not redistribute trained models without permission.
