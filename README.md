# Brain Tumor Detection using Deep Learning

This project implements a **Convolutional Neural Network (CNN)** using **Transfer Learning with VGG16** to detect different types of brain tumors from MRI scans. The system is deployed using a **Flask-based web interface** for real-time image classification.

---

##  Objective
To build a robust brain tumor classification system that can:
- Detect the presence of a tumor (or absence)
- Classify tumor types into **Pituitary**, **Meningioma**, or **Glioma**
- Be deployed as a user-friendly web application using **Flask**

---

## Approach

###  Dataset
- MRI brain scans from the publicly available dataset with four classes:  
  `['pituitary', 'glioma', 'notumor', 'meningioma']`  
- Directory structure:
- /MRI Images/
├── Training/
└── Testing/
- 
###  Preprocessing
- Image augmentation (brightness, contrast)
- Normalization and resizing to `128x128`
- Encoded labels for classification

###  Model Architecture
- **VGG16** base model with `imagenet` weights (top removed)
- Last 3 convolutional layers unfrozen
- Custom head:
- `Flatten → Dropout(0.3) → Dense(128, ReLU) → Dropout(0.2) → Softmax`
- Optimizer: `Adam`, Loss: `sparse_categorical_crossentropy`

###  Training Summary
| Epoch | Accuracy | Loss  |
|-------|----------|-------|
| 1     | 73.6%    | 0.635 |
| 5     | 97.2%    | 0.082 |

###  Evaluation Metrics
- Accuracy: **95.0%**
- Precision, Recall, F1-score (per class)
- Confusion Matrix & ROC-AUC Curves (multi-class)

---

##  Web Deployment (Flask)

- Upload MRI scan through the web interface
- Real-time prediction of tumor type and confidence score
- Output image with classification label

###  Running the App


pip install -r requirements.txt
python app.py


### Results
Classification Accuracy: ~95%

Model Generalization: High recall across all tumor types

Real-Time Detection: Sub-second prediction latency

Web UI: Lightweight, intuitive, browser-accessible

### Sample Prediction
makefile
Copy
Edit
Input: MRI Image
Output: Tumor: Glioma (Confidence: 96.3%)
