# 🫁 Chest X-ray Classification using CNN

A deep learning project focused on classifying chest X-ray images into multiple categories using Convolutional Neural Networks (CNNs), with an emphasis on model performance, robustness, and real-world applicability.

---

## 📌 Problem
Manual analysis of chest X-rays is time-consuming and requires expertise. This project aims to assist in automated diagnosis by building a model that can accurately classify X-ray images.

---

## 🎯 Objective
To develop a CNN-based model that can reliably classify chest X-ray images while maintaining strong generalization across unseen data.

---

## 🗂️ Dataset
- Multi-class chest X-ray dataset from Kaggle  
- Classes: * Normal, Pneumonia, COVID-19*  

### Preprocessing Steps
- Image resizing  
- Normalization  
- Data augmentation (rotation, flipping, etc.)  

---

## 🧠 Model Architecture
- Convolutional layers with ReLU activation  
- MaxPooling layers for spatial reduction  
- Fully connected dense layers  
- Dropout for regularization  

<img width="476" height="272" alt="image" src="https://github.com/user-attachments/assets/ba5ea820-b975-40dd-9e55-2d193b8ad621" />


---

## ⚙️ Training Details
- Loss Function: Categorical Crossentropy  
- Optimizer: Adam  
- Evaluation: Accuracy + Cross-validation  

---

## 📊 Results

- ✅ Test Accuracy: **95.5%**  
- ✅ Cross-validation Accuracy: **~94% (5-fold)**  

---

### 📉 Confusion Matrix
<img width="572" height="458" alt="image" src="https://github.com/user-attachments/assets/d4490ff3-9f8c-4453-a92e-c298c72debd7" />


---

### 📈 Training Performance
<img width="912" height="370" alt="image" src="https://github.com/user-attachments/assets/3ceba0f1-724c-460c-a0fb-a90638e5361a" />


---


## 🔎 Key Insights
- The model generalizes well across classes  
- Data augmentation significantly improved performance  
- Minor misclassifications occur in visually similar cases  

---

## 🚀 Future Improvements
- Implement transfer learning (ResNet, EfficientNet)  
- Add explainability using Grad-CAM  
- Deploy as a web-based diagnostic tool  

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python train.py
