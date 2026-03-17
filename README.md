# ❤️ Heart Disease Prediction using Machine Learning

## 📌 Project Overview
This project implements multiple **Machine Learning algorithms** on the **Heart Disease UCI Dataset** to analyze patient data, predict heart disease, and discover patterns in medical features.

The goal is to demonstrate the use of:
- Regression
- Classification
- Clustering
- Outlier Detection

---

## 📊 Dataset
Dataset used: **Heart Disease UCI Dataset (Kaggle)**  

Features include:
- Age  
- Sex  
- Chest Pain Type  
- Blood Pressure  
- Cholesterol  
- Heart Rate  
- etc.

Target:
- `0` → No Heart Disease  
- `1` → Heart Disease  

---

## ⚙️ Algorithms Implemented

### 1. Linear Regression
- Predicts **cholesterol level**
- Evaluation Metric: **Mean Squared Error (MSE)**

### 2. Naive Bayes
- Classifies patients into **disease / no disease**
- Evaluation Metric: **Accuracy (~78%)**

### 3. K-Means Clustering
- Groups patients into clusters based on similarity

### 4. K-Medoids Clustering
- Similar to K-Means but uses **actual data points (medoids)**

### 5. DBSCAN
- Detects **dense clusters and outliers**
- Identifies abnormal patient patterns

---

## 📈 Visualizations
The project includes graphs for better understanding:
- Linear Regression → Actual vs Predicted Plot  
- Naive Bayes → Confusion Matrix  
- K-Means → Cluster Visualization  
- K-Medoids → Cluster Visualization  
- DBSCAN → Cluster + Outlier Detection  

---

## 🛠️ Technologies Used
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Scikit-learn-extra  
- Matplotlib  
- Seaborn  

---

## 📂 Project Structure
