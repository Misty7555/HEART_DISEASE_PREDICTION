💓 Heart Disease Prediction using Machine Learning
📌 Project Description

This project presents a comprehensive machine learning approach to analyze and understand heart disease data using multiple algorithms. It integrates both supervised and unsupervised learning techniques to perform prediction, classification, clustering, and anomaly detection on patient health records. The project demonstrates how machine learning can assist in extracting meaningful insights from healthcare data.

🎯 Aim of the Project

The main aim of this project is to apply different machine learning algorithms to:

Predict cholesterol levels using regression techniques

Classify patients based on the presence of heart disease

Group patients with similar health characteristics

Detect abnormal or outlier patient cases

📊 Dataset Information

Dataset: Heart Disease Dataset

Number of records: ~1000+ patients

Number of features: 14

Key Features:

Age

Sex

Chest pain type

Resting blood pressure

Cholesterol level

Maximum heart rate

Exercise-induced angina

and more

Target Variable:

0 → No Heart Disease

1 → Heart Disease

⚙️ Data Preprocessing

To ensure the dataset is suitable for machine learning models, the following steps were performed:

Removal of duplicate records

Handling of missing values

Conversion of categorical features into numerical values using Label Encoding

Feature scaling using StandardScaler for distance-based algorithms

🤖 Machine Learning Algorithms Used
🔹 1. Linear Regression

Type: Supervised Learning (Regression)

Purpose: Predict cholesterol levels

Evaluation Metrics: Mean Squared Error (MSE), Root Mean Squared Error (RMSE)

Insight: Achieved moderate prediction accuracy due to complex medical relationships

🔹 2. Naive Bayes

Type: Supervised Learning (Classification)

Purpose: Predict whether a patient has heart disease

Evaluation Metric: Accuracy (~75–85%)

Additional Tool: Confusion Matrix

Insight: Performed well but limited by feature independence assumption

🔹 3. K-Means Clustering

Type: Unsupervised Learning

Purpose: Group patients based on similarity

Insight: Successfully divided patients into clusters representing different health profiles

🔹 4. K-Medoids Clustering

Type: Unsupervised Learning

Purpose: Robust clustering

Key Advantage: Uses actual data points (medoids), making it less sensitive to outliers

Insight: Produced more stable clusters compared to K-Means

🔹 5. DBSCAN

Type: Unsupervised Learning

Purpose: Detect dense clusters and identify outliers

Key Feature: Does not require predefined number of clusters

Insight: Identified abnormal patient cases as outliers

📈 Visualizations

The project includes various visualizations to better understand model performance:

Actual vs Predicted Plot (Linear Regression)

Residual Plot (Error Analysis)

Confusion Matrix (Classification Evaluation)

Cluster Plots (K-Means & K-Medoids)

DBSCAN Plot (Outlier Detection)

Combined Dashboard (All models in one view)

📌 Results Summary

Linear Regression achieved moderate prediction accuracy

Naive Bayes achieved ~77% classification accuracy

K-Means and K-Medoids successfully grouped patients into meaningful clusters

DBSCAN detected a realistic number of outliers (~40 abnormal cases)

⚠️ Limitations

Dataset lacks complete medical information (e.g., lifestyle, genetics)

Medical data is highly variable and noisy

Linear models may not capture non-linear relationships

Some features may overlap between classes

🚀 Future Improvements

Use advanced models such as Random Forest or Gradient Boosting

Perform hyperparameter tuning

Include more real-world features

Apply deep learning techniques for better accuracy

🧠 Key Learnings

Importance of data preprocessing in ML

Difference between supervised and unsupervised learning

Role of visualization in interpreting models

Challenges in working with real-world healthcare data

🎯 Conclusion

This project demonstrates how machine learning techniques can be effectively applied to healthcare data for prediction, classification, clustering, and anomaly detection. While the models provide reasonable performance, further improvements can enhance accuracy and reliability in real-world applications.

⭐ Author

Developed as part of a Machine Learning project to explore real-world applications of AI in healthcare.
