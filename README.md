# Phishing Website Detection Using Machine Learning

## 1. Project Overview

Phishing is a cyberattack in which attackers create deceptive websites to steal sensitive information such as login credentials, banking details, and personal data.

This project uses machine learning classification algorithms to identify phishing and legitimate websites based on their extracted website features.

**Objective:** Build and evaluate machine learning models to classify websites as phishing or legitimate and deploy the best-performing model through a Streamlit web application.

## 2. Dataset Description

- **Dataset:** UCI Phishing Websites
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/327/phishing+websites)
- **Instances:** 11,055
- **Features:** 30
- **Target:** Website classification

**Target encoding:**
- `1` — Phishing
- `0` — Legitimate

## 3. Project Methodology

The project follows these stages:

1. Dataset collection and preprocessing.
2. Exploratory Data Analysis (EDA).
3. Duplicate feature-vector analysis.
4. Group-aware, stratified train-test splitting.
5. Model training and hyperparameter tuning.
6. Performance evaluation and comparison.
7. Model selection and serialization.
8. Streamlit application development and deployment.

Stratified group splitting was used to keep identical feature vectors within the same split and reduce the risk of duplicate-record leakage between training and testing.

## 4. Machine Learning Algorithms

Five classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

Hyperparameter tuning was performed using RandomizedSearchCV with grouped cross-validation.

## 5. Model Performance Comparison

The models were evaluated on the held-out test dataset.

| Model | Accuracy | ROC-AUC | Precision | Recall | F1-Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 91.60% | 97.38% | 91.19% | 89.29% | 90.23% | 0.8288 |
| Decision Tree | 93.99% | 96.10% | 94.33% | 91.68% | 92.99% | 0.8776 |
| KNN | **94.08%** | **98.33%** | 93.97% | **92.31%** | **93.13%** | **0.8795** |
| Gaussian Naive Bayes | 90.11% | 96.30% | 87.72% | 89.81% | 88.75% | 0.7994 |
| SVM | 91.69% | 97.34% | 91.65% | 88.98% | 90.30% | 0.8306 |

## 6. Final Model Selection

**Selected Model: K-Nearest Neighbors (KNN)**

KNN achieved the strongest overall test performance among the five evaluated models.

Key results:

- Accuracy: 94.08%
- ROC-AUC: 98.33%
- Precision: 93.97%
- Recall: 92.31%
- F1-Score: 93.13%
- MCC: 0.8795

The Decision Tree achieved slightly higher precision (94.33%), while KNN performed better on accuracy, ROC-AUC, recall, F1-score, and MCC.

Therefore, KNN was selected as the final model based on its overall test performance.

## 7. Deployment

The selected KNN pipeline was serialized using Joblib and integrated into a Streamlit web application.

The application allows users to:
- Enter encoded website feature values.
- Submit the features for prediction.
- View the predicted classification.
- View the model-estimated phishing probability.

**Live Application:** [Open Streamlit App](https://phishingwebsitedetection-ml.streamlit.app/)

## 8. Limitations and Future Improvements

- The application currently accepts pre-extracted, encoded feature values rather than automatically analyzing a raw URL.
- The model's performance depends on the dataset and feature representation used during training.
- Future work could include automatic URL feature extraction, testing on newer phishing datasets, additional algorithms, and further model optimization.

## 9. Conclusion

This project demonstrates the application of machine learning techniques to phishing website detection. Five classification algorithms were trained, tuned, and evaluated using multiple performance metrics.

Among the evaluated models, KNN achieved the strongest overall test performance, with 94.08% accuracy and a 93.13% F1-score. The trained pipeline was saved and integrated into a Streamlit application to demonstrate model deployment.

The project provides an end-to-end machine learning workflow, from dataset analysis and model evaluation to application development.
