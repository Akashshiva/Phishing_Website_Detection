# Phishing Website Detection Using Machine Learning
## 1. Problem Statement

Phishing websites imitate legitimate websites to trick users into disclosing sensitive information. The objective of this project is to build and compare supervised machine-learning classifiers that identify phishing and legitimate websites from encoded website features.

The project implements five classification algorithms, evaluates them using six metrics, and demonstrates the saved models through a Streamlit application.

## 2. Dataset Description

- **Dataset:** Phishing Websites Dataset
- **Source:** [UCI Machine Learning Repository – Dataset 327](https://archive.ics.uci.edu/dataset/327/phishing)
- **Instances:** 11,055
- **Input features:** 30
- **Problem type:** Binary classification
- **Missing values:** The UCI dataset documentation reports no missing values.

**Target encoding used in this project:**
- `1` — Phishing website (positive class)
- `0` — Legitimate website

The original dataset labels were transformed from `-1` (phishing) and `+1` (legitimate) into the encoding above.

Identical feature vectors were kept within the same partition using grouped, stratified splitting to reduce duplicate-feature leakage between training and testing. The experiment used 8,841 training records and 2,214 testing records (approximately 80:20).

## 3. GitHub Repository

[Phishing_Website_Detection – GitHub Repository](https://github.com/Akashshiva/Phishing_Website_Detection)

The repository includes the notebook source, Streamlit application, requirements file, held-out test CSV, and five serialized model pipelines in the `model/` directory.

## 4. Machine Learning Models

The five required classification algorithms are:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

Hyperparameters were tuned using RandomizedSearchCV with 3-fold Stratified Group Cross-Validation. The selected models were evaluated on the held-out test dataset. Phishing (class 1) is treated as the positive class.

## 5. Model Performance Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 91.60% | 97.38% | 91.19% | 89.29% | 90.23% | 0.8288 |
| Decision Tree | 93.99% | 96.10% | 94.33% | 91.68% | 92.99% | 0.8776 |
| K-Nearest Neighbors | **94.08%** | **98.33%** | 93.97% | **92.31%** | **93.13%** | **0.8795** |
| Gaussian Naive Bayes | 90.11% | 96.30% | 87.72% | 89.81% | 88.75% | 0.7994 |
| Support Vector Machine | 91.69% | 97.34% | 91.65% | 88.98% | 90.30% | 0.8306 |

## 6. Model Performance Observations

| Model | Observation |
|---|---|
| Logistic Regression | Achieved 91.60% accuracy and 90.23% F1 Score. It provides a reliable baseline and has a relatively small training-testing accuracy gap. |
| Decision Tree | Achieved 93.99% accuracy and 92.99% F1 Score. It provides strong classification performance, with some evidence of overfitting. |
| K-Nearest Neighbors | Achieved the highest testing accuracy (94.08%), AUC (98.33%), recall (92.31%), F1 Score (93.13%), and MCC (0.8795). It also had the largest training-testing accuracy gap among the five models. |
| Gaussian Naive Bayes | Achieved 90.11% accuracy and 88.75% F1 Score. It was the weakest of the five evaluated classifiers on the reported test metrics. |
| Support Vector Machine | Achieved 91.69% accuracy and 90.30% F1 Score. It showed a relatively small training-testing accuracy gap but did not outperform KNN or Decision Tree on test accuracy. |

## 7. Generalization Gap Analysis

The accuracy generalization gap is calculated as training accuracy minus testing accuracy.

| Model | Training Accuracy | Testing Accuracy | Accuracy Gap |
|---|---:|---:|---:|
| Logistic Regression | 93.01% | 91.60% | 1.41 percentage points |
| Decision Tree | 97.86% | 93.99% | 3.87 percentage points |
| KNN | 98.94% | 94.08% | 4.85 percentage points |
| Gaussian Naive Bayes | 91.03% | 90.11% | 0.92 percentage points |
| Support Vector Machine | 93.17% | 91.69% | 1.48 percentage points |

KNN achieved the strongest test performance but also the largest accuracy gap. This indicates a greater difference between training and unseen-test performance and should be considered when interpreting the result.

## 8. Overall Winner

**Selected model: K-Nearest Neighbors (KNN)**

KNN was selected based on its overall held-out test performance. It achieved the highest accuracy, AUC, recall, F1 Score, and MCC among the five required classifiers. Decision Tree achieved slightly higher precision (94.33% compared with 93.97%).

KNN's training-testing accuracy gap was larger than the other models, so this limitation is acknowledged.

## 9. Streamlit Application

**Live application:** [Open PhishGuard – Streamlit Community Cloud](https://phishingwebsitedetection-ml.streamlit.app/)

The application loads the saved classification pipelines and evaluates an uploaded test CSV without retraining. It provides:

- CSV test-dataset upload
- Model-selection dropdown
- Six evaluation metrics
- Confusion matrix
- Classification report
- Comparison of the five models
- Downloadable prediction results

The uploaded test CSV must contain all required input features and the `is_phishing` target column.

## 10. Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Streamlit
- GitHub

## 11. Conclusion

This project demonstrates an end-to-end machine-learning workflow for phishing website detection, including dataset analysis, group-aware data splitting, model implementation, hyperparameter tuning, evaluation, model serialization, and deployment.

Among the five required classifiers, KNN achieved the strongest overall held-out test performance, with 94.08% accuracy and a 93.13% F1 Score. The saved model pipelines are integrated into the Streamlit application to demonstrate evaluation on uploaded test data without retraining.
