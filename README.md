# Phishing Website Detection Using Machine Learning

## 1. Problem Statement

Phishing is a major cybersecurity threat in which attackers create fraudulent websites to deceive users into revealing sensitive information, such as passwords, banking credentials, and personal data.

The objective of this project is to develop and evaluate machine learning classification models capable of identifying phishing and legitimate websites based on their website characteristics.

The project aims to compare the performance of five machine learning algorithms and identify the most suitable model for phishing website detection.

The primary objectives are:

- To preprocess and analyze the phishing website dataset.
- To implement five machine learning classification algorithms.
- To evaluate the models using multiple performance metrics.
- To compare the classification performance and generalization ability of the models.
- To identify the best-performing model for phishing website detection.

## 2. Dataset Description

The dataset used in this project contains website-related features that help distinguish phishing websites from legitimate websites.

The dataset is used to train and evaluate supervised machine learning classification models.

### Dataset Information

| Attribute | Description |
|---|---|
| Dataset Name | [Enter the exact dataset name] |
| Dataset Source | [Enter the dataset source or URL] |
| Number of Instances | [Enter the number of rows] |
| Number of Features | [Enter the number of input features] |
| Target Variable | Website classification |
| Classification Type | Binary classification |
| Classes | Phishing and Legitimate |

### Dataset Characteristics

The dataset consists of website characteristics that serve as input features for the classification models.

The target variable represents whether a website is phishing or legitimate.

The dataset was prepared for machine learning by performing the necessary preprocessing, feature preparation, and train-test splitting.

**Note:** Replace the placeholders with the actual dataset information used in the Jupyter Notebook.

## 3. GitHub Repository Link

The complete project, including the dataset-related files, Jupyter Notebook, model evaluation results, and documentation, is maintained in the following GitHub repository.

**Repository:** Phishing Website Detection

**GitHub Link:** [Insert your GitHub repository URL here]

The repository contains the implementation of all five machine learning models, their evaluation results, and the documentation required for the assignment.

## 4. Models Used

Five supervised machine learning classification algorithms were implemented and evaluated for phishing website detection.

The following models were considered:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

### 4.1 Model Performance Comparison

The five classification algorithms were evaluated using six performance metrics: Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 91.60% | 97.38% | 91.19% | 89.29% | 90.23% | 0.8288 |
| Decision Tree | 93.99% | 96.10% | 94.33% | 91.68% | 92.99% | 0.8776 |
| K-Nearest Neighbors | **94.08%** | **98.33%** | 93.97% | **92.31%** | **93.13%** | **0.8795** |
| Gaussian Naive Bayes | 90.11% | 96.30% | 87.72% | 89.81% | 88.75% | 0.7994 |
| Support Vector Machine | 91.69% | 97.34% | 91.65% | 88.98% | 90.30% | 0.8306 |

### 4.2 Individual Model Observations

| Model | Observations |
|---|---|
| Logistic Regression | Achieved 91.60% accuracy and 90.23% F1 Score. It provides a reliable baseline with relatively low overfitting. |
| Decision Tree | Achieved 93.99% accuracy and 92.99% F1 Score. It provides strong classification performance but shows some overfitting. |
| K-Nearest Neighbors | Achieved the highest accuracy, F1 Score, AUC, and MCC. It demonstrated the strongest overall testing performance. |
| Gaussian Naive Bayes | Achieved 90.11% accuracy and 88.75% F1 Score. It performed comparatively poorly among the evaluated classifiers. |
| Support Vector Machine | Achieved 91.69% accuracy and 90.30% F1 Score. It provided reasonable performance but did not outperform KNN or Decision Tree. |

### 4.3 Generalization Gap Analysis

The generalization gap was calculated as the difference between training and testing accuracy.

| Model | Training Accuracy | Testing Accuracy | Accuracy Gap |
|---|---:|---:|---:|
| Logistic Regression | 93.01% | 91.60% | 1.41% |
| Decision Tree | 97.86% | 93.99% | 3.87% |
| K-Nearest Neighbors | 98.94% | 94.08% | 4.85% |
| Gaussian Naive Bayes | 91.03% | 90.11% | 0.92% |
| Support Vector Machine | 93.17% | 91.69% | 1.48% |

KNN achieved the highest testing performance but also exhibited the largest accuracy gap. Decision Tree demonstrated a similar tendency toward overfitting.

Logistic Regression, Gaussian Naive Bayes, and SVM exhibited smaller generalization gaps.

### 4.4 Final Model Selection

#### Selected Model: K-Nearest Neighbors (KNN)

K-Nearest Neighbors was selected as the best-performing model based on its overall testing performance.

The model achieved:

- Accuracy: 94.08%
- AUC: 98.33%
- Precision: 93.97%
- Recall: 92.31%
- F1 Score: 93.13%
- MCC: 0.8795

KNN achieved the highest accuracy, AUC, F1 Score, and MCC among the five evaluated classifiers.

Its high AUC indicates strong discrimination between phishing and legitimate websites.

Although KNN exhibited some overfitting, its testing performance remained superior to the other evaluated models.

Therefore, **K-Nearest Neighbors was selected as the final model for phishing website detection.**

### 4.5 Conclusion

The experimental results demonstrate that machine learning algorithms can effectively identify phishing websites using website characteristics.

Among the five evaluated classifiers, KNN achieved the strongest overall testing performance, followed closely by Decision Tree.

The findings highlight the importance of hyperparameter tuning, appropriate cross-validation, and comprehensive evaluation using multiple performance metrics.

The selected KNN model will be used for the subsequent model deployment and prediction application.
