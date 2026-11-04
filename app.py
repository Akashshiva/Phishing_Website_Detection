
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="Phishing Website Detection",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Phishing Website Detection")
st.write("Evaluate phishing detection models using an uploaded test dataset.")

# Model file locations
MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Gaussian Naive Bayes": "model/gaussian_nb.joblib",
    "SVM": "model/svm.joblib"
}

# Model selection
selected_model = st.selectbox(
    "Select a trained model",
    list(MODEL_FILES.keys())
)

# Upload test dataset
uploaded_file = st.file_uploader(
    "Upload your test CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        test_data = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Dataset")
        st.write("Dataset shape:", test_data.shape)
        st.dataframe(test_data.head())

        # Check target column
        target_column = "is_phishing"

        if target_column not in test_data.columns:
            st.error(
                "The uploaded CSV must contain the "
                "'is_phishing' target column."
            )
            st.stop()

        # Separate features and target
        X_test = test_data.drop(columns=[target_column])
        y_test = test_data[target_column]

        # Check target values
        if not set(y_test.unique()).issubset({0, 1}):
            st.error(
                "Target values must be 0 (Legitimate) "
                "or 1 (Phishing)."
            )
            st.stop()

        if y_test.nunique() < 2:
            st.error(
                "The test dataset must contain both "
                "phishing and legitimate examples "
                "to calculate ROC AUC."
            )
            st.stop()

        # Load selected saved model
        model = joblib.load(MODEL_FILES[selected_model])

        # Verify input feature columns
        expected_features = list(model.feature_names_in_)

        missing_features = [
            feature for feature in expected_features
            if feature not in X_test.columns
        ]

        extra_features = [
            feature for feature in X_test.columns
            if feature not in expected_features
        ]

        if missing_features:
            st.error(
                f"Missing required features: {missing_features}"
            )
            st.stop()

        if extra_features:
            st.warning(
                f"Extra columns will be ignored: {extra_features}"
            )

        # Keep the same feature order as training
        X_test = X_test[expected_features]

        # Predict using saved model
        y_pred = model.predict(X_test)

        # Probability/decision scores for ROC AUC
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]

        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)

        else:
            y_score = None

        # Evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(
            y_test, y_pred, zero_division=0
        )
        recall = recall_score(
            y_test, y_pred, zero_division=0
        )
        f1 = f1_score(
            y_test, y_pred, zero_division=0
        )
        mcc = matthews_corrcoef(y_test, y_pred)

        if y_score is not None:
            auc = roc_auc_score(y_test, y_score)
        else:
            auc = np.nan

        # Display results
        st.subheader("Evaluation Metrics")

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", f"{accuracy:.4f}")
        col2.metric("ROC AUC", f"{auc:.4f}")
        col3.metric("Precision", f"{precision:.4f}")

        col4, col5, col6 = st.columns(3)

        col4.metric("Recall", f"{recall:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC", f"{mcc:.4f}")

        # Confusion matrix
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(
            y_test, y_pred, labels=[0, 1]
        )

        cm_df = pd.DataFrame(
            cm,
            index=["Actual Legitimate", "Actual Phishing"],
            columns=["Predicted Legitimate", "Predicted Phishing"]
        )

        st.dataframe(cm_df)

        # Classification report
        st.subheader("Classification Report")

        report = classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Legitimate", "Phishing"],
            zero_division=0
        )

        st.text(report)

        # Download predictions
        results = test_data.copy()
        results["predicted_phishing"] = y_pred

        st.download_button(
            label="Download Predictions CSV",
            data=results.to_csv(index=False),
            file_name="phishing_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error processing dataset: {e}")
