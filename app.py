
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Phishing Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# 2. CUSTOM STYLING
# ==========================================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

div[data-testid="stMetric"] {
    background-color: rgba(128, 128, 128, 0.07);
    border: 1px solid rgba(128, 128, 128, 0.20);
    padding: 16px;
    border-radius: 12px;
}

div[data-testid="stMetricValue"] {
    font-size: 1.65rem;
    font-weight: 700;
}

[data-testid="stSidebar"] {
    border-right: 1px solid rgba(128, 128, 128, 0.2);
}

.section-heading {
    font-size: 1.2rem;
    font-weight: 650;
    margin-bottom: 12px;
}

.small-caption {
    color: #8b949e;
    font-size: 0.85rem;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# 3. APPLICATION HEADER
# ==========================================================

st.title("🛡️ Phishing Website Detection")

st.markdown(
    "Machine Learning Dashboard for Website Security "
    "and Phishing Classification"
)

st.caption(
    "Evaluate and compare trained machine learning "
    "models using an uploaded test dataset."
)

st.divider()


# ==========================================================
# 4. MODEL CONFIGURATION
# ==========================================================

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "K-Nearest Neighbors": "model/knn.joblib",
    "Gaussian Naive Bayes": "model/gaussian_nb.joblib",
    "Support Vector Machine": "model/svm.joblib"
}

MODEL_DESCRIPTIONS = {
    "Logistic Regression":
        "Linear classification using probability estimation.",

    "Decision Tree":
        "Tree-based classification using feature conditions.",

    "K-Nearest Neighbors":
        "Distance-based classification using nearest observations.",

    "Gaussian Naive Bayes":
        "Probabilistic classification using Gaussian assumptions.",

    "Support Vector Machine":
        "Linear classification using an optimal decision boundary."
}


# ==========================================================
# 5. LOAD SAVED MODELS
# ==========================================================

@st.cache_resource(show_spinner=False)
def load_model(model_path):

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    return joblib.load(model_path)


# ==========================================================
# 6. EVALUATION FUNCTION
# ==========================================================

def evaluate_model(model, test_data):

    target_column = "is_phishing"

    # Separate input features and target
    X_test = test_data.drop(columns=[target_column])
    y_test = pd.to_numeric(
        test_data[target_column],
        errors="coerce"
    )

    # Validate target
    if y_test.isna().any():
        raise ValueError("Target column contains invalid values.")

    if not set(y_test.unique()).issubset({0, 1}):
        raise ValueError(
            "Target values must be 0 or 1."
        )

    if y_test.nunique() != 2:
        raise ValueError(
            "Test dataset must contain both classes."
        )

    # Obtain feature names from the fitted pipeline
    expected_features = list(model.feature_names_in_)

    missing_features = [
        col for col in expected_features
        if col not in X_test.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    extra_features = [
        col for col in X_test.columns
        if col not in expected_features
    ]

    # Reorder features according to training
    X_test = X_test[expected_features].copy()

    # Validate numerical inputs
    numeric_data = X_test.apply(
        pd.to_numeric,
        errors="coerce"
    )

    invalid_values = (
        numeric_data.isna() & X_test.notna()
    )

    if invalid_values.any().any():
        raise ValueError(
            "Uploaded features contain non-numeric values."
        )

    X_test = numeric_data.replace(
        [np.inf, -np.inf], np.nan
    )

    # Predict using saved model
    y_pred = model.predict(X_test)

    # Obtain scores for AUC
    if hasattr(model, "predict_proba"):

        classes = list(model.classes_)
        positive_index = classes.index(1)

        y_score = model.predict_proba(X_test)[
            :, positive_index
        ]

    elif hasattr(model, "decision_function"):

        y_score = model.decision_function(X_test)

    else:
        raise ValueError(
            "Model does not support AUC score calculation."
        )

    # Calculate all six metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),

        "AUC": roc_auc_score(y_test, y_score),

        "Precision": precision_score(
            y_test, y_pred,
            pos_label=1,
            zero_division=0
        ),

        "Recall": recall_score(
            y_test, y_pred,
            pos_label=1,
            zero_division=0
        ),

        "F1 Score": f1_score(
            y_test, y_pred,
            pos_label=1,
            zero_division=0
        ),

        "MCC": matthews_corrcoef(
            y_test, y_pred
        )
    }

    # Generate classification report
    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Legitimate", "Phishing"],
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).T

    # Generate confusion matrix
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    # Prepare prediction dataset
    predictions = test_data.copy()

    predictions["predicted_class"] = y_pred

    predictions["predicted_label"] = np.where(
        y_pred == 1,
        "Phishing",
        "Legitimate"
    )

    predictions["prediction_result"] = np.where(
        y_pred == y_test.to_numpy(),
        "Correct",
        "Incorrect"
    )

    return {
        "metrics": metrics,
        "report": report_df,
        "confusion_matrix": cm,
        "predictions": predictions,
        "extra_features": extra_features
    }


# ==========================================================
# 7. SIDEBAR
# ==========================================================

st.sidebar.title("⚙️ Dashboard Controls")

st.sidebar.markdown("### Model Configuration")

view_mode = st.sidebar.radio(
    "Evaluation Mode",
    [
        "Single Model Evaluation",
        "Compare All Models"
    ]
)

selected_model = st.sidebar.selectbox(
    "Select Classification Model",
    list(MODEL_FILES.keys())
)

st.sidebar.caption(
    MODEL_DESCRIPTIONS[selected_model]
)

st.sidebar.divider()

st.sidebar.markdown("### Dataset Upload")

uploaded_file = st.sidebar.file_uploader(
    "Upload Test Dataset",
    type=["csv"],
    help="Upload a CSV containing all 30 input features "
         "and the is_phishing target column."
)

st.sidebar.info(
    "Target encoding:\n\n"
    "0 = Legitimate\n\n"
    "1 = Phishing"
)

st.sidebar.divider()

st.sidebar.markdown("### About")

st.sidebar.caption(
    "Developed using Python, Scikit-learn, "
    "Pandas and Streamlit."
)

st.sidebar.caption(
    "Models are loaded from saved Joblib pipelines. "
    "No model retraining is performed."
)


# ==========================================================
# 8. MAIN APPLICATION
# ==========================================================

if uploaded_file is None:

    st.info(
        "Please upload your test CSV using the sidebar "
        "to begin model evaluation."
    )

    st.markdown("### Application Features")

    col1, col2, col3 = st.columns(3)

    col1.markdown(
        "**📁 Dataset Upload**\n\n"
        "Upload the held-out test dataset."
    )

    col2.markdown(
        "**🤖 Model Selection**\n\n"
        "Evaluate five trained classifiers."
    )

    col3.markdown(
        "**📊 Performance Analysis**\n\n"
        "View metrics and classification results."
    )

    st.stop()


# ==========================================================
# 9. READ AND VALIDATE DATASET
# ==========================================================

try:

    test_data = pd.read_csv(uploaded_file)

except Exception as error:

    st.error(f"Unable to read CSV file: {error}")
    st.stop()


if test_data.empty:

    st.error("Uploaded dataset is empty.")
    st.stop()


if "is_phishing" not in test_data.columns:

    st.error(
        "Required target column 'is_phishing' is missing."
    )
    st.stop()


# Dataset information
total_records = len(test_data)

total_features = len(test_data.columns) - 1

phishing_count = (
    pd.to_numeric(
        test_data["is_phishing"],
        errors="coerce"
    ) == 1
).sum()

legitimate_count = total_records - phishing_count


# ==========================================================
# 10. DATASET SUMMARY
# ==========================================================

st.subheader("📁 Uploaded Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Records",
    f"{total_records:,}"
)

col2.metric(
    "Input Features",
    total_features
)

col3.metric(
    "Phishing Records",
    f"{phishing_count:,}"
)

col4.metric(
    "Legitimate Records",
    f"{legitimate_count:,}"
)

st.caption(
    f"Uploaded file: {uploaded_file.name} | "
    f"Size: {uploaded_file.size / 1024:.2f} KB"
)


with st.expander("Preview Uploaded Dataset"):

    st.dataframe(
        test_data.head(10),
        use_container_width=True
    )

    st.caption(
        f"Dataset dimensions: {test_data.shape}"
    )


st.divider()


# ==========================================================
# 11. RUN MODEL EVALUATION
# ==========================================================

if view_mode == "Single Model Evaluation":

    models_to_evaluate = [selected_model]

else:

    models_to_evaluate = list(MODEL_FILES.keys())


all_results = {}
failed_models = {}

with st.spinner("Evaluating selected classification models..."):

    for model_name in models_to_evaluate:

        try:

            model = load_model(
                MODEL_FILES[model_name]
            )

            all_results[model_name] = evaluate_model(
                model,
                test_data
            )

        except Exception as error:

            failed_models[model_name] = str(error)


if failed_models:

    for model_name, error in failed_models.items():

        st.error(f"{model_name}: {error}")

    st.stop()


if selected_model not in all_results:

    selected_model = models_to_evaluate[0]

selected_result = all_results[selected_model]


# ==========================================================
# 12. CREATE DASHBOARD TABS
# ==========================================================

tab_overview, tab_compare, tab_confusion, tab_report, tab_predictions = st.tabs(
    [
        "📊 Overview",
        "📈 Model Comparison",
        "🎯 Confusion Matrix",
        "📋 Classification Report",
        "📥 Predictions"
    ]
)


# ==========================================================
# 13. OVERVIEW TAB
# ==========================================================

with tab_overview:

    st.subheader("Model Performance Overview")

    st.markdown(
        f"Selected Model: **{selected_model}**"
    )

    metrics = selected_result["metrics"]

    # First row of metrics
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Accuracy",
        f"{metrics['Accuracy']:.2%}"
    )

    col2.metric(
        "ROC AUC",
        f"{metrics['AUC']:.2%}"
    )

    col3.metric(
        "Precision",
        f"{metrics['Precision']:.2%}"
    )

    # Second row of metrics
    col4, col5, col6 = st.columns(3)

    col4.metric(
        "Recall",
        f"{metrics['Recall']:.2%}"
    )

    col5.metric(
        "F1 Score",
        f"{metrics['F1 Score']:.2%}"
    )

    col6.metric(
        "MCC",
        f"{metrics['MCC']:.4f}"
    )

    st.divider()

    st.subheader("Classification Summary")

    correct = (
        selected_result["predictions"]
        ["prediction_result"]
        .eq("Correct")
        .sum()
    )

    incorrect = total_records - correct

    col1, col2 = st.columns(2)

    col1.metric(
        "Correct Predictions",
        f"{correct:,}"
    )

    col2.metric(
        "Incorrect Predictions",
        f"{incorrect:,}"
    )

    accuracy_summary = pd.DataFrame({
        "Prediction Result": [
            "Correct",
            "Incorrect"
        ],
        "Count": [
            correct,
            incorrect
        ]
    })

    st.bar_chart(
        accuracy_summary.set_index("Prediction Result")
    )


# ==========================================================
# 14. MODEL COMPARISON TAB
# ==========================================================

with tab_compare:

    st.subheader("Comparison of Classification Models")

    if len(all_results) == 1:

        st.info(
            "Select 'Compare All Models' from the sidebar "
            "to display the performance of all five classifiers."
        )

    else:

        comparison_df = pd.DataFrame({
            name: result["metrics"]
            for name, result in all_results.items()
        }).T

        st.dataframe(
            comparison_df.style.format("{:.4f}").highlight_max(
                axis=0,
                color="#184D35"
            ),
            use_container_width=True
        )

        st.markdown("### Performance Comparison Chart")

        chart_metrics = st.multiselect(
            "Select metrics to compare",
            options=list(comparison_df.columns),
            default=[
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score"
            ]
        )

        if chart_metrics:

            st.bar_chart(
                comparison_df[chart_metrics],
                use_container_width=True
            )

        # Rank by F1
        ranking_df = comparison_df.sort_values(
            by="F1 Score",
            ascending=False
        )

        st.markdown("### Model Ranking by F1 Score")

        st.dataframe(
            ranking_df[["F1 Score", "Accuracy", "AUC", "MCC"]]
            .style.format("{:.4f}"),
            use_container_width=True
        )

        winner = ranking_df.index[0]

        st.success(
            f"Highest testing F1 Score: {winner}"
        )


# ==========================================================
# 15. CONFUSION MATRIX TAB
# ==========================================================

with tab_confusion:

    st.subheader("Confusion Matrix")

    cm = selected_result["confusion_matrix"]

    fig, ax = plt.subplots(figsize=(7, 5))

    image = ax.imshow(
        cm,
        cmap="Blues"
    )

    ax.set_xticks(
        [0, 1],
        labels=["Legitimate", "Phishing"]
    )

    ax.set_yticks(
        [0, 1],
        labels=["Legitimate", "Phishing"]
    )

    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")

    ax.set_title(
        f"Confusion Matrix - {selected_model}"
    )

    # Display cell values
    threshold = cm.max() / 2

    for i in range(2):

        for j in range(2):

            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color=(
                    "white"
                    if cm[i, j] > threshold
                    else "black"
                ),
                fontsize=14,
                fontweight="bold"
            )

    fig.colorbar(image, ax=ax)

    st.pyplot(fig)
    plt.close(fig)

    # Display individual confusion matrix values
    tn, fp, fn, tp = cm.ravel()

    col1, col2 = st.columns(2)

    col1.metric("True Negatives", f"{tn:,}")
    col2.metric("False Positives", f"{fp:,}")

    col3, col4 = st.columns(2)

    col3.metric("False Negatives", f"{fn:,}")
    col4.metric("True Positives", f"{tp:,}")


# ==========================================================
# 16. CLASSIFICATION REPORT TAB
# ==========================================================

with tab_report:

    st.subheader("Classification Report")

    report_df = selected_result["report"].copy()

    st.dataframe(
        report_df.style.format(
            {
                "precision": "{:.4f}",
                "recall": "{:.4f}",
                "f1-score": "{:.4f}",
                "support": "{:.0f}"
            }
        ),
        use_container_width=True
    )

    st.caption(
        "The report presents Precision, Recall, F1 Score "
        "and Support for legitimate and phishing classes."
    )


# ==========================================================
# 17. PREDICTIONS TAB
# ==========================================================

with tab_predictions:

    st.subheader("Prediction Results")

    prediction_df = selected_result["predictions"]

    st.dataframe(
        prediction_df,
        use_container_width=True,
        height=400
    )

    st.markdown("### Download Results")

    csv_data = prediction_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Prediction Results",
        data=csv_data,
        file_name="phishing_predictions.csv",
        mime="text/csv",
        type="primary"
    )


# ==========================================================
# 18. FOOTER
# ==========================================================

st.divider()

st.caption(
    "Phishing Website Detection | "
    "Machine Learning Project | "
    "BITS Pilani M.Tech AIML"
)

st.caption(
    "All evaluations use saved models. "
    "No retraining is performed on uploaded test data."
)
