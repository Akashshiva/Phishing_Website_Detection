
import os
from pathlib import Path

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
    page_title="PhishGuard | Security Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent


# ==========================================================
# 2. CUSTOM UI / UX DESIGN
# ==========================================================

st.markdown("""
<style>

/* Main application */
.stApp {
    background: linear-gradient(
        135deg,
        #08111F 0%,
        #101B2D 55%,
        #0B1524 100%
    );
}

/* Main content spacing */
.stMainBlockContainer,
div[data-testid="stMainBlockContainer"] {
    max-width: 1500px;
    padding-top: 4.5rem !important;
    padding-bottom: 3rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0D1829;
    border-right: 1px solid #24334B;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #E8F0FC;
}

/* Main headings */
h1, h2, h3 {
    color: #F3F7FF !important;
    font-weight: 700 !important;
}

p, label {
    color: #C4D0E0;
}

/* Header */
.hero {
    background: linear-gradient(
        115deg,
        #122D4B,
        #123D59,
        #102D42
    );
    padding: 30px;
    border-radius: 18px;
    border: 1px solid #28506C;
    margin-bottom: 25px;
}

.hero-title {
    color: #FFFFFF;
    font-size: 34px;
    font-weight: 800;
    letter-spacing: -1px;
}

.hero-subtitle {
    color: #B9D2E9;
    font-size: 15px;
    margin-top: 8px;
}

.hero-tag {
    display: inline-block;
    background: #164D51;
    color: #7CE8D0;
    border: 1px solid #267C77;
    border-radius: 20px;
    padding: 5px 12px;
    font-size: 12px;
    margin-top: 15px;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: linear-gradient(
        145deg,
        #17283E,
        #132238
    );
    border: 1px solid #2A405D;
    border-radius: 14px;
    padding: 19px;
    min-height: 115px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.15);
}

div[data-testid="stMetricLabel"] {
    color: #AFC2D8;
    font-size: 13px;
}

div[data-testid="stMetricValue"] {
    color: #FFFFFF;
    font-size: 26px;
    font-weight: 750;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-size: 14px;
    font-weight: 600;
    color: #AFC2D8;
}

/* Dataframes */
div[data-testid="stDataFrame"] {
    border: 1px solid #2A405D;
    border-radius: 12px;
}

/* Buttons */
.stButton > button {
    border-radius: 10px;
    border: 1px solid #2788BA;
    background: #14577C;
    color: white;
    font-weight: 650;
}

.stButton > button:hover {
    background: #1976A5;
    border-color: #54BCEB;
    color: white;
}

/* Download button */
.stDownloadButton > button {
    border-radius: 10px;
    background: #087D70;
    color: white;
    border: 1px solid #23B7A4;
    font-weight: 650;
}

/* Section cards */
.panel {
    background: #142238;
    border: 1px solid #293E59;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 15px;
}

.panel-title {
    color: #EAF3FF;
    font-size: 17px;
    font-weight: 700;
    margin-bottom: 8px;
}

/* Small captions */
.muted {
    color: #91A8C3;
    font-size: 13px;
}

/* Status */
.success-text {
    color: #61DDB8;
    font-weight: 650;
}

.warning-text {
    color: #F5C16C;
    font-weight: 650;
}

/* Dividers */
hr {
    border-color: #2B405B;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# 3. APPLICATION HEADER
# ==========================================================

st.html("""
<div style="
    background: linear-gradient(115deg, #122D4B, #123D59, #102D42);
    padding: 32px;
    border-radius: 18px;
    border: 1px solid #28506C;
    margin-bottom: 25px;
">

    <div style="
        color: #FFFFFF;
        font-size: 38px;
        font-weight: 800;
        letter-spacing: -1px;
    ">
        🛡️ PhishGuard
    </div>

    <div style="
        color: #B9D2E9;
        font-size: 16px;
        margin-top: 10px;
    ">
        Intelligent Phishing Website Detection
        &nbsp; | &nbsp;
        Machine Learning Security Analytics
    </div>

    <div style="
        display: inline-block;
        background: #164D51;
        color: #7CE8D0;
        border: 1px solid #267C77;
        border-radius: 20px;
        padding: 6px 14px;
        font-size: 12px;
        margin-top: 18px;
    ">
        ● SECURITY ANALYTICS DASHBOARD
    </div>

</div>
""")


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
        "Probabilistic classification based on Gaussian assumptions.",

    "Support Vector Machine":
        "Linear classification using an optimal decision boundary."
}


# ==========================================================
# 5. LOAD SAVED MODELS
# ==========================================================

@st.cache_resource(show_spinner=False)
def load_model(model_path):

    path = BASE_DIR / model_path

    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    return joblib.load(path)


# ==========================================================
# 6. MODEL EVALUATION FUNCTION
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
        raise ValueError("Invalid target values detected.")

    if not set(y_test.unique()).issubset({0, 1}):
        raise ValueError(
            "Target labels must be 0 or 1."
        )

    if y_test.nunique() != 2:
        raise ValueError(
            "Both legitimate and phishing classes are required."
        )

    # Retrieve expected feature names
    expected_features = list(model.feature_names_in_)

    # Check missing features
    missing_features = [
        col for col in expected_features
        if col not in X_test.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    # Identify extra columns
    extra_features = [
        col for col in X_test.columns
        if col not in expected_features
    ]

    # Preserve training feature order
    X_test = X_test[expected_features].copy()

    # Validate numeric features
    numeric_data = X_test.apply(
        pd.to_numeric,
        errors="coerce"
    )

    invalid_values = numeric_data.isna() & X_test.notna()

    if invalid_values.any().any():
        raise ValueError(
            "Non-numeric feature values detected."
        )

    X_test = numeric_data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Generate predictions
    y_pred = model.predict(X_test)

    # Calculate continuous scores for AUC
    if hasattr(model, "predict_proba"):

        model_classes = list(model.classes_)

        positive_index = model_classes.index(1)

        y_score = model.predict_proba(X_test)[
            :, positive_index
        ]

    elif hasattr(model, "decision_function"):

        y_score = model.decision_function(X_test)

        if list(model.classes_)[1] != 1:
            y_score = -y_score

    else:
        raise ValueError(
            "Model does not support AUC calculation."
        )

    # Calculate evaluation metrics
    metrics = {

        "Accuracy": accuracy_score(
            y_test,
            y_pred
        ),

        "AUC": roc_auc_score(
            y_test,
            y_score
        ),

        "Precision": precision_score(
            y_test,
            y_pred,
            pos_label=1,
            zero_division=0
        ),

        "Recall": recall_score(
            y_test,
            y_pred,
            pos_label=1,
            zero_division=0
        ),

        "F1 Score": f1_score(
            y_test,
            y_pred,
            pos_label=1,
            zero_division=0
        ),

        "MCC": matthews_corrcoef(
            y_test,
            y_pred
        )
    }

    # Classification report
    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=[
            "Legitimate",
            "Phishing"
        ],
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).T

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    # Prepare prediction results
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
# 7. SIDEBAR CONTROLS
# ==========================================================

st.sidebar.markdown("## ⚙️ Control Center")

# Evaluation mode
st.sidebar.markdown("### Evaluation Mode")

view_mode = st.sidebar.radio(
    "Choose evaluation mode",
    [
        "Single Model",
        "Compare All Models"
    ],
    help="Compare all five saved classifiers or evaluate one model."
)

# Model selection
st.sidebar.markdown("### Model Selection")

selected_model = st.sidebar.selectbox(
    "Choose classifier",
    list(MODEL_FILES.keys())
)

st.sidebar.caption(
    MODEL_DESCRIPTIONS[selected_model]
)

st.sidebar.divider()

# Dataset upload
st.sidebar.markdown("### 📁 Dataset Upload")

uploaded_file = st.sidebar.file_uploader(
    "Upload test dataset",
    type=["csv"],
    help="Upload CSV file"
)

st.sidebar.divider()

st.sidebar.markdown("### About PhishGuard")

st.sidebar.caption(
    "A machine learning application developed "
    "for phishing website classification."
)

st.sidebar.caption(
    "BITS Pilani | M.Tech AIML"
)


# ==========================================================
# 8. WELCOME SCREEN
# ==========================================================

if uploaded_file is None:

    st.markdown("## Welcome to PhishGuard")

    st.markdown(
        "Analyze phishing detection models using "
        "your prepared test dataset."
    )

    st.info(
        "Upload your test CSV from the sidebar "
        "to start the evaluation."
    )

    st.markdown("### Dashboard Capabilities")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="panel">
            <div class="panel-title">📁 Dataset Analysis</div>
            <div class="muted">
                Upload your test dataset and inspect
                its features and class distribution.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="panel">
            <div class="panel-title">🤖 Model Evaluation</div>
            <div class="muted">
                Evaluate five trained machine learning
                classification models.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="panel">
            <div class="panel-title">📊 Performance Analytics</div>
            <div class="muted">
                Explore performance metrics, confusion
                matrices and classification reports.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Supported Classification Models")

    model_col1, model_col2 = st.columns(2)

    with model_col1:
        for name in list(MODEL_FILES.keys())[:3]:
            st.markdown(f"🔹 {name}")

    with model_col2:
        for name in list(MODEL_FILES.keys())[3:]:
            st.markdown(f"🔹 {name}")

    st.divider()

    st.caption(
        "PhishGuard | Machine Learning Security Analytics"
    )

    st.stop()


# ==========================================================
# 9. READ AND VALIDATE UPLOADED DATASET
# ==========================================================

try:

    test_data = pd.read_csv(uploaded_file)

except Exception as error:

    st.error(f"Unable to read uploaded CSV: {error}")
    st.stop()


if test_data.empty:

    st.error("The uploaded dataset is empty.")
    st.stop()


if "is_phishing" not in test_data.columns:

    st.error(
        "The required target column 'is_phishing' is missing."
    )
    st.stop()


# Validate target
target_values = pd.to_numeric(
    test_data["is_phishing"],
    errors="coerce"
)

if target_values.isna().any():

    st.error("Invalid values found in the target column.")
    st.stop()

if not set(target_values.unique()).issubset({0, 1}):

    st.error("Target labels must be either 0 or 1.")
    st.stop()

if target_values.nunique() != 2:

    st.error(
        "The test dataset must contain both classes "
        "to calculate AUC."
    )
    st.stop()

test_data["is_phishing"] = target_values.astype(int)


# ==========================================================
# 10. DATASET OVERVIEW
# ==========================================================

st.markdown("## 📁 Dataset Overview")

total_records = len(test_data)

total_features = len(test_data.columns) - 1

phishing_count = int(
    (test_data["is_phishing"] == 1).sum()
)

legitimate_count = int(
    (test_data["is_phishing"] == 0).sum()
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Total Records",
    f"{total_records:,}"
)

c2.metric(
    "Input Features",
    total_features
)

c3.metric(
    "Phishing Records",
    f"{phishing_count:,}"
)

c4.metric(
    "Legitimate Records",
    f"{legitimate_count:,}"
)

st.caption(
    f"File: {uploaded_file.name}  |  "
    f"Size: {uploaded_file.size / 1024:.2f} KB"
)

with st.expander("🔍 Preview Dataset"):

    st.dataframe(
        test_data.head(10),
        use_container_width=True
    )

    st.caption(
        f"Dataset dimensions: {test_data.shape}"
    )


# ==========================================================
# 11. MODEL EVALUATION
# ==========================================================

if view_mode == "Single Model":

    models_to_evaluate = [selected_model]

else:

    models_to_evaluate = list(MODEL_FILES.keys())


all_results = {}
failed_models = {}

with st.spinner(
    "Analyzing dataset and evaluating saved models..."
):

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

    st.error("Some models could not be evaluated.")

    for name, error in failed_models.items():
        st.write(f"**{name}:** {error}")

    if not all_results:
        st.stop()


# Determine detailed model
detail_model = (
    selected_model
    if selected_model in all_results
    else next(iter(all_results))
)

selected_result = all_results[detail_model]


# ==========================================================
# 12. DASHBOARD NAVIGATION
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
# 13. OVERVIEW
# ==========================================================

with tab_overview:

    st.markdown("### Model Performance Overview")

    st.markdown(
        f"Detailed evaluation for **{detail_model}**"
    )

    metrics = selected_result["metrics"]

    # Row 1
    m1, m2, m3 = st.columns(3)

    m1.metric(
        "Accuracy",
        f"{metrics['Accuracy']:.2%}"
    )

    m2.metric(
        "ROC AUC",
        f"{metrics['AUC']:.2%}"
    )

    m3.metric(
        "Precision",
        f"{metrics['Precision']:.2%}"
    )

    # Row 2
    m4, m5, m6 = st.columns(3)

    m4.metric(
        "Recall",
        f"{metrics['Recall']:.2%}"
    )

    m5.metric(
        "F1 Score",
        f"{metrics['F1 Score']:.2%}"
    )

    m6.metric(
        "MCC",
        f"{metrics['MCC']:.4f}"
    )

    st.divider()

    st.markdown("### Prediction Summary")

    predictions = selected_result["predictions"]

    correct = int(
        (predictions["prediction_result"] == "Correct").sum()
    )

    incorrect = total_records - correct

    p1, p2, p3 = st.columns(3)

    p1.metric(
        "Correct Predictions",
        f"{correct:,}"
    )

    p2.metric(
        "Incorrect Predictions",
        f"{incorrect:,}"
    )

    p3.metric(
        "Prediction Accuracy",
        f"{correct / total_records:.2%}"
    )

    st.markdown("### Dataset Class Distribution")

    class_data = pd.DataFrame({
        "Website Type": [
            "Legitimate",
            "Phishing"
        ],
        "Records": [
            legitimate_count,
            phishing_count
        ]
    })

    st.bar_chart(
        class_data.set_index("Website Type"),
        color="#20BFA8"
    )


# ==========================================================
# 14. MODEL COMPARISON
# ==========================================================

with tab_compare:

    st.markdown("### 📊 Model Performance Comparison")

    # Select evaluation metrics
    selected_metrics = st.multiselect(
        "Choose evaluation metrics",
        options=list(comparison_df.columns),
        default=[
            "Accuracy",
            "AUC",
            "Precision",
            "Recall",
            "F1 Score"
        ]
    )

    if selected_metrics:

        # Define metric colors
        metric_colors = {
            "Accuracy": "#20BFA8",
            "AUC": "#3B82F6",
            "Precision": "#A78BFA",
            "Recall": "#F59E0B",
            "F1 Score": "#F472B6",
            "MCC": "#22D3EE"
        }

        # Create grouped vertical bar chart
        fig, ax = plt.subplots(figsize=(13, 6))

        models = comparison_df.index.tolist()

        x = np.arange(len(models))

        total_width = 0.8
        bar_width = total_width / len(selected_metrics)

        # Plot each selected metric
        for i, metric in enumerate(selected_metrics):

            offset = (
                i - (len(selected_metrics) - 1) / 2
            ) * bar_width

            positions = x + offset

            ax.bar(
                positions,
                comparison_df[metric].values,
                width=bar_width * 0.9,
                label=metric,
                color=metric_colors[metric],
                edgecolor="white",
                linewidth=0.3
            )

        # Chart formatting
        ax.set_title(
            "Machine Learning Model Performance Comparison",
            fontsize=15,
            fontweight="bold",
            pad=18
        )

        ax.set_xlabel(
            "Classification Models",
            fontsize=11,
            fontweight="bold",
            labelpad=12
        )

        ax.set_ylabel(
            "Performance Score",
            fontsize=11,
            fontweight="bold",
            labelpad=12
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            models,
            rotation=15,
            ha="right"
        )

        # MCC can range from -1 to 1
        lower_limit = -1 if "MCC" in selected_metrics else 0
        ax.set_ylim(lower_limit, 1.05)

        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.25
        )

        ax.set_axisbelow(True)

        ax.legend(
            title="Evaluation Metrics",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.20),
            ncol=3,
            frameon=False
        )

        fig.tight_layout()

        # Display chart
        st.pyplot(fig, use_container_width=True)

        plt.close(fig)

        st.caption(
            "Compare the performance of all five classifiers. "
            "Higher scores indicate better performance, "
            "while MCC ranges from -1 to +1."
        )


# ==========================================================
# 15. CONFUSION MATRIX
# ==========================================================

with tab_confusion:

    st.markdown("### Confusion Matrix")

    st.caption(
        f"Selected model: {detail_model}"
    )

    cm = selected_result["confusion_matrix"]

    fig, ax = plt.subplots(figsize=(7, 5))

    fig.patch.set_facecolor("#142238")
    ax.set_facecolor("#142238")

    image = ax.imshow(
        cm,
        cmap="GnBu"
    )

    ax.set_xticks(
        [0, 1],
        labels=["Legitimate", "Phishing"]
    )

    ax.set_yticks(
        [0, 1],
        labels=["Legitimate", "Phishing"]
    )

    ax.set_xlabel(
        "Predicted Class",
        color="white"
    )

    ax.set_ylabel(
        "Actual Class",
        color="white"
    )

    ax.tick_params(colors="white")

    ax.set_title(
        "Classification Outcomes",
        color="white",
        pad=15
    )

    threshold = cm.max() / 2

    for i in range(2):

        for j in range(2):

            ax.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
                color=(
                    "white"
                    if cm[i, j] > threshold
                    else "#102438"
                )
            )

    colorbar = fig.colorbar(
        image,
        ax=ax
    )

    colorbar.ax.yaxis.set_tick_params(
        color="white"
    )

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=False
    )

    plt.close(fig)

    tn, fp, fn, tp = cm.ravel()

    st.markdown("### Classification Outcomes")

    a, b = st.columns(2)

    a.metric("True Negatives", f"{tn:,}")
    b.metric("False Positives", f"{fp:,}")

    c, d = st.columns(2)

    c.metric("False Negatives", f"{fn:,}")
    d.metric("True Positives", f"{tp:,}")


# ==========================================================
# 16. CLASSIFICATION REPORT
# ==========================================================

with tab_report:

    st.markdown("### Detailed Classification Report")

    report_df = selected_result["report"].copy()

    st.dataframe(
        report_df.style.format({
            "precision": "{:.4f}",
            "recall": "{:.4f}",
            "f1-score": "{:.4f}",
            "support": "{:.0f}"
        }),
        use_container_width=True
    )

    st.caption(
        "The report includes precision, recall, F1 Score "
        "and support for both website classes."
    )


# ==========================================================
# 17. PREDICTION RESULTS
# ==========================================================

with tab_predictions:

    st.markdown("### Website Classification Results")

    prediction_df = selected_result["predictions"]

    filter_option = st.selectbox(
        "Filter predictions",
        [
            "All Records",
            "Correct Predictions",
            "Incorrect Predictions",
            "Predicted Phishing",
            "Predicted Legitimate"
        ]
    )

    if filter_option == "Correct Predictions":

        display_df = prediction_df[
            prediction_df["prediction_result"] == "Correct"
        ]

    elif filter_option == "Incorrect Predictions":

        display_df = prediction_df[
            prediction_df["prediction_result"] == "Incorrect"
        ]

    elif filter_option == "Predicted Phishing":

        display_df = prediction_df[
            prediction_df["predicted_class"] == 1
        ]

    elif filter_option == "Predicted Legitimate":

        display_df = prediction_df[
            prediction_df["predicted_class"] == 0
        ]

    else:

        display_df = prediction_df

    st.caption(
        f"Displaying {len(display_df):,} of {total_records:,} records."
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        height=450
    )

    st.markdown("### Export Results")

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

footer_left, footer_right = st.columns(2)

with footer_left:

    st.markdown(
        "**🛡️ PhishGuard**  \n"
        "Intelligent Phishing Website Detection"
    )

with footer_right:

    st.markdown(
        "<div style='text-align:right;color:#91A8C3;'>"
        "BITS Pilani | M.Tech AIML<br>"
        "Machine Learning Classification Project"
        "</div>",
        unsafe_allow_html=True
    )

st.caption(
    "Evaluation is performed using saved models. "
    "No model retraining is performed on uploaded test data."
)
