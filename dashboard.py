# ==============================================================
# STREAMLIT DASHBOARD – Mental Health Predictor
# Dataset: mental_health_social_media_datasetCA.xlsx
# Task: Predict mental_state and suggest general precautions
# ==============================================================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------
# 1. Load and Preprocess Data
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("mental_health_social_media_datasetCA.xlsx")

    # Convert date to datetime and derive features
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek

    # Drop non-useful column
    if "person_name" in df.columns:
        df = df.drop(columns=["person_name"])
    df = df.drop(columns=["date"])

    # Handle missing numeric values
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # Label Encoding for categorical features
    le_gender = LabelEncoder()
    le_platform = LabelEncoder()
    le_mental = LabelEncoder()

    df["gender_enc"] = le_gender.fit_transform(df["gender"])
    df["platform_enc"] = le_platform.fit_transform(df["platform"])
    df["mental_state_enc"] = le_mental.fit_transform(df["mental_state"])

    return df, le_gender, le_platform, le_mental

@st.cache_resource
def train_model(df):
    # Features and target
    feature_cols = [
        "age",
        "gender_enc",
        "platform_enc",
        "daily_screen_time_min",
        "social_media_time_min",
        "negative_interactions_count",
        "positive_interactions_count",
        "sleep_hours",
        "physical_activity_min",
        "anxiety_level",
        "stress_level",
        "mood_level",
        "month",
        "day_of_week",
    ]
    X = df[feature_cols]
    y = df["mental_state_enc"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate once
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return model, feature_cols, acc, (y_test, y_pred)

# -------------------------
# 2. Streamlit UI
# -------------------------
st.set_page_config(page_title="Mental Health Predictor", layout="centered")

st.title("🧠 Mental Health & Social Media – Predictive Dashboard")
st.write(
    """
This dashboard uses a machine learning model trained on a mental health and social media usage dataset  
to **predict mental_state** based on your inputs and suggest some **general wellness precautions**.

> ⚠️ This is **not** a medical tool. It is for **educational purposes only**.
"""
)

# Load data and model
df, le_gender, le_platform, le_mental = load_data()
model, feature_cols, model_acc, (y_test_eval, y_pred_eval) = train_model(df)

st.sidebar.header("📊 Model Info")
st.sidebar.write(f"Model Type: RandomForestClassifier")
st.sidebar.write(f"Test Accuracy: **{model_acc:.3f}**")

# -------------------------
# 3. User Input Section
# -------------------------
st.header("📥 Enter Your Details")

# Get unique values for dropdowns from original columns
gender_options = sorted(df["gender"].unique())
platform_options = sorted(df["platform"].unique())

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", min_value=13, max_value=70, value=25, step=1)
    gender = st.selectbox("Gender", gender_options)
    platform = st.selectbox("Primary Social Media Platform", platform_options)
    daily_screen_time_min = st.slider(
        "Total Daily Screen Time (minutes)", min_value=30, max_value=900, value=300, step=15
    )
    social_media_time_min = st.slider(
        "Daily Social Media Time (minutes)", min_value=10, max_value=600, value=180, step=10
    )

with col2:
    negative_interactions_count = st.slider(
        "Negative Interactions per Day (approx.)", min_value=0, max_value=20, value=1, step=1
    )
    positive_interactions_count = st.slider(
        "Positive Interactions per Day (approx.)", min_value=0, max_value=20, value=3, step=1
    )
    sleep_hours = st.slider(
        "Average Sleep Hours per Night", min_value=3.0, max_value=12.0, value=7.0, step=0.5
    )
    physical_activity_min = st.slider(
        "Physical Activity per Day (minutes)", min_value=0, max_value=180, value=30, step=5
    )
    anxiety_level = st.slider(
        "Self-rated Anxiety Level (0–10)", min_value=0, max_value=10, value=3, step=1
    )
    stress_level = st.slider(
        "Self-rated Stress Level (0–10)", min_value=0, max_value=10, value=4, step=1
    )
    mood_level = st.slider(
        "Self-rated Mood Level (0–10, higher = better)", min_value=0, max_value=10, value=6, step=1
    )

# Use today's date for month and day_of_week
today = datetime.today()
month = today.month
day_of_week = today.weekday()

# Encode categorical inputs
gender_enc = le_gender.transform([gender])[0]
platform_enc = le_platform.transform([platform])[0]

# Create input DataFrame
user_data = pd.DataFrame(
    [[
        age,
        gender_enc,
        platform_enc,
        daily_screen_time_min,
        social_media_time_min,
        negative_interactions_count,
        positive_interactions_count,
        sleep_hours,
        physical_activity_min,
        anxiety_level,
        stress_level,
        mood_level,
        month,
        day_of_week,
    ]],
    columns=feature_cols,
)

st.subheader("📋 Your Input Summary")
st.write(user_data)

# -------------------------
# 4. Predict Mental State
# -------------------------
if st.button("🔍 Predict Mental State"):
    pred_enc = model.predict(user_data)[0]
    pred_label = le_mental.inverse_transform([pred_enc])[0]

    st.subheader("🧾 Prediction Result")
    st.write(f"**Predicted Mental State:** `{pred_label}`")

    # -------------------------
    # 5. Suggest Precautions (General)
    # -------------------------
    st.subheader("🛡 Suggested General Precautions (Non-medical)")
    
    # Default suggestions
    suggestions = []

    # Very general rule-based mapping
    if "stress" in pred_label.lower():
        suggestions.append("Try to reduce daily screen time, especially close to bedtime.")
        suggestions.append("Maintain a regular sleep schedule with at least 7–8 hours of sleep.")
        suggestions.append("Include daily physical activity like walking, stretching, or light exercise.")
        suggestions.append("Limit exposure to negative or toxic interactions on social media.")
        suggestions.append("Talk to a trusted friend, family member, or counselor if you feel overwhelmed.")
    else:
        suggestions.append("Maintain a healthy balance between online and offline life.")
        suggestions.append("Continue your current sleep and physical activity habits.")
        suggestions.append("Use social media mindfully and avoid unnecessary negative content.")
        suggestions.append("Practice relaxation techniques like deep breathing or short breaks from screens.")

    # Extra checks based on numeric inputs
    if daily_screen_time_min > 360:
        suggestions.append("Your total screen time is quite high. Consider setting screen time limits or taking regular breaks.")
    if sleep_hours < 6:
        suggestions.append("Your sleep duration seems low. Try to improve sleep hygiene and aim for 7–8 hours.")
    if physical_activity_min < 20:
        suggestions.append("Try to include at least 20–30 minutes of physical movement in your day.")
    if anxiety_level >= 7 or stress_level >= 7:
        suggestions.append(
            "If you are consistently feeling very stressed or anxious, please consider talking to a mental health professional."
        )

    for i, s in enumerate(suggestions, start=1):
        st.write(f"**{i}. {s}**")

    st.info(
        "⚠️ This tool is for learning and awareness only. "
        "It does not replace professional mental health advice or diagnosis."
    )


# -------------------------
# 6. Model Evaluation (Optional Section)
# -------------------------
with st.expander("📊 View Model Evaluation on Test Set"):
    X = df[feature_cols]
    y = df["mental_state_enc"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model_eval = RandomForestClassifier(n_estimators=300, random_state=42)
    model_eval.fit(X_train, y_train)
    y_pred_eval = model_eval.predict(X_test)

    st.write("**Accuracy:**", round(accuracy_score(y_test, y_pred_eval), 3))
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred_eval))
    st.text("Confusion Matrix:")
    st.write(pd.DataFrame(confusion_matrix(y_test, y_pred_eval)))
