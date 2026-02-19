import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
from datetime import datetime

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Lifestyle & Energy Predictor",
    layout="wide"
)

st.title("🧠 AI Lifestyle & Energy Predictor")
st.write("Predict your daily energy using AI based on your habits.")

# ---------------- LOAD MODEL ----------------
@st.cache_data
def load_data_and_model():
    data = {
        'sleep_hours': [7, 5, 8, 6, 4, 9, 7],
        'steps': [8000, 3000, 10000, 6000, 2000, 12000, 7500],
        'workout_intensity': [2, 0, 3, 1, 0, 3, 2],
        'junk_food_level': [1, 3, 1, 2, 3, 1, 2],
        'screen_time': [4, 7, 3, 5, 8, 2, 4],
        'stress_level': [3, 8, 2, 5, 9, 2, 4],
        'energy_score': [80, 40, 90, 70, 30, 95, 75]
    }

    df = pd.DataFrame(data)
    X = df.drop('energy_score', axis=1)
    y = df['energy_score']

    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)

    return df, X, model

train_df, X, model = load_data_and_model()

# ---------------- INPUT SECTION ----------------
st.subheader("Enter Today's Lifestyle Details")

col1, col2, col3 = st.columns(3)

with col1:
    sleep_hours = st.slider("Sleep hours", 3.0, 10.0, 7.0, 0.5)
    steps = st.number_input("Steps walked", 0, 30000, 8000, 500)

with col2:
    workout_intensity = st.selectbox("Workout intensity", ['None', 'Light', 'Medium', 'Heavy'])
    junk_food_level = st.selectbox("Junk food intake", ['Low', 'Medium', 'High'])

with col3:
    screen_time = st.slider("Screen time (hours)", 1.0, 12.0, 5.0, 0.5)
    stress_level = st.slider("Stress level (1-10)", 1, 10, 4)

# Mapping categories
workout_map = {'None': 0, 'Light': 1, 'Medium': 2, 'Heavy': 3}
junk_map = {'Low': 1, 'Medium': 2, 'High': 3}

input_row = pd.DataFrame({
    'sleep_hours': [sleep_hours],
    'steps': [steps],
    'workout_intensity': [workout_map[workout_intensity]],
    'junk_food_level': [junk_map[junk_food_level]],
    'screen_time': [screen_time],
    'stress_level': [stress_level]
})

# ---------------- SESSION STATE INIT ----------------
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame()

# ---------------- PREDICT ----------------
if st.button("Predict Energy"):

    pred = model.predict(input_row)[0]
    pred_int = int(pred)

    st.subheader(f"🔋 Predicted Energy Score: {pred_int}/100")

    # Suggestions
    suggestions = []

    if sleep_hours < 7:
        suggestions.append("Increase sleep to 7–8 hours.")
    if steps < 6000:
        suggestions.append("Try reaching at least 6000–8000 steps.")
    if junk_food_level == "High":
        suggestions.append("Reduce junk food intake.")
    if screen_time > 7:
        suggestions.append("Reduce screen exposure at night.")
    if stress_level > 6:
        suggestions.append("Practice stress reduction techniques.")

    st.subheader("💡 Suggestions")

    if suggestions:
        for s in suggestions:
            st.write("• " + s)
    else:
        st.success("Your lifestyle looks balanced. Maintain consistency!")

    # Feature Importance
    feat_imp = model.feature_importances_
    imp_df = pd.DataFrame({
        'feature': X.columns,
        'importance': feat_imp
    }).sort_values('importance', ascending=False)

    st.subheader("📊 Feature Importance (Model View)")
    fig = px.bar(imp_df, x='feature', y='importance')
    st.plotly_chart(fig, use_container_width=True)

    # Save Option
    if st.button("Save Today"):
        new_entry = input_row.copy()
        new_entry["energy_score"] = pred_int
        new_entry["date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        st.session_state.history = pd.concat(
            [st.session_state.history, new_entry],
            ignore_index=True
        )

        st.success("Today's data saved!")

# ---------------- HISTORY SECTION ----------------
st.subheader("📅 History Log")

if not st.session_state.history.empty:

    st.dataframe(st.session_state.history)

    trend_fig = px.line(
        st.session_state.history,
        x="date",
        y="energy_score",
        markers=True,
        title="Energy Score Trend"
    )

    st.plotly_chart(trend_fig, use_container_width=True)

else:
    st.info("No saved history yet. Predict and save your first entry.")

# ---------------- TRAINING DATA ----------------
st.subheader("📚 Sample Training Data (Demo)")
st.dataframe(train_df)
