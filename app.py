import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -------------------- CONFIG --------------------
st.set_page_config(
    page_title="Iris Prediction App",
    page_icon="🌸",
    layout="wide"
)

# -------------------- LOAD MODEL (CACHED) --------------------
@st.cache_resource
def load_model():
    return joblib.load("random_forest_iris_model.pkl")

model = load_model()

target_names = ["Setosa", "Versicolor", "Virginica"]

# -------------------- TITLE --------------------
st.title("🌸 Iris Flower Prediction App")
st.write("A Random Forest powered web app with advanced features")

# -------------------- SIDEBAR --------------------
st.sidebar.header("🌼 Input Features")

# Example presets
example = st.sidebar.selectbox(
    "Choose Example",
    ["Custom", "Setosa Example", "Versicolor Example", "Virginica Example"]
)

if example == "Setosa Example":
    sepal_length, sepal_width, petal_length, petal_width = 5.1, 3.5, 1.4, 0.2
elif example == "Versicolor Example":
    sepal_length, sepal_width, petal_length, petal_width = 6.0, 2.9, 4.5, 1.5
elif example == "Virginica Example":
    sepal_length, sepal_width, petal_length, petal_width = 6.5, 3.0, 5.5, 2.0
else:
    sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
    sepal_width  = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
    petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
    petal_width  = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# -------------------- PREDICTION --------------------
features = np.array([
    sepal_length, sepal_width, petal_length, petal_width
]).reshape(1, -1)

if st.button("🔮 Predict"):
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    st.success(f"🌺 Prediction: **{target_names[prediction]}**")
    st.write(f"Confidence: **{round(max(probabilities) * 100, 2)}%**")

    # Probability bar chart
    st.subheader("📊 Prediction Probabilities")
    prob_df = pd.DataFrame({
        "Class": target_names,
        "Probability": probabilities
    })
    st.bar_chart(prob_df.set_index("Class"))

# -------------------- FEATURE IMPORTANCE --------------------
st.subheader("🧠 Feature Importance")
importance = model.feature_importances_
features_names = [
    "Sepal Length", "Sepal Width", "Petal Length", "Petal Width"
]

fig, ax = plt.subplots()
ax.barh(features_names, importance)
ax.set_xlabel("Importance")
ax.set_title("Random Forest Feature Importance")
st.pyplot(fig)

# -------------------- BATCH CSV PREDICTION --------------------
st.subheader("📂 Batch Prediction (Upload CSV)")
uploaded_file = st.file_uploader(
    "Upload CSV with 4 columns: sepal_length, sepal_width, petal_length, petal_width",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    preds = model.predict(df)
    df["Prediction"] = [target_names[i] for i in preds]
    st.dataframe(df)

    st.download_button(
        "⬇️ Download Results",
        df.to_csv(index=False),
        "iris_predictions.csv",
        "text/csv"
    )

# -------------------- FOOTER --------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit & Random Forest")
