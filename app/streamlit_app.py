"""Streamlit UI for iris prediction."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
from src.predict import predict

st.set_page_config(page_title="Iris Classifier", page_icon="🌸")

st.title("🌸 Iris Flower Classifier")
st.write("Enter flower measurements to predict species.")

col1, col2 = st.columns(2)
with col1:
    sepal_l = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1, 0.1)
    sepal_w = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5, 0.1)
with col2:
    petal_l = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4, 0.1)
    petal_w = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2, 0.1)

model_name = st.selectbox("Model", ["svm", "random_forest", "knn", "logistic"])

if st.button("🔮 Predict"):
    label, proba = predict([sepal_l, sepal_w, petal_l, petal_w],
                           model_name=model_name)
    st.success(f"Predicted Species: **{label}**")
    if proba is not None:
        st.bar_chart({"probability": proba})