import streamlit as st
import numpy as np
import pickle

model = pickle.load(open("diabetes_model.pkl", "rb"))

st.title("Diabetes Prediction App")

preg = st.number_input("Pregnancies", 0, 20, 1)
glu = st.number_input("Glucose", 0, 200, 100)
bp = st.number_input("Blood Pressure", 0, 150, 70)
skin = st.number_input("Skin Thickness", 0, 100, 20)
ins = st.number_input("Insulin", 0, 900, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 0, 120, 30)

if st.button("Predict"):
    features = np.array([[preg, glu, bp, skin, ins, bmi, dpf, age]])
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    if pred == 1:
        st.error(f"Diabetes detected (Risk: {prob:.2f})")
    else:
        st.success(f"No Diabetes (Risk: {prob:.2f})")