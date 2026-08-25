import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="InsuranceAI Predictor",
    page_icon="💳",
    layout="centered"
)

# Custom Styling (Warm Brown Palette based on notebook theme)
st.markdown("""
    <style>
    .main {
        background-color: #FDFBF7;
    }
    .stButton>button {
        background-color: #3E1F14;
        color: #FFFFFF;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #A67564;
        color: #FFFFFF;
    }
    .prediction-card {
        background-color: #D9B18E;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #3E1F14;
        text-align: center;
        color: #3E1F14;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained model and expected feature columns
@st.cache_resource
def load_assets():
    model = joblib.load('insurance_rf_model.pkl')
    columns = joblib.load('model_columns.pkl')
    return model, columns

try:
    model, model_columns = load_assets()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model files: {e}")
    model_loaded = False

# Application Title
st.title("💳 InsuranceAI Cost Predictor")
st.write("Enter the details below to estimate your medical insurance charges.")

st.divider()

# Input Form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
        bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
        sex = st.selectbox("Sex", ["male", "female"])
        
    with col2:
        children = st.number_input("Number of Children", min_value=0, max_value=10, value=0, step=1)
        smoker = st.selectbox("Smoker", ["no", "yes"])
        region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

    submit_button = st.form_submit_button("Calculate Estimated Charges")

# Prediction logic
if submit_button and model_loaded:
    input_data = {col: 0 for col in model_columns}
    
    input_data['age'] = age
    input_data['bmi'] = bmi
    input_data['children'] = children
    
    if 'sex' in input_data:
        input_data['sex'] = 1 if sex == 'male' else 0
    if 'smoker' in input_data:
        input_data['smoker'] = 1 if smoker == 'yes' else 0

    region_col = f"region_{region}"
    if region_col in input_data:
        input_data[region_col] = 1

    df_input = pd.DataFrame([input_data])
    prediction = model.predict(df_input)[0]
    
    st.divider()
    st.markdown(
        f"""
        <div class="prediction-card">
            <h3 style="margin: 0; color: #3E1F14;">Estimated Annual Insurance Charge</h3>
            <h1 style="margin: 10px 0; color: #3E1F14;">${prediction:,.2f}</h1>
        </div>
        """, 
        unsafe_allow_html=True
    )
