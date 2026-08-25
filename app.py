import streamlit as st
import pandas as pd
import joblib

# 1. Page Configuration
st.set_page_config(
    page_title="InsuranceAI | Health Cost Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 950px;
    }

    /* Hero Banner Header - Dark Slate */
    .hero-banner {
        background-color: #1E293B;
        padding: 25px 30px;
        border-radius: 16px;
        color: white;
        margin-bottom: 25px;
    }
    .hero-banner h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        color: #FFFFFF !important;
    }
    .hero-banner p {
        margin-top: 6px;
        margin-bottom: 0;
        font-size: 0.95rem;
        color: #94A3B8;
    }

    /* Form Card Container */
    .stForm {
        background: #FFFFFF;
        border-radius: 16px;
        padding: 25px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    /* Calculate Button: Light Grey by Default -> Dark Slate Theme on Hover */
    div.stButton > button {
        background-color: #F1F5F9 !important;
        color: #334155 !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        border: 1px solid #CBD5E1 !important;
        transition: all 0.3s ease !important;
    }
    div.stButton > button:hover {
        background-color: #1E293B !important;
        color: #FFFFFF !important;
        border-color: #1E293B !important;
        box-shadow: 0 4px 12px rgba(30, 41, 59, 0.25) !important;
    }

    /* Estimated Price Result Card */
    .output-card {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border-radius: 16px;
        padding: 28px;
        color: white;
        text-align: center;
        margin-top: 25px;
        border: 2px solid #334155;
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.3);
    }
    .output-card h4 {
        margin: 0;
        color: #94A3B8 !important;
        font-size: 0.95rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .output-card h1 {
        margin: 10px 0 0 0;
        color: #FFFFFF !important;
        font-size: 3.2rem;
        font-weight: 700;
    }

    /* Model Performance Card (Matched Theme Colors) */
    .perf-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 24px;
        margin-top: 25px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    }
    .perf-title {
        color: #1E293B;
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 18px;
    }
    .perf-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 0;
        border-bottom: 1px solid #F1F5F9;
        font-size: 0.95rem;
    }
    .perf-label {
        color: #64748B;
        font-weight: 500;
    }
    .perf-value {
        color: #0F172A;
        font-weight: 700;
    }
    .progress-bar-bg {
        background-color: #F1F5F9;
        border-radius: 8px;
        height: 10px;
        width: 100%;
        margin-top: 15px;
        overflow: hidden;
    }
    .progress-bar-fill {
        background-color: #1E293B;
        height: 100%;
        border-radius: 8px;
    }

    /* Sidebar Background */
    section[data-testid="stSidebar"] {
        background-color: #F8FAFC;
        border-right: 1px solid #E2E8F0;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Model & Asset Loader
@st.cache_resource
def load_model_assets():
    model = joblib.load('insurance_rf_model.pkl')
    columns = joblib.load('model_columns.pkl')
    return model, columns

try:
    model, model_columns = load_model_assets()
    model_loaded = True
except Exception as e:
    st.error(f"⚠️ Could not load trained model files (`insurance_rf_model.pkl` / `model_columns.pkl`): {e}")
    model_loaded = False

# 4. Sidebar Content
with st.sidebar:
    st.markdown("### 📊 Project Context")
    st.write("This application leverages a Random Forest Regression model trained on demographic and lifestyle health metrics to estimate annual medical coverage charges.")
    st.markdown("---")
    st.markdown("#### 💡 Top Influencing Factors")
    st.markdown("- **1. Smoker Status**")
    st.markdown("- **2. Body Mass Index (BMI)**")
    st.markdown("- **3. Age**")

# 5. Hero Banner
st.markdown("""
    <div class="hero-banner">
        <h1>💳 Health Insurance Cost Estimator</h1>
        <p>Provide your personal health parameters below to compute an estimated annual charge.</p>
    </div>
""", unsafe_allow_html=True)

# 6. Form Inputs
if model_loaded:
    with st.form("prediction_form"):
        st.markdown('### 📋 Personal & Demographic Attributes')
        
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            age = st.number_input("Age (Years)", min_value=18, max_value=100, value=30, step=1)
            sex = st.selectbox("Biological Sex", ["male", "female"])
            children = st.number_input("Number of Dependent Children", min_value=0, max_value=10, value=0, step=1)

        with col2:
            bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
            smoker = st.selectbox("Smoking Status", ["no", "yes"])
            region = st.selectbox("US Region of Residence", ["northeast", "northwest", "southeast", "southwest"])

        st.markdown("---")
        submit_button = st.form_submit_button("✨ Calculate Estimated Charge", use_container_width=True)

    # 7. Prediction Logic
    if submit_button:
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
        
        st.markdown(
            f"""
            <div class="output-card">
                <h4>Estimated Annual Insurance Charge</h4>
                <h1>${prediction:,.2f}</h1>
            </div>
            """, 
            unsafe_allow_html=True
        )

# 8. Model Performance Card (Matching Overall Theme)
r2_score_val = 90.02
r2_score_str = f"{r2_score_val:.2f}%"

st.markdown(
    f"""
    <div class="perf-card">
        <div class="perf-title">Model Performance</div>
        <div class="perf-row">
            <span class="perf-label">Model</span>
            <span class="perf-value">Random Forest</span>
        </div>
        <div class="perf-row">
            <span class="perf-label">RMSE</span>
            <span class="perf-value">$4,282.68</span>
        </div>
        <div class="perf-row" style="border-bottom: none;">
            <span class="perf-label">R² Score</span>
            <span class="perf-value">{r2_score_str}</span>
        </div>
        <div class="progress-bar-bg">
            <div class="progress-bar-fill" style="width: {r2_score_val}%;"></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
