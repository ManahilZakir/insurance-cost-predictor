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

# 2. Complete Custom CSS (Monochrome & Slate Modern Theme)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1000px;
    }

    /* Hero Banner Header - Dark Slate */
    .hero-banner {
        background-color: #1E293B;
        padding: 30px;
        border-radius: 16px;
        color: white;
        margin-bottom: 25px;
        border: 1px solid #334155;
    }
    .hero-banner h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        color: #FFFFFF !important;
    }
    .hero-banner p {
        margin-top: 8px;
        margin-bottom: 0;
        font-size: 0.95rem;
        color: #94A3B8;
    }

    /* Form Container Card */
    .stForm {
        background: #FFFFFF;
        border-radius: 16px;
        padding: 25px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    /* Form Section Headers */
    .section-header {
        color: #0F172A;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 15px;
    }

    /* Output Card - Midnight Dark */
    .output-card {
        background-color: #0F172A;
        border-radius: 16px;
        padding: 30px;
        color: white;
        text-align: center;
        margin-top: 25px;
        border: 1px solid #334155;
    }
    .output-card h4 {
        margin: 0;
        color: #94A3B8 !important;
        font-size: 0.9rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .output-card h1 {
        margin: 10px 0 0 0;
        color: #FFFFFF !important;
        font-size: 3rem;
        font-weight: 700;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #F8FAFC;
        border-right: 1px solid #E2E8F0;
    }

    /* Dark Matte Action Button */
    div.stButton > button {
        background-color: #0F172A !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        border: none !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button:hover {
        background-color: #334155 !important;
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

# 4. Sidebar Panel
with st.sidebar:
    st.markdown("### 📊 Model Context")
    st.info("**Model Type:** Random Forest Regressor")
    st.write("Calculates estimated health insurance expenses based on demographic details.")
    
    st.markdown("---")
    st.markdown("#### 💡 Primary Drivers")
    st.markdown("- **Smoker Status**")
    st.markdown("- **BMI Value**")
    st.markdown("- **Age Group**")

# 5. Main Hero Banner UI
st.markdown("""
    <div class="hero-banner">
        <h1>💳 Health Insurance Cost Estimator</h1>
        <p>Provide your personal health parameters below to compute an estimated annual charge.</p>
    </div>
""", unsafe_allow_html=True)

# 6. User Inputs Form Layout
if model_loaded:
    with st.form("prediction_form"):
        st.markdown('<div class="section-header">📋 Personal & Demographic Attributes</div>', unsafe_allow_html=True)
        
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
        submit_button = st.form_submit_button("Calculate Estimated Premium", use_container_width=True)

    # 7. Prediction Logic Execution
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
