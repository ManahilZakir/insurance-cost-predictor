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

# 2. Complete Custom CSS (Stitch / Material Design Aesthetic)
st.markdown("""
    <style>
    /* Google Material Font Setup */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Top Padding Adjustment */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }

    /* Hero Banner Header */
    .hero-banner {
        background: linear-gradient(135deg, #1A73E8 0%, #0D47A1 100%);
        padding: 30px;
        border-radius: 20px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px rgba(26, 115, 232, 0.2);
    }
    .hero-banner h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff !important;
    }
    .hero-banner p {
        margin-top: 8px;
        margin-bottom: 0;
        font-size: 1rem;
        color: #E8F0FE;
    }

    /* Form Container Card */
    .stForm {
        background: #FFFFFF;
        border-radius: 20px;
        padding: 25px;
        border: 1px solid #E0E0E0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.03);
    }

    /* Form Section Headers */
    .section-header {
        color: #1A73E8;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Modern Output Hero Card */
    .output-card {
        background: linear-gradient(135deg, #0F9D58 0%, #0B8043 100%);
        border-radius: 20px;
        padding: 30px;
        color: white;
        text-align: center;
        margin-top: 25px;
        box-shadow: 0 12px 30px rgba(15, 157, 88, 0.25);
        animation: fadeIn 0.5s ease-in-out;
    }
    .output-card h4 {
        margin: 0;
        color: #E6F4EA !important;
        font-size: 1rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .output-card h1 {
        margin: 10px 0 0 0;
        color: #FFFFFF !important;
        font-size: 3.2rem;
        font-weight: 700;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #F8F9FA;
        border-right: 1px solid #E8EAED;
    }

    /* Primary Button Custom Styling */
    div.stButton > button {
        background-color: #1A73E8 !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        border: none !important;
        transition: all 0.3s ease !important;
    }
    div.stButton > button:hover {
        background-color: #1557B0 !important;
        box-shadow: 0 4px 12px rgba(26, 115, 232, 0.3) !important;
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

# 4. Sidebar Details & Model Context Panel
with st.sidebar:
    st.markdown("### 📊 Model Info & Context")
    st.info("**Model Type:** Random Forest Regressor")
    st.write("This app uses a Machine Learning model trained on demographic and lifestyle data to predict individual annual medical charges.")
    
    st.markdown("---")
    st.markdown("#### 💡 Feature Weight Overview")
    st.markdown("- **Smoker Status:** Primary cost driver")
    st.markdown("- **BMI:** Secondary influence")
    st.markdown("- **Age:** Gradual impact")
    
    st.markdown("---")
    st.caption("Developed with Python, Scikit-Learn, and Streamlit.")

# 5. Main Hero Banner UI
st.markdown("""
    <div class="hero-banner">
        <h1>💳 Health Insurance Cost Estimator</h1>
        <p>Provide your personal health and demographic parameters below to compute a real-time annual charge estimate.</p>
    </div>
""", unsafe_allow_html=True)

# 6. User Inputs Form Layout
if model_loaded:
    with st.form("prediction_form"):
        st.markdown('<div class="section-header">📋 Enter Demographic & Health Attributes</div>', unsafe_allow_html=True)
        
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
        submit_button = st.form_submit_button("✨ Calculate Estimated Premium", use_container_width=True)

    # 7. Prediction Logic Execution
    if submit_button:
        # Build zero-initialized payload
        input_data = {col: 0 for col in model_columns}
        
        # Numeric Features
        input_data['age'] = age
        input_data['bmi'] = bmi
        input_data['children'] = children
        
        # Categorical Encodings
        if 'sex' in input_data:
            input_data['sex'] = 1 if sex == 'male' else 0
        if 'smoker' in input_data:
            input_data['smoker'] = 1 if smoker == 'yes' else 0

        # One-Hot Encoded Region Mapping
        region_col = f"region_{region}"
        if region_col in input_data:
            input_data[region_col] = 1

        # Predict with Model
        df_input = pd.DataFrame([input_data])
        prediction = model.predict(df_input)[0]
        
        # Render Styled Output Display Card
        st.markdown(
            f"""
            <div class="output-card">
                <h4>Estimated Annual Medical Coverage Charge</h4>
                <h1>${prediction:,.2f}</h1>
            </div>
            """, 
            unsafe_allow_html=True
        )
