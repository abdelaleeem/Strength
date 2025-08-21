# app.py
import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

# --- Page Configuration ---
st.set_page_config(
    page_title="28 days strength Prediction App",
    page_icon="⚙️",
    layout="wide"
)

# --- Load Models ---
@st.cache_resource
def load_models():
    """Loads the pickled machine learning models."""
    try:
        model_dir = Path(__file__).parent
        with open(model_dir / 'linear_model.pkl', 'rb') as f:
            linear_model = pickle.load(f)
        with open(model_dir / 'RF_model.pkl', 'rb') as f:
            RF_model = pickle.load(f)
        # New models for setting time
        with open(model_dir / 'linear_setting_model.pkl', 'rb') as f:
            linear_setting_model = pickle.load(f)
        with open(model_dir / 'RF_setting_model.pkl', 'rb') as f:
            RF_setting_model = pickle.load(f)

        return linear_model, RF_model, linear_setting_model, RF_setting_model
    except FileNotFoundError:
        st.error("Model files not found. Please ensure pickle files are present in the app directory.")
        return None, None, None, None

linear_model, RF_model, linear_setting_model, RF_setting_model = load_models()

# --- Define Final Feature Order (for strength models only) ---
MODEL_FEATURES_ORDER = [
    'CK Factor ', 'ADD', 'Blaine', 'I.R',
    'Bazalt Type_Fine Basalt',
    'Bazalt Type_Mixed'
]

# --- Define Feature Order for Setting Time ---
SETTING_FEATURES_ORDER = ['Temp ', 'C3A', 'I.R', 'Blaine']

# --- Main Application ---
st.title("⚙️ Prediction Dashboard")
st.markdown("This app combines predictions from ML models with Vade-Macum correlations.")

# --- Layout ---
col1, col2 = st.columns([1, 2])

# --- Input Widgets (Right Column) ---
with col2:
    st.header("Input Features")
    f_col1, f_col2, f_col3 = st.columns(3)

    # Formula Inputs
    with f_col1:
        st.subheader("Formula Inputs")
        Na_eq = st.number_input("Na_eq", min_value=0.0, value=0.5, step=0.01, format="%.2f")
        C3S = st.number_input("C3S", min_value=0.0, value=59.0, step=0.5, format="%.1f")
        C2S = st.number_input("C2S", min_value=0.0, value=10.0, step=0.5, format="%.1f")
        C4AF = st.number_input("C4AF", min_value=0.0, value=10.0, step=0.1, format="%.1f")
        MgO = st.number_input("MgO", min_value=0.0, value=2.0, step=0.1, format="%.1f")
        S_Alk = st.number_input("S/Alk", min_value=0.0, value=1.0, step=0.1, format="%.1f")

    # Strength Model Inputs
    with f_col2:
        st.subheader("Strength Model Inputs")
        CK_Factor = st.number_input("CK Factor", min_value=0.0, value=65.0, step=0.5, format="%.2f")
        ADD = st.number_input("ADD", min_value=0.0, value=15.0, step=0.5, format="%.1f")
        Blaine = st.number_input("Blaine", min_value=0, value=4200, step=50)
        IR = st.number_input("I.R", min_value=0.0, value=10.0, step=0.5, format="%.2f")

    # Categorical + Setting Time Inputs
    with f_col3:
        st.subheader("Other Inputs")
        bazalt_type = st.selectbox("Bazalt Type", ["El Sonny", 'Fine Basalt', 'Mixed'])
        ga_dosage = st.selectbox("GA Dosage", ['Mapei C570'])
        temp = st.number_input("Mill Outlet Temperature", min_value=0.0, value=120.0, step=1.0, format="%.1f")
        C3A = st.number_input("C3A", min_value=0.0, value=7.0, step=0.5, format="%.1f")

# --- Prediction and Display Logic ---
if linear_model and RF_model and linear_setting_model and RF_setting_model:
    # ---------- Strength Prediction ----------
    total_effect = (-10 * Na_eq + 0.6 * C3S + 0.5 * C2S - 0.5 * C4AF - 0.6 * MgO + 1.5 * S_Alk)
    model_input_data = {
        'CK Factor ': [CK_Factor],
        'ADD': [ADD],
        'Blaine': [Blaine],
        'I.R': [IR]
    }
    model_df = pd.DataFrame(model_input_data)
    model_df['Bazalt Type_Fine Basalt'] = 1 if bazalt_type == 'Fine Basalt' else 0
    model_df['Bazalt Type_Mixed'] = 1 if bazalt_type == 'Mixed' else 0
    final_model_df = model_df[MODEL_FEATURES_ORDER]

    pred_linear = linear_model.predict(final_model_df)
    pred_RF = RF_model.predict(final_model_df)
    total_strength = pred_linear[0] + pred_RF[0] + total_effect

    # ---------- Setting Time Prediction ----------
    setting_input_data = {
        'Temp ': [temp],
        'C3A': [C3A],
        'I.R': [IR],
        'Blaine': [Blaine]
    }
    setting_df = pd.DataFrame(setting_input_data)[SETTING_FEATURES_ORDER]

    pred_setting_linear = linear_setting_model.predict(setting_df)
    pred_setting_RF = RF_setting_model.predict(setting_df)
    total_setting_time = pred_setting_linear[0] + pred_setting_RF[0]

    # ---------- Display ----------
    with col1:
        st.header("Prediction Results")

        # Strength
        st.subheader("🧱 28-days Strength")
        st.metric(label="Final Combined Prediction", value=f"{total_strength:.2f}")
        with st.expander("Strength Calculation Details"):
            st.markdown(f"**Vade-Macum**: `{total_effect:.2f}`")
            st.markdown(f"**Linear Model Prediction**: `{pred_linear[0]:.2f}`")
            st.markdown(f"**Random Forest Prediction**: `{pred_RF[0]:.2f}`")
            st.markdown(f"**Final Sum** = {total_strength:.2f}")
            st.dataframe(final_model_df)

        # Setting Time
        st.subheader("⏳ Initial Setting Time")
        st.metric(label="Predicted Setting Time", value=f"{total_setting_time:.2f} min")
        with st.expander("Setting Time Calculation Details"):
            st.markdown(f"**Linear Model Prediction**: `{pred_setting_linear[0]:.2f}`")
            st.markdown(f"**Random Forest Residual Model Prediction**: `{pred_setting_RF[0]:.2f}`")
            st.markdown(f"**Final Sum** = {total_setting_time:.2f}")
            st.dataframe(setting_df)

else:
    with col1:
        st.header("Prediction Results")
        st.info("Waiting for models to load or for input.")
