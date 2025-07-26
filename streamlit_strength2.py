# app.py
import streamlit as st
import pandas as pd
import pickle

# --- Page Configuration ---
st.set_page_config(
    page_title="28 days strength Prediction App",
    page_icon="⚙️",
    layout="wide"
)

# --- Load Models ---s
@st.cache_resource
def load_models():
    """Loads the pickled machine learning models."""
    try:
        with open('D:\cem mills files/linear_model.pkl', 'rb') as f:
            linear_model = pickle.load(f)
        # Corrected filename to match the creation script
        with open('D:\cem mills files/RF_model.pkl', 'rb') as f:
            RF_model = pickle.load(f)
        return linear_model, RF_model
    except FileNotFoundError:
        st.error("Model files not found. Please run the `create_models.py` script first.")
        return None, None

linear_model, RF_model = load_models()

# --- Define Final Feature Order (for models only) ---
MODEL_FEATURES_ORDER = [
    'CK Factor ', 'ADD', 'Blaine', 'I.R',
    'Bazalt Type_Fine Basalt',
       'Bazalt Type_Mixed'
]

# --- Main Application ---
st.title("⚙️ Prediction Dashboard")
st.markdown("This app combines predictions from two ML models with a Vade-Macum correlations with R-squared of 0.897 ,The prediction updates automatically as you change the inputs.")

# --- Layout ---
col1, col2 = st.columns([1, 2])

# --- Input Widgets (Right Column) ---
with col2:
    st.header("Input Features")
    f_col1, f_col2, f_col3 = st.columns(3)

    # Features for the 'total_effect' formula
    with f_col1:
        st.subheader("Formula Inputs")
        Na_eq = st.number_input("Na_eq", min_value=0.0, value=0.5, step=0.01, format="%.2f")
        C3S = st.number_input("C3S", min_value=0.0, value=59.0, step=0.5, format="%.1f")
        C2S = st.number_input("C2S", min_value=0.0, value=10.0, step=0.5, format="%.1f")
        C4AF = st.number_input("C4AF", min_value=0.0, value=10.0, step=0.1, format="%.1f")
        MgO = st.number_input("MgO", min_value=0.0, value=2.0, step=0.1, format="%.1f")
        S_Alk = st.number_input("S/Alk", min_value=0.0, value=1.0, step=0.1, format="%.1f")

    # Numerical features for the ML models
    with f_col2:
        st.subheader("Model Inputs (Numerics) for Regression")
        CK_Factor = st.number_input("CK Factor", min_value=0.0, value=65.0, step=0.5, format="%.2f")
        ADD = st.number_input("ADD", min_value=0.0, value=15.0, step=0.5, format="%.1f")
        #GYP = st.number_input("GYP", min_value=0.0, value=10.0, step=0.5, format="%.1f")
        Blaine = st.number_input("Blaine", min_value=0, value=4200, step=50)
        IR = st.number_input("I.R", min_value=0.0, value=10.0, step=0.5, format="%.2f")
        #Sieve_45 = st.number_input("Sieve 45 m", min_value=0.0, value=2.0, step=0.1, format="%.1f")
        #Sieve_90 = st.number_input("Sieve 90 m", min_value=0.0, value=0.3, step=0.1, format="%.1f")

    # Categorical features for the ML models
    with f_col3:
        st.subheader("Model Inputs (Categorical) for Regression")
        bazalt_type = st.selectbox("Bazalt Type", ["El Sonny" ,'Fine Basalt', 'Mixed'])
        ga_dosage = st.selectbox("GA Dosage", ['Mapei C570'])
        #cement_type = st.selectbox("Cement Type", ['CEM V B(P) 32.5 N', 'Pozz'])


# --- Prediction and Display Logic ---
# This block runs automatically if the models are loaded.
if linear_model is not None and RF_model is not None:
    # 1. Calculate `total_effect` from its specific inputs
    total_effect = (-10 * Na_eq + 0.6 * C3S + 0.5 * C2S - 0.5 * C4AF - 0.6 * MgO + 1.5 * S_Alk)

    # 2. Create DataFrame for the ML models with correct column names
    model_input_data = {
        'CK Factor ': [CK_Factor], # Note the required trailing space
        'ADD': [ADD], 'Blaine': [Blaine], 'I.R': [IR]
        
    }
    model_df = pd.DataFrame(model_input_data)

    # 3. Perform One-Hot Encoding according to the final feature list
    
    model_df['Bazalt Type_Fine Basalt'] = 1 if bazalt_type == 'Fine Basalt' else 0
    
    model_df['Bazalt Type_Mixed'] = 1 if bazalt_type == 'Mixed' else 0
    
    # 4. Ensure DataFrame has the exact feature order required by the models
    final_model_df = model_df[MODEL_FEATURES_ORDER]

    # 5. Make predictions
    pred_linear = linear_model.predict(final_model_df)
    pred_RF = RF_model.predict(final_model_df)

    # 6. Sum the results: (Model 1) + (Model 2) + (Formula)
    total_prediction = pred_linear[0] + pred_RF[0] + total_effect

    # 7. Display the result in the left column
    with col1:
        st.header("Prediction Result")
        st.metric(label="Final Combined Prediction", value=f"{total_prediction:.4f}")
        with st.expander("Show Calculation Details"):
            st.markdown(f"**Calculated `Vade-Macum`**: `{total_effect:.4f}`")
            st.markdown(f"**Linear Model Prediction**: `{pred_linear[0]:.4f}`")
            st.markdown(f"**Random Forest Regressor Model Prediction**: `{pred_RF[0]:.4f}`")
            st.markdown(f"**Final Sum**: `{total_effect:.4f} + {pred_linear[0]:.4f} + {pred_RF[0]:.4f} = {total_prediction:.4f}`")
            st.write("Data Sent to ML Models (Final Order):")
            st.dataframe(final_model_df)
else:
    # Display a placeholder if models aren't loaded
    with col1:
        st.header("Prediction Result")
        st.info("Waiting for models to load or for input.")