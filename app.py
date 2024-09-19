import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load models
models = {
    "Random Forest": joblib.load('Models/random_forest_model.pkl'),
    "XGBoost": joblib.load('Models/xgb_model.pkl'),
    "LightGBM": joblib.load('Models/lgb_model.pkl'),
    "CatBoost": joblib.load('Models/cb_model.pkl'),
    "HistGradientBoost": joblib.load('Models/hist_Boost_model.pkl'),
}

# App title
st.title("Predictive Model Selector")

# Model selection
model_choice = st.selectbox("Choose a model:", list(models.keys()))

# User input
def user_input():
    inputs = {}
    for i in range(22): 
        value = st.number_input(f"Column{i}", format="%.15f", value=None, help="Leave empty for NaN value")
        inputs[f"Column{i}"] = value if value is not None else np.nan
    return pd.DataFrame([inputs])

input_data = user_input()

# Predict button
if st.button("Predict"):
    model = models[model_choice]
    try:
        prediction = model.predict(input_data)
        st.write(f"Prediction with {model_choice}: {prediction[0]}")
    except Exception as e:
        st.write(f"Error: {e}")

# Option to show predictions from all models
if st.checkbox("Show predictions from all models"):
    predictions = {}
    for name, model in models.items():
        try:
            pred = model.predict(input_data)
            predictions[name] = pred[0]
        except Exception as e:
            predictions[name] = f"Error: {e}"
    
    # Create a DataFrame for displaying predictions
    predictions_df = pd.DataFrame(list(predictions.items()), columns=['Model', 'Prediction'])
    st.write(predictions_df)
