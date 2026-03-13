import streamlit as st
import pandas as pd
import joblib
import os

# Set the title of the web app
st.set_page_config(page_title="Product Return Predictor", page_icon="📦", layout="centered")

# --- UI HEADER ---
st.title("📦 E-Commerce Return Predictor")
st.write("""
Welcome! This Machine Learning app predicts whether a customer is likely to **return a purchased item** 
based on their age, the product category, and the price they paid.
""")

st.markdown("---")

# --- SIDEBAR: USER INPUT ---
st.sidebar.header("📝 Enter Purchase Details")

# Ensure the model exists before trying to load it
if os.path.exists('models/return_predictor_model.pkl') and os.path.exists('models/model_features.pkl'):
    # Load the trained model and feature names
    model = joblib.load('models/return_predictor_model.pkl')
    model_features = joblib.load('models/model_features.pkl')

    # Sliders and Dropdowns for user input
    age = st.sidebar.slider("Customer Age", min_value=18, max_value=80, value=30)
    category = st.sidebar.selectbox("Product Category", ["Electronics", "Clothing", "Home", "Books", "Beauty"])
    amount = st.sidebar.number_input("Item Price ($)", min_value=1.0, max_value=2000.0, value=150.0, step=10.0)

    # --- MAKE PREDICTION ---
    if st.sidebar.button("Predict Return", type="primary"):
        # 1. Format the user's input into a DataFrame that looks exactly like our training data
        input_data = pd.DataFrame({'age': [age], 'category': [category], 'amount': [amount]})
        
        # 2. Get dummies (One-Hot encode the category) just like we did in training
        input_data_encoded = pd.get_dummies(input_data, columns=['category'])

        # 3. CRUCIAL FIX: Ensure our user input has ALL the columns the model was trained on
        # If the user selected 'Clothing', we need 'category_Electronics' to exist but be 0/False
        for feature in model_features:
            if feature not in input_data_encoded.columns:
                input_data_encoded[feature] = 0
                
        # Reorder columns to match the exact order the model expects
        input_data_encoded = input_data_encoded[model_features]

        # 4. Make the Prediction
        prediction = model.predict(input_data_encoded)[0]
        probability = model.predict_proba(input_data_encoded)[0][1] # Probability of '1' (Return)

        # --- DISPLAY RESULTS ---
        st.subheader("🤖 Prediction Result:")
        
        if prediction == 1:
            st.error(f"⚠️ **High Risk of Return!** (Probability: {probability:.1%})")
            st.write("The model suggests this customer is likely to return this item. You might want to review shipping/return policies for this demographic.")
        else:
            st.success(f"✅ **Low Risk of Return.** (Return Probability: {probability:.1%})")
            st.write("The model suggests this item will likely be kept by the customer.")
else:
    st.error("Model files not found! Please run `train_model.py` first to generate the models.")
