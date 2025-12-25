import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Page Config
st.set_page_config(page_title="Boston House Price Predictor", layout="wide")

st.title("🏡 Boston House Price Prediction App")
st.write("This app uses a **Polynomial Regression Model** (86% accuracy) to predict house prices.")

# 1. Load and Prepare Model (Internal Logic)
@st.cache_data
def get_trained_model():
    df = pd.read_csv(r"C:\Users\alanb\Downloads\Boston House Prices.csv")
    
    # Clean Outliers
    Q1, Q3 = df['MEDV'].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    df = df[(df['MEDV'] >= Q1 - 1.5 * IQR) & (df['MEDV'] <= Q3 + 1.5 * IQR)]
    
    X = df.drop('MEDV', axis=1)
    y = df['MEDV']
    
    # Polynomial Transformation (Degree 2)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_poly)
    
    # Train
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    return model, poly, scaler, X.columns

model, poly, scaler, _ = get_trained_model()

# 2. User Input Sidebar
st.sidebar.header("Input House Details")

def user_input_features():
    inputs = {}
    col1, col2 = st.sidebar.columns(2)
    
    # Inputs with default values based on dataset averages
    inputs['CRIM'] = col1.number_input('Crime Rate', value=3.61)
    inputs['ZN'] = col2.number_input('Res. Zone %', value=11.36)
    inputs['INDUS'] = col1.number_input('Non-Retail %', value=11.13)
    inputs['CHAS'] = col2.selectbox('Near River? (1=Yes, 0=No)', [0, 1])
    inputs['NOX'] = col1.number_input('Nitric Oxide', value=0.55)
    inputs['RM'] = col2.number_input('Avg Rooms', value=6.28)
    inputs['AGE'] = col1.number_input('Old Houses %', value=68.57)
    inputs['DIS'] = col2.number_input('Dist to Work', value=3.79)
    inputs['RAD'] = col1.number_input('Hwy Access Index', value=9.54)
    inputs['TAX'] = col2.number_input('Tax Rate', value=408.23)
    inputs['PTRATIO'] = col1.number_input('Pupil-Teacher Ratio', value=18.45)
    inputs['B'] = col2.number_input('B-Index', value=356.67)
    inputs['LSTAT'] = st.sidebar.slider('Lower Status %', 0.0, 40.0, 12.65)
    
    return pd.DataFrame([inputs])

input_df = user_input_features()

# 3. Prediction Logic
st.subheader("Model Prediction")

# Process the input just like the training data
input_poly = poly.transform(input_df)
input_scaled = scaler.transform(input_poly)
prediction = model.predict(input_scaled)

# Display Result
st.success(f"### Predicted House Price: ${round(prediction[0], 2)}k")

st.info("The prediction is based on the 13 neighborhood factors provided in the sidebar.")
