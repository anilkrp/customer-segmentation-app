import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

@st.cache_data
def load_data(uploaded_file):
    """Load data from uploaded file with caching"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

def preprocess_data(df):
    """Preprocess the retail data"""
    df_clean = df.copy()
    
    # Standardize column names
    df_clean.columns = df_clean.columns.str.lower().str.replace(' ', '_')
    
    # Convert date
    df_clean['date'] = pd.to_datetime(df_clean['date'])
    
    # Extract date features
    df_clean['year'] = df_clean['date'].dt.year
    df_clean['month'] = df_clean['date'].dt.month
    df_clean['day_of_week'] = df_clean['date'].dt.day_name()
    df_clean['quarter'] = df_clean['date'].dt.quarter
    
    # Ensure numeric columns
    numeric_columns = ['quantity', 'price_per_unit', 'total_amount']
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Remove duplicates
    initial_shape = df_clean.shape[0]
    df_clean = df_clean.drop_duplicates()
    if initial_shape != df_clean.shape[0]:
        st.info(f"ℹ️ Removed {initial_shape - df_clean.shape[0]} duplicate rows")
    
    return df_clean

def validate_data(df):
    """Validate data structure"""
    required_columns = ['transaction_id', 'date', 'customer_id', 'total_amount']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"❌ Missing required columns: {missing_columns}")
        return False
    
    return True