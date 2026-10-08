# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 16:59:32 2026

@author: barber
"""

import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title='Alzheimer Gene Expression Analysis System',
    page_icon='🧬',
    layout='wide'
)

# --- Custom CSS Styling ---
st.markdown(
    """
    <style>
    .main > div {
        padding-top: 2rem;
    }
    h1 {
        color: #1E88E5;
        font-family: 'Helvetica Neue', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Display Project Logo ---
try:
    img = Image.open('NeuroGene.jpeg')
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(img, caption='NeuroGene Project - HIERO 2026', width=300)
except FileNotFoundError:
    st.warning('⚠️ Logo image "NeuroGene.jpeg" not found. Please place it in the project folder.')

st.title('🧬 Alzheimer Gene Expression Analysis System')
st.markdown('***')
st.write(
    'Interactive web application built for **HIERO 2026** to classify and analyze gene expression samples.'
)
st.markdown('***')

# --- Load Trained Model and Feature Names ---
@st.cache_resource
def load_assets():
    model = joblib.load('alzheimer_model.pkl')
    try:
        genes = joblib.load('model_genes.pkl')
    except:
        genes = None
    return model, genes

try:
    model, expected_genes = load_assets()
    st.sidebar.success('Model and assets loaded successfully!')
except Exception as e:
    st.sidebar.error('❌ Required files not found. Ensure alzheimer_model.pkl exists.')
    st.stop()

# --- Sidebar File Uploader ---
st.sidebar.header('1. Data Input')
uploaded_file = st.sidebar.file_uploader(
    'Choose a pre-processed CSV data file', type=['csv']
)

if uploaded_file is not None:
    try:
        input_data = pd.read_csv(uploaded_file)
        st.subheader('Uploaded Data Preview:')
        st.dataframe(input_data.head(), height=200)
        st.info(f'Data loaded successfully. Shape: {input_data.shape[0]} rows, {input_data.shape[1]} columns.')

        if st.sidebar.button('🚀 Run Prediction'):
            with st.spinner('Processing data and running classification...'):
                model_input = input_data.copy()

                # Drop non-feature columns if present
                for col in ['ID_REF', 'Patient', 'Stage']:
                    if col in model_input.columns:
                        model_input = model_input.drop(columns=[col])

                # Standardize column names to uppercase
                model_input.columns = (
                    pd.Index(model_input.columns)
                    .astype(str)
                    .str.strip()
                    .str.upper()
                )

                # Handle duplicate columns by averaging
                model_input = model_input.T.groupby(level=0).mean().T

                # Align features strictly using expected model genes or feature_names_in_
                if expected_genes is not None:
                    # تنظيف أسماء الجينات المتوقعة لتتوافق مع المدخلات
                    clean_expected = [str(g).strip().upper() for g in expected_genes]
                    model_input = model_input.reindex(columns=clean_expected, fill_value=0)
                elif hasattr(model, 'feature_names_in_'):
                    model_input = model_input.reindex(columns=model.feature_names_in_, fill_value=0)

                # Perform prediction
                predictions = model.predict(model_input)
                probabilities = model.predict_proba(model_input)

                results_df = input_data.copy()
                results_df['Final Prediction'] = np.where(
                    predictions == 1, 'Alzheimer (AD)', 'Control'
                )
                results_df['Prediction Score'] = np.max(probabilities, axis=1)

                st.markdown('***')
                st.subheader('🏆 Final Classification Results:')

                # Styled dataframe using .map (compatible with Pandas modern versions)
                def color_prediction(val):
                    color = '#ffcccc' if val == 'Alzheimer (AD)' else '#ccffcc'
                    return f'background-color: {color}'

                st.dataframe(
                    results_df.style.map(
                        color_prediction, subset=['Final Prediction']
                    )
                    .format({'Prediction Score': '{:.4f}'}),
                    height=600
                )

                ad_count = sum(predictions == 1)
                ctrl_count = sum(predictions == 0)
                st.write(f'📊 Summary: AD Cases: {ad_count}, Control Cases: {ctrl_count}')

    except Exception as err:
        st.error(
            '❌ An error occurred during data processing and model prediction. Please check your input file format. Error details: '
            f'{err}'
        )
else:
    st.markdown(
        """
        <div style="background-color: #e3f2fd; padding: 20px; border-radius: 10px; border: 1px solid #90caf9;">
            <h3 style="color: #0d47a1;">👋 Welcome to NeuroGene Classifier</h3>
            <p style="color: #1565c0; font-size: 1.1em;">
            To begin the analysis, please upload a pre-processed Gene Expression CSV file 
            via the sidebar on the left.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('***')

# --- Footer ---
st.markdown(
    """
    <div style="text-align: center; color: #757575; padding-top: 50px;">
        <p>NeuroGene Project © 2026</p>
        <p>Developed for HIERO 2026 - Hybrid Intelligence for Engineering, Robotics, and Optimization</p>
    </div>
    """,
    unsafe_allow_html=True,
)