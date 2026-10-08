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
from PIL import Image # مكتبة ضرورية لعرض الصورة

warnings.filterwarnings('ignore')

# إعدادات الصفحة
st.set_page_config(
    page_title='Alzheimer Gene Expression Analysis System',
    page_icon='🧬',
    layout='wide' # استخدام عرض الصفحة بالكامل لتنسيق أجمل
)

# --- تنسيق احترافي للواجهة (CSS) ---
# جعل الخطوط أوضح وتوسيع الحاوية الرئيسية
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
    .stAlert > div {
        font-family: 'Courier New', monospace;
    }
    .reportview-container .markdown-text-box {
        font-family: 'Arial', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



try:

    img = Image.open('NeuroGene.jpeg')

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(img, caption='NeuroGene Project - HIERO 2026', width=300)
except FileNotFoundError:

    st.warning(
        '⚠️ Logo image "NeuroGene.jpg" not found. Please place it in the project'
        ' folder.'
    )

st.title('🧬 Alzheimer Gene Expression Analysis System')
st.markdown(
    '***'
) 
st.write(
    'Interactive web application built for **HIERO 2026** to classify and analyze'
    ' gene expression samples.'
)
st.markdown(
    '***'
)



@st.cache_resource
def load_model():
    return joblib.load('alzheimer_model.pkl')


try:
    model = load_model()
    st.sidebar.success('Model loaded successfully!')
except Exception as e:
    st.sidebar.error(
        '❌ Model file not found. Please ensure alzheimer_model.pkl is in the'
        ' folder.'
    )
    st.stop()


st.sidebar.header('1. Data Input')
uploaded_file = st.sidebar.file_uploader(
    'Choose a pre-processed CSV data file', type=['csv']
)

if uploaded_file is not None:
    try:

        input_data = pd.read_csv(uploaded_file)
        st.subheader('Uploaded Data Preview:')
        st.dataframe(input_data.head(), height=200)
        st.info(
            f'Data loaded successfully. Shape: {input_data.shape[0]} rows,'
            f' {input_data.shape[1]} columns.'
        )

        if st.sidebar.button('🚀 Run Prediction'):
            with st.spinner('Processing data and running classification...'):

                model_input = input_data.copy()


                for col in ['ID_REF', 'Patient', 'Stage']:
                    if col in model_input.columns:
                        model_input = model_input.drop(columns=[col])


                model_input.columns = (
                    pd.Index(model_input.columns)
                    .astype(str)
                    .str.strip()
                    .str.upper()
                )


                model_input = model_input.T.groupby(level=0).mean().T


                
                GENES = joblib.load('alzheimer_model.pkl')  # 4576 gene symbols (uppercase)
                if len(model_input) < 2:
                    st.error('Need at least 2 samples (the model was trained on per-gene z-scored data).')
                    st.stop()

                missing = [g for g in GENES if g not in model_input.columns]
                if len(missing) > 0.05 * len(GENES):
                    st.error(f'{len(missing)} of {len(GENES)} model genes are missing from the file. Check gene naming.')
                    st.stop()

                X = model_input.apply(pd.to_numeric, errors='coerce')
                sd = X.std(ddof=1).replace(0, 1).fillna(1)
                X = ((X - X.mean()) / sd).fillna(0)          # per-gene z-score, same as training
                model_input = X.reindex(columns=GENES, fill_value=0.0)
                st.success(f'Aligned to model genes: {len(GENES) - len(missing)}/{len(GENES)} matched.')


                predictions = model.predict(model_input.values)
                probabilities = model.predict_proba(model_input.values)


                results_df = input_data.copy()
                results_df['Final Prediction'] = np.where(
                    predictions == 1, 'Alzheimer (AD)', 'Control'
                )
                results_df['Prediction Score'] = np.max(probabilities, axis=1)
                
                st.markdown('***')
                st.subheader('🏆 Final Classification Results:')
                

                def color_prediction(val):
                    color = '#ffcccc' if val == 'Alzheimer (AD)' else '#ccffcc'
                    return f'background-color: {color}'

                st.dataframe(
                    results_df.style.applymap(
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
            '❌ An error occurred during data processing and model prediction. Please'
            f' check your input file format. Error details: {err}'
        )
else:

    st.markdown(
        """
        <div style="background-color: #e3f2fd; padding: 20px; border-radius: 10px; border: 1px solid #90caf9;">
            <h3 style="color: #0d47a1;">👋 Welcome to NeuroGene Classifier</h3>
            <p style="color: #1565c0; font-size: 1.1em;">
            To begin the analysis, please upload a pre-processed Gene Expression CSV file 
            via the sidebar on the left. 
            <br><br>
            Ensure the file contains only the relevant gene expression values 
            (after filtering and standardization) and does not include non-numeric columns 
            like Patient ID or Stage, or that you handle them properly if you included them.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('***')



st.markdown(
    """
    <div style="text-align: center; color: #757575; padding-top: 50px;">
        <p>NeuroGene Project © 2026</p>
        <p>Developed for HIERO 2026 - Hybrid Intelligence for Engineering, Robotics, and Optimization</p>
    </div>
    """,
    unsafe_allow_html=True,
)