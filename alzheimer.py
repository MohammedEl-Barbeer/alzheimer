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

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title='Alzheimer Gene Expression Classifier', page_icon='🧬'
)

st.title('🧬 Alzheimer Gene Expression Analysis System')
st.write(
    'Interactive web application built for HIERO 2026 to classify and analyze'
    ' gene expression samples.'
)


# Load the trained model
@st.cache_resource
def load_model():
  return joblib.load('alzheimer_model.pkl')


try:
  model = load_model()
  st.success('Model loaded successfully!')
except Exception as e:
  st.warning('Model file not found. Please make sure alzheimer_model.pkl exists.')

# Sidebar for file upload
st.sidebar.header('Data Input')
uploaded_file = st.sidebar.file_uploader(
    'Choose a CSV data file', type=['csv']
)

if uploaded_file is not None:
  input_data = pd.read_csv(uploaded_file)
  st.subheader('Uploaded Data Preview:')
  st.dataframe(input_data.head())

  if st.button('Predict'):
    try:
      model_input = input_data.copy()

      # Drop non-feature columns if present
      for col in ['ID_REF', 'Patient', 'Stage']:
        if col in model_input.columns:
          model_input = model_input.drop(columns=[col])

      # Standardize column names to uppercase
      model_input.columns = (
          pd.Index(model_input.columns).astype(str).str.strip().str.upper()
      )

      # Handle duplicate columns by averaging
      model_input = model_input.T.groupby(level=0).mean().T

      # Align columns with what the model expects during training
      if hasattr(model, 'feature_names_in_'):
        expected_features = model.feature_names_in_
        model_input = model_input.reindex(
            columns=expected_features, fill_value=0
        )
      else:
        # Fallback if feature_names_in_ is missing from pipeline steps, align using pipeline structure
        # If the pipeline uses SelectKBest/StandardScaler from a fixed set of shared genes:
        pass

      # Perform prediction
      predictions = model.predict(model_input)
      probabilities = model.predict_proba(model_input)

      input_data['Prediction (0: Control, 1: AD)'] = predictions
      input_data['Confidence Score'] = np.max(probabilities, axis=1)

      st.subheader('Final Classification Results:')
      st.dataframe(input_data)

    except Exception as err:
      st.error(
          'An error occurred during data processing and model prediction:'
          f' {err}'
      )
else:
  st.info(
      'Please upload a CSV file containing gene expression readings via the'
      ' sidebar to start the analysis.'
  )