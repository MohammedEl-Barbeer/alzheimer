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


# --- إضافة اللوجو والعنوان ---
try:
    # تحميل الصورة وعرضها بحجم مناسب
    img = Image.open('NeuroGene.jpeg')
    # عرض الصورة في المنتصف بحجم 300 بكسل عرض
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(img, caption='NeuroGene Project - HIERO 2026', width=300)
except FileNotFoundError:
    # في حال لم يتم العثور على الصورة، يتم تجاهلها
    st.warning(
        '⚠️ Logo image "NeuroGene.jpg" not found. Please place it in the project'
        ' folder.'
    )

st.title('🧬 Alzheimer Gene Expression Analysis System')
st.markdown(
    '***'
) # خط فاصل أفقي جمالي
st.write(
    'Interactive web application built for **HIERO 2026** to classify and analyze'
    ' gene expression samples.'
)
st.markdown(
    '***'
)


# --- تحميل النموذج المدرب ---
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
    st.stop() # إيقاف التطبيق إذا لم يتم العثور على النموذج

# --- الشريط الجانبي لرفع الملفات ---
st.sidebar.header('1. Data Input')
uploaded_file = st.sidebar.file_uploader(
    'Choose a pre-processed CSV data file', type=['csv']
)

if uploaded_file is not None:
    try:
        # قراءة البيانات وعرض المعاينة
        input_data = pd.read_csv(uploaded_file)
        st.subheader('Uploaded Data Preview:')
        st.dataframe(input_data.head(), height=200)
        st.info(
            f'Data loaded successfully. Shape: {input_data.shape[0]} rows,'
            f' {input_data.shape[1]} columns.'
        )

        if st.sidebar.button('🚀 Run Prediction'):
            with st.spinner('Processing data and running classification...'):
                # --- معالجة البيانات ومطابقة الأعمدة (الجزء الحرج) ---
                model_input = input_data.copy()

                # إسقاط الأعمدة غير المتعلقة بالجينات
                for col in ['ID_REF', 'Patient', 'Stage']:
                    if col in model_input.columns:
                        model_input = model_input.drop(columns=[col])

                # توحيد أسماء الأعمدة
                model_input.columns = (
                    pd.Index(model_input.columns)
                    .astype(str)
                    .str.strip()
                    .str.upper()
                )

                # معالجة التكرار في أسماء الجينات بأخذ المتوسط
                model_input = model_input.T.groupby(level=0).mean().T

                # مطابقة أعمدة النموذج تماماً
                if hasattr(model, 'feature_names_in_'):
                    expected_features = model.feature_names_in_
                    # إضافة الأعمدة المفقودة بقيمة 0، وإسقاط الزائدة
                    model_input = model_input.reindex(
                        columns=expected_features, fill_value=0
                    )
                    st.success(
                        f'Data successfully aligned with model features: {model_input.shape[1]} features matched.'
                    )
                else:
                    st.warning(
                        '⚠️ Warning: Model feature names not found. Prediction might fail if input format differs.'
                    )

                # --- تنفيذ التنبؤ ---
                # إذا واجهتك خطأ X has 9746 features هنا، فالمشكلة في الملف المرفوع
                predictions = model.predict(model_input)
                probabilities = model.predict_proba(model_input)

                # --- عرض النتائج بتنسيق احترافي ---
                results_df = input_data.copy()
                results_df['Final Prediction'] = np.where(
                    predictions == 1, 'Alzheimer (AD)', 'Control'
                )
                results_df['Prediction Score'] = np.max(probabilities, axis=1)
                
                st.markdown('***')
                st.subheader('🏆 Final Classification Results:')
                
                # استخدام التنسيق الشرطي لإبراز النتائج في جدول النتائج
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
                
                # إحصائيات سريعة للنتائج
                ad_count = sum(predictions == 1)
                ctrl_count = sum(predictions == 0)
                st.write(f'📊 Summary: AD Cases: {ad_count}, Control Cases: {ctrl_count}')

    except Exception as err:
        st.error(
            '❌ An error occurred during data processing and model prediction. Please'
            f' check your input file format. Error details: {err}'
        )
else:
    # رسالة ترحيبية في الجزء الرئيسي في حالة عدم وجود ملف
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


# --- تذييل الصفحة (Footer) ---
st.markdown(
    """
    <div style="text-align: center; color: #757575; padding-top: 50px;">
        <p>NeuroGene Project © 2026</p>
        <p>Developed for HIERO 2026 - Hybrid Intelligence for Engineering, Robotics, and Optimization</p>
    </div>
    """,
    unsafe_allow_html=True,
)