# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 16:59:32 2026

@author: barber
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import warnings
warnings.filterwarnings("ignore")


# إعدادات صفحة الويب
st.set_page_config(
    page_title='Alzheimer Gene Expression Classifier', page_icon='🧬'
)

st.title('🧬 نظام تحليل التعبير الجيني لمرض الزهايمر')
st.write(
    'تطبيق ويب تفاعلي مبني ضمن فعاليات HIERO 2026 لتصنيف وتحليل عينات التعبير'
    ' الجيني.'
)



# تحميل النموذج المحفوظ مسبقاً
@st.cache_resource
def load_model():
  # تأكد من وضع ملف الوديل في نفس المجلد
  return joblib.load('alzheimer_model.pkl')


try:
  model = load_model()
  st.success('تم تحميل النموذج بنجاح!')
except Exception as e:
  st.warning(
      'لم يتم العثور على ملف النموذج المحفوظ. الرجاء التأكد من رفع ملف'
      ' alzheimer_model.pkl'
  )

# شريط جانبي لرفع الملفات
st.sidebar.header('إدخال البيانات')
uploaded_file = st.sidebar.file_uploader(
    'اختر ملف البيانات (CSV)', type=['csv']
)

if uploaded_file is not None:
  input_data = pd.read_csv(uploaded_file)
  st.subheader('معاينة البيانات المدخلة:')
  st.dataframe(input_data.head())

  if st.button('تنفيذ التنبؤ (Predict)'):
    try:
      # استبعاد الأعمدة النصية أو غير الرقمية مثل ID_REF أو Patient أو Stage إذا وجدت
      # سنقوم بأخذ الأعمدة الرقمية فقط التي يتوقعها الموديل
      model_input = input_data.select_dtypes(include=[np.number])

      # إذا كان عمود ID_REF خارج التصنيف، نتأكد من إسقاطه إن وجد كعمود نصي
      if 'ID_REF' in input_data.columns:
        model_input = input_data.drop(columns=['ID_REF'], errors='ignore')
      if 'Patient' in model_input.columns:
        model_input = model_input.drop(columns=['Patient'], errors='ignore')
      if 'Stage' in model_input.columns:
        model_input = model_input.drop(columns=['Stage'], errors='ignore')

      # تنفيذ التنبؤ باستخدام النماذج الرقمية فقط
      predictions = model.predict(model_input)
      probabilities = model.predict_proba(model_input)

      input_data['Prediction (0: Control, 1: AD)'] = predictions
      input_data['Confidence Score'] = np.max(probabilities, axis=1)

      st.subheader('نتائج التصنيف النهائية:')
      st.dataframe(input_data)

    except Exception as err:
      st.error(
          f'حدث خطأ أثناء معالجة البيانات وتطبيق النموذج: تأكد من توافق'
          f' الأعمدة. الخطأ: {err}'
      )