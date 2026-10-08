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
      # تنفيذ التنبؤ باستخدام النموذج
      predictions = model.predict(input_data)
      probabilities = model.predict_proba(input_data)

      input_data['Prediction (0: Control, 1: AD)'] = predictions
      input_data['Confidence Score'] = np.max(probabilities, axis=1)

      st.subheader('نتائج التصنيف النهائية:')
      st.dataframe(input_data)

    except Exception as err:
      st.error(
          f'حدث خطأ أثناء معالجة البيانات وتطبيق النموذج: تأكد من توافق'
          f' الأعمدة. الخطأ: {err}'
      )
else:
  st.info(
      'الرجاء إرفاق ملف CSV يحتوي على قراءات الجينات عبر القائمة الجانبية لبدء'
      ' التحليل.'
  )