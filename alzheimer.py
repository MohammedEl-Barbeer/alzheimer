if uploaded_file is not None:
  input_data = pd.read_csv(uploaded_file)
  st.subheader('معاينة البيانات المدخلة:')
  st.dataframe(input_data.head())

  if st.button('تنفيذ التنبؤ (Predict)'):
    try:
      # تجهيز البيانات المدخلة لتتوافق مع توقعات الموديل (إسقاط الأعمدة غير الرقمية)
      model_input = input_data.copy()
      if 'ID_REF' in model_input.columns:
        model_input = model_input.drop(columns=['ID_REF'], errors='ignore')
      if 'Patient' in model_input.columns:
        model_input = model_input.drop(columns=['Patient'], errors='ignore')
      if 'Stage' in model_input.columns:
        model_input = model_input.drop(columns=['Stage'], errors='ignore')

      # توحيد أسماء الأعمدة لتكون بحروف كبيرة ومطابقة تماماً لتدريب الموديل
      model_input.columns = (
          pd.Index(model_input.columns).astype(str).str.strip().str.upper()
      )

      # إذا كان الموديل يحفظ أسماء الأعمدة أو يتوقع عدداً محدداً، سنقوم بمطابقتها
      # بما أن الموديل يتوقع عدد ميزات معين (مثل 4576 جين)، سنقوم بأخذ أول 4576 عموداً رقمياً متوافقاً أو ضبط الفلترة
      # الأفضل: إذا كانت البيانات المدخلة هي نفسها الملف الخام، نحتاج لتطبيق نفس تقطيع الجينات
      if hasattr(model, 'feature_names_in_'):
        expected_features = model.feature_names_in_
        # تصفية الأعمدة لتشمل فقط الموجودة في تدريب النموذج
        model_input = model_input.reindex(
            columns=expected_features, fill_value=0
        )

      # تنفيذ التنبؤ باستخدام النموذج المحفوظ
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