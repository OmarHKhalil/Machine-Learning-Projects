# Weather Forecasting System using Machine Learning

## Overview

This folder contains weather forecasting notebooks that predict whether it will rain tomorrow using the `weatherAUS.csv` dataset.

الهدف من المشروع هو تحويل مشكلة تنبؤ “هل ستمطر غداً؟” إلى سير عمل ML واضح وقابل للتكرار: تنظيف البيانات، ترميز المتغيرات، ثم تدريب نماذج تصنيف ومقارنتها. تم بناؤه كتمرين عملي لتعلم كيفية التعامل مع بيانات حقيقية تحتوي على قيم مفقودة ومتغيرات فئوية. القيمة المضافة تتمثل في أن القارئ يتعلم كيف ترتبط خطوة المعالجة المسبقة مباشرةً بالتحسن في Accuracy ومع فهم `confusion matrix` لقراءة الأخطاء بشكل أعمق.

## Summary Table

| Dataset          | Model                    | Key Result                                           |
| ---------------- | ------------------------ | ---------------------------------------------------- |
| `weatherAUS.csv` | `DecisionTreeClassifier` | Delivered mid-80% accuracy on rain prediction.       |
| `weatherAUS.csv` | `RandomForestClassifier` | Achieved accuracy near 85.7% on the evaluated split. |
| `weatherAUS.csv` | `OneHotEncoder`          | Encoded categorical weather features.                |
| `weatherAUS.csv` | `MinMaxScaler`           | Scaled numeric weather inputs.                       |

## Approach

- Load and clean the weather dataset.
- Encode categorical variables and drop missing target values.
- Train Decision Tree and Random Forest classifiers.
- Evaluate models using accuracy and confusion matrix analysis.

## Project Structure

```
Weather Forecasting System using Machine Learning/
├── Decision Tree.ipynb
└── Random Forest.ipynb
```

## Notes

This project demonstrates binary weather classification modeling using real-world rainfall data.

## Model Results

| Model / Notebook                                 | Key metrics (excerpt)                                            |
| ------------------------------------------------ | ---------------------------------------------------------------- |
| `DecisionTreeClassifier` (`Decision Tree.ipynb`) | Reported accuracies across runs: ~79%–85% (multiple runs logged) |
| `RandomForestClassifier` (`Random Forest.ipynb`) | Reported accuracies: up to 85.68% in evaluation runs             |
 
## Key Features
- استخدام تقنيات: `pandas` لتنظيف البيانات، `OneHotEncoder` + `MinMaxScaler` لتهيئة الميزات، ثم مقارنة `DecisionTreeClassifier` و`RandomForestClassifier`.
- دقة عالية: تصل دقة التنبؤ بهطول المطر إلى ~`85.68%` مع Random Forest، مع نطاق ~`79%–85%` للـ Decision Tree.
- تكامل خارجي: يعتمد على ملف `weatherAUS.csv` المحلي داخل المجلد (بدون تكامل خارجي مثل Gemini API).

## Requirements / Installation
- Python: `3.9+`
- التثبيت:
  `pip install -r requirements.txt`
- البيانات: تأكد من وجود `weatherAUS.csv` داخل هذا المجلد (أو عدّل مسارات القراءة داخل الـ notebook).

## Workflow / Pipeline
1. تحميل وتنظيف بيانات الطقس.
2. ترميز السمات الفئوية وإسقاط حالات target المفقودة.
3. تقسيم بيانات التدريب/الاختبار (مثال: 85% تدريب و15% اختبار).
4. تدريب Decision Tree و Random Forest.
5. تقييم الأداء باستخدام Accuracy و`confusion matrix`.

## Results / Metrics
- Accuracy:
  - Decision Tree: ~`79%–85%`
  - Random Forest: حتى ~`85.68%`
- Confusion Matrix: تُستخدم داخل notebook ضمن قسم تقييم التصنيف.

## Usage
1. افتح `Decision Tree.ipynb` أو `Random Forest.ipynb` في Jupyter:
   `jupyter notebook "Decision Tree.ipynb"`
2. شغّل الخلايا بالترتيب.

## Authors / Credits
- Contributors: e-Machine-Learning-Projects (لا توجد أسماء محددة داخل notebooks/README).
