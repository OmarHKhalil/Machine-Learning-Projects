# Healthcare Insurance Cost Predictor

## Overview

This notebook predicts health insurance charges using the `insurance.csv` dataset and evaluates linear regression models using feature-based RMSE.

الهدف من المشروع هو بناء نموذج انحدار يتنبأ بتكاليف التأمين الصحي من سمات ديموغرافية وعادات/خصائص مرتبطة بالحياة. تم بناؤه كتمرين يوضح أن جودة التنبؤ لا تعتمد على النموذج فقط، بل تعتمد على تجهيز الميزات: ترميز الفئات، اختيار مقياس التطبيع، وبناء خط تقييم يعتمد على RMSE. القيمة المضافة هنا هي إتاحة مثال عملي لمشروع regression “واقعي” يمكن أن يستفيد منه القارئ في فهم كيفية قياس الخطأ وتحسينه خطوة بخطوة.

## Summary Table

| Dataset         | Model              | Key Result                                                                   |
| --------------- | ------------------ | ---------------------------------------------------------------------------- |
| `insurance.csv` | `LinearRegression` | Reported RMSE losses such as 8461.95 and 4991.99 for different feature sets. |
| `insurance.csv` | `OneHotEncoder`    | Encoded categorical variables for regression.                                |
| `insurance.csv` | `MinMaxScaler`     | Scaled numeric inputs for modeling.                                          |
| `insurance.csv` | `StandardScaler`   | Standardized numeric features across experiments.                            |

## Approach

- Clean and inspect the insurance dataset.
- Encode categorical variables and scale numeric inputs.
- Train Linear Regression models on medical cost targets.
- Evaluate performance using RMSE.

## Project Structure

```
Healthcare Insurance Cost Predictor/
└── Scikit Learning.ipynb
```

## Notes

This notebook is a regression-focused project for estimating insurance charges from demographic and lifestyle features.

## Model Results

| Model / Notebook                                                | Key metrics (excerpt)                                                                                                                                                          |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `LinearRegression` / other regressors (`Scikit Learning.ipynb`) | Reported RMSE losses: 8461.95, 4991.99, 11049.66, 10711.00 for different experiments; classification-style accuracy reports show ~84.2%–85.4% for some label-based evaluations |
 
## Key Features
- استخدام تقنيات: ترميز المتغيرات الفئوية بـ `OneHotEncoder`، تطبيع/standardization (`MinMaxScaler` و`StandardScaler`)، ثم تدريب نماذج `LinearRegression` (و/أو regressors أخرى) لتقدير تكاليف التأمين.
- أداء واضح: RMSE مُسجل لـ `LinearRegression` بأرقام مثل ~`8461.95` و`4991.99` و`11049.66` و`10711.00` (حسب مجموعات ميزات مختلفة).
- تكامل خارجي: يعتمد على ملف `insurance.csv` المحلي (لا يوجد تكامل خدمات خارجية مثل Gemini API).

## Requirements / Installation
- Python: `3.9+`
- التثبيت:
  `pip install -r requirements.txt`
- البيانات: ضع `insurance.csv` داخل هذا المجلد أو عدّل مسار القراءة في notebook.

## Workflow / Pipeline
1. تحميل البيانات وفحص الأعمدة/القيم المفقودة.
2. ترميز المتغيرات الفئوية.
3. قياس/تطبيع الميزات الرقمية.
4. تقسيم البيانات إلى تدريب/اختبار (مثال: 85% تدريب و15% اختبار).
5. تدريب نماذج الانحدار ومقارنة إعدادات الميزات.
6. تقييم الأداء باستخدام RMSE (وأحياناً تقارير مشابهة للتصنيف حسب notebook).

## Results / Metrics
- RMSE:
  - ~`8461.95` و`4991.99` و`11049.66` و`10711.00` (حسب التجارب).
- Accuracy/F1: غير مُعتَمدة كمؤشر رئيسي لهذه المشكلة لأنها انحدار، لكن الـ notebook يذكر “classification-style” accuracy في بعض السيناريوهات.
- Confusion Matrix: غير مطبّقة (Regression).

## Usage
1. افتح `Scikit Learning.ipynb` في Jupyter:
   `jupyter notebook "Scikit Learning.ipynb"`
2. شغّل الخلايا بالترتيب.

## Authors / Credits
- Contributors: e-Machine-Learning-Projects (لا توجد أسماء محددة داخل notebooks/README).
