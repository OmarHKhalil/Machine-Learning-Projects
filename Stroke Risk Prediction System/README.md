# Stroke Risk Prediction System

## Overview

This notebook builds a stroke risk classification pipeline using the `healthcare.csv` dataset. It prepares medical and demographic features to estimate stroke likelihood.

الهدف من المشروع هو تقديم نموذج تصنيف يقدّر احتمال الإصابة بالسكتة (stroke) بناءً على سمات طبية وديموغرافية، مع التركيز على أن الجودة تبدأ من تجهيز البيانات وليس من اختيار الخوارزمية فقط. تم بناؤه كحالة تعليمية واقعية لتعلم كيفية التعامل مع قيم مفقودة، تنظيف الأعمدة، ثم ترميز السمات وتدريب مصنفات مع تقييم بـ `confusion matrix` و`classification report`. القيمة المضافة هي إيضاح كيف يمكن تحويل بيانات خام إلى إشارات قابلة للتعلم، مع فهم حدود النموذج وأثر عدم توازن الفئات في مهام طبية حساسة.

## Summary Table

| Dataset          | Model                  | Key Result                                                               |
| ---------------- | ---------------------- | ------------------------------------------------------------------------ |
| `healthcare.csv` | `KNeighborsClassifier` | Achieved classification accuracy around 96% on selected evaluation runs. |
| `healthcare.csv` | `LabelEncoder`         | Encoded categorical medical and demographic fields.                      |
| `healthcare.csv` | `MinMaxScaler`         | Scaled numeric features for classifier performance.                      |

## Approach

- Fill missing BMI values with mean imputation.
- Remove duplicate and redundant columns.
- Encode categorical features and apply MinMaxScaler.
- Train classification models and evaluate with confusion matrices and classification reports.

## Project Structure

```
Stroke Risk Prediction System/
└── Stroke_Risk_Prediction.ipynb
```

## Notes

This notebook illustrates medical risk modeling with careful preprocessing and classifier evaluation.

## Model Results

| Model / Notebook                                                            | Key metrics (excerpt)                                                                                                                                                                                                                      |
| --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `KNeighborsClassifier` / other classifiers (`Stroke_Risk_Prediction.ipynb`) | Reported accuracies across runs: ~91%–96%; some small-sample runs show low precision/recall for the positive (stroke) class (precision/recall ~0.10–0.28), while other runs (different splits/labels) report balanced per-class F1 ≈ 0.96. |
 
## Key Features
- استخدام تقنيات: معالجة القيم المفقودة (BMI) عبر mean imputation، ثم ترميز الفئات بـ `LabelEncoder`، وتطبيع الميزات بـ `MinMaxScaler`، وتدريب نماذج تصنيف (مثل `KNeighborsClassifier` مع نماذج أخرى).
- دقة عالية: نطاق `accuracy` تقريباً ~`91%–96%` مع ملاحظة حساسية النتائج لعدد العينات/تقسيم البيانات.
- تقييم شامل: يعرض `classification reports` و`confusion matrix` لفهم أداء التصنيف على فئة “stroke”.
- تكامل خارجي: لا يوجد تكامل مثل Gemini API؛ يعتمد على `healthcare.csv` المحلي.

## Requirements / Installation
- Python: `3.9+`
- التثبيت:
  `pip install -r requirements.txt`
- البيانات: ضع `healthcare.csv` داخل هذا المجلد أو عدّل مسارات القراءة داخل الـ notebook.

## Workflow / Pipeline
1. تنظيف البيانات وإزالة الأعمدة/النسخ المكررة.
2. تعبئة BMI المفقود (mean imputation).
3. ترميز الفئات وتحويلها إلى قيم رقمية (`LabelEncoder`/ترميز متعدد).
4. تطبيع الميزات (`MinMaxScaler`).
5. تقسيم البيانات إلى تدريب/اختبار (مثال: 85% تدريب و15% اختبار).
6. تدريب نماذج التصنيف وتقييمها عبر Accuracy/F1 و`confusion matrix`.

## Results / Metrics
- Accuracy: ~`91%–96%` (حسب runs/اختيارات split).
- Precision/Recall (فئة stroke): ~`0.10–0.28` في بعض runs (وفق README).
- F1-Score: يمكن أن يصل إلى ~`0.96` (حسب README في runs أخرى).
- Confusion Matrix: موجودة ضمن قسم التقييم في الـ notebook.

## Usage
1. افتح `Stroke_Risk_Prediction.ipynb` في Jupyter:
   `jupyter notebook "Stroke_Risk_Prediction.ipynb"`
2. شغّل الخلايا بالترتيب.

## Authors / Credits
- Contributors: e-Machine-Learning-Projects (لا توجد أسماء محددة داخل notebooks/README).
