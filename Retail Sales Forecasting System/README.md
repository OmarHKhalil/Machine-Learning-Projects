# Retail Sales Forecasting System

## Overview

This project forecasts store sales using the Rossmann retail dataset. It demonstrates feature engineering, encoding, and XGBoost regression modeling.

الهدف من المشروع هو تحويل بيانات مبيعات متسلسلة إلى تنبؤات قابلة للاستخدام في قرارات التخطيط للمخزون والميزانية. تم بناؤه كتمرين عملي لتعلّم “كيف نفكّر” في بيانات التنبؤ: هندسة ميزات الوقت، ترميز المتغيرات الفئوية، ثم استخدام نماذج قوية مثل `XGBoost` مع تقييم يعتمد على RMSE. القيمة المضافة هنا هي تقديم سير عمل متكامل من تنظيف البيانات إلى تقييم موثوق يمكن تكراره على مشاكل forecasting مشابهة.

## Summary Table

| Dataset                                                                                             | Model           | Key Result                                                      |
| --------------------------------------------------------------------------------------------------- | --------------- | --------------------------------------------------------------- |
| `rossmann-store-sales/train.csv`, `rossmann-store-sales/test.csv`, `rossmann-store-sales/store.csv` | `XGBRegressor`  | Achieved best RMSE around 1217.19 and CV RMSE around 1740–1750. |
| `rossmann-store-sales/train.csv`, `rossmann-store-sales/test.csv`, `rossmann-store-sales/store.csv` | `OneHotEncoder` | Encoded categorical store and date-related features.            |
| `rossmann-store-sales/train.csv`, `rossmann-store-sales/test.csv`, `rossmann-store-sales/store.csv` | `MinMaxScaler`  | Scaled numeric features for modeling.                           |

## Approach

- Load and merge Rossmann train, test, and store datasets.
- Parse dates and engineer time-related features.
- Encode categorical variables and scale numeric features.
- Train XGBoost regression and tune hyperparameters.
- Evaluate performance with RMSE.

## Project Structure

```
Retail Sales Forecasting System/
└── XGBoost.ipynb
```

## Notes

This notebook is a Kaggle-style forecasting workflow for retail store sales.

## Model Results

| Model / Notebook                 | Key metrics (excerpt)                                                                                                                                                                         |
| -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `XGBRegressor` (`XGBoost.ipynb`) | Reported Mean RMSE (one config): 2393.39; cross-val RMSE examples include Mean RMSE ≈ 1745.38 (KFold) and cross_val_predict RMSE ≈ 1751.20; many train/validation RMSEs logged per experiment |
 
## Key Features
- استخدام تقنيات: `pandas` لمعالجة ودمج بيانات Rossmann، `OneHotEncoder` لترميز السمات الفئوية، `MinMaxScaler` لتسوية القيم الرقمية، و`XGBoost (XGBRegressor)` لتحسين نموذج الانحدار.
- دقة/أداء قوي: أفضل RMSE ~`1217.19` مع متوسط RMSE عبر KFold ~`1745.38` (حسب نتائج notebook).
- تكامل خارجي: يعتمد أسلوب Kaggle/OpenDatasets داخل الـ notebook لتحضير/تحميل البيانات (يظهر ذلك عند استخدام `opendatasets`).

## Requirements / Installation
- Python: `3.9+`
- `requirements.txt`: يوجد ملف مشترك في جذر المستودع.
- التثبيت:
  `pip install -r requirements.txt`
- البيانات: أضف ملفات Rossmann المطلوبة ضمن مجلد مثل `rossmann-store-sales/` (مثل `train.csv`, `test.csv`, `store.csv`) أو عدّل مسارات الملفات داخل الـ notebook.

## Workflow / Pipeline
1. تحميل ودمج بيانات التدريب/الاختبار/المتاجر من Rossmann.
2. معالجة التواريخ واستخراج ميزات زمنية (time-related features).
3. ترميز السمات الفئوية وتوسيع/تطبيع الميزات الرقمية.
4. تقسيم البيانات إلى تدريب/اختبار (مثال: 85% تدريب و15% اختبار).
5. تدريب `XGBRegressor` وتحسين hyperparameters.
6. التقييم باستخدام RMSE.

## Results / Metrics
- RMSE عبر التحقق: `Mean RMSE ≈ 1745.38` و`RMSE cross_val_predict ≈ 1751.20`.
- أفضل تجربة (Grid Search): `RMSE ≈ 1217.19`.
- Confusion Matrix: غير مطبّق لأن هذه مهمة `Regression`.

## Usage
1. افتح الملف `XGBoost.ipynb` باستخدام Jupyter:
   `jupyter notebook "XGBoost.ipynb"`
2. شغّل الخلايا بالترتيب (من الأعلى للأسفل).

## Authors / Credits
- Contributors: e-Machine-Learning-Projects (لا توجد أسماء محددة داخل notebooks/README).
