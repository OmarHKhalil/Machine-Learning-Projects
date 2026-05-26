# Taxi Fare Prediction System

## Overview

This notebook predicts NYC taxi fares using a Kaggle dataset. It includes data preparation, datetime parsing, and regression model evaluation for fare estimation.

الهدف من المشروع هو بناء خط أنابيب انحدار (Regression) كامل لتقدير أجور التاكسي من بيانات واقعية، مع التركيز على الجزء الذي يصنع الفرق عادةً: استخراج ميزات الزمن والموقع وربطها بالقيم المستهدفة. تم بناؤه كتمرين لفهم كيف تؤثر المعالجة المسبقة (مثل parsing للتاريخ/الوقت وتهيئة المتغيرات العددية) على RMSE/MAE. القيمة المضافة هي توفير مثال عملي لكيفية تحويل بيانات “فوضوية” إلى ميزات جاهزة للتعلم ثم مقارنة نماذج متعددة.

## Summary Table

| Dataset               | Model                   | Key Result                                             |
| --------------------- | ----------------------- | ------------------------------------------------------ |
| NYC taxi fare dataset | `LinearRegression`      | Provided a baseline fare prediction model.             |
| NYC taxi fare dataset | `RandomForestRegressor` | Modeled non-linear relationships in fare data.         |
| NYC taxi fare dataset | `XGBRegressor`          | Used gradient boosting regression for fare estimation. |
| NYC taxi fare dataset | `StandardScaler`        | Standardized continuous input features.                |

## Approach

- Download and load the Kaggle taxi fare dataset.
- Parse pickup datetime and select relevant geographic, temporal, and passenger features.
- Prepare training and test datasets with proper dtypes.
- Train regression models and compare their performance.

## Project Structure

```
Taxi Fare Prediction System/
└── nyc-taxi-fare-prediction-filled.ipynb
```

## Notes

This notebook covers regression modeling for taxi fare prediction and Kaggle dataset preparation.

## Model Results

| Model / Notebook                                                  | Key metrics (excerpt)                                                                                              |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `RandomForestRegressor` (`nyc-taxi-fare-prediction-filled.ipynb`) | Model printed with params: `max_depth=10`, `n_estimators=50` (report shows trained RandomForestRegressor instance) |
| `LinearRegression` / `XGBRegressor`                               | Baseline and boosted models included; no RMSE/MAE values extracted in metric_extraction.json for these runs        |
 
## Key Features
- استخدام تقنيات: `pandas` لتجهيز بيانات NYC Taxi، تحليل التاريخ/الزمن، ثم نمذجة انحدار باستخدام `LinearRegression` و`RandomForestRegressor` و`XGBRegressor`.
- دقة/أداء: مقارنة بين خط أساس (Linear Regression) ونماذج غير خطية (Random Forest وXGBoost) لتقدير أجرة التاكسي (التقييم عادةً عبر RMSE/MAE داخل notebook).
- تكامل خارجي: يعتمد على تنزيل مجموعة Kaggle عبر `opendatasets` داخل الـ notebook (إن تم تفعيلها).

## Requirements / Installation
- Python: `3.9+`
- التثبيت:
  `pip install -r requirements.txt`
- البيانات: ضع مجموعة بيانات NYC taxi داخل المجلد أو عدّل مسار القراءة في notebook (تعتمد على Kaggle/OpenDatasets في التحميل).

## Workflow / Pipeline
1. تنزيل/تحميل بيانات NYC taxi.
2. استخراج ميزات زمنية من pickup datetime وتحضير السمات الجغرافية/الركاب.
3. تقسيم البيانات إلى تدريب/اختبار (مثال: 85% تدريب و15% اختبار).
4. تدريب نماذج الانحدار (Linear/RF/XGBoost) ومقارنة الأداء.
5. تقييم باستخدام مقاييس انحدار (مثل RMSE/MAE حسب notebook).

## Results / Metrics
- Metrics: README لا يذكر أرقام RMSE/MAE بشكل صريح (غير مستخرجة نصياً)، لكن الـ notebook يحتوي قسم تقييم نماذج الانحدار.
- Confusion Matrix: غير مطبّقة (Regression).

## Usage
1. افتح `nyc-taxi-fare-prediction-filled.ipynb` في Jupyter:
   `jupyter notebook "nyc-taxi-fare-prediction-filled.ipynb"`
2. شغّل الخلايا من الأعلى للأسفل.

## Authors / Credits
- Contributors: e-Machine-Learning-Projects (لا توجد أسماء محددة داخل notebooks/README).
