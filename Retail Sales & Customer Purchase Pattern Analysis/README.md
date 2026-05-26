# Retail Sales & Customer Purchase Pattern Analysis

## Overview

This notebook analyzes retail transactions from `Online Retail.csv` and uses clustering to identify customer purchase patterns.

الهدف من هذا المشروع هو استخراج أنماط شراء العملاء دون الاعتماد على “ملصقات” جاهزة: عبر تحليل معاملات البيع (transactions) نستخدم `KMeans` لاستخراج segments، و`IsolationForest` لاكتشاف المعاملات الشاذة. تم بناؤه لأجل تعليم كيفية تحويل بيانات تجارية إلى ميزات قابلة للتجميع، ثم قراءة النتائج كقيمة عمل حقيقية (مثل تحديد شرائح عملاء ذات سلوك مشابه أو معاملات غير معتادة). القيمة المضافة هي ربط خطوات المعالجة والتجميع بتطبيقات التسويق/التحسين (segmentation & anomaly detection).

## Summary Table

| Dataset             | Model             | Key Result                                                       |
| ------------------- | ----------------- | ---------------------------------------------------------------- |
| `Online Retail.csv` | `KMeans`          | Identified customer segments from transaction data.              |
| `Online Retail.csv` | `IsolationForest` | Detected anomalies in retail transactions.                       |
| `Online Retail.csv` | `PCA`             | Reduced feature dimensionality for clustering and visualization. |
| `Online Retail.csv` | `StandardScaler`  | Standardized numeric retail features.                            |
| `Online Retail.csv` | `MinMaxScaler`    | Scaled numeric data for algorithm performance.                   |

## Approach

- Clean the dataset by removing duplicates and missing values.
- Extract numeric sales features and scale them.
- Apply PCA for dimension reduction.
- Use KMeans clustering and IsolationForest for customer segmentation and anomaly detection.

## Project Structure

```
Retail Sales & Customer Purchase Pattern Analysis/
└── Online_Retail.ipynb
```

## Notes

The notebook combines retail analytics with unsupervised learning to reveal customer purchase segments.

## Model Results

| Model / Notebook                              | Key metrics (excerpt)                                                            |
| --------------------------------------------- | -------------------------------------------------------------------------------- |
| `KMeans` / clustering (`Online_Retail.ipynb`) | Silhouette Score reported: 0.9964 (noted in notebook for a chosen configuration) |
 
## Key Features
- استخدام تقنيات: تنظيف البيانات وإزالة القيم المكررة/الناقصة، ثم `StandardScaler`/`MinMaxScaler`، وبعدها `PCA` لتقليل الأبعاد و`KMeans` لتقسيم العملاء إلى segments، مع `IsolationForest` لاكتشاف الشذوذات (anomalies).
- دقة/جودة عالية في التجميع: `Silhouette Score` مذكور بقيمة `0.9964` (حسب README/اختيار configuration في الـ notebook).
- تكامل خارجي: لا يوجد تكامل خدمات (مثل Gemini API)؛ يعتمد على `Online Retail.csv` المحلي.

## Requirements / Installation
- Python: `3.9+`
- التثبيت:
  `pip install -r requirements.txt`
- البيانات: ضع `Online Retail.csv` داخل هذا المجلد أو عدّل مسار القراءة داخل `Online_Retail.ipynb`.

## Workflow / Pipeline
1. تحميل البيانات وتنظيفها (duplicates/missing).
2. استخراج ميزات رقمية لطلبات العملاء scale للميزات.
3. تطبيق PCA لتقليل الأبعاد للتجميع بشكل أفضل.
4. تدريب KMeans لاستخراج clusters/segments.
5. تشغيل IsolationForest لاكتشاف المعاملات الشاذة.

## Results / Metrics
- Silhouette Score: `0.9964` لـ KMeans (مذكور في README).
- Accuracy/F1: غير مطبّقة (تعلم غير مُشرف).
- Confusion Matrix: غير موجودة (لعدم وجود labels).

## Usage
1. افتح `Online_Retail.ipynb` في Jupyter:
   `jupyter notebook "Online_Retail.ipynb"`
2. شغّل الخلايا بالترتيب.

## Authors / Credits
- Contributors: e-Machine-Learning-Projects (لا توجد أسماء محددة داخل notebooks/README).
