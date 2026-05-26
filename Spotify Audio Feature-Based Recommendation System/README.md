# Spotify Audio Feature-Based Recommendation System

## Overview

This notebook analyzes audio features from `genres_v2.csv` and applies clustering to support recommendation-style grouping of songs.

الهدف من هذا المشروع هو تحويل فكرة “التوصية” إلى تقسيم عملي مبني على البيانات: بدلاً من نموذج توصيات معقّد، نستخدم التعلم غير المُشرف لتجميع الأغاني وفق تشابه خصائصها الصوتية. تم بناؤه كتمرين يعلّم كيفية التعامل مع بيانات وصفية (metadata/features)، ثم ترميزها وتطبيعها، وأخيراً استخدام `KMeans` لتوليد مجموعات يمكن استخدامها كنمط توصية/تجميع. القيمة المضافة هي منحك فهمًا لكيف ولماذا تتشكل مجموعات الأغاني، وما الذي يعنيه ذلك لقرارات مثل اقتراح Playlists أو تصنيف المحتوى.

## Summary Table

| Dataset         | Model          | Key Result                                         |
| --------------- | -------------- | -------------------------------------------------- |
| `genres_v2.csv` | `KMeans`       | Clustered songs based on audio feature similarity. |
| `genres_v2.csv` | `LabelEncoder` | Encoded genre labels for analysis.                 |
| `genres_v2.csv` | `MinMaxScaler` | Scaled audio features before clustering.           |

## Approach

- Clean and preprocess Spotify track metadata.
- Encode genre labels with LabelEncoder.
- Scale audio features with MinMaxScaler.
- Use the elbow method and KMeans clustering for song grouping.

## Project Structure

```
Spotify Audio Feature-Based Recommendation System/
└── spotify.ipynb
```

## Notes

The notebook explores audio feature clustering and recommendation-style segmentation.

## Model Results

| Model / Notebook                        | Key metrics (excerpt)                                                                                                                                  |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `KMeans` / clustering (`spotify.ipynb`) | No numeric evaluation metrics were extracted for this project in `metric_extraction.json`; notebook focuses on clustering examples and visualizations. |
 
## Key Features
- استخدام تقنيات: `pandas` للتنظيف، `LabelEncoder` لترميز الأنواع، `MinMaxScaler` لتوحيد نطاق السمات الصوتية، و`KMeans` للتجميع.
- قيمة عملية: يقدّم “Recommendation-style” grouping عبر تجميع الأغاني وفق تشابه الميزات (بدون نظام توصيات صريح يعتمد على feedback).
- تكامل خارجي: يعتمد على `genres_v2.csv` المحلي داخل المجلد (بدون تكامل خارجي مثل Gemini API).

## Requirements / Installation
- Python: `3.9+`
- التثبيت:
  `pip install -r requirements.txt`
- البيانات: ضع `genres_v2.csv` داخل هذا المجلد أو عدّل مسارات القراءة داخل `spotify.ipynb`.

## Workflow / Pipeline
1. تحميل ومعالجة بيانات `genres_v2.csv`.
2. ترميز فئات الأنواع بـ `LabelEncoder`.
3. ضبط/Scale للميزات الصوتية.
4. اختيار عدد العناقيد (Elbow method).
5. تطبيق KMeans وإنشاء visualizations للتجميع.

## Results / Metrics
- مقاييس رقمية: لا توجد أرقام مستخرجة نصياً ضمن README؛ التقييم يعتمد على جودة visualizations في notebook.
- Confusion Matrix: غير مطبّقة (تعلم غير مُشرف).

## Usage
1. افتح `spotify.ipynb` في Jupyter:
   `jupyter notebook "spotify.ipynb"`
2. شغّل الخلايا بالترتيب.

## Authors / Credits
- Contributors: e-Machine-Learning-Projects (لا توجد أسماء محددة داخل notebooks/README).
