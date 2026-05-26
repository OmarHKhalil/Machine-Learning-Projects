# Unsupervised Learning and Recommendations

## Overview

This tutorial notebook applies unsupervised learning techniques to the Iris dataset. It is designed to teach clustering and dimensionality reduction.

الهدف من هذا الـ notebook هو فهم “ماذا نعرف بدون تسميات” عبر التعلم غير المُشرف: أي كيف يمكن استخراج بنية البيانات (clusters) وتصورها حتى عندما لا نملك labels جاهزة. تم بناؤه كتعليمي يربط بين خطوات مثل `MinMaxScaler` و`KMeans/DBSCAN` ثم تقليل الأبعاد بـ `PCA` و`t-SNE`. القيمة المضافة هي تعليمك ليس فقط تشغيل الخوارزمية، بل اختيار معلماتها وفهم الفرق بين الأساليب عندما تختلف طبيعة البيانات.

## Summary Table

| Dataset      | Model          | Key Result                                   |
| ------------ | -------------- | -------------------------------------------- |
| Iris dataset | `KMeans`       | Clustered iris samples into groups.          |
| Iris dataset | `DBSCAN`       | Demonstrated density-based clustering.       |
| Iris dataset | `PCA`          | Reduced dimensionality for visualization.    |
| Iris dataset | `TSNE`         | Visualized cluster structure in 2D.          |
| Iris dataset | `MinMaxScaler` | Scaled numeric features prior to clustering. |

## Approach

- Scale numeric features with MinMaxScaler.
- Apply KMeans and DBSCAN clustering.
- Use PCA and t-SNE for dimensionality reduction and visualization.

## Project Structure

```
Unsupervised Learning and Recommendations/
└── Unsupervised_Learning&Recommendations.ipynb
```

## Notes

This notebook is an educational resource on unsupervised learning and cluster visualization.

## Model Results

| Model / Notebook                                         | Key metrics (excerpt)                                                                           |
| -------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `KMeans` (`Unsupervised_Learning&Recommendations.ipynb`) | Notebook config: `n_clusters=8` shown in init; no explicit numeric evaluation metrics extracted |
| `PCA` / `TSNE`                                           | Used for visualization; no numeric metrics extracted                                            |
 
## Key Features
- استخدام تقنيات: `MinMaxScaler` لتوحيد نطاق الميزات، ثم `KMeans` و`DBSCAN` لاكتشاف clusters، بالإضافة إلى `PCA` و`t-SNE` لتصور الهيكل غير الخطي.
- دقة/جودة الأداء: في التعلم غير الموجه لا توجد Accuracy/F1 تقليدية؛ التقييم هنا يعتمد على جودة التجميع بصرياً ومعاني clusters (وقد يظهر Silhouette score داخل notebook حسب الإعدادات).
- تكامل خارجي: لا يعتمد على خدمات مثل Gemini API؛ يستخدم مجموعة `Iris` المتاحة عبر `scikit-learn`.

## Requirements / Installation
- Python: `3.9+`
- التثبيت:
  `pip install -r requirements.txt`
- البيانات: يستخدم `sklearn.datasets.load_iris` (لا تحتاج CSV خارجية).

## Workflow / Pipeline
1. تحميل بيانات Iris.
2. Scale للميزات باستخدام MinMaxScaler.
3. تطبيق KMeans/DBSCAN لاستخراج مجموعات (clusters).
4. تقليل الأبعاد بـ PCA ثم تصور بـ t-SNE.

## Results / Metrics
- Clustering quality: `n_clusters=8` لـ KMeans (حسب config في notebook).
- Silhouette / مقاييس رقمية: غير مستخرجة نصياً من README، لكن قد تكون مطروحة داخل notebook.
- Confusion Matrix: غير مطبّقة (تعلم غير موجه).

## Usage
1. افتح `Unsupervised_Learning&Recommendations.ipynb` في Jupyter:
   `jupyter notebook "Unsupervised_Learning&Recommendations.ipynb"`
2. شغّل الخلايا بالترتيب.

## Authors / Credits
- Contributors: e-Machine-Learning-Projects (لا توجد أسماء محددة داخل notebooks/README).
