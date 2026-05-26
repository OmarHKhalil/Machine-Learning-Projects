# Titanic Survival Predictive Analytics

## Overview

This notebook predicts passenger survival on the Titanic using `Titanic-Dataset.csv`. It covers data cleaning, feature engineering, and classifier comparison.

الهدف من المشروع هو بناء نموذج يتنبأ باحتمالية نجاة الركاب اعتماداً على السمات المتاحة، مع إظهار أن النجاح لا يأتي فقط من “الخوارزمية” بل من تجهيز البيانات جيداً. تم بناؤه لتقديم تجربة تعليمية لكيفية تطبيق خطوات أساسية (تنظيف، هندسة ميزات مشتقة مثل `family size`، ترميز One-Hot) ثم مقارنة عدة مصنفات لمعرفة ما الذي يعمل بشكل أفضل. القيمة المضافة هي أن القارئ يرى سبب التحسن من منظور هندسة الميزات وتقييم الأداء وليس مجرد رقم دقة.

## Summary Table

| Dataset               | Model                    | Key Result                                                 |
| --------------------- | ------------------------ | ---------------------------------------------------------- |
| `Titanic-Dataset.csv` | `DecisionTreeClassifier` | Provided a decision tree baseline for survival prediction. |
| `Titanic-Dataset.csv` | `RandomForestClassifier` | Compared ensemble performance with tree-based modeling.    |
| `Titanic-Dataset.csv` | `LogisticRegression`     | Offered a linear classification approach.                  |
| `Titanic-Dataset.csv` | `KNeighborsClassifier`   | Used KNN to predict survival from passenger features.      |
| `Titanic-Dataset.csv` | `OneHotEncoder`          | Encoded categorical passenger features.                    |
| `Titanic-Dataset.csv` | `MinMaxScaler`           | Scaled numeric features for classification.                |

## Approach

- Impute missing age and embarked values.
- Drop irrelevant columns such as PassengerId, Name, and Cabin.
- Create derived features such as family size.
- One-hot encode categorical variables.
- Train and compare Decision Tree, Logistic Regression, and Random Forest classifiers.

## Project Structure

```
Titanic Survival Predictive Analytics/
└── Titanic_Survival_Predictive_Analytics.ipynb
```

## Notes

This is a classic classification project for survival prediction with feature engineering and model comparison.

## Model Results

| Model / Notebook                                                         | Key metrics (excerpt)                                  |
| ------------------------------------------------------------------------ | ------------------------------------------------------ |
| `DecisionTreeClassifier` (`Titanic_Survival_Predictive_Analytics.ipynb`) | Reported accuracies in runs: ~78%–81%; example: 78.36% |
| `LogisticRegression` (`Titanic_Survival_Predictive_Analytics.ipynb`)     | Reported accuracy: ~79.85%                             |
| `RandomForestClassifier` (`Titanic_Survival_Predictive_Analytics.ipynb`) | Reported accuracy: ~81.72%; weighted avg F1 ≈ 0.81     |
| `Ensemble Methods`                                                       | Reported accuracy: ~80.97%                             |
 
## Key Features
- استخدام تقنيات: تنظيف البيانات (Imputation)، استخراج ميزات (مثل `family size`)، One-Hot Encoding، ثم مقارنة عدة نماذج تصنيف (Decision Tree/Logistic Regression/Random Forest/KNN).
- دقة عالية: أفضل نماذج تصل إلى ~81.72% accuracy (Random Forest) مع weighted avg F1 ≈ 0.81.
- تكامل خارجي: يعتمد على ملف `Titanic-Dataset.csv` داخل المجلد (لا يوجد تكامل خدمات مثل Gemini API).

## Requirements / Installation
- Python: `3.9+`
- التثبيت:
  `pip install -r requirements.txt`
- البيانات: ضع `Titanic-Dataset.csv` داخل هذا المجلد، أو عدّل مسار القراءة داخل الـ notebook إن كانت البيانات بمكان آخر.

## Workflow / Pipeline
1. تنظيف القيم المفقودة واستبعاد الأعمدة غير المفيدة.
2. إنشاء ميزات مشتقة (مثل حجم العائلة).
3. One-hot encode للسمات الفئوية وتطبيع/تجهيز السمات الرقمية.
4. تقسيم البيانات إلى تدريب/اختبار (مثال: 85% تدريب و15% اختبار).
5. تدريب ومقارنة النماذج.
6. تقييم باستخدام Accuracy و F1 (وغالباً `confusion matrix`/`classification report` داخل notebook).

## Results / Metrics
- Accuracy:
  - Decision Tree: ~78%–81% (مثال 78.36%)
  - Logistic Regression: ~79.85%
  - Random Forest: ~81.72%
  - Ensemble: ~80.97%
- F1-Score: weighted avg F1 ≈ 0.81 (Random Forest).
- Confusion Matrix: يتم استخدامها في تقييم التصنيف داخل الـ notebook (إن كانت الرسومات مطروحة في الخلايا).

## Usage
1. افتح `Titanic_Survival_Predictive_Analytics.ipynb` في Jupyter:
   `jupyter notebook "Titanic_Survival_Predictive_Analytics.ipynb"`
2. شغّل الخلايا من الأعلى للأسفل.

## Authors / Credits
- Contributors: e-Machine-Learning-Projects (لا توجد أسماء محددة داخل notebooks/README).
