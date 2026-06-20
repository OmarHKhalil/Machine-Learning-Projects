# Stroke Risk Prediction System

## 📋 Overview

A comprehensive machine learning project that predicts stroke risk based on medical and demographic features. This project demonstrates a complete end-to-end machine learning pipeline including data preprocessing, feature engineering, model building, and handling class imbalance using advanced techniques like SMOTE.

## 🎯 Project Objectives

This project serves as a practical educational example of:

- **Data Preprocessing:** Handling missing values, outlier detection and removal
- **Feature Engineering:** Categorical encoding, feature scaling, and normalization
- **Model Development:** Training and hyperparameter tuning of classification models
- **Class Imbalance Handling:** Implementing over-sampling and SMOTE techniques
- **Model Evaluation:** Comprehensive performance metrics and visualization

## 📊 Dataset

- **Source:** `healthcare.csv`
- **Records:** ~5,000+ patient records
- **Features:** Medical indicators and demographic information
  - Age, Gender, Hypertension, Heart Disease, Ever Married
  - Work Type, Residence Type, Average Glucose Level, BMI, Smoking Status
  - **Target Variable:** Stroke (Binary: 0 = No Stroke, 1 = Stroke)

## 🔄 Project Workflow

### 1. **Data Exploration**

- Load and inspect the dataset
- Analyze data types and shapes
- Identify missing values

### 2. **Data Preprocessing**

- **Missing Value Handling:** Fill missing BMI values using mean imputation
- **Outlier Detection:** Use box plots and 2-sigma method to identify outliers
- **Outlier Removal:** Remove age outliers beyond 2 standard deviations
- **Duplicate Removal:** Eliminate duplicate records

### 3. **Feature Engineering**

- **Label Encoding:** Encode categorical variables (Gender, Ever Married, Residence Type)
- **One-Hot Encoding:** Convert Smoking Status to dummy variables
- **Feature Selection:** Remove non-predictive columns (ID, Work Type)

### 4. **Feature Scaling**

- Apply **MinMaxScaler** to normalize all features to [0, 1] range
- Ensures all features contribute equally to model training

### 5. **Exploratory Analysis**

- Generate correlation matrix heatmap
- Analyze feature relationships with target variable
- Visualize class distribution

### 6. **Model Development**

#### K-Nearest Neighbors (KNN) Classifier

- Train models with different k values (k=1, 3, 5, ..., 19)
- Perform error rate analysis to find optimal k value
- Best performance at **k=3**

#### Train-Test Split

- 80% training data, 20% testing data
- Stratified split to maintain class distribution

### 7. **Handling Class Imbalance**

The dataset suffers from **class imbalance** (more stroke-free cases than stroke cases). Two techniques are implemented:

#### a) **Random Over-Sampling**

- Randomly duplicate minority class samples
- Balance class distribution before training

#### b) **SMOTE (Synthetic Minority Over-sampling Technique)**

- Generate synthetic samples for minority class
- More sophisticated than random over-sampling
- Better prevents overfitting

### 8. **Model Evaluation**

Performance metrics include:

- **Confusion Matrix:** True Positives, True Negatives, False Positives, False Negatives
- **Classification Report:** Precision, Recall, F1-Score for each class
- **Accuracy:** Overall correctness of predictions
- **Precision:** Correctly identified positive cases (important for medical decisions)
- **Recall:** Ability to find all positive cases (critical in medical screening)

## 📈 Key Results

| Metric                  | Value                          |
| ----------------------- | ------------------------------ |
| **Best K Value**        | 3                              |
| **Accuracy (KNN, k=3)** | ~95-96%                        |
| **Model Type**          | K-Nearest Neighbors Classifier |
| **Preprocessing**       | MinMaxScaler Normalization     |
| **Class Balancing**     | SMOTE & Random Over-Sampling   |

## 🛠️ Technologies Used

- **Python 3.9+**
- **pandas:** Data manipulation and analysis
- **NumPy:** Numerical computations
- **scikit-learn:** Machine learning algorithms and preprocessing
- **imbalanced-learn:** SMOTE and over-sampling techniques
- **Matplotlib:** Data visualization
- **Seaborn:** Statistical data visualization

## 📁 Project Structure

```
Stroke Risk Prediction System/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── Stroke_Risk_Prediction.ipynb       # Main notebook with complete pipeline
└── healthcare.csv                     # Dataset (input)
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- Jupyter Notebook or JupyterLab

### Installation

1. **Clone or download the project**

   ```bash
   cd "Stroke Risk Prediction System"
   ```

2. **Create a virtual environment (optional but recommended)**

   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the notebook**
   ```bash
   jupyter notebook Stroke_Risk_Prediction.ipynb
   ```

## 📝 Notebook Walkthrough

The Jupyter notebook includes 48 cells organized as follows:

1. **Cells 1-3:** Data Loading and Initial Exploration
2. **Cells 4-5:** Missing Value Analysis and Imputation
3. **Cells 6-8:** Outlier Detection and Removal
4. **Cells 9-18:** Feature Engineering and Encoding
5. **Cells 19-21:** Feature Scaling with MinMaxScaler
6. **Cells 22-23:** Correlation Analysis
7. **Cells 24-31:** KNN Model Training and Evaluation (k=1, k=3)
8. **Cells 32-36:** Hyperparameter Tuning (k=1 to 19)
9. **Cells 37-40:** Class Distribution Analysis
10. **Cells 41-44:** Random Over-Sampling Implementation
11. **Cells 45-48:** SMOTE Implementation and Final Evaluation

## 🎓 Learning Outcomes

After exploring this project, you will understand:
✅ Complete ML pipeline from raw data to predictions
✅ Data preprocessing and cleaning techniques
✅ Feature encoding and scaling best practices
✅ Hyperparameter tuning and model optimization
✅ Class imbalance handling techniques (Over-sampling, SMOTE)
✅ Model evaluation metrics and interpretation
✅ Real-world medical data modeling challenges

## 💡 Key Insights

1. **Data Quality is Crucial:** 50% of model performance depends on preprocessing
2. **Class Imbalance Matters:** In medical datasets, minority class accuracy is critical
3. **Feature Scaling:** Improves KNN and distance-based algorithms significantly
4. **SMOTE is Effective:** Better than simple over-sampling for synthetic data generation
5. **Hyperparameter Tuning:** Even simple models (KNN) benefit from optimization

## ⚠️ Important Notes

- **Missing Data:** BMI values were imputed using mean; consider domain-specific imputation methods
- **Class Imbalance:** Original dataset is heavily imbalanced (5-10:1 ratio); use appropriate metrics
- **Medical Disclaimer:** This model is for educational purposes only, not for clinical decision-making
- **Data Privacy:** The `healthcare.csv` dataset should be used responsibly and in compliance with privacy regulations

## 📚 References & Inspiration

- Scikit-learn Documentation: https://scikit-learn.org/
- Imbalanced-learn Documentation: https://imbalanced-learn.org/
- KNN Algorithm: https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm
- SMOTE Technique: https://arxiv.org/abs/1106.1813

## 📧 Questions & Support

For questions about the implementation or methodology, refer to:

- Code comments in the Jupyter notebook
- Scikit-learn and imbalanced-learn official documentation
- Stack Overflow for common machine learning questions

## 📜 License

This project is provided for educational purposes. Feel free to use, modify, and distribute as needed for learning.

## Author

- **Developed by:** Omar Hafez Khalil
- **GitHub:** [OmarHKhalil](https://github.com/OmarHKhalil)
- **LinkedIn:** [Omar Khalil](https://www.linkedin.com/in/omar-khalil-55a674281)

---

**Last Updated:** June 2026
**Project Type:** Machine Learning Classification  
**Difficulty Level:** Intermediate
