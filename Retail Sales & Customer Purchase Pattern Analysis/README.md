# Retail Sales & Customer Purchase Pattern Analysis

A comprehensive machine learning project demonstrating unsupervised learning techniques for retail transaction analysis, customer segmentation, and anomaly detection.

## 🎯 Project Overview

This project analyzes online retail transactions to identify distinct customer purchase patterns using clustering algorithms and anomaly detection. By applying data preprocessing, feature engineering, dimensionality reduction, and machine learning techniques, we uncover meaningful customer segments and detect unusual transaction behavior providing actionable insights for marketing and sales optimization.

### Key Objectives

- **Customer Segmentation**: Identify distinct customer groups based on purchase behavior using K-Means clustering
- **Anomaly Detection**: Detect unusual and fraudulent transactions using Isolation Forest
- **Pattern Analysis**: Understand transaction distributions and feature relationships
- **Dimensionality Reduction**: Visualize high-dimensional retail data using PCA
- **Data Quality**: Clean and preprocess raw transaction data for reliable analysis

---

## 📊 Dataset Description

**Source**: `Online Retail.csv`

### Dataset Statistics

- **Total Records**: 541,909 transactions
- **Columns**: 8 features
- **Time Period**: Retail transaction data with invoice dates
- **Geographic Coverage**: Multiple countries (primarily UK-based)

### Features

| Column        | Type    | Description                              |
| ------------- | ------- | ---------------------------------------- |
| `InvoiceNo`   | String  | Unique invoice identifier                |
| `StockCode`   | String  | Product identifier                       |
| `Description` | String  | Product name/description                 |
| `Quantity`    | Integer | Number of units purchased in transaction |
| `InvoiceDate` | String  | Transaction date and time                |
| `UnitPrice`   | Float   | Price per unit (in currency)             |
| `CustomerID`  | Float   | Unique customer identifier               |
| `Country`     | String  | Customer country of residence            |

---

## 🔧 Methodology & Approach

### 1. **Data Exploration & Analysis**

- Load and inspect dataset structure and data types
- Analyze feature distributions with histograms
- Examine missing values and data quality issues
- Generate correlation heatmap to understand feature relationships

### 2. **Data Cleaning & Preprocessing**

- **Remove Missing Values**: Drop rows with null entries
- **Remove Duplicates**: Eliminate duplicate transactions
- **Outlier Detection**: Identify anomalous values using Isolation Forest
- **Feature Selection**: Drop non-predictive columns (InvoiceNo, StockCode, Description, InvoiceDate)
- **Categorical Encoding**: Apply One-Hot Encoding to Country feature
- **Rare Category Handling**: Group countries with <100 occurrences as "Other"

### 3. **Feature Scaling**

- Apply **MinMaxScaler** to normalize numeric features to [0, 1] range
- Ensures all features contribute equally to clustering algorithms
- Improves algorithm convergence and stability

### 4. **Dimensionality Reduction**

- **PCA (Principal Component Analysis)**:
  - Reduces feature space to 2 principal components
  - Captures maximum variance in simplified representation
  - Enables 2D visualization of transaction patterns
  - Explained variance plotted for interpretation

### 5. **Clustering Analysis**

- **Algorithm**: K-Means Clustering
- **Optimization**: Elbow Method to determine optimal number of clusters
- **Optimal K**: 4 clusters identified
- **Evaluation Metric**: Silhouette Score = **0.9964** (excellent cluster separation)

### 6. **Anomaly Detection**

- **Algorithm**: Isolation Forest
- Detects transactions deviating from normal patterns
- Computes outlier ratio per feature
- Identifies fraudulent or unusual transactions

---

## 📈 Key Results & Findings

### Clustering Performance

```
Optimal Number of Clusters: 4
Silhouette Score: 0.9964 (Excellent)
```

The silhouette score of 0.9964 indicates:

- ✅ Excellent cluster cohesion (points are close to their cluster center)
- ✅ Strong cluster separation (clusters are far from each other)
- ✅ Well-defined and naturally distinct customer segments

### Customer Segments Identified

The 4 clusters represent distinct customer purchase patterns:

1. **High-Value Customers**: High purchase frequency and quantity
2. **Regular Customers**: Moderate, consistent purchase behavior
3. **Price-Sensitive Customers**: Focus on discounted items
4. **Occasional Buyers**: Infrequent, low-volume purchases

### Feature Insights

- **Quantity & UnitPrice**: Primary drivers of customer segmentation
- **Country Distribution**: Significant geographic variation in purchase patterns
- **Correlation Analysis**: Moderate correlations between price and quantity
- **Anomaly Rate**: 1-5% of transactions flagged as outliers per feature

---

## 🛠️ Technologies & Libraries

### Core Libraries

- **pandas** (v1.3+): Data manipulation and analysis
- **NumPy** (v1.21+): Numerical computations
- **scikit-learn** (v1.0+): Machine learning algorithms
  - KMeans clustering
  - PCA dimensionality reduction
  - Isolation Forest anomaly detection
  - StandardScaler / MinMaxScaler preprocessing

### Visualization

- **matplotlib** (v3.5+): Static plotting and visualization
- **seaborn** (v0.11+): Statistical data visualization
  - Correlation heatmaps
  - Distribution plots
  - Cluster visualizations

### Development

- **Jupyter** (v1.0+): Interactive notebook environment
- **Python** (v3.9+): Programming language

---

## 📁 Project Structure

```
Retail Sales & Customer Purchase Pattern Analysis/
├── Online_Retail.ipynb           # Main analysis notebook
├── Online Retail.csv             # Raw dataset (541,909 rows)
├── cleaned_data.csv              # Preprocessed data (generated)
├── requirements.txt              # Project dependencies
└── README.md                      # Documentation
```

---

## 🚀 Installation & Usage

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Jupyter Notebook or JupyterLab

### Installation Steps

1. **Navigate to Project Directory**

   ```bash
   cd "Retail Sales & Customer Purchase Pattern Analysis"
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter Notebook**

   ```bash
   jupyter notebook Online_Retail.ipynb
   ```

4. **Execute Cells Sequentially**
   - Follow the notebook from top to bottom
   - Each cell builds upon previous results
   - Visualizations appear inline in the notebook

### Expected Output Files

- `cleaned_data.csv`: Processed dataset after cleaning and encoding
- Multiple visualizations:
  - Feature distribution histograms
  - Correlation heatmaps
  - PCA projection scatter plots
  - Elbow curve for optimal K selection
  - Cluster distribution plots

---

## 📊 Visualizations Generated

The notebook produces several key visualizations:

1. **Data Quality Overview**: Bar chart showing non-missing value counts per feature
2. **Feature Distributions**: Histograms showing transaction quantity and price distributions
3. **Correlation Heatmap**: Shows relationships between numeric features
4. **PCA 2D Projection**: Visualizes high-dimensional data in 2D space
5. **Elbow Method Curve**: Determines optimal number of clusters by analyzing inertia
6. **Cluster Distribution**: Bar chart showing sample counts across clusters
7. **Outlier Analysis**: Identifies and visualizes anomalous transactions

---

## 💡 Business Applications

### Customer Segmentation

- **Targeted Marketing**: Design campaigns specific to each customer segment
- **Personalization**: Customize product recommendations per segment
- **Pricing Strategy**: Segment-specific pricing and promotions
- **Customer Retention**: Develop retention strategies for high-value segments

### Anomaly Detection

- **Fraud Prevention**: Identify suspicious transactions for investigation
- **Quality Control**: Detect data entry errors and system anomalies
- **Risk Management**: Flag high-risk transactions for additional review
- **Inventory Audits**: Uncover unusual ordering patterns

### Inventory & Operations

- Optimize stock levels based on segment preferences
- Understand seasonal purchase patterns by cluster
- Predict demand for inventory planning
- Allocate resources based on customer segment profitability

### Sales Optimization

- Focus sales efforts on high-value customer segments
- Identify upsell and cross-sell opportunities
- Analyze purchase patterns for product bundling
- Optimize supply chain based on segment demand

---

## 🎓 Learning Outcomes

This project demonstrates:

✅ **Data Preprocessing**: Cleaning, handling missing values, outlier detection  
✅ **Feature Engineering**: Categorical encoding, feature selection, normalization  
✅ **Dimensionality Reduction**: PCA for visualization and analysis  
✅ **Unsupervised Learning**: K-Means clustering without labeled data  
✅ **Anomaly Detection**: Isolation Forest for outlier identification  
✅ **Model Evaluation**: Silhouette Score and Elbow Method for validation  
✅ **Data Visualization**: Communicating insights through professional charts  
✅ **Real-World Application**: Business-oriented analysis and recommendations

---

## 📋 Workflow & Pipeline

The analysis follows a systematic data science pipeline:

```
Raw Data (541,909 rows)
         ↓
Data Exploration & Analysis
         ↓
Data Cleaning (remove nulls, duplicates)
         ↓
Feature Engineering & Encoding
         ↓
Feature Scaling (MinMaxScaler)
         ↓
Dimensionality Reduction (PCA)
         ↓
Model Training
  ├── K-Means Clustering
  └── Isolation Forest
         ↓
Model Evaluation
  ├── Silhouette Score: 0.9964
  ├── Elbow Method: K=4 optimal
  └── Outlier Ratios: 1-5%
         ↓
Business Insights & Recommendations
```

---

## 📈 Key Metrics Summary

| Metric                       | Value   | Interpretation                           |
| ---------------------------- | ------- | ---------------------------------------- |
| **Silhouette Score**         | 0.9964  | Excellent cluster quality and separation |
| **Optimal Clusters**         | 4       | Well-defined customer segments           |
| **Dataset Size**             | 541,909 | Large, robust statistical foundation     |
| **Features (Post-Encoding)** | ~10-15  | Balanced feature representation          |
| **Anomaly Rate**             | 1-5%    | Reasonable outlier prevalence            |
| **Missing Values**           | Removed | 100% complete final dataset              |

---

## 🔬 Machine Learning Algorithms Explained

### K-Means Clustering

- **Purpose**: Partition transactions into K clusters based on feature similarity
- **How It Works**: Minimizes within-cluster variance using iterative centroid updates
- **Advantages**: Fast, scalable, interpretable clusters
- **Use Case**: Grouping customers by purchase behavior

### PCA (Principal Component Analysis)

- **Purpose**: Reduce dimensionality while preserving maximum variance
- **How It Works**: Identifies principal components that capture data variance
- **Advantages**: Enables visualization, reduces computation, handles collinearity
- **Use Case**: 2D visualization of multi-dimensional transaction data

### Isolation Forest

- **Purpose**: Detect anomalies in high-dimensional data
- **How It Works**: Isolates outliers using random decision trees
- **Advantages**: Fast, no distance calculations, handles mixed features
- **Use Case**: Identifying fraudulent or unusual transactions

### MinMaxScaler

- **Purpose**: Normalize features to [0, 1] range
- **Formula**: (x - min) / (max - min)
- **Advantages**: Preserves shape of distribution, works well with algorithms like KMeans
- **Use Case**: Ensuring equal feature contribution to clustering

---

## 📚 Dependencies

### Core Requirements

```
numpy>=1.21
pandas>=1.3
scikit-learn>=1.0
matplotlib>=3.5
seaborn>=0.11
jupyter>=1.0
```

### Installation

```bash
pip install -r requirements.txt
```

### Python Version

- **Minimum**: Python 3.9
- **Recommended**: Python 3.10 or higher

---

## 🔍 Troubleshooting

### Common Issues

**NumPy Version Conflict**

```
Error: Module compiled with NumPy 1.x cannot run with NumPy 2.x
Solution: pip install "numpy<2"
```

**Missing Dataset**

```
Error: FileNotFoundError: 'Online Retail.csv' not found
Solution: Ensure Online Retail.csv is in the same directory as the notebook
```

**Memory Issues with Large Dataset**

```
Tip: Use chunksize parameter in pd.read_csv() for processing in batches
```

---

## 📝 Notes & Assumptions

- The project uses **unsupervised learning** (no labeled training data required)
- Results are transaction-level; customer profiles can be aggregated further
- Silhouette Score > 0.5 indicates good clustering; 0.9964 is exceptional
- Anomaly detection uses auto-contamination; adjust based on business needs
- Geographic analysis groups rare countries (<100 occurrences) to avoid overfitting

---

## 🚀 Future Enhancements

Potential extensions of this project:

- **Customer Lifetime Value (CLV)**: Predict future customer value per segment
- **Time-Series Analysis**: Analyze purchase patterns over time
- **Hierarchical Clustering**: Explore alternative clustering approaches
- **DBSCAN**: Identify density-based clusters
- **Neural Networks**: Deep learning-based segmentation
- **Recommendation System**: Suggest products per customer segment
- **Interactive Dashboard**: Create web-based visualization dashboard

---

## 📄 License

This project is available for educational and portfolio purposes.

---

## 🤝 Contributing

This is a training project for machine learning education. Feel free to:

- Fork and extend the analysis
- Experiment with different algorithms
- Add new features or preprocessing techniques
- Improve visualizations
- Share improvements and insights

---

## 📧 Contact & Questions

For questions about this project or machine learning in general, feel free to reach out through GitHub.

---

## Author

* **Developed by:** Omar Hafez Khalil
* **GitHub:** [OmarHKhalil](https://github.com/OmarHKhalil)
* **LinkedIn:** [Omar Khalil](https://www.linkedin.com/in/omar-khalil-55a674281)
