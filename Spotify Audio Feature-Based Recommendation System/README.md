# 🎵 Spotify Audio Feature-Based Recommendation System

A machine learning project that leverages **unsupervised clustering** to group Spotify songs based on their audio features. This project demonstrates data preprocessing, feature engineering, and KMeans clustering for music recommendation and analysis.

![Spotify](https://img.shields.io/badge/Spotify-API-1DB954?style=for-the-badge&logo=spotify)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37726?style=for-the-badge&logo=jupyter)

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Key Insights](#key-insights)
- [Contributing](#contributing)

## 📌 Overview

This project demonstrates a **recommendation-style song grouping system** built on unsupervised learning principles. Rather than implementing a complex collaborative filtering algorithm, it uses **KMeans clustering** to automatically group songs with similar audio characteristics.

**Key Objectives:**

- Clean and preprocess Spotify metadata
- Extract and encode categorical features (genre, track type)
- Scale audio features to a normalized range
- Apply KMeans clustering with optimal hyperparameters
- Visualize and evaluate cluster quality
- Generate actionable song groupings for recommendations

**Real-World Applications:**

- Playlist generation and curation
- Music discovery recommendations
- Genre/mood classification
- Content-based filtering systems
- Music production analysis

## 📊 Dataset

**Dataset:** `genres_v2.csv`

### Audio Features Analyzed:

| Feature              | Description                                | Scale         |
| -------------------- | ------------------------------------------ | ------------- |
| **danceability**     | How suitable a track is for dancing (0-1)  | 0.0-1.0       |
| **energy**           | Intensity and activity measured (0-1)      | 0.0-1.0       |
| **key**              | Estimated overall key of track             | 0-11 (C to B) |
| **loudness**         | Overall loudness in decibels (dB)          | -60 to 0      |
| **mode**             | Major (1) or Minor (0)                     | 0-1           |
| **speechiness**      | Presence of spoken words (0-1)             | 0.0-1.0       |
| **acousticness**     | Confidence measure of acoustic sound (0-1) | 0.0-1.0       |
| **instrumentalness** | Likelihood track contains no vocals (0-1)  | 0.0-1.0       |
| **liveness**         | Presence of audience in recording (0-1)    | 0.0-1.0       |
| **valence**          | Musical positiveness conveyed (0-1)        | 0.0-1.0       |
| **tempo**            | Overall estimated tempo in BPM             | 0-300+        |
| **duration_ms**      | Track length in milliseconds               | Variable      |

### Data Characteristics:

- **Total Rows:** ~1000+ Spotify tracks
- **Audio Features:** 12 continuous numeric features
- **Categorical Features:** Genre (10+ categories), Track Type
- **Missing Values:** Handled through removal
- **Duplicates:** Removed during preprocessing

## 🎯 Features

✅ **Data Preprocessing**

- Missing value detection and removal
- Duplicate record elimination
- Irrelevant feature removal (URIs, IDs, URLs)

✅ **Feature Engineering**

- Label encoding for categorical variables (genre, type)
- MinMax scaling for feature normalization (0-1 range)
- Preserved numerical feature relationships

✅ **Clustering Analysis**

- Elbow Method for optimal cluster determination
- KMeans clustering with k=4
- Silhouette score evaluation (0.55)

✅ **Visualization**

- Genre vs. Danceability analysis
- Liveness vs. Danceability cluster scatter plot
- Cluster centers visualization
- Elbow curve for hyperparameter selection

✅ **Model Evaluation**

- Silhouette Score: 0.55 (acceptable clustering quality)
- Within-Cluster Sum of Squares (WCSS) analysis
- Visual cluster separation assessment

## 📁 Project Structure

```
Spotify Audio Feature-Based Recommendation System/
├── spotify.ipynb                    # Main Jupyter notebook with full analysis
├── genres_v2.csv                    # Dataset with 1000+ Spotify tracks
├── requirements.txt                 # Python package dependencies
└── README.md                        # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Virtual environment (recommended)

### Step 1: Clone/Download the Repository

```bash
cd "Spotify Audio Feature-Based Recommendation System"
```

### Step 2: Create Virtual Environment (Optional but Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import pandas, sklearn, matplotlib; print('All dependencies installed successfully!')"
```

## 💻 Usage

### Running the Notebook

1. **Start Jupyter Notebook:**

```bash
jupyter notebook
```

2. **Open `spotify.ipynb`** in your browser

3. **Run All Cells** (Ctrl+A, Ctrl+Enter) or run cells sequentially:
   - **Cell 1:** Import libraries
   - **Cell 2-5:** Load and explore data
   - **Cell 6-9:** Data cleaning and preprocessing
   - **Cell 10-14:** Feature encoding and visualization
   - **Cell 15-18:** Feature scaling (MinMaxScaler)
   - **Cell 19-23:** Elbow method analysis
   - **Cell 24-28:** KMeans clustering and evaluation

## 🔬 Methodology

### 1. **Data Exploration & Cleaning**

```
✓ Load CSV data
✓ Examine structure and data types
✓ Check for missing values
✓ Remove duplicates
✓ Drop irrelevant columns (URIs, IDs, URLs)
```

### 2. **Feature Preprocessing**

```
✓ Label encode categorical variables (genre, type)
✓ Inspect unique values
✓ Create visualization of genre distribution
```

### 3. **Feature Scaling**

```
✓ Apply MinMaxScaler normalization
✓ Transform features to [0, 1] range
✓ Preserve relationships between features
```

### 4. **Optimal Cluster Selection (Elbow Method)**

```
Algorithm:
  For k = 1 to 10:
    - Fit KMeans with k clusters
    - Calculate WCSS (Within-Cluster Sum of Squares)
    - Record inertia value

  Plot WCSS vs Number of Clusters
  Identify "elbow" point where curve flattens
  Select k at elbow point
```

### 5. **KMeans Clustering**

```
Parameters:
  - n_clusters: 4 (from elbow analysis)
  - algorithm: 'k-means++'
  - random_state: Set for reproducibility

Output:
  - Cluster labels for each song
  - Cluster centers in feature space
```

### 6. **Evaluation & Validation**

```
Silhouette Score:
  - Measures cluster cohesion and separation
  - Range: -1 (poor) to 1 (excellent)
  - Our Result: 0.55 (acceptable)

Interpretation:
  - Clusters are reasonably separated
  - Some overlap is natural in music data
  - Value suggests distinct groupings
```

### 7. **Visualization**

- Scatter plots with cluster labels
- Cluster center markers
- Feature relationship analysis

## 📈 Results

### Clustering Performance

| Metric                   | Value                    | Interpretation        |
| ------------------------ | ------------------------ | --------------------- |
| **Optimal Clusters (k)** | 4                        | From elbow method     |
| **Silhouette Score**     | 0.55                     | Acceptable separation |
| **Data Points**          | ~1000+                   | Spotify tracks        |
| **Features**             | 12 audio + 2 categorical | Total 14 dimensions   |

### Cluster Insights

**Cluster 0:** High Energy, High Danceability

- Characteristics: Upbeat, dance-friendly tracks
- Use Case: Party/workout playlists

**Cluster 1:** Low Energy, High Acousticness

- Characteristics: Acoustic, calm songs
- Use Case: Relaxation/study playlists

**Cluster 2:** High Valence, High Liveness

- Characteristics: Happy, live performance tracks
- Use Case: Feel-good/concert playlists

**Cluster 3:** Mixed Characteristics

- Characteristics: Diverse genre blend
- Use Case: Discovery/mixed playlists

## 🛠️ Technologies Used

| Technology           | Purpose                   | Version |
| -------------------- | ------------------------- | ------- |
| **Python**           | Programming Language      | 3.9+    |
| **Pandas**           | Data Manipulation         | 1.3+    |
| **NumPy**            | Numerical Computing       | 1.20+   |
| **scikit-learn**     | Machine Learning          | 0.24+   |
| **Matplotlib**       | Data Visualization        | 3.3+    |
| **Seaborn**          | Statistical Visualization | 0.11+   |
| **Jupyter Notebook** | Interactive Development   | 6.0+    |

## 🤝 Contributing

Contributions are welcome! To improve this project:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/improvement`)
3. **Commit** your changes (`git commit -m 'Add improvement'`)
4. **Push** to the branch (`git push origin feature/improvement`)
5. **Submit** a Pull Request

### Contribution Ideas

- Add new audio features
- Implement additional clustering algorithms
- Create interactive visualizations
- Improve documentation
- Add unit tests
- Optimize performance

## 👨‍💻 Author

**Omar Hafez Khalil**

---

## Happy Learning! 🎵📊
