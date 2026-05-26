
# California Housing Price Predictor (Geospatial Clustering)

## Overview

This notebook applies density-based clustering to California housing coordinates to identify spatial patterns in the dataset. It focuses on geospatial segmentation rather than direct price regression. The goal is to discover spatial patterns by grouping similar regions using latitude and longitude. This project serves as an educational exercise in feature selection, hyperparameter tuning (`eps` and `min_samples`), and evaluating cluster quality via the silhouette score.

## Summary Table

| Dataset | Model | Key Result |
| --- | --- | --- |
| `housing.csv` | `DBSCAN` | Identified geographic clusters with a silhouette score. |
| `housing.csv` | `StandardScaler` | Scaled location coordinates to ensure uniform impact on clustering. |

## DBSCAN Experiments

| Experiment | Model Parameters | Silhouette Score | Observation |
| --- | --- | --- | --- |
| Experiment 1 | `eps=0.33, min_samples=18` | 0.6668 | Produced strong density-based clusters. |
| Experiment 2 | `eps=1.00, min_samples=2` | 0.8999 | Highest score; potentially over-segmented or too granular. |
| Experiment 3 | `eps=0.76, min_samples=15` | 0.8492 | Strong performance with stable dense regions. |
| Experiment 4 | `eps=0.76, min_samples=15` | 0.4146 | Lower quality, but identifies broader meaningful spatial groups. |

## Approach

1. **Load** the California housing dataset.
2. **Select** longitude and latitude as the primary clustering input.
3. **Scale** geographic coordinates using `StandardScaler` to normalize the input space.
4. **Tune** DBSCAN parameters (`eps` and `min_samples`) and evaluate using the silhouette score.

## Project Structure

```text
California Housing Price Predictor/
├── housing_DBSCAN.ipynb
└── housing.csv

```

## Key Features

* **Feature Scaling:** Uses `StandardScaler` to harmonize the coordinate range, allowing DBSCAN to detect non-linear dense clusters effectively.
* **Clustering Quality:** Explores the trade-off between silhouette scores and geographic interpretability.
* **Environment:** Self-contained analysis relying on local `housing.csv` data.

## Workflow / Pipeline

1. Load `housing.csv`.
2. Extract geographic features (longitude/latitude).
3. Apply `StandardScaler`.
4. Perform grid search or iterative testing for `eps` and `min_samples`.
5. Extract clusters and visualize spatial groupings.

## Usage

1. Open the notebook: `jupyter notebook "housing_DBSCAN.ipynb"`
2. Execute the cells sequentially to reproduce the clustering results and visualizations.

## Authors / Credits

* **Author:** Omar Hafez Khalil
