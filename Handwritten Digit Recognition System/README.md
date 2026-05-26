
# Handwritten Digit Recognition System

## Overview

This project applies digit recognition and dimensionality reduction to the scikit-learn `load_digits` dataset. It demonstrates classification and two visualization techniques.

The objective of this project is to provide a practical "pipeline" for digit classification while testing how preprocessing and dimensionality reduction steps affect model performance. It is built as an educational exercise to understand the relationship between `StandardScaler`, `PCA`/`t-SNE`, and the ability of `LogisticRegression` to separate classes. The added value is that the reader does not just run a model, but observes how reduced representations can make data easier to learn and how that translates to classification quality.

## Summary Table

| Dataset | Model / Technique | Key Result |
| --- | --- | --- |
| `scikit-learn load_digits` | `LogisticRegression` | Trains digit classifiers on raw and reduced features. |
| `scikit-learn load_digits` | `PCA` | Reduces dimensionality for classification and visualization. |
| `scikit-learn load_digits` | `TSNE` | Visualizes digit clusters in a 2D embedding space. |
| `scikit-learn load_digits` | `StandardScaler` | Standardizes pixel intensity features before modeling. |

## Approach

* Standardize features using `StandardScaler`.
* Train `LogisticRegression` classifiers.
* Use `PCA` for dimensionality reduction and classification on the reduced feature space.
* Visualize digit clusters with `t-SNE` embeddings.

## Project Structure

```
Handwritten Digit Recognition System/
├── load_digits_PCA.ipynb
└── load_digits_TSNE.ipynb
└── README.md
```

## Notes

The project is a practical introduction to digit classification, PCA dimensionality reduction, and t-SNE visualization.

## Key Features

* **Techniques:** Uses `StandardScaler` for `load_digits` normalization, `LogisticRegression` for classification, `PCA` for dimensionality reduction, and `t-SNE` for clustering visualization.
* **Performance:** The notebook achieves high accuracy for the `LogisticRegression` algorithm (score 0.69-0.97 in notebook outputs).
* **Integration:** Self-contained; data source is directly from `scikit-learn`.

## Requirements / Installation

* **Python:** 3.9+
* **Data:** No CSV downloads required; uses `sklearn.datasets.load_digits`.

## Workflow / Pipeline

1. Load the `load_digits` dataset.
2. Split data (e.g., 85% train, 15% test).
3. Normalize inputs using `StandardScaler`.
4. Train `LogisticRegression`.
5. Apply `PCA` and re-train/evaluate on the reduced space.
6. Visualize results using `t-SNE`.

## Results / Metrics

* **Accuracy:** Logistic Regression score is approximately 0.69-0.97.

## Usage

1. Open the notebooks in Jupyter:
`jupyter notebook "load_digits_PCA.ipynb"`
`jupyter notebook "load_digits_TSNE.ipynb"`
2. Run the cells sequentially.

## Authors / Credits

* **Contributors:** Omar Hafez Khalil