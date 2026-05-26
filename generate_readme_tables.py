import json
import re
from pathlib import Path
from collections import defaultdict

# Load the extracted metrics
metrics_file = Path('metric_extraction.json')
data = json.loads(metrics_file.read_text(encoding='utf-8'))

def parse_metrics_from_text(text):
    """Extract metrics from various output formats"""
    metrics = {}
    
    # Parse various metric formats
    # Accuracy patterns
    acc_patterns = [
        r'accuracy\s*=\s*([\d.]+)',
        r'test\s+accuracy\s*=\s*([\d.]+)',
        r'Accuracy\s*=\s*([\d.]+)',
        r'accuracy\s+([\d.]+)',
    ]
    
    # Precision patterns
    prec_patterns = [
        r'precision\s*is\s*([\d.]+)',
        r'Precision\s*is\s*([\d.]+)',
    ]
    
    # Recall patterns
    recall_patterns = [
        r'recall\s+is\s*([\d.]+)',
        r'recall[^=]*=\s*([\d.]+)',
    ]
    
    # F1 patterns
    f1_patterns = [
        r'f1\s+is\s*([\d.]+)',
        r'f1[^=]*=\s*([\d.]+)',
    ]
    
    # RMSE patterns
    rmse_patterns = [
        r'rmse\s*:?\s*([\d.]+)',
        r'RMSE\s*:?\s*([\d.]+)',
        r'test rmse\s*:?\s*([\d.]+)',
    ]
    
    # R² patterns
    r2_patterns = [
        r'r\s*[²2]\s*:?\s*([\d.]+)',
        r'R\s*[²2]\s*:?\s*([\d.]+)',
        r'r\s*squared\s*:?\s*([\d.]+)',
    ]
    
    # Silhouette
    silhouette_patterns = [
        r'silhouette\s*(?:score)?\s*:?\s*([\d.]+)',
    ]
    
    text_lower = text.lower()
    
    # Try each pattern
    for pattern in acc_patterns:
        match = re.search(pattern, text_lower)
        if match:
            try:
                metrics['accuracy'] = round(float(match.group(1)), 4)
                break
            except:
                pass
    
    for pattern in prec_patterns:
        match = re.search(pattern, text_lower)
        if match:
            try:
                metrics['precision'] = round(float(match.group(1)), 4)
                break
            except:
                pass
    
    for pattern in recall_patterns:
        match = re.search(pattern, text_lower)
        if match:
            try:
                metrics['recall'] = round(float(match.group(1)), 4)
                break
            except:
                pass
    
    for pattern in f1_patterns:
        match = re.search(pattern, text_lower)
        if match:
            try:
                metrics['f1'] = round(float(match.group(1)), 4)
                break
            except:
                pass
    
    for pattern in rmse_patterns:
        match = re.search(pattern, text_lower)
        if match:
            try:
                metrics['rmse'] = round(float(match.group(1)), 4)
                break
            except:
                pass
    
    for pattern in r2_patterns:
        match = re.search(pattern, text_lower)
        if match:
            try:
                metrics['r2'] = round(float(match.group(1)), 4)
                break
            except:
                pass
    
    for pattern in silhouette_patterns:
        match = re.search(pattern, text_lower)
        if match:
            try:
                metrics['silhouette'] = round(float(match.group(1)), 4)
                break
            except:
                pass
    
    # Try to parse classification report format for precision/recall/f1
    if 'weighted avg' in text:
        lines = text.split('\n')
        for line in lines:
            if 'weighted avg' in line or 'macro avg' in line:
                parts = line.split()
                try:
                    if 'weighted avg' in line:
                        # Format: weighted avg       0.xx      0.xx      0.xx
                        idx = parts.index('avg')
                        if idx + 3 < len(parts):
                            metrics['precision'] = round(float(parts[idx + 1]), 4)
                            metrics['recall'] = round(float(parts[idx + 2]), 4)
                            metrics['f1'] = round(float(parts[idx + 3]), 4)
                except:
                    pass
    
    return metrics

def identify_model(notebook_name):
    """Identify the model type from notebook name"""
    name = notebook_name.upper()
    
    if 'DT' in name or 'DECISION' in name or 'TREE' in name:
        return "Decision Tree"
    elif 'KNN' in name or 'K-NN' in name:
        return "K-Nearest Neighbors"
    elif 'LR' in name or ('LINEAR' in name and 'REGRESSION' in name):
        return "Linear Regression"
    elif 'RF' in name or 'RANDOM' in name or 'FOREST' in name:
        return "Random Forest"
    elif 'XGB' in name or 'XGBOOST' in name:
        return "XGBoost"
    elif 'SVC' in name or 'SVM' in name:
        return "SVM/SVC"
    elif 'DBSCAN' in name:
        return "DBSCAN"
    elif 'TSNE' in name or 'T-SNE' in name:
        return "t-SNE"
    elif 'PCA' in name:
        return "PCA"
    elif 'LOGISTIC' in name or 'BANK' in name or 'STROKE' in name or 'TITANIC' in name:
        return "Logistic Regression"
    
    return None

# Parse all data
parsed = defaultdict(dict)
for project, entries in data.items():
    for entry in entries:
        notebook = entry['notebook']
        text = entry['out']
        
        model = identify_model(notebook)
        if not model:
            continue
        
        metrics = parse_metrics_from_text(text)
        if metrics:
            # Keep best metrics (most fields populated)
            if model not in parsed[project]:
                parsed[project][model] = metrics
            else:
                if len(metrics) > len(parsed[project][model]):
                    parsed[project][model] = metrics

# Generate markdown tables
tables_output = defaultdict(str)

for project in sorted(parsed.keys()):
    models = parsed[project]
    if not models:
        continue
    
    # Determine which columns to use
    all_metrics = set()
    for metrics in models.values():
        all_metrics.update(metrics.keys())
    
    # Order columns: accuracy, precision, recall, f1, then other metrics
    ordered_cols = ['accuracy', 'precision', 'recall', 'f1', 'rmse', 'r2', 'silhouette']
    columns = ['Model'] + [col for col in ordered_cols if col in all_metrics]
    
    # Build table
    table = "| " + " | ".join(columns) + " |\n"
    table += "| " + " | ".join(["---"] * len(columns)) + " |\n"
    
    for model in sorted(models.keys()):
        metrics = models[model]
        row = [model]
        for col in columns[1:]:  # Skip 'Model'
            val = metrics.get(col, '')
            if isinstance(val, (int, float)):
                row.append(f"{val:.4f}")
            else:
                row.append(str(val))
        table += "| " + " | ".join(row) + " |\n"
    
    tables_output[project] = table

# Save to file
output = {}
for project, table in tables_output.items():
    output[project] = table

with open('model_tables.json', 'w') as f:
    json.dump(output, f, indent=2)

# Print summary
print("Generated tables for", len(tables_output), "projects")
for proj in sorted(tables_output.keys())[:3]:
    print(f"\n{proj}:")
    print(tables_output[proj])
