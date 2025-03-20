import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix

def calculate_fairness_metrics(X_test, y_test, y_proba, protected_attribute):
    """Calculate fairness metrics for a protected attribute."""
    if protected_attribute not in X_test.columns:
        print(f"Warning: {protected_attribute} column not found in data")
        return None, None
    
    # Initialize metrics
    metrics_by_group = {}
    y_pred = (y_proba > 0.5).astype(int)
    
    # Calculate metrics for each group
    for group in X_test[protected_attribute].unique():
        mask = X_test[protected_attribute] == group
        if sum(mask) < 10:  # Skip if too few samples
            continue
            
        group_true = y_test[mask]
        group_pred = y_pred[mask]
        group_proba = y_proba[mask]
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(group_true, group_pred).ravel()
        
        metrics_by_group[group] = {
            'Sample Size': sum(mask),
            'Mortality Rate': group_true.mean(),
            'Prediction Rate': group_pred.mean(),
            'True Positive Rate': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'False Positive Rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'ROC AUC': roc_auc_score(group_true, group_proba)
        }
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame.from_dict(metrics_by_group, orient='index')
    
    return metrics_df, metrics_by_group

def analyze_age_fairness(X_test, y_test, y_proba):
    """Analyze fairness metrics across age groups."""
    if 'age' not in X_test.columns:
        print("Warning: age column not found in data")
        return None
    
    # Convert age to numeric
    X_test = X_test.copy()
    X_test['age'] = pd.to_numeric(X_test['age'], errors='coerce')
    
    # Create age groups
    bins = [0, 30, 50, 70, float('inf')]
    labels = ['Young', 'Middle', 'Senior', 'Elderly']
    X_test['age_group'] = pd.cut(X_test['age'], bins=bins, labels=labels)
    
    # Calculate fairness metrics for age groups
    metrics_df, _ = calculate_fairness_metrics(X_test, y_test, y_proba, 'age_group')
    
    return metrics_df 