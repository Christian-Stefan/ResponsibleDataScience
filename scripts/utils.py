import pandas as pd
import numpy as np
import os
import pickle
from sklearn.metrics import (
    roc_auc_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    accuracy_score,
    precision_recall_curve,
    auc
)
import config


def print_dataset_info(df):
    """Print basic information about the dataset."""
    print("\nDataset Information:")
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nData Types:")
    print(df.dtypes)
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nBasic Statistics:")
    print(df.describe())


def describe_variable(variable_name, description_dict):
    if variable_name in description_dict:
        print(f"Description of {variable_name}:")
        for key, value in description_dict[variable_name].items():
            print(f"  {key}: {value}")
    else:
        print(f"Variable {variable_name} not found in the data dictionary.")


def print_mortality_stats(df, target_col='hospital_death'):
    if target_col in df.columns:
        number_of_deads = np.sum(df[target_col] == 1)
        number_of_survivors = np.sum(df[target_col] == 0)
        mortality_rate = number_of_deads / len(df[target_col])
        print(f"Dead: {number_of_deads} ({mortality_rate:.2%})")
        print(f"Survived: {number_of_survivors} ({1 - mortality_rate:.2%})")
        print(f"Total: {len(df[target_col])}")
    else:
        print(f"Column '{target_col}' not found in the dataset.")


def compare_models(models_results, metric='roc_auc'):
    """Compare models based on a specified metric."""
    comparison = {}
    for name, results in models_results.items():
        if metric in results:
            comparison[name] = results[metric]
    
    comparison_df = pd.DataFrame.from_dict(comparison, orient='index', columns=[metric])
    comparison_df = comparison_df.sort_values(metric, ascending=False)
    
    best_model = comparison_df.index[0]
    print(f"\nBest model: {best_model} ({metric}: {comparison_df[metric][0]:.3f})")
    
    return comparison_df, best_model


def get_optimal_threshold(y_true, y_proba, metric='f1'):
    thresholds = np.linspace(0.01, 0.99, 99)
    scores = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        if metric == 'f1':
            score = f1_score(y_true, y_pred)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        scores.append(score)
    best_idx = np.argmax(scores)
    optimal_threshold = thresholds[best_idx]
    best_score = scores[best_idx]
    return optimal_threshold, best_score


def calculate_performance_metrics(y_true, y_pred, y_proba=None):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
    return metrics


def save_model(model, filename):
    """Save a trained model to a file."""
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filename}")


def load_model(filename):
    """Load a trained model from a file."""
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model


def export_model_performance(model_name, X_test, y_test, model, filename='model_performance.csv'):
    """Export model performance metrics to a CSV file."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = calculate_performance_metrics(y_test, y_pred, y_proba)
    
    performance = {
        'model_name': model_name,
        'roc_auc': metrics['roc_auc'],
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1']
    }
    
    performance_df = pd.DataFrame([performance])
    file_exists = os.path.isfile(filename)
    if file_exists:
        performance_df.to_csv(filename, mode='a', header=False, index=False)
    else:
        performance_df.to_csv(filename, index=False)
    print(f"Model performance exported to {filename}")


def calculate_disparate_impact(y_true, y_pred, sensitive_features):
    """Calculate disparate impact ratio for different groups."""
    disparate_impact = {}
    
    for group in sensitive_features.unique():
        mask = sensitive_features == group
        if mask.sum() > 0:
            y_pred_group = y_pred[mask]
            prediction_rate = np.mean(y_pred_group)
            disparate_impact[group] = prediction_rate
    
    # Calculate ratio relative to majority group
    majority_group = max(disparate_impact.items(), key=lambda x: x[1])[0]
    majority_rate = disparate_impact[majority_group]
    
    for group in disparate_impact:
        disparate_impact[group] = disparate_impact[group] / majority_rate
    
    return pd.Series(disparate_impact)


def calculate_equal_opportunity(y_true, y_pred, sensitive_features):
    """Calculate equal opportunity difference for different groups."""
    equal_opportunity = {}
    
    for group in sensitive_features.unique():
        mask = sensitive_features == group
        if mask.sum() > 0:
            y_true_group = y_true[mask]
            y_pred_group = y_pred[mask]
            true_positive_rate = np.mean(y_pred_group[y_true_group == 1])
            equal_opportunity[group] = true_positive_rate
    
    # Calculate difference from majority group
    majority_group = max(equal_opportunity.items(), key=lambda x: x[1])[0]
    majority_rate = equal_opportunity[majority_group]
    
    for group in equal_opportunity:
        equal_opportunity[group] = equal_opportunity[group] - majority_rate
    
    return pd.Series(equal_opportunity)


def calculate_predictive_parity(y_true, y_pred, sensitive_features):
    """Calculate predictive parity difference for different groups."""
    predictive_parity = {}
    
    for group in sensitive_features.unique():
        mask = sensitive_features == group
        if mask.sum() > 0:
            y_true_group = y_true[mask]
            y_pred_group = y_pred[mask]
            positive_predictive_value = np.mean(y_true_group[y_pred_group == 1])
            predictive_parity[group] = positive_predictive_value
    
    # Calculate difference from majority group
    majority_group = max(predictive_parity.items(), key=lambda x: x[1])[0]
    majority_rate = predictive_parity[majority_group]
    
    for group in predictive_parity:
        predictive_parity[group] = predictive_parity[group] - majority_rate
    
    return pd.Series(predictive_parity)


def calculate_balanced_accuracy(y_true, y_pred):
    """Calculate balanced accuracy score."""
    sensitivity = recall_score(y_true, y_pred)
    specificity = recall_score(y_true, y_pred, pos_label=0)
    return (sensitivity + specificity) / 2


def calculate_f1_score(y_true, y_pred):
    """Calculate F1 score."""
    return f1_score(y_true, y_pred)


def calculate_auc_roc(y_true, y_pred_proba):
    """Calculate AUC-ROC score."""
    return roc_auc_score(y_true, y_pred_proba)


def calculate_auc_pr(y_true, y_pred_proba):
    """Calculate AUC-PR score."""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    return auc(recall, precision)


def calculate_calibration_error(y_true, y_pred_proba, n_bins=10):
    """Calculate calibration error using equal-width binning."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred_proba, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    calibration_error = 0
    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        if mask.sum() > 0:
            bin_true = y_true[mask]
            bin_pred = y_pred_proba[mask]
            bin_mean_true = np.mean(bin_true)
            bin_mean_pred = np.mean(bin_pred)
            calibration_error += np.abs(bin_mean_true - bin_mean_pred) * mask.sum()
    
    return calibration_error / len(y_true)
