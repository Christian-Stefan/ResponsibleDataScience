import pandas as pd
import numpy as np
import os
import pickle
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score


def print_dataset_info(df):
    print("Number of samples\t:", df.shape[0])
    print("Number of features\t:", df.shape[1])


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
    comparison = {}
    for model_name, results in models_results.items():
        if metric == 'roc_auc':
            comparison[model_name] = results['roc_auc']
        elif metric == 'accuracy':
            comparison[model_name] = results['accuracy']
        elif metric in ['precision', 'recall', 'f1']:
            comparison[model_name] = results['report']['1'][metric]
    comparison_df = pd.DataFrame({metric: comparison}).sort_values(metric, ascending=False)
    best_model = comparison_df.index[0]
    return comparison_df, best_model


def create_submission_file(ids, predictions, filename='submission.csv'):
    submission = pd.DataFrame({
        'encounter_id': ids,
        'hospital_death': predictions
    })
    submission.to_csv(filename, index=False)
    print(f"Submission file created: {filename}")


def log_experiment_results(experiment_name, params, metrics, filename='experiment_log.csv'):
    log_entry = {
        'experiment_name': experiment_name,
        'timestamp': pd.Timestamp.now(),
    }
    for param_name, param_value in params.items():
        log_entry[f'param_{param_name}'] = param_value
    for metric_name, metric_value in metrics.items():
        log_entry[f'metric_{metric_name}'] = metric_value
    log_df = pd.DataFrame([log_entry])
    file_exists = os.path.isfile(filename)
    if file_exists:
        log_df.to_csv(filename, mode='a', header=False, index=False)
    else:
        log_df.to_csv(filename, index=False)
    print(f"Experiment logged to {filename}")


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
