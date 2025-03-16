"""
Script to perform in-depth analysis on the trained models and results.
"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, precision_recall_curve

import data_processing as dp
import modeling as mdl
import visualization as viz
import utils
import config

def main():
    """Run analysis pipeline."""
    # Create directories if they don't exist
    analysis_dir = os.path.join(config.RESULTS_DIR, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Load preprocessed data
    print("Loading preprocessed data...")
    with open(os.path.join(config.RESULTS_DIR, 'processed_data.pkl'), 'rb') as f:
        data = pickle.load(f)
    
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    numerical_cols = data['numerical_cols']
    categorical_cols = data['categorical_cols']
    
    # Load best model (assuming it exists from run_modeling.py)
    best_strategy = "simple"  # Default, but will look for better ones
    
    # Try to find the best strategy from previous results
    try:
        with open(os.path.join(config.RESULTS_DIR, 'feature_importance.csv'), 'r') as f:
            # The filename could indicate the best strategy
            pass
    except FileNotFoundError:
        print("Feature importance file not found, using default strategy.")
    
    # Load best model
    print(f"\nLoading best model (strategy: {best_strategy})...")
    try:
        with open(os.path.join(config.MODELS_DIR, f'rf_{best_strategy}.pkl'), 'rb') as f:
            best_model = pickle.load(f)
    except FileNotFoundError:
        print(f"Model file not found. Run run_modeling.py first.")
        return
    
    # Analyze model performance across different body systems
    print("\nAnalyzing model performance across different body systems...")
    body_system_results = mdl.analyze_by_body_system(
        X_train, y_train, X_test, y_test, 
        numerical_cols, categorical_cols, best_strategy
    )
    
    if body_system_results is not None:
        # Compare AUC scores across body systems
        body_system_auc = {system: results['results']['roc_auc'] 
                          for system, results in body_system_results.items()}
        
        auc_df = pd.DataFrame({'AUC': body_system_auc}).sort_values('AUC', ascending=False)
        auc_df.to_csv(os.path.join(analysis_dir, 'body_system_auc.csv'))
        
        # Plot AUC by body system
        plt.figure(figsize=(12, 6))
        sns.barplot(x=auc_df.index, y=auc_df['AUC'])
        plt.title('ROC AUC Score by Body System')
        plt.ylabel('AUC Score')
        plt.xlabel('Body System')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, 'body_system_auc.png'))
        plt.close()
        
        # Save feature importance by body system
        feature_names = mdl.get_feature_names(best_model.named_steps['preprocessor'])
        
        for system, results in body_system_results.items():
            # Clean system name for filename
            system_clean = system.replace('/', '_').replace(' ', '_').lower()
            
            # Save feature importance data
            importance_df = pd.DataFrame({
                'Feature': feature_names[:len(results['feature_importances'])],
                'Importance': results['feature_importances']
            }).sort_values('Importance', ascending=False)
            
            importance_df.to_csv(os.path.join(
                analysis_dir, f'feature_importance_{system_clean}.csv'))
    
    # Find optimal threshold for classification
    print("\nFinding optimal classification threshold...")
    y_proba = best_model.predict_proba(X_test)[:, 1]
    optimal_threshold, best_f1 = utils.get_optimal_threshold(y_test, y_proba, metric='f1')
    
    print(f"Optimal threshold: {optimal_threshold:.3f} (F1: {best_f1:.3f})")
    
    # Evaluate with optimal threshold
    y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
    print("\nPerformance with optimal threshold:")
    print(classification_report(y_test, y_pred_optimal))
    
    # Save precision-recall curve
    plt.figure(figsize=(8, 6))
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    plt.plot(recall, precision)
    plt.axvline(recall[np.argmax(precision * recall)], color='r', linestyle='--', 
               label=f'Optimal threshold: {optimal_threshold:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig(os.path.join(analysis_dir, 'precision_recall_curve.png'))
    plt.close()
    
    # Generate detailed statistics
    print("\nGenerating detailed statistics...")
    
    # Mortality rate by body system
    if 'apache_2_bodysystem' in X_test.columns:
        body_system_mortality = {}
        for body_system in X_test['apache_2_bodysystem'].unique():
            if pd.isna(body_system):
                continue
            
            mask = X_test['apache_2_bodysystem'] == body_system
            mortality_rate = y_test[mask].mean()
            count = mask.sum()
            body_system_mortality[body_system] = {'mortality_rate': mortality_rate, 'count': count}
        
        body_system_df = pd.DataFrame(body_system_mortality).T
        body_system_df = body_system_df.sort_values('mortality_rate', ascending=False)
        body_system_df.to_csv(os.path.join(analysis_dir, 'mortality_by_body_system.csv'))
        
        # Plot mortality rate by body system
        plt.figure(figsize=(12, 6))
        sns.barplot(x=body_system_df.index, y=body_system_df['mortality_rate'])
        plt.title('Mortality Rate by Body System')
        plt.ylabel('Mortality Rate')
        plt.xlabel('Body System')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, 'mortality_by_body_system.png'))
        plt.close()
    
    print(f"\nAnalysis completed. Results saved to {analysis_dir}")

if __name__ == "__main__":
    main()
