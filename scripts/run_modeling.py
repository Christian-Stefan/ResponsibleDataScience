#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to run the modeling pipeline using preprocessed data.
"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

import data_processing as dp
import modeling as mdl
import visualization as viz
import utils
import config
from data_processing import load_processed_data

def main():
    """Run the modeling pipeline."""
    print("Starting modeling pipeline...")
    
    # Load processed data
    print("\nLoading processed data...")
    processed_data = load_processed_data(
        os.path.join(config.RESULTS_DIR, 'processed_data.pkl')
    )
    if processed_data is None:
        print("Failed to load processed data. Exiting...")
        return
    
    X_train = processed_data['X_train']
    X_test = processed_data['X_test']
    y_train = processed_data['y_train']
    y_test = processed_data['y_test']
    numerical_cols = processed_data['numerical_cols']
    categorical_cols = processed_data['categorical_cols']
    
    # Create directories if they don't exist
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    
    # Set visualization style
    viz.set_visualization_style()
    
    # Train and evaluate models with different imputation strategies
    print("\nTraining and evaluating models...")
    models_results = mdl.train_and_evaluate_all_imputation_strategies(
        X_train, X_test, y_train, y_test,
        numerical_cols, categorical_cols,
        config.IMPUTATION_STRATEGIES
    )
    
    # Plot confusion matrices
    print("\nPlotting confusion matrices...")
    for strategy, results in models_results.items():
        viz.plot_enhanced_confusion_matrix(
            y_test, results['y_pred'],
            strategy
        )
    
    # Plot ROC curves
    print("\nPlotting ROC curves...")
    viz.plot_roc_curves_comparison(
        y_test, {strategy: results['y_pred_proba'] 
                 for strategy, results in models_results.items()}
    )
    
    # Plot Precision-Recall curves
    print("\nPlotting Precision-Recall curves...")
    viz.plot_precision_recall_curves_comparison(
        y_test, {strategy: results['y_pred_proba'] 
                 for strategy, results in models_results.items()}
    )
    
    # Find best model
    print("\nFinding best model...")
    comparison_df, best_strategy = utils.compare_models(
        {s: {'roc_auc': results['auc']} for s, results in models_results.items()}, 'roc_auc'
    )
    print(f"\nBest model configuration: {best_strategy}")
    print("\nModel Comparison:")
    print(comparison_df)
    
    # Save best model
    print("\nSaving best model...")
    best_model = models_results[best_strategy]['model']
    with open(os.path.join(config.MODELS_DIR, f'best_model_{best_strategy}.pkl'), 'wb') as f:
        pickle.dump(best_model, f)
    
    print("\nModeling pipeline completed successfully!")

if __name__ == '__main__':
    main()
