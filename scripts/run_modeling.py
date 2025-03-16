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

def main():
    """Run modeling pipeline."""
    # Create directories if they don't exist
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    
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
    
    # Set up plots directory
    plots_dir = os.path.join(config.RESULTS_DIR, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Train and evaluate models with different imputation strategies
    print("\nTraining models with different imputation strategies...")
    pipelines, scores = {}, {}
    
    for strategy in config.IMPUTATION_STRATEGIES:
        print(f"\nTraining model with {strategy} imputation...")
        
        # Create preprocessor
        preprocessor = dp.create_preprocessing_pipeline(
            numerical_cols, categorical_cols, strategy
        )
        
        # Create and train the pipeline
        pipeline = mdl.create_random_forest_pipeline(
            preprocessor, 
            use_class_weights=True, 
            n_estimators=config.RF_PARAMS['n_estimators'],
            max_depth=config.RF_PARAMS['max_depth'],
            min_samples_split=config.RF_PARAMS['min_samples_split'],
            random_state=config.RF_PARAMS['random_state']
        )
        
        pipeline.fit(X_train, y_train)
        
        # Save model
        with open(os.path.join(config.MODELS_DIR, f'rf_{strategy}.pkl'), 'wb') as f:
            pickle.dump(pipeline, f)
        
        # Evaluate
        results, y_pred, y_proba = mdl.evaluate_model(
            pipeline, X_test, y_test, f"{strategy.capitalize()} Imputation"
        )
        
        # Store pipeline and results
        pipelines[strategy] = pipeline
        scores[strategy] = results
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        viz.plot_confusion_matrix(
            y_test, y_pred, f"Confusion Matrix - {strategy.capitalize()} Imputation"
        )
        plt.savefig(os.path.join(plots_dir, f'confusion_matrix_{strategy}.png'))
        plt.close()
        
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        viz.plot_roc_curve(
            y_test, y_proba, title=f"ROC Curve - {strategy.capitalize()} Imputation"
        )
        plt.savefig(os.path.join(plots_dir, f'roc_curve_{strategy}.png'))
        plt.close()
    
    # Compare imputation strategies
    print("\nComparing imputation strategies...")
    comparison_df, best_strategy = utils.compare_models(
        {s: scores[s] for s in config.IMPUTATION_STRATEGIES}, 'roc_auc'
    )
    
    print(f"Best imputation strategy: {best_strategy}")
    print(comparison_df)
    
    # Analyze feature importance
    print("\nAnalyzing feature importance...")
    feature_names = mdl.get_feature_names(pipelines[best_strategy].named_steps['preprocessor'])
    feature_importances = pipelines[best_strategy].named_steps['classifier'].feature_importances_
    
    # Save feature importance plot
    plt.figure(figsize=(12, 8))
    importance_df = viz.plot_feature_importance(
        feature_names, feature_importances, top_n=20, 
        title=f"Top 20 Feature Importances - {best_strategy.capitalize()} Imputation"
    )
    plt.savefig(os.path.join(plots_dir, 'feature_importance.png'))
    plt.close()
    
    # Save feature importance data
    importance_df.to_csv(os.path.join(config.RESULTS_DIR, 'feature_importance.csv'))
    
    # Compare with APACHE IV prediction if available
    print("\nComparing with APACHE IV prediction...")
    apache_result = mdl.compare_with_apache_iv(X_test, y_test, pipelines[best_strategy].predict_proba(X_test)[:, 1])
    
    if apache_result is not None:
        apache_auc, our_auc = apache_result
        
        # Plot comparison
        # First, get the rows where APACHE predictions are not NaN
        valid_mask = ~X_test['apache_4a_hospital_death_prob'].isna()
        
        if valid_mask.sum() > 0:
            plt.figure(figsize=(10, 8))
            viz.plot_multiple_roc_curves(
                y_test[valid_mask], 
                {
                    f"Our Model ({best_strategy})": pipelines[best_strategy].predict_proba(X_test[valid_mask])[:, 1],
                    "APACHE IV": X_test.loc[valid_mask, 'apache_4a_hospital_death_prob']
                },
                "ROC Curves - Our Model vs. APACHE IV"
            )
            plt.savefig(os.path.join(plots_dir, 'roc_comparison_apache.png'))
            plt.close()
    
    # Analyze fairness by ethnicity if available
    print("\nAnalyzing fairness by ethnicity...")
    best_model = pipelines[best_strategy]
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    metrics_df = mdl.analyze_fairness_by_ethnicity(X_test, y_test, y_pred, y_proba)
    
    if metrics_df is not None:
        # Save metrics by ethnicity
        metrics_df.to_csv(os.path.join(config.RESULTS_DIR, 'fairness_metrics.csv'))
        
        # Plot metrics by ethnicity
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x=metrics_df.index, y=metrics_df['auc'])
        plt.title('ROC AUC Score by Ethnicity')
        plt.ylabel('AUC Score')
        plt.xlabel('Ethnicity')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'fairness_auc.png'))
        plt.close()
    
    print(f"\nModeling done and results saved to {config.RESULTS_DIR} and {config.MODELS_DIR}")

if __name__ == "__main__":
    main()
