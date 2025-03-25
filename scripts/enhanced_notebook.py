import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, recall_score, confusion_matrix
from sklearn.inspection import partial_dependence, permutation_importance
from fairlearn.metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio,
    true_positive_rate_difference,
    false_positive_rate_difference,
    true_positive_rate,
    false_positive_rate
)
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
import shap
import lime
import lime.lime_tabular
import sys
import warnings
import glob
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules
import data_processing as dp
import visualization as viz
import modeling as mdl
import utils
import config

def main():
    """Main function to run the analysis pipeline."""

    
    try:
        # Set visualization style
        print("Setting visualization style...")
        viz.set_visualization_style()
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Create directories if they don't exist
        print("Creating output directories...")
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        os.makedirs(config.MODELS_DIR, exist_ok=True)
        
        # Ensure matplotlib is using the Agg backend
        matplotlib.use('Agg')
        
        # Load and preprocess data
        print("\nLoading and preprocessing data")
        df, description_dict = dp.load_data(config.DATA_DIR)
        if df is None:
            print("Error: Failed to load data. Exiting...")
            return
        utils.print_dataset_info(df)
        
        # Analyze missing values
        print("\nAnalyzing missing values")
        missing_percentages, high_missing_cols = dp.analyze_missing_values(df)
        viz.plot_missing_values(missing_percentages)
        
        # Split data for modeling
        print("\nSplitting data for modeling")
        X, y = dp.split_features_target(df, config.TARGET_COLUMN)
        X_train, X_test, y_train, y_test = dp.create_train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
        )
        
        # Filter high missing columns and identify column types
        print("\nFiltering high missing columns")
        X_train_filtered, X_test_filtered, high_missing_cols = dp.filter_high_missing_columns(
            X_train, X_test, threshold=config.MISSING_THRESHOLD_HIGH
        )
        numerical_cols, categorical_cols = dp.identify_column_types(X_train_filtered)
        
        # Save processed data
        print("\nSaving processed data")
        dp.save_processed_data(
            X_train_filtered, X_test_filtered, y_train, y_test,
            numerical_cols, categorical_cols, description_dict,
            os.path.join(config.RESULTS_DIR, 'processed_data.pkl')
        )
        
        # Data exploration
        print("\nPerforming data exploration")
        viz.plot_target_distribution(df)
        viz.plot_age_distribution_by_mortality(X_train_filtered, y_train)
        
        # Analyze physiological parameters
        print("\nAnalyzing physiological parameters")
        physiological_vars = ['heart_rate_apache', 'map_apache', 'resprate_apache', 'temp_apache', 'wbc_apache', 'creatinine_apache']
        viz.plot_physiological_vars_by_mortality(X_train_filtered, y_train, physiological_vars)
        
        # Visualize categorical features
        print("\nVisualizing categorical features")
        key_categorical = ['ethnicity', 'apache_2_bodysystem', 'apache_3j_bodysystem', 'icu_type']
        viz.plot_categorical_features_by_mortality(X_train_filtered, y_train, key_categorical)
        
        # Correlation analysis
        print("\nPerforming correlation analysis")
        viz.plot_correlation_matrix_of_clinical_features(X_train_filtered, y_train)
        
        # Model Training and Evaluation
        print("\nModel Training and Evaluation")
        
        # Check if we have existing trained models
        model_files = glob.glob(os.path.join(config.MODELS_DIR, '*.pkl'))
        if model_files:
            print("Loading existing trained models...")
            latest_model = max(model_files, key=os.path.getctime)
            with open(latest_model, 'rb') as f:
                best_model = pickle.load(f)
            print(f"Loaded model from {latest_model}")
        else:
            print("Training new models...")
            model_results = mdl.train_and_evaluate_all_models(
                X_train_filtered, X_test_filtered, y_train, y_test,
                numerical_cols, categorical_cols
            )
            
            model_metrics = pd.DataFrame({
                name: result['metrics'] for name, result in model_results.items()
            }).T
            
            best_model_name = model_metrics['roc_auc'].idxmax()
            best_model = model_results[best_model_name]['model']
            
            os.makedirs(config.MODELS_DIR, exist_ok=True)
            model_path = os.path.join(config.MODELS_DIR, f'best_model_{best_model_name}.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(best_model, f)
            print(f"Saved best model ({best_model_name}) to {model_path}")
            
            model_metrics.to_csv(os.path.join(config.RESULTS_DIR, 'model_comparison.csv'))
        
        # Generate predictions from best model
        y_pred = best_model.predict(X_test_filtered)
        y_pred_proba = best_model.predict_proba(X_test_filtered)[:, 1]
        
        # Calculate and print metrics
        metrics = mdl.calculate_metrics(y_test, y_pred, y_pred_proba)
        print("\nModel Performance Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.3f}")
        
        # Create confusion matrices
        print("\nCreating confusion matrices...")
        viz.plot_all_confusion_matrices(y_test, y_pred, y_pred_proba)
        
        # Compare with APACHE IV
        print("\nComparing with APACHE IV...")
        apache_comparison = mdl.compare_with_apache_iv(best_model, X_test_filtered, y_test)
        if apache_comparison:
            print(f"Our model AUC: {apache_comparison['our_auc']:.3f}")
            print(f"APACHE IV AUC: {apache_comparison['apache_auc']:.3f}")
            print(f"Number of samples compared: {apache_comparison['n_samples']}")
        
        # Analyze fairness
        print("\nAnalyzing fairness...")
        fairness_df, fairness_metrics = mdl.analyze_fairness_by_ethnicity(X_test_filtered, y_test, y_pred, y_pred_proba)
        if fairness_df is not None:
            print("\nFairness Metrics by Ethnicity:")
            print(fairness_df)
            fairness_df.to_csv(os.path.join(config.RESULTS_DIR, 'fairness_metrics.csv'))
            
            # Calculate group fairness metrics
            print("\nCalculating group fairness metrics...")
            ethnicity_fairness_metrics = mdl.analyze_fairness_metrics(
                y_test, y_pred, y_pred_proba, X_test_filtered['ethnicity']
            )
            print("\nGroup Fairness Metrics for Ethnicity:")
            for metric_name, value in ethnicity_fairness_metrics.items():
                print(f"{metric_name}: {value:.4f}")
            
            pd.DataFrame([ethnicity_fairness_metrics]).to_csv(
                os.path.join(config.RESULTS_DIR, 'group_fairness_metrics.csv')
            )
            
            # Train a fairness constrained model
            print("\nTraining model with fairness constraints...")
            try:
                constrained_model = mdl.train_fairness_constrained_model(
                    X_train_filtered, y_train, 
                    X_train_filtered['ethnicity'], 
                    constraint_type='demographic_parity'
                )
                
                constrained_pred = constrained_model.predict(X_test_filtered)
                constrained_proba = constrained_model.predict_proba(X_test_filtered)[:,1]
                
                print("\nOriginal vs Fairness-Constrained Model:")
                original_metrics = mdl.calculate_metrics(y_test, y_pred, y_pred_proba)
                constrained_metrics = mdl.calculate_metrics(y_test, constrained_pred, constrained_proba)
                
                metrics_comparison = pd.DataFrame({
                    'Original Model': original_metrics,
                    'Constrained Model': constrained_metrics
                })
                print(metrics_comparison)
                metrics_comparison.to_csv(os.path.join(config.RESULTS_DIR, 'fairness_constrained_comparison.csv'))
                
                constrained_fairness = mdl.analyze_fairness_metrics(
                    y_test, constrained_pred, constrained_proba, X_test_filtered['ethnicity']
                )
                
                fairness_comparison = pd.DataFrame({
                    'Original Model': ethnicity_fairness_metrics,
                    'Constrained Model': constrained_fairness
                })
                fairness_comparison.to_csv(os.path.join(config.RESULTS_DIR, 'fairness_metrics_comparison.csv'))
                
            except Exception as e:
                print(f"Error training fairness-constrained model: {str(e)}")
        
        # Analyze performance by body system
        print("\nAnalyzing performance by body system...")
        performance_df, performance_metrics = mdl.analyze_performance_by_body_system(X_test_filtered, y_test, best_model)
        if performance_df is not None:
            print("\nPerformance Metrics by Body System:")
            print(performance_df)
            performance_df.to_csv(os.path.join(config.RESULTS_DIR, 'performance_by_body_system.csv'))
        
        # Analyze classification thresholds
        print("\nAnalyzing classification thresholds...")
        threshold_analysis = mdl.analyze_classification_thresholds(best_model, X_test_filtered, y_test)
        print("\nClassification Threshold Analysis:")
        print(threshold_analysis)
        threshold_analysis.to_csv(os.path.join(config.RESULTS_DIR, 'threshold_analysis.csv'))
        
        # SHAP Analysis
        print("\nPerforming SHAP analysis (this may take a while)...")
        shap_cache_file = os.path.join(config.RESULTS_DIR, 'shap_values.pkl')
        shap_data = mdl.analyze_with_shap_cached(best_model, X_test_filtered, cache_file=shap_cache_file)
        if shap_data:
            print("SHAP analysis completed successfully")
        
        # LIME Analysis
        print("\nPerforming LIME analysis...")
        feature_names = numerical_cols + categorical_cols
        lime_explanation = mdl.explain_with_lime(
            best_model, 
            X_train_filtered, 
            X_test_filtered.iloc[0], 
            feature_names=feature_names, 
            class_names=['Survive', 'Die']
        )
        if lime_explanation:
            print("LIME analysis completed successfully")
        
        # Intersectional Fairness Analysis
        print("\nAnalyzing intersectional fairness...")
        X_test_with_age_groups = X_test_filtered.copy()
        X_test_with_age_groups['age'] = pd.to_numeric(X_test_with_age_groups['age'], errors='coerce')
        X_test_with_age_groups['age_group'] = pd.cut(
            X_test_with_age_groups['age'],
            bins=[0, 30, 50, 70, float('inf')],
            labels=['<30', '30-50', '50-70', '>70']
        )
        
        protected_attributes = ['ethnicity', 'gender', 'age_group']
        intersectional_results = mdl.analyze_intersectional_fairness(
            best_model, 
            X_test_with_age_groups, 
            y_test, 
            protected_attributes
        )
        print("\nIntersectional Fairness Results:")
        print(intersectional_results)
        intersectional_results.to_csv(os.path.join(config.RESULTS_DIR, 'intersectional_fairness.csv'))
        
        # Analyze feature importance using permutation importance
        print("\nAnalyzing feature importance with permutation importance...")
        perm_importance = permutation_importance(
            best_model, X_test_filtered, y_test, 
            n_repeats=10, random_state=42, n_jobs=-1
        )
        
        perm_importance_df = pd.DataFrame({
            'Feature': X_test_filtered.columns,
            'Importance': perm_importance.importances_mean,
            'Std': perm_importance.importances_std
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 10 features by permutation importance:")
        print(perm_importance_df.head(10))
        perm_importance_df.to_csv(os.path.join(config.RESULTS_DIR, 'permutation_importance.csv'))
        
        viz.plot_feature_importance(
            perm_importance_df, 
            title='Feature Importance (Permutation)',
            save_path=os.path.join(config.RESULTS_DIR, 'permutation_importance.png')
        )
        
        # Analyze partial dependence for top features
        print("\nAnalyzing partial dependence for top features...")
        top_features = perm_importance_df['Feature'].head(5).tolist()
        numerical_top_features = [f for f in top_features if f in numerical_cols]
        
        if numerical_top_features:
            print(f"Calculating partial dependence for {len(numerical_top_features)} features...")
            
            for feature in tqdm(numerical_top_features, desc="Calculating partial dependence"):
                feature_values = X_train_filtered[feature].dropna()
                grid_points = np.linspace(
                    feature_values.quantile(0.05),
                    feature_values.quantile(0.95),
                    num=20
                )
                
                X_temp = X_train_filtered.copy()
                pdp_values = []
                
                for value in grid_points:
                    X_temp[feature] = value
                    predictions = best_model.predict_proba(X_temp)[:, 1]
                    pdp_values.append(np.mean(predictions))
                
                plt.figure(figsize=(10, 6))
                plt.plot(grid_points, pdp_values)
                plt.xlabel(feature)
                plt.ylabel('Partial dependence')
                plt.title(f'Partial Dependence Plot for {feature}')
                plt.grid(True)
                plt.savefig(os.path.join(config.RESULTS_DIR, f'partial_dependence_{feature}.png'))
                plt.close()
            
            print("Partial dependence analysis completed successfully")
        
        print("\nAnalysis pipeline completed successfully!")
        
    except Exception as e:
        print(f"Error in main function: {e}")
        print("Analysis pipeline failed.")
        plt.close('all')

if __name__ == '__main__':
    main()


# In[76]:




