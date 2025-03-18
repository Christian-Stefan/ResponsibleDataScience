import sklearn
try:
    from sklearn.exceptions import UnsetMetadataPassedError
except ImportError:
    class UnsetMetadataPassedError(Exception):
        pass
    sklearn.exceptions.UnsetMetadataPassedError = UnsetMetadataPassedError

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_recall_curve, roc_curve, 
    precision_score, recall_score, f1_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import shap
from lime.lime_tabular import LimeTabularExplainer
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate

def create_random_forest_pipeline(preprocessor, use_class_weights=True, n_estimators=100, max_depth=10,
                                  min_samples_split=10, random_state=1):
    class_weight = 'balanced' if use_class_weights else None
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            class_weight=class_weight,
            n_jobs=-1
        ))
    ])
    return pipeline

def train_model(pipeline, X_train, y_train):
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_model(pipeline, X_test, y_test, model_name="Model"):
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_proba)
    accuracy = (y_pred == y_test).mean()
    report = classification_report(y_test, y_pred, output_dict=True)
    print(f"Evaluation Results - {model_name}:")
    print(f"ROC AUC: {roc_auc:.3f}")
    print(f"Accuracy: {accuracy:.3f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    results = {'model_name': model_name, 'roc_auc': roc_auc, 'accuracy': accuracy, 'report': report}
    return results, y_pred, y_proba

def create_and_evaluate_imputation_strategies(X_train, X_test, y_train, y_test, numerical_cols, categorical_cols,
                                              imputation_strategies=['simple', 'median']):
    from data_processing import create_preprocessing_pipeline
    pipelines = {}
    scores = {}
    for strategy in imputation_strategies:
        print(f"\nTraining model with {strategy} imputation...")
        preprocessor = create_preprocessing_pipeline(numerical_cols, categorical_cols, strategy)
        pipelines[strategy] = create_random_forest_pipeline(preprocessor)
        pipelines[strategy].fit(X_train, y_train)
        scores[strategy], _, _ = evaluate_model(pipelines[strategy], X_test, y_test,
                                                f"{strategy.capitalize()} Imputation")
    return pipelines, scores

def train_and_evaluate_all_imputation_strategies(X_train, X_test, y_train, y_test, numerical_cols, categorical_cols, 
                                                imputation_strategies, rf_params):
    """Train and evaluate models with different imputation strategies."""
    from data_processing import create_preprocessing_pipeline
    
    pipelines, scores = {}, {}
    
    for strategy in imputation_strategies:
        print(f"\nTraining model with {strategy} imputation...")
        
        # Create preprocessor
        preprocessor = create_preprocessing_pipeline(
            numerical_cols, categorical_cols, strategy
        )
        
        # Create and train the pipeline
        pipeline = create_random_forest_pipeline(
            preprocessor, 
            use_class_weights=True, 
            n_estimators=rf_params['n_estimators'],
            max_depth=rf_params['max_depth'],
            min_samples_split=rf_params['min_samples_split'],
            random_state=rf_params['random_state']
        )
        
        pipeline.fit(X_train, y_train)
        
        # Evaluate
        results, y_pred, y_proba = evaluate_model(
            pipeline, X_test, y_test, f"{strategy.capitalize()} Imputation"
        )
        
        # Store pipeline and results
        pipelines[strategy] = pipeline
        scores[strategy] = results
    
    return pipelines, scores

def create_smote_pipeline(preprocessor, random_state=1):
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=random_state)),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            random_state=random_state,
            n_jobs=-1
        ))
    ])
    return pipeline

def hyperparameter_tuning(X_train, y_train, preprocessor, param_grid=None, cv=3, scoring='roc_auc'):
    if param_grid is None:
        param_grid = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [10, 20, None],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4]
        }
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1))
    ])
    print("Starting grid search. This might take some time...")
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    print("Best parameters:")
    print(grid_search.best_params_)
    return grid_search, grid_search.best_estimator_

def get_feature_names(column_transformer):
    output_features = []
    for name, pipe, features in column_transformer.transformers_:
        if name != 'remainder':
            if hasattr(pipe, 'get_feature_names_out'):
                if isinstance(features, slice):
                    features = list(column_transformer._feature_names_in[features])
                current_features = pipe.get_feature_names_out(features)
                output_features.extend(current_features.tolist())
            else:
                output_features.extend(features)
    return output_features

def compare_with_apache_iv(X_test, y_test, y_proba_best):
    try:
        if 'apache_4a_hospital_death_prob' not in X_test.columns:
            print("APACHE IV predictions not found in the dataset.")
            return None
        y_test_array = y_test.values if isinstance(y_test, pd.Series) else y_test
        apache_proba = X_test['apache_4a_hospital_death_prob'].values
        valid_mask = ~np.isnan(apache_proba)
        if np.sum(valid_mask) < 10:
            print(f"Too few valid APACHE IV predictions: {np.sum(valid_mask)} (need at least 10)")
            return None
        y_test_valid = y_test_array[valid_mask]
        apache_proba_valid = apache_proba[valid_mask]
        y_proba_best_valid = y_proba_best[valid_mask]
        apache_auc = roc_auc_score(y_test_valid, apache_proba_valid)
        our_auc = roc_auc_score(y_test_valid, y_proba_best_valid)
        print(f"APACHE IV AUC: {apache_auc:.3f}")
        print(f"Our model AUC (on same subset): {our_auc:.3f}")
        print(f"Valid samples for comparison: {np.sum(valid_mask)} out of {len(y_test)}")
        return (apache_auc, our_auc)
    except Exception as e:
        print(f"Error comparing with APACHE IV: {str(e)}")
        print("Continuing without APACHE IV comparison")
        return None

def analyze_fairness_by_ethnicity(X_test, y_test, y_pred, y_proba):
    """Analyze fairness metrics across different ethnic groups."""
    if 'ethnicity' not in X_test.columns:
        print("Ethnicity information not available in the dataset.")
        return None, None
    
    ethnicities = X_test['ethnicity'].dropna().unique()
    print("\nFairness Metrics by Ethnicity:")
    
    fairness_metrics = {}
    metrics_by_ethnicity = {}
    
    for ethnicity in ethnicities:
        ethnicity_indices = X_test['ethnicity'] == ethnicity
        if sum(ethnicity_indices) < 10:
            continue
        
        y_test_group = y_test[ethnicity_indices]
        y_pred_group = y_pred[ethnicity_indices]
        y_proba_group = y_proba[ethnicity_indices]
        
        try:
            # Calculate metrics
            fpr, tpr, _ = roc_curve(y_test_group, y_proba_group)
            
            fairness_metrics[ethnicity] = {
                'count': sum(ethnicity_indices),
                'mortality_rate': y_test_group.mean(),
                'prediction_rate': y_pred_group.mean(),
                'auc': roc_auc_score(y_test_group, y_proba_group),
                'selection_rate': selection_rate(y_test_group, y_pred_group),
                'true_positive_rate': true_positive_rate(y_test_group, y_pred_group),
                'false_positive_rate': false_positive_rate(y_test_group, y_pred_group)
            }
            
            metrics_by_ethnicity[ethnicity] = {
                'fpr': fpr,
                'tpr': tpr,
                'roc_auc': roc_auc_score(y_test_group, y_proba_group)
            }
        except Exception as e:
            print(f"Error calculating metrics for ethnicity '{ethnicity}': {str(e)}")
            continue
    
    fairness_df = pd.DataFrame(fairness_metrics).T
    fairness_df['count'] = fairness_df['count'].astype(int)
    fairness_df = fairness_df.sort_values('count', ascending=False)
    
    return fairness_df, metrics_by_ethnicity

def analyze_performance_by_body_system(X_test, y_test, model):
    """Analyze model performance across different body systems."""
    if 'apache_2_bodysystem' not in X_test.columns:
        print("Body system information not available in the dataset.")
        return None, None
    
    body_system_metrics = {}
    body_system_mortality = {}
    
    for body_system in X_test['apache_2_bodysystem'].dropna().unique():
        mask = X_test['apache_2_bodysystem'] == body_system
        if mask.sum() > 20:
            # Calculate performance metrics for this body system
            mortality_rate = y_test[mask].mean()
            count = mask.sum()
            y_pred_system = model.predict(X_test[mask])
            y_proba_system = model.predict_proba(X_test[mask])[:, 1]
            
            try:
                auc_system = roc_auc_score(y_test[mask], y_proba_system)
                precision_system = precision_score(y_test[mask], y_pred_system)
                recall_system = recall_score(y_test[mask], y_pred_system)
                f1_system = f1_score(y_test[mask], y_pred_system)
                fpr, tpr, _ = roc_curve(y_test[mask], y_proba_system)
                
                body_system_metrics[body_system] = {
                    'fpr': fpr,
                    'tpr': tpr,
                    'roc_auc': auc_system
                }
                
                body_system_mortality[body_system] = {
                    'mortality_rate': mortality_rate, 
                    'count': count,
                    'auc': auc_system,
                    'precision': precision_system,
                    'recall': recall_system,
                    'f1': f1_system,
                    'prediction_rate': y_pred_system.mean()
                }
            except Exception as e:
                print(f"Error calculating metrics for body system '{body_system}': {str(e)}")
                continue
    
    body_system_df = pd.DataFrame(body_system_mortality).T
    body_system_df['count'] = body_system_df['count'].astype(int)
    body_system_df = body_system_df.sort_values('auc', ascending=False)
    
    return body_system_df, body_system_metrics

def analyze_by_body_system(X_train, y_train, X_test, y_test, numerical_cols, categorical_cols, best_strategy='simple'):
    from data_processing import create_preprocessing_pipeline
    if 'apache_2_bodysystem' not in X_test.columns:
        print("Body system information not available in the dataset.")
        return None
    body_systems = X_test['apache_2_bodysystem'].unique()
    results_by_system = {}
    for body_system in body_systems:
        if pd.isna(body_system):
            continue
        indices_train = X_train['apache_2_bodysystem'] == body_system
        indices_test = X_test['apache_2_bodysystem'] == body_system
        if sum(indices_train) < 100 or sum(indices_test) < 20:
            continue
        print(f"\nAnalyzing body system: {body_system}")
        print(f"Training samples: {sum(indices_train)}, Test samples: {sum(indices_test)}")
        X_train_system = X_train[indices_train]
        y_train_system = y_train[indices_train]
        X_test_system = X_test[indices_test]
        y_test_system = y_test[indices_test]
        try:
            preprocessor = create_preprocessing_pipeline(numerical_cols, categorical_cols, best_strategy)
            pipe_system = create_random_forest_pipeline(preprocessor)
            pipe_system.fit(X_train_system, y_train_system)
            results, y_pred_system, y_proba_system = evaluate_model(
                pipe_system, X_test_system, y_test_system, f"Body System: {body_system}"
            )
            results_by_system[body_system] = {
                'results': results,
                'model': pipe_system,
                'feature_importances': pipe_system.named_steps['classifier'].feature_importances_
            }
        except Exception as e:
            print(f"Error analyzing body system '{body_system}': {str(e)}")
            continue
    return results_by_system

def analyze_classification_thresholds(model, X_test, y_test):
    """Find optimal threshold for classification."""
    y_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    
    # Calculate F1 scores for different thresholds
    f1_scores = []
    for threshold in thresholds:
        y_pred_temp = (y_proba >= threshold).astype(int)
        f1 = f1_score(y_test, y_pred_temp)
        f1_scores.append(f1)
    
    best_f1_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_f1_idx]
    best_f1 = f1_scores[best_f1_idx]
    
    print(f"Optimal threshold: {optimal_threshold:.3f} (F1: {best_f1:.3f})")
    
    # Find threshold where precision equals recall (balanced)
    balanced_idx = np.argmin(np.abs(precision[:-1] - recall[:-1]))
    balanced_threshold = thresholds[balanced_idx]
    balanced_precision = precision[balanced_idx]
    balanced_recall = recall[balanced_idx]
    
    print(f"Balanced threshold: {balanced_threshold:.3f} (Precision = {balanced_precision:.3f}, Recall = {balanced_recall:.3f})")
    
    # Evaluate with optimal threshold
    y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
    print("\nPerformance with optimal threshold (maximizing F1):")
    print(classification_report(y_test, y_pred_optimal))
    
    # Evaluate with balanced threshold
    y_pred_balanced = (y_proba >= balanced_threshold).astype(int)
    print("\nPerformance with balanced threshold (equal precision and recall):")
    print(classification_report(y_test, y_pred_balanced))
    
    return {
        'optimal_threshold': optimal_threshold,
        'best_f1': best_f1,
        'balanced_threshold': balanced_threshold,
        'balanced_precision': balanced_precision,
        'balanced_recall': balanced_recall,
        'precision': precision,
        'recall': recall,
        'thresholds': thresholds,
        'f1_scores': f1_scores,
        'best_f1_idx': best_f1_idx
    }

def analyze_with_shap(model, X_test, feature_names):
    """Analyze model with SHAP."""
    try:
        # Create a sample of the data for SHAP analysis
        X_sample = X_test.sample(min(100, len(X_test)), random_state=42)
        
        # Create the preprocessor
        preprocessor = model.named_steps['preprocessor']
        X_sample_processed = preprocessor.transform(X_sample)
        
        # Create a SHAP explainer with the classifier
        classifier = model.named_steps['classifier']
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_sample_processed)
        
        # Plot SHAP summary
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values[1], X_sample_processed, feature_names=feature_names)
        
        return shap_values
    except Exception as e:
        print(f"SHAP analysis failed: {str(e)}")
        print("Continue with other analyses")
        return None

def plot_heatmap(data, title="Heatmap", xlabel="X", ylabel="Y"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(data, annot=True, fmt=".2f", cmap="viridis")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()

def explain_with_lime(model, X_train, instance, feature_names, class_names, num_features=10):
    explainer = LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=feature_names,
        class_names=class_names,
        mode='classification'
    )
    explanation = explainer.explain_instance(
        data_row=instance,
        predict_fn=model.predict_proba,
        num_features=num_features
    )
    return explanation

def explain_with_shap(model, X_sample):
    explainer = shap.Explainer(model.predict_proba, X_sample)
    shap_values = explainer(X_sample)
    shap.summary_plot(shap_values, X_sample)
    return shap_values

def create_lasso_pipeline(preprocessor, C=1.0, random_state=42):
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            penalty='l1',
            solver='saga',
            C=C,
            random_state=random_state,
            max_iter=10000
        ))
    ])
    return pipeline

def explain_lasso_with_lime(lasso_model, X_train, instance, feature_names, class_names, num_features=10):
    return explain_with_lime(lasso_model, X_train, instance, feature_names, class_names, num_features)

def evaluate_fairness_with_fairlearn(y_true, y_pred, sensitive_features):
    metrics = {
        'selection_rate': selection_rate,
        'true_positive_rate': true_positive_rate,
        'false_positive_rate': false_positive_rate
    }
    mf = MetricFrame(metrics=metrics, y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features)
    print("Fairness metrics by group:")
    print(mf.by_group)
    return mf
