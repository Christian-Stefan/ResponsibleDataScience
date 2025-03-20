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
    precision_score, recall_score, f1_score, accuracy_score, auc, confusion_matrix
)
import shap
from lime.lime_tabular import LimeTabularExplainer
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pytorch_tabular import TabularModel
from pytorch_tabular.models import TabNetModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from tqdm import tqdm, trange

# Try importing imbalanced-learn, fall back to sklearn's Pipeline if not available
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    IMBALANCED_LEARN_AVAILABLE = True
except ImportError:
    ImbPipeline = Pipeline
    SMOTE = None
    IMBALANCED_LEARN_AVAILABLE = False
    print("Warning: imbalanced-learn not available. Using sklearn's Pipeline instead.")

from data_processing import create_preprocessing_pipeline
import config
from sklearn.inspection import partial_dependence, permutation_importance

def create_random_forest_pipeline(preprocessor, use_class_weights=True, n_estimators=100, max_depth=10,
                                  min_samples_split=10, random_state=1):
    """Create a pipeline with Random Forest classifier."""
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
    """Train the model."""
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_model(pipeline, X_test, y_test, model_name="Model"):
    """Evaluate the model's performance."""
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
        scores[strategy], _ = evaluate_model(pipelines[strategy], X_test, y_test,
                                                f"{strategy.capitalize()} Imputation")
    return pipelines, scores

def create_tabular_transformer_pipeline(preprocessor, learning_rate=0.01, batch_size=1024, max_epochs=10):
    """Create a pipeline with Tabular Transformer model."""
    # Define data configuration
    data_config = DataConfig(
        target=['hospital_death'],
        continuous_cols=preprocessor.named_transformers_['num'].get_feature_names_out(),
        categorical_cols=preprocessor.named_transformers_['cat'].get_feature_names_out(),
        continuous_feature_transform='standard_scaler',
        categorical_feature_transform='one_hot',
        handle_unknown_categories=True,
        handle_missing_values=True,
        num_workers=0
    )
    
    # Define trainer configuration
    trainer_config = TrainerConfig(
        max_epochs=max_epochs,
        batch_size=batch_size,
        accelerator='cpu',
        devices=1,
        early_stopping_patience=5,
        early_stopping_mode='min',
        checkpoints='best'
    )
    
    # Define optimizer configuration
    optimizer_config = OptimizerConfig(
        optimizer='Adam',
        optimizer_params={'lr': learning_rate},
        lr_scheduler='ReduceLROnPlateau',
        lr_scheduler_params={'patience': 3}
    )
    
    # Create TabNet model
    tabular_model = TabularModel(
        data_config=data_config,
        model_config=TabNetModel(
            task='classification',
            learning_rate=learning_rate,
            hidden_dims=[32, 16, 8],
            num_shared_layers=2,
            num_independent_layers=2,
            num_decision_steps=3,
            mask_type='sparsemax',
            num_attn_blocks=2,
            num_attn_heads=4,
            dropout=0.1
        ),
        optimizer_config=optimizer_config,
        trainer_config=trainer_config
    )
    
    # Create pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', tabular_model)
    ])
    
    return pipeline

def train_and_evaluate_all_imputation_strategies(X_train, X_test, y_train, y_test, numerical_cols, categorical_cols, strategies):
    """Train and evaluate models with different imputation strategies."""
    results = {}
    
    for strategy in strategies:
        print(f"\nTraining model with {strategy} imputation...")
        
        # Create preprocessing pipeline
        preprocessor = create_preprocessing_pipeline(numerical_cols, categorical_cols)
        
        # Create classifier
        classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=1,
            class_weight='balanced'  # Handle class imbalance
        )
        
        # Create and fit pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', classifier)
        ])
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        results[strategy] = {
            'model': pipeline,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'auc': auc_score,
            'pr_auc': pr_auc
        }
    
    return results

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
        ('classifier', RandomForestClassifier(random_state=1, class_weight='balanced', n_jobs=-1))
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

def get_feature_names(preprocessor):
    """Get feature names after preprocessing."""
    feature_names = []
    
    # Get numerical feature names
    if hasattr(preprocessor, 'named_transformers_') and 'num' in preprocessor.named_transformers_:
        num_features = preprocessor.named_transformers_['num'].get_feature_names_out()
        feature_names.extend(num_features)
    
    # Get categorical feature names
    if hasattr(preprocessor, 'named_transformers_') and 'cat' in preprocessor.named_transformers_:
        cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out()
        feature_names.extend(cat_features)
    
    return feature_names

def compare_with_apache_iv(model, X_test, y_test):
    """Compare model performance with APACHE IV predictions."""
    if 'apache_4a_hospital_death_prob' not in X_test.columns:
        print("APACHE IV predictions not available in the dataset.")
        return None
    
    # Get valid predictions
    valid_mask = ~X_test['apache_4a_hospital_death_prob'].isna()
    if valid_mask.sum() == 0:
        print("No valid APACHE IV predictions available.")
        return None
    
    X_valid = X_test[valid_mask]
    y_valid = y_test[valid_mask]
    
    # Get our model's predictions
    our_pred_proba = model.predict_proba(X_valid)[:, 1]
    our_auc = roc_auc_score(y_valid, our_pred_proba)
    
    # Get APACHE IV predictions
    apache_pred_proba = X_valid['apache_4a_hospital_death_prob']
    apache_auc = roc_auc_score(y_valid, apache_pred_proba)
    
    return {
        'our_auc': our_auc,
        'apache_auc': apache_auc,
        'n_samples': valid_mask.sum()
    }

def analyze_fairness_by_ethnicity(X_test, y_test, y_pred, y_proba):
    """Analyze model fairness across different ethnicities."""
    if 'ethnicity' not in X_test.columns:
        return None, None
    
    metrics_by_ethnicity = {}
    fairness_metrics = []
    
    for ethnicity in X_test['ethnicity'].unique():
        mask = X_test['ethnicity'] == ethnicity
        if mask.sum() > 0:
            group_preds = y_pred[mask]
            group_true = y_test[mask]
            group_proba = y_proba[mask]
            
            # Calculate metrics
            tn, fp, fn, tp = confusion_matrix(group_true, group_preds).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            metrics = {
                'n_samples': mask.sum(),
                'mortality_rate': group_true.mean(),
                'prediction_rate': group_preds.mean(),
                'true_positive_rate': recall_score(group_true, group_preds),
                'false_positive_rate': 1 - specificity,
                'auc': roc_auc_score(group_true, group_proba)
            }
            
            metrics_by_ethnicity[ethnicity] = metrics
            fairness_metrics.append(pd.Series(metrics, name=ethnicity))
    
    fairness_df = pd.concat(fairness_metrics)
    return fairness_df, metrics_by_ethnicity

def analyze_performance_by_body_system(X_test, y_test, model):
    """Analyze model performance by body system."""
    import pandas as pd
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
    
    # Initialize lists to store results
    results = []
    
    # Get unique body systems
    body_systems = X_test['apache_3j_bodysystem'].unique()
    
    # Calculate metrics for each body system
    for system in body_systems:
        mask = X_test['apache_3j_bodysystem'] == system
        if mask.sum() < 10:  # Skip if too few samples
            continue
            
        X_system = X_test[mask]
        y_system = y_test[mask]
        
        # Get predictions
        y_pred = model.predict(X_system)
        y_pred_proba = model.predict_proba(X_system)[:, 1]
        
        # Calculate metrics
        metrics = {
            'body_system': system,
            'n_samples': len(y_system),
            'mortality_rate': y_system.mean(),
            'prediction_rate': y_pred.mean(),
            'auc': roc_auc_score(y_system, y_pred_proba),
            'precision': precision_score(y_system, y_pred),
            'recall': recall_score(y_system, y_pred),
            'f1': f1_score(y_system, y_pred)
        }
        results.append(metrics)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Sort by number of samples
    df = df.sort_values('n_samples', ascending=False)
    
    # Calculate overall metrics
    overall_metrics = calculate_metrics(y_test, model.predict(X_test), model.predict_proba(X_test)[:, 1])
    
    return df, overall_metrics

def analyze_classification_thresholds(model, X_test, y_test):
    """Analyze model performance across different classification thresholds."""
    y_proba = model.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.1, 0.9, 0.05)
    results = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        # Calculate metrics
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        results.append({
            'threshold': threshold,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv
        })
    
    return pd.DataFrame(results)

def analyze_with_shap_cached(model, X_test, feature_names=None, cache_file=None):
    """Analyze model using SHAP values with caching support."""
    import os
    import pickle
    import shap
    
    if cache_file and os.path.exists(cache_file):
        print("Loading cached SHAP values...")
        try:
            with open(cache_file, 'rb') as f:
                shap_data = pickle.load(f)
                print("Successfully loaded SHAP values from cache")
                return shap_data
        except Exception as e:
            print(f"Error loading cached SHAP values: {str(e)}")
            print("Recalculating SHAP values...")
    
    try:
        # Get the classifier and preprocessor from the pipeline
        classifier = model.named_steps['classifier']
        preprocessor = model.named_steps['preprocessor']
        
        # Transform the test data
        print("Transforming test data...")
        X_test_transformed = preprocessor.transform(X_test)
        
        # Get feature names if not provided
        if feature_names is None:
            feature_names = preprocessor.get_feature_names_out()
        
        # Create background dataset for SHAP
        print("Creating background dataset...")
        background_data = X_test_transformed[:min(1000, len(X_test_transformed))]
        
        # Create prediction function for SHAP
        def model_predict(x):
            return classifier.predict_proba(x)[:, 1]
        
        # Create SHAP explainer with progress bar
        print("Creating SHAP explainer...")
        explainer = shap.KernelExplainer(model_predict, background_data)
        
        # Calculate SHAP values for a subset of test data
        X_test_subset = X_test_transformed[:min(50, len(X_test_transformed))]
        print("Calculating SHAP values...")
        with tqdm(total=len(X_test_subset), desc="SHAP analysis") as pbar:
            shap_values = []
            for i in range(len(X_test_subset)):
                shap_values.append(explainer.shap_values(X_test_subset[i:i+1])[0])
                pbar.update(1)
            shap_values = np.array(shap_values)
        
        # Prepare return data
        shap_data = {
            'shap_values': shap_values,
            'feature_names': feature_names,
            'X_test_subset': X_test_subset,
            'background_data': background_data
        }
        
        # Cache the results if cache_file is provided
        if cache_file:
            try:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                with open(cache_file, 'wb') as f:
                    pickle.dump(shap_data, f)
                print(f"Cached SHAP values to {cache_file}")
            except Exception as e:
                print(f"Error caching SHAP values: {str(e)}")
        
        return shap_data
        
    except Exception as e:
        print(f"Error in SHAP analysis: {str(e)}")
        return None

def explain_with_lime(model, X_train, instance, feature_names=None, class_names=None):
    """Explain a prediction using LIME.
    
    Args:
        model: The trained model
        X_train: Training data used to train the model
        instance: The instance to explain
        feature_names: List of feature names
        class_names: List of class names
    """
    import lime
    import lime.lime_tabular
    import numpy as np
    import os
    import config
    import pandas as pd
    
    try:
        # Get preprocessed training data
        preprocessor = model.named_steps['preprocessor']
        X_train_transformed = preprocessor.transform(X_train)
        
        # Convert to dense array if sparse
        if hasattr(X_train_transformed, 'toarray'):
            X_train_transformed = X_train_transformed.toarray()
        
        # Get feature names if not provided
        if feature_names is None:
            try:
                # Get feature names from preprocessor
                num_features = preprocessor.named_transformers_['num'].get_feature_names_out()
                cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out()
                feature_names = np.concatenate([num_features, cat_features])
            except (AttributeError, ValueError) as e:
                print(f"Warning: Could not get feature names from preprocessor: {str(e)}")
                feature_names = [f"feature_{i}" for i in range(X_train_transformed.shape[1])]
        
        # Ensure feature_names matches the number of features
        n_features = X_train_transformed.shape[1]
        if len(feature_names) != n_features:
            print(f"Warning: Number of feature names ({len(feature_names)}) doesn't match number of features ({n_features})")
            feature_names = [f"feature_{i}" for i in range(n_features)]
        
        # Create LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train_transformed,
            feature_names=feature_names,
            class_names=class_names if class_names else ['Survive', 'Die'],
            mode='classification',
            training_labels=None,
            discretize_continuous=True
        )
        
        # Transform the instance
        instance_df = pd.DataFrame([instance])
        instance_transformed = preprocessor.transform(instance_df)
        if hasattr(instance_transformed, 'toarray'):
            instance_transformed = instance_transformed.toarray()
        
        # Ensure instance_transformed is 2D
        if len(instance_transformed.shape) == 1:
            instance_transformed = instance_transformed.reshape(1, -1)
        
        # Create prediction function that works with preprocessed data
        def predict_fn(x):
            if len(x.shape) == 1:
                x = x.reshape(1, -1)
            return model.named_steps['classifier'].predict_proba(x)
        
        # Generate explanation
        explanation = explainer.explain_instance(
            instance_transformed[0],  # Use first row since we only have one instance
            predict_fn,
            num_features=10,
            top_labels=1,
            num_samples=5000
        )
        
        if explanation is not None:
            # Save explanation visualization
            output_path = os.path.join(config.RESULTS_DIR, 'lime_explanation.html')
            explanation.save_to_file(output_path)
            print(f"LIME explanation saved to {output_path}")
            return explanation
        else:
            print("Failed to generate LIME explanation")
            return None
        
    except Exception as e:
        print(f"Error in LIME explanation: {str(e)}")
        import traceback
        print(traceback.format_exc())  # Print full traceback for debugging
        return None

def plot_heatmap(data, title="Heatmap", xlabel="X", ylabel="Y"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(data, annot=True, fmt=".2f", cmap="viridis")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()

def create_lasso_pipeline(preprocessor, C=1.0, random_state=1):
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

def create_model_pipeline(preprocessor, classifier):
    """Create full model pipeline."""
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    return pipeline

def analyze_intersectional_fairness(model, X_test, y_test, protected_attributes):
    """Analyze fairness metrics across intersections of protected attributes."""
    results = []
    
    # Create progress bar for protected attributes
    pbar = tqdm(protected_attributes, desc="Analyzing fairness")
    
    # Create all possible combinations of protected attributes
    for attr in pbar:
        pbar.set_description(f"Analyzing {attr}")
        if attr not in X_test.columns:
            pbar.write(f"Protected attribute {attr} not available in the dataset.")
            continue
        
        for value in X_test[attr].unique():
            mask = X_test[attr] == value
            if mask.sum() > 0:
                X_subgroup = X_test[mask]
                y_subgroup = y_test[mask]
                
                # Get predictions
                y_pred = model.predict(X_subgroup)
                y_pred_proba = model.predict_proba(X_subgroup)[:, 1]
                
                # Calculate metrics
                prediction_rate = np.mean(y_pred)
                true_positive_rate = np.mean(y_pred[y_subgroup == 1])
                false_positive_rate = np.mean(y_pred[y_subgroup == 0])
                auc_score = roc_auc_score(y_subgroup, y_pred_proba)
                
                results.append({
                    'attribute': attr,
                    'value': value,
                    'n_samples': mask.sum(),
                    'prediction_rate': prediction_rate,
                    'true_positive_rate': true_positive_rate,
                    'false_positive_rate': false_positive_rate,
                    'auc': auc_score
                })
    
    return pd.DataFrame(results)

def get_model_candidates():
    """Get dictionary of candidate models for evaluation."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    import xgboost as xgb
    from pytorch_tabular import TabularModel
    from pytorch_tabular.models import TabNetModel
    from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
    
    models = {
        'logistic_regression': {
            'model': LogisticRegression(random_state=1),
            'params': {
                'classifier__C': [0.001, 0.01, 0.1, 1, 10],
                'classifier__class_weight': ['balanced', None],
                'classifier__max_iter': [1000]
            }
        },
        'decision_tree': {
            'model': DecisionTreeClassifier(random_state=1),
            'params': {
                'classifier__max_depth': [3, 5, 7, 10],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4],
                'classifier__class_weight': ['balanced', None]
            }
        },
        'random_forest': {
            'model': RandomForestClassifier(random_state=1),
            'params': {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__max_depth': [5, 10, 15],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4],
                'classifier__class_weight': ['balanced', None]
            }
        },
        'xgboost': {
            'model': xgb.XGBClassifier(random_state=1),
            'params': {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__max_depth': [3, 5, 7],
                'classifier__learning_rate': [0.01, 0.1, 0.3],
                'classifier__subsample': [0.8, 0.9, 1.0],
                'classifier__colsample_bytree': [0.8, 0.9, 1.0],
                'classifier__scale_pos_weight': [1, 5, 10]
            }
        },
        'tabnet': {
            'model': create_tabular_transformer_pipeline(None),  # preprocessor will be added later
            'params': {
                'classifier__model_config__num_decision_steps': [3, 5],
                'classifier__model_config__num_shared_layers': [1, 2],
                'classifier__model_config__num_independent_layers': [1, 2],
                'classifier__model_config__hidden_dims': [[32, 16], [64, 32]],
                'classifier__trainer_config__max_epochs': [10, 20],
                'classifier__trainer_config__batch_size': [512, 1024],
                'classifier__optimizer_config__learning_rate': [0.01, 0.001]
            }
        }
    }
    
    return models

def train_and_evaluate_all_models(X_train, X_test, y_train, y_test, numerical_cols, categorical_cols):
    """Train and evaluate all model candidates with hyperparameter tuning."""
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    import pandas as pd
    
    # Get preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(numerical_cols, categorical_cols)
    
    # Get model candidates
    models = get_model_candidates()
    results = {}
    
    # Create progress bar
    model_pbar = tqdm(models.items(), desc="Training models", total=len(models))
    
    for model_name, model_config in model_pbar:
        model_pbar.set_description(f"Training {model_name}")
        try:
            if model_name == 'tabnet':
                # Special handling for TabNet
                pipeline = create_tabular_transformer_pipeline(
                    preprocessor,
                    learning_rate=0.01,
                    batch_size=1024,
                    max_epochs=10
                )
                # Fit TabNet directly
                pipeline.fit(X_train, y_train)
                best_model = pipeline
            else:
                # Create standard pipeline
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', model_config['model'])
                ])
                
                # Create grid search
                grid_search = GridSearchCV(
                    pipeline,
                    model_config['params'],
                    cv=5,
                    scoring='roc_auc',
                    n_jobs=-1,
                    verbose=1
                )
                
                # Fit grid search
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
            
            # Make predictions
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
            
            # Store results
            results[model_name] = {
                'model': best_model,
                'best_params': grid_search.best_params_ if model_name != 'tabnet' else None,
                'best_score': grid_search.best_score_ if model_name != 'tabnet' else metrics['roc_auc'],
                'metrics': metrics
            }
            
            model_pbar.write(f"{model_name} training completed.")
            if model_name != 'tabnet':
                model_pbar.write(f"Best parameters: {grid_search.best_params_}")
                model_pbar.write(f"Best cross-validation score: {grid_search.best_score_:.3f}")
            model_pbar.write(f"Test set ROC AUC: {metrics['roc_auc']:.3f}")
            
        except Exception as e:
            model_pbar.write(f"Error training {model_name}: {str(e)}")
            continue
    
    return results

def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calculate comprehensive set of evaluation metrics."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, confusion_matrix,
        balanced_accuracy_score, brier_score_loss
    )
    
    # Calculate confusion matrix values
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'specificity': specificity, 
        'negative_predictive_value': npv,
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'pr_auc': average_precision_score(y_true, y_pred_proba),
        'brier_score': brier_score_loss(y_true, y_pred_proba)
    }
    
    return metrics

def analyze_fairness_metrics(y_true, y_pred, y_pred_proba, sensitive_features):
    """Calculate comprehensive fairness metrics for a protected attribute."""
    from fairlearn.metrics import (
        demographic_parity_difference,
        demographic_parity_ratio,
        equalized_odds_difference, 
        equalized_odds_ratio,
        true_positive_rate_difference,
        false_positive_rate_difference
    )
    
    # Ensure data types are appropriate for fairness metrics
    y_true = pd.Series(y_true).astype(int)
    y_pred = pd.Series(y_pred).astype(int)
    y_pred_proba = pd.Series(y_pred_proba).astype(float)
    
    # Convert sensitive_features to string to ensure no comparison issues
    if isinstance(sensitive_features, pd.Series):
        sensitive_features = sensitive_features.astype(str)
    else:
        sensitive_features = pd.Series(sensitive_features).astype(str)
    
    # Calculate group fairness metrics
    dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features)
    dp_ratio = demographic_parity_ratio(y_true, y_pred, sensitive_features=sensitive_features)
    eo_diff = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_features)
    eo_ratio = equalized_odds_ratio(y_true, y_pred, sensitive_features=sensitive_features)
    tpr_diff = true_positive_rate_difference(y_true, y_pred, sensitive_features=sensitive_features)
    fpr_diff = false_positive_rate_difference(y_true, y_pred, sensitive_features=sensitive_features)
    
    fairness_metrics = {
        'demographic_parity_difference': dp_diff,
        'demographic_parity_ratio': dp_ratio,
        'equalized_odds_difference': eo_diff,
        'equalized_odds_ratio': eo_ratio,
        'true_positive_rate_difference': tpr_diff,
        'false_positive_rate_difference': fpr_diff
    }
    
    return fairness_metrics

def train_fairness_constrained_model(X_train, y_train, sensitive_features, constraint_type='demographic_parity'):
    """Train a model with fairness constraints using fairlearn.
    
    Args:
        X_train: Training features
        y_train: Training target
        sensitive_features: Protected attribute
        constraint_type: Type of fairness constraint ('demographic_parity' or 'equalized_odds')
        
    Returns:
        A model trained with fairness constraints
    """
    from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    
    # Ensure data types are appropriate for fairness constraints
    y_train = pd.Series(y_train).astype(int)
    
    # Convert sensitive_features to string to ensure no comparison issues
    if isinstance(sensitive_features, pd.Series):
        sensitive_features = sensitive_features.astype(str)
    else:
        sensitive_features = pd.Series(sensitive_features).astype(str)
    
    print("Setting up fairness constrained model...")
    
    # Create preprocessing pipeline for categorical features
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns
    numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(drop='first', sparse=False), categorical_features)
        ],
        remainder='drop'
    )
    
    # Base estimator with preprocessing
    estimator = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=1))
    ])
    
    # Select constraint
    if constraint_type == 'demographic_parity':
        constraint = DemographicParity()
        print("Using Demographic Parity constraint")
    elif constraint_type == 'equalized_odds':
        constraint = EqualizedOdds()
        print("Using Equalized Odds constraint")
    else:
        raise ValueError(f"Unsupported constraint type: {constraint_type}")
    
    # Create mitigator
    mitigator = ExponentiatedGradient(
        estimator=estimator,
        constraints=constraint,
        eps=0.01
    )
    
    # Custom progress callback for fairness constrained training
    pbar = tqdm(desc="Training fairness constrained model", total=100)
    last_iter = [0]
    
    def progress_callback(iteration_number):
        # Update progress bar (assume max 100 iterations)
        if iteration_number > last_iter[0]:
            pbar.update(iteration_number - last_iter[0])
            last_iter[0] = iteration_number
    
    print("Starting fairness constrained training (this may take a while)...")
    mitigator.fit(X_train, y_train, sensitive_features=sensitive_features, progress_callback=progress_callback)
    pbar.close()
    
    return mitigator
