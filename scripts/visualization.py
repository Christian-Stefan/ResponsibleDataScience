import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, confusion_matrix, classification_report, auc

def set_visualization_style():
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams['figure.figsize'] = (10, 6)

def plot_missing_values(missing_percentages, top_n=20):
    plt.figure(figsize=(12, 8))
    missing_percentages.sort_values(ascending=False).head(top_n).plot(kind='bar')
    plt.title(f'Percentage of Missing Values in Top {top_n} Features', fontsize=14)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Missing Values (%)', fontsize=12)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def plot_target_distribution(df, target_col='hospital_death'):
    if target_col in df.columns:
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(x=target_col, data=df, palette="Set2")
        
        # Add count and percentage annotations
        total = len(df[target_col])
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width()/2.,
                    height + 0.1,
                    f'{height} ({height/total:.1%})',
                    ha="center") 
        
        plt.title('Mortality Outcome Distribution', fontsize=14)
        plt.xlabel('Mortality (0 = Survived, 1 = Died)', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.show()
        
        # Calculate and display death toll
        number_of_deaths = np.sum(df[target_col] == 1)
        number_of_survivors = np.sum(df[target_col] == 0)
        print(f"Dead: {number_of_deaths} ({number_of_deaths/total:.2%}) Survived: {number_of_survivors} ({number_of_survivors/total:.2%}) out of a total of {total}")
    else:
        print("Column 'hospital_death' not found in the dataset.")

def plot_age_distribution_by_mortality(X, y):
    """Plot age distribution by mortality outcome."""
    if 'age' in X.columns:
        plt.figure(figsize=(12, 6))
        sns.histplot(data=X.join(y), x='age', hue='hospital_death', 
                     bins=30, kde=True, palette='Set1', alpha=0.6)
        plt.title('Age Distribution by Mortality Outcome', fontsize=14)
        plt.xlabel('Age', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.legend(['Survived', 'Died'])
        plt.tight_layout()
        plt.show()
    else:
        print("Column 'age' not found in the dataset.")

def plot_physiological_vars_by_mortality(X, y, physiological_vars):
    """Plot physiological variables by mortality using boxplots."""
    available_vars = [var for var in physiological_vars if var in X.columns]

    if available_vars:
        plt.figure(figsize=(16, 4 * len(available_vars) // 2))
        for i, var in enumerate(available_vars, 1):
            plt.subplot(len(available_vars) // 2 + len(available_vars) % 2, 2, i)
            try:
                sns.boxplot(x='hospital_death', y=var, data=X.join(y),
                            palette='Set3')
                plt.title(f'{var} by Mortality', fontsize=12)
                plt.xlabel('Hospital Death', fontsize=10)
                plt.ylabel(var, fontsize=10)
            except Exception as e:
                print(f"Error plotting {var}: {str(e)}")
        plt.tight_layout()
        plt.show()
    else:
        print("None of the specified physiological variables found in the dataset.")

def plot_categorical_features_by_mortality(X, y, categorical_features):
    """Plot categorical features by mortality."""
    available_categorical = [col for col in categorical_features if col in X.columns]

    if available_categorical:
        for col in available_categorical:
            plt.figure(figsize=(14, 8))
            # Get value counts
            value_counts = X[col].value_counts()
            # Plot only top 15 categories if there are many
            if len(value_counts) > 15:
                top_categories = value_counts.index[:15].tolist()
                filtered_data = X[X[col].isin(top_categories)]
                filtered_y = y[filtered_data.index]
                
                # Create a countplot with percentages
                ax = sns.countplot(y=col, data=filtered_data.join(filtered_y), 
                                  hue='hospital_death', palette='viridis',
                                  order=value_counts.index[:15])
                plt.title(f'Top 15 Categories for {col} by Mortality Outcome', fontsize=14)
                plt.xlabel('Count', fontsize=12)
                plt.ylabel(col, fontsize=12)
            else:
                # Create a countplot for all categories
                ax = sns.countplot(y=col, data=X.join(y), 
                                  hue='hospital_death', palette='viridis',
                                  order=value_counts.index)
                plt.title(f'{col} by Mortality Outcome', fontsize=14)
                plt.xlabel('Count', fontsize=12)
                plt.ylabel(col, fontsize=12)
            
            plt.tight_layout()
            plt.show()
    else:
        print("None of the specified categorical features found in the dataset.")

def plot_correlation_matrix_of_clinical_features(X, y):
    """Plot correlation matrix of key clinical features."""
    if len(X.select_dtypes(include=['int64', 'float64']).columns) > 0:
        # Select a subset of numeric columns that might be clinically relevant
        clinical_cols = ['age', 'apache_4a_hospital_death_prob', 'apache_4a_icu_death_prob',
                         'heart_rate', 'map_apache', 'resp_rate', 'temperature', 'gcs_eyes_apache',
                         'gcs_motor_apache', 'gcs_verbal_apache', 'glucose_apache', 'creatinine_apache']
        
        # Filter to only include available columns
        available_clinical_cols = [col for col in clinical_cols if col in X.columns]
        
        if len(available_clinical_cols) > 0:
            # Include the target variable
            corr_data = X[available_clinical_cols].join(y)
            
            # Calculate correlation matrix
            corr_matrix = corr_data.corr()
            
            # Plot heatmap
            plt.figure(figsize=(14, 12))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', 
                       mask=mask, vmin=-1, vmax=1, center=0,
                       square=True, linewidths=.5)
            plt.title('Correlation Matrix of Key Clinical Features', fontsize=16)
            plt.tight_layout()
            plt.show()

def plot_enhanced_confusion_matrix(y_test, y_pred, strategy="Model"):
    """Plot an enhanced confusion matrix with counts and percentages."""
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix Raw Values for {strategy}:")
    print(cm)
    
    # Calculate percentages
    cm_percent = cm / cm.sum()
    
    # Create annotations with count and percentage
    annot = np.empty_like(cm, dtype=object)  # Changed from str to object
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f'{cm[i, j]:,d}\n({cm_percent[i, j]:.1%})'  # Added thousands separator

    # Plot heatmap with larger font size
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', 
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'],
                cbar=False, annot_kws={'size': 12})  # Increased font size
    plt.title(f'Confusion Matrix - {strategy.capitalize()} Imputation', fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_roc_curves_comparison(y_test, model_probas_dict):
    """Plot ROC curves for different models."""
    plt.figure(figsize=(10, 8))
    for strategy, y_proba in model_probas_dict.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_score = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, linewidth=2, label=f'{strategy.capitalize()} (AUC = {auc_score:.3f})')

    # Add random baseline
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Comparing Imputation Strategies', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_precision_recall_curves_comparison(y_test, model_probas_dict):
    """Plot Precision-Recall curves for different models."""
    plt.figure(figsize=(10, 8))
    for strategy, y_proba in model_probas_dict.items():
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        auc_pr = auc(recall, precision)  # Area under the PR curve
        plt.plot(recall, precision, linewidth=2, label=f'{strategy.capitalize()} (AUC-PR = {auc_pr:.3f})')

    # Add baseline based on positive class prevalence
    baseline = np.mean(y_test)
    plt.axhline(y=baseline, color='k', linestyle='--', label=f'Baseline (prevalence = {baseline:.3f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves - Comparing Imputation Strategies', fontsize=14)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_model_vs_apache_comparison(X_test, y_test, best_model, best_strategy, apache_auc, our_auc):
    """Plot comparison between our model and APACHE IV predictions."""
    valid_mask = ~np.isnan(X_test['apache_4a_hospital_death_prob'].values)

    if np.sum(valid_mask) > 0:
        if isinstance(y_test, pd.Series):
            y_test_valid = y_test.iloc[valid_mask]
        else:
            y_test_valid = y_test[valid_mask]

        if hasattr(X_test, 'iloc'):
            X_test_valid = X_test.iloc[valid_mask]
        else:
            X_test_valid = X_test[valid_mask]

        our_proba = best_model.predict_proba(X_test_valid)[:, 1]
        apache_proba = X_test['apache_4a_hospital_death_prob'].values[valid_mask]

        # Plot ROC curves
        plt.figure(figsize=(10, 8))

        fpr_our, tpr_our, _ = roc_curve(y_test_valid, our_proba)
        plt.plot(fpr_our, tpr_our, linewidth=2, label=f'Our Model ({best_strategy}) (AUC = {our_auc:.3f})')

        fpr_apache, tpr_apache, _ = roc_curve(y_test_valid, apache_proba)
        plt.plot(fpr_apache, tpr_apache, linewidth=2, label=f'APACHE IV (AUC = {apache_auc:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.500)')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Our Model vs. APACHE IV', fontsize=14)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # Create a calibration plot
        plt.figure(figsize=(10, 8))
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Fix for our model calibration curve
        our_bins = np.clip(np.digitize(our_proba, bin_edges) - 1, 0, n_bins - 1)
        bin_counts_our = np.bincount(our_bins, minlength=n_bins)
        # Avoid division by zero
        bin_counts_our = np.maximum(bin_counts_our, 1)
        our_bin_means = np.bincount(our_bins, weights=our_proba, minlength=n_bins) / bin_counts_our
        our_bin_true = np.bincount(our_bins, weights=y_test_valid.astype(int), minlength=n_bins) / bin_counts_our

        # Fix for APACHE calibration curve
        apache_bins = np.clip(np.digitize(apache_proba, bin_edges) - 1, 0, n_bins - 1)
        bin_counts_apache = np.bincount(apache_bins, minlength=n_bins)
        # Avoid division by zero
        bin_counts_apache = np.maximum(bin_counts_apache, 1)
        apache_bin_means = np.bincount(apache_bins, weights=apache_proba, minlength=n_bins) / bin_counts_apache
        apache_bin_true = np.bincount(apache_bins, weights=y_test_valid.astype(int),
                                      minlength=n_bins) / bin_counts_apache

        # Filter out bins with no data
        valid_our_mask = bin_counts_our > 1
        valid_apache_mask = bin_counts_apache > 1

        plt.plot(our_bin_means[valid_our_mask], our_bin_true[valid_our_mask], 's-',
                 label=f'Our Model ({best_strategy})')
        plt.plot(apache_bin_means[valid_apache_mask], apache_bin_true[valid_apache_mask], 'o-', label='APACHE IV')
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')

        plt.xlabel('Mean Predicted Probability', fontsize=12)
        plt.ylabel('Fraction of Positives', fontsize=12)
        plt.title('Calibration Plot - Our Model vs. APACHE IV', fontsize=14)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
def plot_fairness_metrics(fairness_df, metrics_by_ethnicity):
    """Plot fairness metrics by ethnicity."""
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 2, 1)
    ax = sns.barplot(x=fairness_df.index, y=fairness_df['auc'])
    plt.title('ROC AUC by Ethnicity', fontsize=12)
    plt.ylabel('AUC Score', fontsize=10)
    plt.xlabel('Ethnicity', fontsize=10)
    plt.xticks(rotation=45)
    plt.ylim(0.5, 1.0)
    
    plt.subplot(2, 2, 2)
    ax = sns.barplot(x=fairness_df.index, y=fairness_df['true_positive_rate'])
    plt.title('True Positive Rate by Ethnicity', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=10)
    plt.xlabel('Ethnicity', fontsize=10)
    plt.xticks(rotation=45)
    plt.ylim(0, 1.0)
    
    plt.subplot(2, 2, 3)
    width = 0.35
    x = np.arange(len(fairness_df.index))
    plt.bar(x - width/2, fairness_df['mortality_rate'], width, label='Actual Mortality Rate')
    plt.bar(x + width/2, fairness_df['prediction_rate'], width, label='Predicted Mortality Rate')
    plt.title('Actual vs. Predicted Mortality Rate by Ethnicity', fontsize=12)
    plt.xlabel('Ethnicity', fontsize=10)
    plt.ylabel('Rate', fontsize=10)
    plt.xticks(x, fairness_df.index, rotation=45)
    plt.legend(loc='best')
    plt.ylim(0, max(fairness_df['mortality_rate'].max(), fairness_df['prediction_rate'].max()) * 1.2)
    
    plt.subplot(2, 2, 4)
    for ethnicity, metrics in metrics_by_ethnicity.items():
        if 'fpr' in metrics and 'tpr' in metrics:
            plt.plot(metrics['fpr'], metrics['tpr'], label=f'{ethnicity} (AUC = {metrics["roc_auc"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=10)
    plt.ylabel('True Positive Rate', fontsize=10)
    plt.title('ROC Curves by Ethnicity', fontsize=12)
    plt.legend(loc="lower right", fontsize=8)
    
    plt.tight_layout()
    plt.show()

def plot_performance_by_body_system(body_system_df, body_system_metrics):
    """Plot model performance by body system."""
    plt.figure(figsize=(16, 14))
    
    plt.subplot(2, 2, 1)
    sns.barplot(x=body_system_df.index, y=body_system_df['auc'])
    plt.title('ROC AUC by Body System', fontsize=14)
    plt.xlabel('Body System', fontsize=10)
    plt.ylabel('AUC Score', fontsize=10)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0.5, 1.0)
    
    plt.subplot(2, 2, 2)
    body_system_df_sorted = body_system_df.sort_values('mortality_rate', ascending=False)
    sns.barplot(x=body_system_df_sorted.index, y=body_system_df_sorted['mortality_rate'])
    plt.title('Mortality Rate by Body System', fontsize=14)
    plt.xlabel('Body System', fontsize=10)
    plt.ylabel('Mortality Rate', fontsize=10)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)
    
    plt.subplot(2, 2, 3)
    body_system_df_count = body_system_df.sort_values('count', ascending=False)
    sns.barplot(x=body_system_df_count.index, y=body_system_df_count['count'])
    plt.title('Sample Count by Body System', fontsize=14)
    plt.xlabel('Body System', fontsize=10)
    plt.ylabel('Count', fontsize=10)
    plt.xticks(rotation=45, ha='right')
    
    plt.subplot(2, 2, 4)
    top_systems = body_system_df_count.head(5).index.tolist()
    for body_system in top_systems:
        if body_system in body_system_metrics and 'fpr' in body_system_metrics[body_system] and 'tpr' in body_system_metrics[body_system]:
            plt.plot(
                body_system_metrics[body_system]['fpr'], 
                body_system_metrics[body_system]['tpr'], 
                label=f'{body_system} (AUC = {body_system_metrics[body_system]["roc_auc"]:.3f})'
            )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=10)
    plt.ylabel('True Positive Rate', fontsize=10)
    plt.title('ROC Curves for Top 5 Body Systems by Sample Size', fontsize=14)
    plt.legend(loc="lower right", fontsize=8)
    
    plt.tight_layout()
    plt.show()

def plot_body_system_analysis_results(body_system_results, body_system_df, feature_names):
    """Plot the results of body system analysis."""
    body_system_auc = {system: results['results']['roc_auc'] for system, results in body_system_results.items()}
    auc_df = pd.DataFrame({'AUC': body_system_auc}).sort_values('AUC', ascending=False)
    print("\nAUC Scores for Models Trained Specifically for Each Body System:")
    display(auc_df)
    
    if body_system_df is not None and len(body_system_df) > 0:
        comparison_df = pd.DataFrame()
        comparison_df['Overall Model AUC'] = body_system_df['auc']
        specialized_aucs = {}
        for system in body_system_df.index:
            if system in auc_df.index:
                specialized_aucs[system] = auc_df.loc[system, 'AUC']
        comparison_df['Specialized Model AUC'] = pd.Series(specialized_aucs)
        comparison_df['Improvement'] = comparison_df['Specialized Model AUC'] - comparison_df['Overall Model AUC']
        comparison_df = comparison_df.sort_values('Improvement', ascending=False)
        
        print("\nComparing Overall vs. Specialized Models by Body System:")
        display(comparison_df)
        
        plt.figure(figsize=(12, 8))
        comparison_df_plot = comparison_df.dropna().head(10)
        x = np.arange(len(comparison_df_plot.index))
        width = 0.35
        plt.bar(x - width/2, comparison_df_plot['Overall Model AUC'], width, label='Overall Model AUC', color='skyblue')
        plt.bar(x + width/2, comparison_df_plot['Specialized Model AUC'], width, label='Specialized Model AUC', color='orange')
        plt.xlabel('Body System', fontsize=12)
        plt.ylabel('AUC Score', fontsize=12)
        plt.title('Comparison of Overall vs. Specialized Models by Body System', fontsize=14)
        plt.xticks(x, comparison_df_plot.index, rotation=45, ha='right')
        plt.legend()
        plt.ylim(0.5, 1.0)
        plt.tight_layout()
        plt.show()
    
    if len(auc_df) > 0:
        top_system = auc_df.index[0]
        print(f"\nTop 10 important features for {top_system}:")
        feature_importances = body_system_results[top_system]['feature_importances']
        feature_names_subset = feature_names[:len(feature_importances)]
        importance_df = pd.DataFrame({
            'Feature': feature_names_subset,
            'Importance': feature_importances
        }).sort_values('Importance', ascending=False).head(10)
        display(importance_df)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
        plt.title(f'Top 10 Features for {top_system}', fontsize=14)
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        plt.show()

def plot_threshold_analysis_results(thresholds_analysis):
    """Plot the results of optimal threshold analysis."""
    optimal_threshold = thresholds_analysis['optimal_threshold']
    best_f1 = thresholds_analysis['best_f1']
    balanced_threshold = thresholds_analysis['balanced_threshold']
    balanced_precision = thresholds_analysis['balanced_precision']
    balanced_recall = thresholds_analysis['balanced_recall']
    precision = thresholds_analysis['precision']
    recall = thresholds_analysis['recall']
    thresholds = thresholds_analysis['thresholds']
    f1_scores = thresholds_analysis['f1_scores']
    best_f1_idx = thresholds_analysis['best_f1_idx']
    
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.plot(recall, precision, 'b-', label='Precision-Recall curve')
    plt.scatter(recall[best_f1_idx], precision[best_f1_idx], marker='o', color='red', label=f'Optimal F1: {best_f1:.3f}')
    plt.scatter(balanced_recall, balanced_precision, marker='s', color='green', label=f'Balanced P-R: {(balanced_precision + balanced_recall)/2:.3f}')
    plt.xlabel('Recall', fontsize=10)
    plt.ylabel('Precision', fontsize=10)
    plt.title('Precision-Recall Curve with Optimal Thresholds', fontsize=12)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(thresholds, f1_scores, 'g-')
    plt.scatter(optimal_threshold, best_f1, marker='o', color='red', label=f'Optimal: {optimal_threshold:.3f}')
    plt.axvline(x=0.5, color='gray', linestyle='--', label='Default (0.5)')
    plt.xlabel('Threshold', fontsize=10)
    plt.ylabel('F1 Score', fontsize=10)
    plt.title('F1 Score vs. Threshold', fontsize=12)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.plot(thresholds, precision[:-1], 'b-', label='Precision')
    plt.plot(thresholds, recall[:-1], 'r-', label='Recall')
    plt.xlabel('Threshold', fontsize=10)
    plt.ylabel('Score', fontsize=10)
    plt.title('Precision and Recall vs. Threshold', fontsize=12)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    plt.plot(thresholds, f1_scores, 'm-', label='F1 Score')
    plt.xlabel('Threshold', fontsize=10)
    plt.ylabel('F1 Score', fontsize=10)
    plt.title('F1 Score vs. Threshold', fontsize=12)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df, method='kendall', figsize=(20, 20)):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = df[numeric_cols].corr(method=method)
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True, vmin=-1, vmax=1)
    plt.title(f'Correlation Matrix of Numeric Features using {method.capitalize()} Method')
    plt.tight_layout()
    plt.show()

def plot_categorical_features(df, categorical_cols=None):
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in categorical_cols:
        plt.figure()
        sns.countplot(y=col, data=df, palette="viridis", order=df[col].value_counts().index)
        plt.title(f'Count Plot for {col}')
        plt.xlabel('Count')
        plt.ylabel(col)
        plt.show()

def plot_apache_mortality_by_bodysystem(df):
    if 'apache_2_bodysystem' in df.columns and 'hospital_death' in df.columns and 'apache_4a_hospital_death_prob' in df.columns:
        df["x_numeric"] = df["apache_2_bodysystem"].astype("category").cat.codes
        categories = df["apache_2_bodysystem"].astype("category").cat.categories
        df_red = df[df["hospital_death"] == 1].copy()
        df_green = df[df["hospital_death"] == 0].copy()
        df_red.loc[:, "x_numeric"] = df_red["x_numeric"] + 0.1
        plt.figure(figsize=(15, 6))
        plt.scatter(df_green["x_numeric"], df_green["apache_4a_hospital_death_prob"], color="green", label="hospital_death = 0", alpha=0.6)
        plt.scatter(df_red["x_numeric"], df_red["apache_4a_hospital_death_prob"], color="red", label="hospital_death = 1", alpha=0.6)
        plt.xticks(ticks=np.arange(len(categories)), labels=categories, rotation=45)
        plt.title("APACHE IV Hospital Death Probability by APACHE II Bodysystem and Actual Outcome (Colored)")
        plt.xlabel("APACHE II Bodysystem")
        plt.ylabel("APACHE IV Hospital Death Probability")
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("Required columns for this plot are not available in the dataset.")

def plot_histograms_by_bodysystem(df):
    if 'apache_2_bodysystem' in df.columns and 'hospital_death' in df.columns and 'apache_4a_hospital_death_prob' in df.columns:
        categories = df["apache_2_bodysystem"].astype("category").cat.categories
        df_green = df[df["hospital_death"] == 0].copy()
        df_red = df[df["hospital_death"] == 1].copy()
        num_categories = len(categories)
        fig, axs = plt.subplots(2, num_categories, figsize=(20, 10), squeeze=False)
        for i, cat in enumerate(categories):
            subset_green = df_green[df_green["apache_2_bodysystem"] == cat]
            axs[0, i].hist(subset_green["apache_4a_hospital_death_prob"], bins=20, color="green", alpha=0.6, edgecolor="black")
            axs[0, i].set_title(f"{cat}\n(Samples: {len(subset_green)})")
            axs[0, i].set_xlabel("Death Prob")
            axs[0, i].set_ylabel("Count")
            subset_red = df_red[df_red["apache_2_bodysystem"] == cat]
            axs[1, i].hist(subset_red["apache_4a_hospital_death_prob"], bins=20, color="red", alpha=0.6, edgecolor="black")
            axs[1, i].set_title(f"{cat}\n(Samples: {len(subset_red)})")
            axs[1, i].set_xlabel("Death Prob")
            axs[1, i].set_ylabel("Count")
        fig.suptitle("Histogram of Apache 4a Death Prob by Bodysystem and Actual Death")
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        plt.show()
    else:
        print("Required columns for this plot are not available in the dataset.")

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def plot_roc_curve(y_true, y_proba, label=None, title="ROC Curve"):
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    if label:
        plt.plot(fpr, tpr, label=f'{label} (AUC = {auc:.3f})')
    else:
        plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    plt.show()
    return auc

def plot_multiple_roc_curves(y_true, y_probas_dict, title="ROC Curves Comparison"):
    plt.figure(figsize=(10, 8))
    for label, y_proba in y_probas_dict.items():
        mask = ~np.isnan(y_proba)
        y_true_valid = y_true[mask]
        y_proba_valid = y_proba[mask]
        if len(y_true_valid) == 0:
            print(f"No valid predictions for {label}. Skipping.")
            continue
        fpr, tpr, _ = roc_curve(y_true_valid, y_proba_valid)
        auc = roc_auc_score(y_true_valid, y_proba_valid)
        plt.plot(fpr, tpr, label=f'{label} (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_precision_recall_curve(y_true, y_proba, label=None, title="Precision-Recall Curve"):
    plt.figure(figsize=(8, 6))
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    if label:
        plt.plot(recall, precision, label=label)
    else:
        plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    if label:
        plt.legend()
    plt.show()

def plot_feature_importance(feature_names, feature_importances, top_n=20, title="Feature Importance"):
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values('Importance', ascending=False)
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(top_n))
    plt.title(title)
    plt.tight_layout()
    plt.show()
    return importance_df