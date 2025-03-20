import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, confusion_matrix, classification_report, auc
import config
import os

# Add color mappings at the top of the file after imports
# Color mappings for consistent visualization
ETHNICITY_COLORS = {
    'Caucasian': '#2ecc71',      # Green
    'African American': '#3498db', # Blue
    'Hispanic': '#e74c3c',        # Red
    'Asian': '#f1c40f',          # Yellow
    'Native American': '#9b59b6', # Purple
    'Other/Unknown': '#95a5a6'    # Gray
}

BODY_SYSTEM_COLORS = {
    'Cardiovascular': '#2ecc71',    # Green
    'Respiratory': '#3498db',       # Blue
    'Neurologic': '#e74c3c',        # Red
    'Gastrointestinal': '#f1c40f',  # Yellow
    'Metabolic': '#9b59b6',         # Purple
    'Trauma': '#1abc9c',            # Turquoise
    'Sepsis': '#e67e22',           # Orange
    'Other': '#95a5a6'             # Gray
}

AGE_GROUP_COLORS = {
    'Young': '#2ecc71',    # Green
    'Middle': '#3498db',   # Blue
    'Senior': '#e74c3c',   # Red
    'Elderly': '#9b59b6'   # Purple
}

# Color schemes
MORTALITY_COLORS = {
    'Survive': '#3498db',  # Blue
    'Die': '#e74c3c'      # Red
}

METRIC_COLORS = {
    'Prediction Rate': '#3498db',    # Blue
    'True Positive Rate': '#2ecc71',  # Green
    'False Positive Rate': '#e74c3c'  # Red
}

def set_visualization_style():
    """Set the visualization style for all plots."""
    # Clear any existing plots
    plt.close('all')
    
    # Force Agg backend
    matplotlib.use('Agg')
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    # Configure matplotlib parameters
    plt.rcParams.update({
        'figure.figsize': config.FIGURE_SIZE,
        'figure.dpi': config.DPI,
        'savefig.dpi': config.DPI,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'figure.autolayout': True,
        'axes.grid': True,
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'image.cmap': 'viridis',
        'axes.grid.which': 'both',
        'grid.alpha': 0.3,
        'lines.linewidth': 2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.max_open_warning': 0  # Suppress warning about too many open figures
    })

def save_plot(fig, filename):
    """Helper function to save plots with proper settings."""
    try:
        # Ensure the results directory exists
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        
        # Make sure the figure is not empty
        if not fig.get_axes():
            print(f"Warning: Empty figure detected for {filename}")
            return
        
        # Draw the figure to make sure all artists are rendered
        fig.canvas.draw()
        
        # Save with high quality settings
        output_path = os.path.join(config.RESULTS_DIR, filename)
        
        # Save with explicit settings
        fig.savefig(
            output_path,
            dpi=config.DPI,
            bbox_inches='tight',
            pad_inches=0.1,
            format='png',
            facecolor='white',
            edgecolor='none',
            transparent=False
        )
        
        # Verify the file was created and has content
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            if file_size > 0:
                print(f"Successfully saved plot to {output_path} (size: {file_size/1024:.1f}KB)")
            else:
                print(f"Warning: Plot file {output_path} is empty")
        else:
            print(f"Warning: Plot file {output_path} was not created")
        
        # Close the figure
        plt.close(fig)
        
    except Exception as e:
        print(f"Error saving plot {filename}: {str(e)}")
        plt.close('all')  # Clean up all figures in case of error

def plot_missing_values(missing_percentages):
    """Plot missing values in the dataset."""
    try:
        # Create a new figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Sort and get top 20 missing values
        missing_data = missing_percentages.sort_values(ascending=False).head(20)
        
        # Create the bar plot
        bars = ax.bar(range(len(missing_data)), missing_data.values)
        
        # Customize the plot
        ax.set_title('Percentage of Missing Values in Top 20 Features', pad=20)
        ax.set_xlabel('Features')
        ax.set_ylabel('Missing Values (%)')
        ax.set_xticks(range(len(missing_data)))
        ax.set_xticklabels(missing_data.index, rotation=90)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot
        save_plot(fig, 'missing_values.png')
        
    except Exception as e:
        print(f"Error in plot_missing_values: {str(e)}")
        plt.close('all')  # Clean up in case of error

def plot_target_distribution(df):
    """Plot the distribution of the target variable."""
    fig = plt.figure(figsize=(8, 6))
    df[config.TARGET_COLUMN].value_counts(normalize=True).plot(kind='bar')
    plt.title('Distribution of Hospital Mortality')
    plt.xlabel('Outcome')
    plt.ylabel('Proportion')
    plt.xticks([0, 1], ['Survive', 'Die'])
    plt.tight_layout()
    save_plot(fig, 'target_distribution.png')

def plot_age_distribution_by_mortality(X, y):
    """Plot age distribution by mortality outcome."""
    fig = plt.figure(figsize=(10, 6))
    sns.boxplot(x=y, y=X['age'], palette=[MORTALITY_COLORS['Survive'], MORTALITY_COLORS['Die']])
    plt.title('Age Distribution by Mortality Outcome')
    plt.xlabel('Mortality')
    plt.ylabel('Age')
    plt.xticks([0, 1], ['Survive', 'Die'])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot(fig, 'age_distribution.png')

def plot_physiological_vars_by_mortality(X, y, variables):
    """Plot physiological variables by mortality outcome."""
    n_vars = len(variables)
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(15, 5*n_rows))
    for i, var in enumerate(variables, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.boxplot(x=y, y=X[var])
        plt.title(f'{var} by Mortality')
        plt.xlabel('Mortality')
        plt.ylabel(var)
        plt.xticks([0, 1], ['Survive', 'Die'])
    plt.tight_layout()
    save_plot(fig, 'physiological_vars.png')

def plot_categorical_features_by_mortality(X, y, features):
    """Plot categorical features by mortality outcome."""
    n_features = len(features)
    n_cols = 2
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(15, 5*n_rows))
    for i, feature in enumerate(features, 1):
        plt.subplot(n_rows, n_cols, i)
        # Create a temporary DataFrame with the feature and target
        temp_df = pd.DataFrame({'feature': X[feature], 'target': y})
        mortality_rates = temp_df.groupby('feature')['target'].mean()
        
        # Choose appropriate color palette based on feature type
        if feature == 'ethnicity':
            colors = [ETHNICITY_COLORS.get(cat, '#95a5a6') for cat in mortality_rates.index]
        elif feature in ['apache_2_bodysystem', 'apache_3j_bodysystem']:
            colors = [BODY_SYSTEM_COLORS.get(cat, '#95a5a6') for cat in mortality_rates.index]
        else:
            # Use default color palette for other features
            colors = sns.color_palette("husl", len(mortality_rates))
        
        # Create bar plot with appropriate colors
        plt.bar(range(len(mortality_rates)), mortality_rates, color=colors)
        plt.title(f'Mortality Rate by {feature}')
        plt.xlabel(feature)
        plt.ylabel('Mortality Rate')
        plt.xticks(range(len(mortality_rates)), mortality_rates.index, rotation=45, ha='right')
    
    plt.tight_layout()
    save_plot(fig, 'categorical_features.png')

def plot_correlation_matrix_of_clinical_features(X, y):
    """Plot correlation matrix of clinical features with improved styling."""
    clinical_features = X.select_dtypes(include=['int64', 'float64']).columns
    correlation_matrix = X[clinical_features].corrwith(y)
    
    fig = plt.figure(figsize=(12, 8))
    
    # Sort and get top 20 features
    top_20_correlations = correlation_matrix.abs().sort_values(ascending=False).head(20)
    features = top_20_correlations.index
    correlations = correlation_matrix[features]
    
    # Create bar plot with gradient colors based on correlation values
    colors = plt.cm.RdBu(np.linspace(0, 1, len(correlations)))
    bars = plt.bar(range(len(correlations)), correlations, color=colors)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    plt.title('Top 20 Features Correlated with Mortality')
    plt.xlabel('Features')
    plt.ylabel('Correlation')
    plt.xticks(range(len(features)), features, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot(fig, 'correlation_matrix.png')

def plot_enhanced_confusion_matrix(y_true, y_pred, strategy):
    """Plot enhanced confusion matrix with percentages."""
    cm = confusion_matrix(y_true, y_pred)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percentage, annot=True, fmt='.2%', cmap='Blues')
    plt.title(f'Confusion Matrix - {strategy.capitalize()} Imputation')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    save_plot(fig, f'confusion_matrix_{strategy}.png')

def plot_roc_curves_comparison(y_true, y_pred_dict):
    """Plot ROC curves for different models."""
    fig = plt.figure(figsize=(10, 8))
    for name, y_pred in y_pred_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc_score = np.trapz(tpr, fpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    plt.tight_layout()
    save_plot(fig, 'roc_curves.png')

def plot_precision_recall_curves_comparison(y_true, y_pred_dict):
    """Plot Precision-Recall curves for different models."""
    fig = plt.figure(figsize=(10, 8))
    for name, y_pred in y_pred_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        auc_score = np.trapz(precision, recall)
        plt.plot(recall, precision, label=f'{name} (AUC = {auc_score:.3f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves Comparison')
    plt.legend()
    plt.tight_layout()
    save_plot(fig, 'precision_recall_curves.png')

def plot_feature_importance(importance_df, title='Feature Importance', top_n=20, figsize=(12, 8), save_path=None):
    """
    Plot feature importance from a DataFrame.
    
    Args:
        importance_df: DataFrame with columns 'Feature' and 'Importance'
        title: Plot title
        top_n: Number of top features to display
        figsize: Figure size
        save_path: Path to save the figure
    """
    # Take top N features
    df = importance_df.head(top_n)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create horizontal bar plot
    sns.barplot(x='Importance', y='Feature', data=df, ax=ax)
    
    # Add title and labels
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    
    # Add grid
    ax.grid(True, axis='x', alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_model_vs_apache_comparison(X_test, y_test, model, strategy, apache_auc, our_auc):
    """Plot comparison between our model and APACHE IV."""
    valid_mask = ~X_test['apache_4a_hospital_death_prob'].isna()
    
    if valid_mask.sum() > 0:
        plt.figure(figsize=(10, 8))
        fpr, tpr, _ = roc_curve(y_test[valid_mask], model.predict_proba(X_test[valid_mask])[:, 1])
        plt.plot(fpr, tpr, label=f'Our Model ({strategy}) (AUC = {our_auc:.3f})')
        
        fpr_apache, tpr_apache, _ = roc_curve(
            y_test[valid_mask], 
            X_test.loc[valid_mask, 'apache_4a_hospital_death_prob']
        )
        plt.plot(fpr_apache, tpr_apache, label=f'APACHE IV (AUC = {apache_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Our Model vs. APACHE IV')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(config.RESULTS_DIR, 'model_vs_apache.png'))
        plt.close()

def plot_performance_by_body_system(body_system_df, metrics):
    """Plot performance metrics by body system."""
    if body_system_df is None or metrics is None:
        print("No body system data available for plotting.")
        return
    
    # Plot AUC scores with color gradient
    fig = plt.figure(figsize=(12, 6))
    colors = plt.cm.RdYlBu(np.linspace(0.2, 0.8, len(body_system_df)))
    body_system_df['auc'].plot(kind='bar', color=colors)
    plt.title('Model Performance (AUC) by Body System')
    plt.xlabel('Body System')
    plt.ylabel('AUC Score')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot(fig, 'body_system_auc.png')
    
    # Plot mortality rates vs prediction rates
    fig = plt.figure(figsize=(12, 6))
    mortality_rates = [metrics[system]['mortality_rate'] for system in body_system_df.index]
    prediction_rates = [metrics[system]['prediction_rate'] for system in body_system_df.index]
    
    x = np.arange(len(body_system_df.index))
    width = 0.35
    
    plt.bar(x - width/2, mortality_rates, width, label='Mortality Rate', color='#3498db')
    plt.bar(x + width/2, prediction_rates, width, label='Prediction Rate', color='#e67e22')
    
    plt.title('Mortality and Prediction Rates by Body System')
    plt.xlabel('Body System')
    plt.ylabel('Rate')
    plt.xticks(x, body_system_df.index, rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot(fig, 'body_system_rates.png')

def plot_body_system_analysis_results(body_system_results, body_system_df, feature_names):
    """Plot detailed body system analysis results."""
    # Plot AUC by body system
    plt.figure(figsize=(12, 6))
    body_system_auc = {system: results['results']['roc_auc'] 
                      for system, results in body_system_results.items()}
    sns.barplot(x=list(body_system_auc.keys()), y=list(body_system_auc.values()))
    plt.title('AUC Score by Body System')
    plt.xlabel('Body System')
    plt.ylabel('AUC Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, 'body_system_auc.png'))
    plt.close()
    
    # Plot feature importance by body system
    for system, results in body_system_results.items():
        plt.figure(figsize=(12, 6))
        importance_df = pd.DataFrame({
            'Feature': feature_names[:len(results['feature_importances'])],
            'Importance': results['feature_importances']
        }).sort_values('Importance', ascending=False)
        
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(10))
        plt.title(f'Top 10 Feature Importances - {system}')
        plt.tight_layout()
        plt.savefig(os.path.join(config.RESULTS_DIR, f'feature_importance_{system}.png'))
        plt.close()

def plot_threshold_analysis_results(thresholds_analysis):
    """Plot results of threshold analysis."""
    plt.figure(figsize=(12, 8))
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        plt.plot(thresholds_analysis['threshold'], thresholds_analysis[metric], label=metric)
    
    plt.xlabel('Classification Threshold')
    plt.ylabel('Score')
    plt.title('Model Performance Across Different Thresholds')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, 'threshold_analysis.png'))
    plt.close()

def plot_fairness_metrics(fairness_df, metrics_by_ethnicity):
    """Plot fairness metrics across different ethnic groups."""
    # Ensure fairness_df has unique index
    fairness_df = fairness_df.reset_index()
    fairness_df = fairness_df.rename(columns={'index': 'Ethnicity'})
    
    # Plot all metrics together
    fig = plt.figure(figsize=(15, 8))
    metrics = ['Mortality Rate', 'True Positive Rate', 'False Positive Rate']
    
    # Create positions for the bars
    x = np.arange(len(fairness_df))
    width = 0.25
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        plt.bar(x + i*width, fairness_df[metric], width, 
               label=metric,
               color=METRIC_COLORS.get(metric, '#95a5a6'))
    
    plt.xlabel('Ethnicity')
    plt.ylabel('Rate')
    plt.title('Fairness Metrics by Ethnicity')
    plt.xticks(x + width, fairness_df['Ethnicity'], rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot(fig, 'fairness_metrics_comparison.png')

def plot_roc_curves_by_ethnicity(X_test, y_test, y_proba):
    """Plot ROC curves for different ethnicities."""
    if 'ethnicity' not in X_test.columns:
        print("Warning: Ethnicity column not found in the dataset.")
        return
    
    fig = plt.figure(figsize=(10, 8))
    
    # Plot ROC curve for each ethnicity
    for ethnicity in X_test['ethnicity'].unique():
        mask = X_test['ethnicity'] == ethnicity
        if mask.sum() > 0:
            fpr, tpr, _ = roc_curve(y_test[mask], y_proba[mask])
            auc_score = roc_auc_score(y_test[mask], y_proba[mask])
            plt.plot(fpr, tpr, label=f'{ethnicity} (AUC = {auc_score:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves by Ethnicity')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    save_plot(fig, 'roc_curves_by_ethnicity.png')

def plot_roc_curves_by_body_system(X_test, y_test, y_proba, top_n=5):
    """Plot ROC curves for the top N body systems by sample size."""
    if 'apache_2_bodysystem' not in X_test.columns:
        print("Warning: apache_2_bodysystem column not found in data")
        return
    
    # Calculate sample counts for each body system
    body_system_counts = X_test['apache_2_bodysystem'].value_counts()
    top_systems = body_system_counts.head(top_n).index
    
    plt.figure(figsize=(10, 8))
    metrics_list = []
    
    for system in top_systems:
        mask = X_test['apache_2_bodysystem'] == system
        if sum(mask) < 10:  # Skip if too few samples
            continue
            
        y_test_system = y_test[mask]
        y_proba_system = y_proba[mask]
        
        fpr, tpr, _ = roc_curve(y_test_system, y_proba_system)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'{system} (AUC = {roc_auc:.2f}, n={sum(mask)})',
                color=BODY_SYSTEM_COLORS.get(system, '#95a5a6'))
        
        # Store metrics for return
        metrics_dict = {
            'Body System': system,
            'Sample Size': sum(mask),
            'ROC AUC': roc_auc
        }
        metrics_list.append(metrics_dict)
    
    metrics_df = pd.DataFrame(metrics_list)
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves by Body System (Top {top_n} by Sample Size)')
    plt.legend(loc="lower right", bbox_to_anchor=(1.2, 0))
    plt.tight_layout()
    save_plot(plt.gcf(), 'roc_curves_by_body_system.png')
    
    return metrics_df

def plot_comprehensive_ethnicity_analysis(X_test, y_test, y_proba):
    """Create a comprehensive visualization of model performance across ethnicities."""
    if 'ethnicity' not in X_test.columns:
        print("Warning: Ethnicity column not found in the dataset.")
        return
    
    # Calculate metrics for each ethnicity
    metrics = {}
    for ethnicity in X_test['ethnicity'].unique():
        mask = X_test['ethnicity'] == ethnicity
        if mask.sum() > 0:
            metrics[ethnicity] = {
                'n_samples': mask.sum(),
                'mortality_rate': y_test[mask].mean(),
                'auc': roc_auc_score(y_test[mask], y_proba[mask])
            }
    
    # Create DataFrame for plotting
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
    metrics_df = metrics_df.sort_values('n_samples', ascending=False)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. ROC AUC by Ethnicity (top left)
    plt.subplot(2, 2, 1)
    sns.barplot(x=metrics_df.index, y=metrics_df['auc'],
                palette=[ETHNICITY_COLORS.get(e, '#95a5a6') for e in metrics_df.index])
    plt.title('ROC AUC by Ethnicity')
    plt.xlabel('Ethnicity')
    plt.ylabel('AUC Score')
    plt.xticks(rotation=45)
    
    # 2. Mortality Rate by Ethnicity (top right)
    plt.subplot(2, 2, 2)
    sns.barplot(x=metrics_df.index, y=metrics_df['mortality_rate'],
                palette=[ETHNICITY_COLORS.get(e, '#95a5a6') for e in metrics_df.index])
    plt.title('Mortality Rate by Ethnicity')
    plt.xlabel('Ethnicity')
    plt.ylabel('Mortality Rate')
    plt.xticks(rotation=45)
    
    # 3. Sample Count by Ethnicity (bottom left)
    plt.subplot(2, 2, 3)
    sns.barplot(x=metrics_df.index, y=metrics_df['n_samples'],
                palette=[ETHNICITY_COLORS.get(e, '#95a5a6') for e in metrics_df.index])
    plt.title('Sample Count by Ethnicity')
    plt.xlabel('Ethnicity')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    # 4. ROC Curves for Top 5 Ethnicities by Sample Size (bottom right)
    plt.subplot(2, 2, 4)
    top_5_ethnicities = metrics_df.head(5).index
    
    for ethnicity in top_5_ethnicities:
        mask = X_test['ethnicity'] == ethnicity
        if mask.sum() > 0:
            fpr, tpr, _ = roc_curve(y_test[mask], y_proba[mask])
            auc_score = metrics_df.loc[ethnicity, 'auc']
            plt.plot(fpr, tpr, 
                    label=f'{ethnicity} (AUC = {auc_score:.3f})',
                    color=ETHNICITY_COLORS.get(ethnicity, '#95a5a6'))
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Top 5 Ethnicities by Sample Size')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    save_plot(fig, 'comprehensive_ethnicity_analysis.png')
    
    return metrics_df

def plot_confusion_matrix(cm, title, filename):
    """Plot confusion matrix with improved styling."""
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.title(title)
    plt.tight_layout()
    save_plot(fig, filename)

def plot_all_confusion_matrices(y_true, y_pred, y_pred_proba):
    """Plot confusion matrices with different thresholds and styling.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    import numpy as np
    
    # Set up the figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot standard confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Confusion Matrix (Default Threshold)')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    # Plot confusion matrix with optimized threshold
    # Find optimal threshold using ROC curve
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    # Create confusion matrix with optimal threshold
    y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
    cm_optimal = confusion_matrix(y_true, y_pred_optimal)
    
    sns.heatmap(cm_optimal, annot=True, fmt='d', cmap='Blues', ax=axes[1])
    axes[1].set_title(f'Confusion Matrix (Optimal Threshold: {optimal_threshold:.2f})')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    
    plt.tight_layout()
    save_plot(plt.gcf(), 'confusion_matrices.png')
    plt.close()

def plot_partial_dependence(pdp_result, X, features, figsize=(15, 10), save_path=None):
    """
    Plot partial dependence plots for the given features.
    
    Args:
        pdp_result: Result object from partial_dependence
        X: Feature data used to compute percentiles
        features: List of feature names or indices
        figsize: Figure size
        save_path: Path to save the figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Unpack pdp_result
    pd_values = pdp_result['average']
    pd_positions = pdp_result['values']
    
    # Create figure with enough subplots for all features
    n_features = len(features)
    n_cols = min(3, n_features)
    n_rows = int(np.ceil(n_features / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    # Plot each feature's partial dependence
    for i, (feature, values, avg_preds) in enumerate(zip(features, pd_positions, pd_values)):
        if i >= len(axes):
            break
            
        # Get feature name if it's an index
        feature_name = X.columns[feature] if isinstance(feature, int) else feature
        
        # Calculate percentiles for feature values
        percentiles = np.percentile(X[feature_name], [5, 95])
        
        # Plot partial dependence
        axes[i].plot(values, avg_preds, 'r-', linewidth=2)
        
        # Add vertical lines at 5th and 95th percentiles
        axes[i].axvline(x=percentiles[0], color='k', linestyle='--', alpha=0.5)
        axes[i].axvline(x=percentiles[1], color='k', linestyle='--', alpha=0.5)
        
        # Add feature distribution as a rug plot
        axes[i].plot(X[feature_name], np.full_like(X[feature_name], avg_preds.min() - 0.1), 
                    'k|', alpha=0.2)
        
        # Set labels and title
        axes[i].set_xlabel(feature_name)
        axes[i].set_ylabel('Predicted probability')
        axes[i].set_title(f'Partial Dependence: {feature_name}')
        axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig