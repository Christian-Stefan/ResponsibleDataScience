import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, confusion_matrix, classification_report

def set_visualization_style():
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams['figure.figsize'] = (10, 6)

def plot_missing_values(missing_percentages, top_n=20):
    plt.figure(figsize=(10, 6))
    missing_percentages.sort_values(ascending=False).head(top_n).plot(kind='bar')
    plt.title(f'Percentage of Missing Values in Top {top_n} Features')
    plt.xlabel('Features')
    plt.ylabel('Missing Values (%)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def plot_target_distribution(df, target_col='hospital_death'):
    if target_col in df.columns:
        plt.figure()
        ax = sns.countplot(x=target_col, data=df, palette="Set2", hue=target_col)
        ax.get_legend().remove()
        plt.title('Mortality Outcome Distribution')
        plt.xlabel('Mortality (0 = Survived, 1 = Died)')
        plt.ylabel('Count')
        plt.show()
        number_of_deads = np.sum(df[target_col] == 1)
        number_of_survivors = np.sum(df[target_col] == 0)
        print(f"Dead: {number_of_deads} Survived: {number_of_survivors} out of a total of {len(df[target_col])}")
    else:
        print(f"Column '{target_col}' not found in the dataset.")

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
