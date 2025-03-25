import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import os

# Set random seed for reproducibility
np.random.seed(42)

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid")

# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

def load_data():
    """
    Load the WiDS 2020 dataset and the data dictionary.
    """
    train_data = pd.read_csv('dataset/training_v2.csv')
    data_dict = pd.read_csv('dataset/WiDS_Datathon_2020_Dictionary.csv')
    return train_data, data_dict

def analyze_basic_stats(df):
    """
    Print dataset overview and info regarding missing values.
    """
    print("\nBasic Dataset Information:")
    print("-" * 50)
    print(f"Number of samples: {df.shape[0]}")
    print(f"Number of features: {df.shape[1]}")
    
    # Data types information
    print("\nFeature Types:")
    print(df.dtypes.value_counts())
    
    # Missing values analysis
    missing_values = df.isnull().sum()
    missing_percentages = (missing_values / len(df)) * 100
    missing_info = pd.DataFrame({
        'Missing Values': missing_values,
        'Missing Percentage': missing_percentages
    }).sort_values('Missing Percentage', ascending=False)
    
    print("\nMissing Values Analysis (Top 10 features with missing values):")
    print(missing_info[missing_info['Missing Values'] > 0].head(10))
    
    return missing_info

def analyze_target_variable(df):
    """
    Analyze the distribution of the target variable (hospital_death).
    """
    target_dist = df['hospital_death'].value_counts(normalize=True) * 100
    
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='hospital_death')
    plt.title('Distribution of Target Variable (Hospital Death)')
    plt.xlabel('Hospital Death (0: Survived, 1: Deceased)')
    plt.ylabel('Count')
    plt.savefig('plots/target_distribution.png')
    plt.close()
    
    print("\nTarget Variable Distribution:")
    print(f"Survival Rate: {target_dist[0]:.2f}%")
    print(f"Mortality Rate: {target_dist[1]:.2f}%")
    
    return target_dist

def analyze_sensitive_features(df):
    """
    Analyze the potential sensitive features and their relations with the target.
    """
    # Age analysis
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='age', hue='hospital_death', multiple="stack", bins=30)
    plt.title('Age Distribution by Outcome')
    plt.savefig('plots/age_distribution.png')
    plt.close()
    
    # Create age groups
    df['age_group'] = pd.cut(df['age'], 
                            bins=[0, 18, 30, 45, 60, 75, 100],
                            labels=['0-18', '19-30', '31-45', '46-60', '61-75', '75+'])
    
    # Plot mortality rates by age group
    age_mortality = df.groupby('age_group')['hospital_death'].mean() * 100
    plt.figure(figsize=(10, 6))
    age_mortality.plot(kind='bar')
    plt.title('Mortality Rate by Age Group')
    plt.ylabel('Mortality Rate (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/age_group_mortality.png')
    plt.close()
    
    print("\nMortality Rate by Age Group:")
    print(age_mortality)
    
    # Gender analysis
    gender_mortality = df.groupby('gender')['hospital_death'].mean() * 100
    plt.figure(figsize=(8, 6))
    gender_mortality.plot(kind='bar')
    plt.title('Mortality Rate by Gender')
    plt.ylabel('Mortality Rate (%)')
    plt.savefig('plots/gender_mortality.png')
    plt.close()
    
    print("\nMortality Rate by Gender:")
    print(gender_mortality)
    
    # Gender and Age Group Interaction
    gender_age_mortality = df.groupby(['gender', 'age_group'])['hospital_death'].mean() * 100
    gender_age_mortality = gender_age_mortality.unstack()
    
    plt.figure(figsize=(12, 6))
    gender_age_mortality.plot(kind='bar', width=0.8)
    plt.title('Mortality Rate by Gender and Age Group')
    plt.ylabel('Mortality Rate (%)')
    plt.xlabel('Gender')
    plt.legend(title='Age Group')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('plots/gender_age_mortality.png')
    plt.close()
    
    print("\nMortality Rate by Gender and Age Group:")
    print(gender_age_mortality)
    
    # Chi-square test for gender and outcome
    gender_death_contingency = pd.crosstab(df['gender'], df['hospital_death'])
    chi2, p_value = stats.chi2_contingency(gender_death_contingency)[:2]
    
    print(f"\nChi-square test for gender and mortality:")
    print(f"Chi-square statistic: {chi2:.2f}")
    print(f"p-value: {p_value:.4f}")
    
    # ANOVA test for age groups and outcome
    age_groups = df.groupby('age_group')['hospital_death'].apply(list)
    f_stat, p_value = stats.f_oneway(*age_groups)
    
    print(f"\nANOVA test for age groups and mortality:")
    print(f"F-statistic: {f_stat:.2f}")
    print(f"p-value: {p_value:.4f}")
    
    return gender_mortality, age_mortality, gender_age_mortality

def analyze_correlations(df):
    """
    Analyze correlation matrix.
    """

    # Get the numerical columns
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns

    # Plot correlation matrix
    correlation_matrix = df[numerical_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, annot=False)
    plt.title('Correlation Matrix of Numerical Features')
    plt.tight_layout()
    plt.savefig('plots/correlation_matrix.png')
    plt.close()
    
    # Get top correlations with target variable
    target_correlations = correlation_matrix['hospital_death'].sort_values(ascending=False)
    # Remove target correlation
    target_correlations = target_correlations.drop('hospital_death')
    print("\nTop 10 Features Correlated with Hospital Death:")
    print(target_correlations.head(10))
    
    # Plot top 10 correlations
    plt.figure(figsize=(12, 6))
    target_correlations.head(10).plot(kind='bar')
    plt.title('Top 10 Features Correlated with Hospital Death')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('plots/top_correlations.png')
    plt.close()
    
    return correlation_matrix

def analyze_ethnicity(df):
    """
    Analyze ethnicity distribution and its relationship with mortality.
    """
    # Ethnicity distribution
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='ethnicity', order=df['ethnicity'].value_counts().index)
    plt.title('Distribution of Ethnicity')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('plots/ethnicity_distribution.png')
    plt.close()
    
    # Mortality rates by ethnicity
    ethnicity_mortality = df.groupby('ethnicity')['hospital_death'].mean() * 100
    ethnicity_counts = df['ethnicity'].value_counts()
    
    # Create a summary DataFrame
    ethnicity_summary = pd.DataFrame({
        'Count': ethnicity_counts,
        'Percentage': ethnicity_counts / len(df) * 100,
        'Mortality_Rate': ethnicity_mortality
    }).sort_values('Count', ascending=False)
    
    plt.figure(figsize=(12, 6))
    ethnicity_mortality.plot(kind='bar')
    plt.title('Mortality Rate by Ethnicity')
    plt.ylabel('Mortality Rate (%)')
    plt.xlabel('Ethnicity')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('plots/ethnicity_mortality.png')
    plt.close()
    
    print("\nEthnicity Distribution and Mortality Rates:")
    print(ethnicity_summary)
    
    # Chi-square test for ethnicity and outcome
    ethnicity_death_contingency = pd.crosstab(df['ethnicity'], df['hospital_death'])
    chi2, p_value = stats.chi2_contingency(ethnicity_death_contingency)[:2]
    
    print(f"\nChi-square test for ethnicity and mortality:")
    print(f"Chi-square statistic: {chi2:.2f}")
    print(f"p-value: {p_value:.4f}")
    
    return ethnicity_summary

def analyze_apache_bodysystem(df):
    """
    Analyze apache_2_bodysystem distribution and its relationship with mortality.
    """

    # Body system distribution
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='apache_2_bodysystem', order=df['apache_2_bodysystem'].value_counts().index)
    plt.title('Distribution of Apache 2 Body System')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('plots/bodysystem_distribution.png')
    plt.close()
    
    # Mortality rates by body system
    bodysystem_mortality = df.groupby('apache_2_bodysystem')['hospital_death'].mean() * 100
    bodysystem_counts = df['apache_2_bodysystem'].value_counts()
    
    # Create a summary DataFrame
    bodysystem_summary = pd.DataFrame({
        'Count': bodysystem_counts,
        'Percentage': bodysystem_counts / len(df) * 100,
        'Mortality_Rate': bodysystem_mortality
    }).sort_values('Count', ascending=False)
    
    plt.figure(figsize=(12, 6))
    bodysystem_mortality.plot(kind='bar')
    plt.title('Mortality Rate by Apache 2 Body System')
    plt.ylabel('Mortality Rate (%)')
    plt.xlabel('Body System')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('plots/bodysystem_mortality.png')
    plt.close()
    
    print("\nApache 2 Body System Distribution and Mortality Rates:")
    print(bodysystem_summary)
    
    # Chi-square test for body system and outcome
    bodysystem_death_contingency = pd.crosstab(df['apache_2_bodysystem'], df['hospital_death'])
    chi2, p_value = stats.chi2_contingency(bodysystem_death_contingency)[:2]
    
    print(f"\nChi-square test for body system and mortality:")
    print(f"Chi-square statistic: {chi2:.2f}")
    print(f"p-value: {p_value:.4f}")
    
    return bodysystem_summary

def analyze_apache4_predictions(df):
    """
    Analyze APACHE 4 probability predictions for potential biases.
    """
    # Overall prediction performance
    apache_prob = df['apache_4a_hospital_death_prob']
    actual = df['hospital_death']
    
    # Remove rows with missing values
    mask = ~apache_prob.isna()
    apache_prob = apache_prob[mask]
    actual = actual[mask]
    
    # Calculate ROC AUC
    roc_auc = roc_auc_score(actual, apache_prob)
    
    # Calculate PR Curve
    precision, recall, _ = precision_recall_curve(actual, apache_prob)
    pr_auc = auc(recall, precision)
    
    print("\nAPACHE 4 Prediction Performance:")
    print(f"ROC AUC: {roc_auc:.3f}")
    print(f"PR AUC: {pr_auc:.3f}")
    
    # Calibration plot
    plt.figure(figsize=(10, 6))
    
    # Create bins for predicted probabilities
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Calculate actual probabilities for each bin
    bin_indices = np.digitize(apache_prob, bins) - 1
    bin_actuals = np.array([actual[bin_indices == i].mean() for i in range(n_bins)])
    bin_counts = np.array([np.sum(bin_indices == i) for i in range(n_bins)])
    
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.scatter(bin_centers, bin_actuals, s=100 * bin_counts / len(actual))
    plt.xlabel('Predicted probability')
    plt.ylabel('Actual probability')
    plt.title('Calibration Plot of APACHE 4 Predictions')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/apache_calibration.png')
    plt.close()
    
    # Analyze prediction bias across groups
    def analyze_group_bias(group_col):
        group_metrics = []
        for group in df[group_col].unique():
            # Filter rows with group and non-missing predictions
            mask = (df[group_col] == group) & ~df['apache_4a_hospital_death_prob'].isna()
            # Calculate ROC AUC for the group if there are samples
            if mask.sum() > 0:
                group_roc = roc_auc_score(df.loc[mask, 'hospital_death'], 
                                        df.loc[mask, 'apache_4a_hospital_death_prob'])
                group_metrics.append({
                    'Group': group,
                    'Count': mask.sum(),
                    'ROC_AUC': group_roc,
                    'Mean_Predicted': df.loc[mask, 'apache_4a_hospital_death_prob'].mean(),
                    'Actual_Rate': df.loc[mask, 'hospital_death'].mean() * 100
                })
        return pd.DataFrame(group_metrics)
    
    # Analyze bias across different groups
    gender_bias = analyze_group_bias('gender')
    ethnicity_bias = analyze_group_bias('ethnicity')
    bodysystem_bias = analyze_group_bias('apache_2_bodysystem')
    
    print("\nPrediction Bias Analysis by Gender:")
    print(gender_bias)
    print("\nPrediction Bias Analysis by Ethnicity:")
    print(ethnicity_bias)
    print("\nPrediction Bias Analysis by Body System:")
    print(bodysystem_bias)
    
    # Plot prediction distributions by gender
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df[~df['apache_4a_hospital_death_prob'].isna()], 
                x='gender', y='apache_4a_hospital_death_prob')
    plt.title('APACHE 4 Prediction Distribution by Gender')
    plt.tight_layout()
    plt.savefig('plots/apache_gender_dist.png')
    plt.close()
    
    # Plot prediction distributions by ethnicity
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df[~df['apache_4a_hospital_death_prob'].isna()], 
                x='ethnicity', y='apache_4a_hospital_death_prob')
    plt.title('APACHE 4 Prediction Distribution by Ethnicity')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('plots/apache_ethnicity_dist.png')
    plt.close()
    
    # Plot prediction distributions by body system
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=df[~df['apache_4a_hospital_death_prob'].isna()], 
                x='apache_2_bodysystem', y='apache_4a_hospital_death_prob')
    plt.title('APACHE 4 Prediction Distribution by Body System')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('plots/apache_bodysystem_dist.png')
    plt.close()
    
    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'gender_bias': gender_bias,
        'ethnicity_bias': ethnicity_bias,
        'bodysystem_bias': bodysystem_bias
    }

def main():
    # Load the data
    print("Loading data...")
    train_data, data_dict = load_data()
    
    # Basic statistics and missing values analysis
    missing_info = analyze_basic_stats(train_data)
    
    # Target variable analysis
    target_dist = analyze_target_variable(train_data)
    
    # Sensitive attributes analysis
    gender_mortality, age_mortality, gender_age_mortality = analyze_sensitive_features(train_data)
    
    # Ethnicity analysis
    ethnicity_summary = analyze_ethnicity(train_data)
    
    # Apache body system analysis
    bodysystem_summary = analyze_apache_bodysystem(train_data)
    
    # Current APACHE 4 predictions analysis
    apache_metrics = analyze_apache4_predictions(train_data)
    
    # Correlation analysis
    correlation_matrix = analyze_correlations(train_data)

if __name__ == "__main__":
    main() 