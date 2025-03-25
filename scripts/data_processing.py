import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
import config

def load_data(data_dir):
    """Load the ICU dataset and its description dictionary."""
    # Load main dataset
    df = pd.read_csv(os.path.join(data_dir, 'training_v2.csv'))
    
    # Load data dictionary
    description_dict = pd.read_csv(os.path.join(data_dir, 'WiDS_Datathon_2020_Dictionary.csv'))
    description_dict = dict(zip(description_dict['Variable Name'], description_dict['Description']))
    
    return df, description_dict

def analyze_missing_values(df):
    """Analyze missing values in the dataset."""
    missing_percentages = df.isnull().mean() * 100
    high_missing_cols = missing_percentages[missing_percentages > config.MISSING_THRESHOLD_HIGH].index.tolist()
    
    print("\nMissing Values Analysis:")
    print(f"Total missing values: {df.isnull().sum().sum()}")
    print(f"Columns with >{config.MISSING_THRESHOLD_HIGH}% missing values: {len(high_missing_cols)}")
    if high_missing_cols:
        print("High missing columns:", high_missing_cols)
    
    return missing_percentages, high_missing_cols

def split_features_target(df, target_column):
    """Split features and target variable."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

def create_train_test_split(X, y, test_size=0.2, random_state=42, stratify_by=None):
    """Create train-test split of the data with optional stratification."""
    if stratify_by is not None and stratify_by in X.columns:
        stratify = X[stratify_by]
    else:
        stratify = None
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)

def identify_column_types(df):
    """Identify numerical and categorical columns."""
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print("\nColumn Types:")
    print(f"Numerical columns: {len(numerical_cols)}")
    print(f"Categorical columns: {len(categorical_cols)}")
    
    return numerical_cols, categorical_cols

def filter_high_missing_columns(X_train, X_test, threshold=80):
    """Filter out columns with high missing values."""
    # Important columns to preserve regardless of missing values
    important_columns = [
        'apache_2_bodysystem',
        'apache_3j_bodysystem',
        'ethnicity',
        'gender',
        'age',
        'apache_4a_hospital_death_prob',
        'apache_4a_icu_death_prob'
    ]
    
    missing_percentages = X_train.isnull().mean() * 100
    high_missing_cols = missing_percentages[missing_percentages > threshold].index.tolist()
    
    # Remove important columns from high_missing_cols
    high_missing_cols = [col for col in high_missing_cols if col not in important_columns]
    
    if high_missing_cols:
        print(f"\nDropping {len(high_missing_cols)} columns with >{threshold}% missing values")
        print("Preserving important columns:", important_columns)
        X_train = X_train.drop(columns=high_missing_cols)
        X_test = X_test.drop(columns=high_missing_cols)
    
    return X_train, X_test, high_missing_cols

def create_preprocessing_pipeline(numerical_cols, categorical_cols, strategy='mean'):
    """Create preprocessing pipeline for numerical and categorical features.
    
    Args:
        numerical_cols: List of numerical column names
        categorical_cols: List of categorical column names
        strategy: Imputation strategy for numerical features ('mean', 'median', or 'most_frequent')
    """
    # Numerical preprocessing
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=strategy)),
        ('scaler', StandardScaler())
    ])
    
    # Categorical preprocessing
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    return preprocessor

def save_processed_data(X_train, X_test, y_train, y_test, numerical_cols, categorical_cols, description_dict, filepath):
    """Save processed data to a pickle file."""
    processed_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols,
        'description_dict': description_dict
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(processed_data, f)

def load_processed_data(filepath):
    """Load processed data from a pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def analyze_feature_distributions(X, numerical_cols, categorical_cols):
    """Analyze distributions of numerical and categorical features."""
    print("\nNumerical Feature Distributions:")
    for col in numerical_cols:
        print(f"\n{col}:")
        print(X[col].describe())
    
    print("\nCategorical Feature Distributions:")
    for col in categorical_cols:
        print(f"\n{col}:")
        print(X[col].value_counts(normalize=True))

def analyze_feature_correlations(X, numerical_cols):
    """Analyze correlations between numerical features."""
    correlation_matrix = X[numerical_cols].corr()
    
    # Find highly correlated features
    high_correlation = np.where(np.abs(correlation_matrix) > 0.8)
    high_correlation = [(correlation_matrix.index[x], correlation_matrix.columns[y], correlation_matrix.iloc[x, y])
                       for x, y in zip(*high_correlation) if x != y and x < y]
    
    if high_correlation:
        print("\nHighly Correlated Features (|correlation| > 0.8):")
        for feat1, feat2, corr in high_correlation:
            print(f"{feat1} - {feat2}: {corr:.3f}")
    
    return correlation_matrix

def analyze_class_balance(y):
    """Analyze the balance of classes in the target variable."""
    class_counts = y.value_counts()
    class_percentages = y.value_counts(normalize=True)
    
    print("\nClass Distribution:")
    print("Counts:")
    print(class_counts)
    print("\nPercentages:")
    print(class_percentages)
    
    return class_counts, class_percentages

def analyze_feature_importance(model, feature_names):
    """Analyze feature importance from the model."""
    if hasattr(model.named_steps['classifier'], 'feature_importances_'):
        importances = model.named_steps['classifier'].feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(importance_df.head(10))
        
        return importance_df
    else:
        print("Model does not have feature_importances_ attribute")
        return None

def analyze_feature_interactions(X, y, feature_pairs):
    """Analyze interactions between pairs of features."""
    for feat1, feat2 in feature_pairs:
        if feat1 in X.columns and feat2 in X.columns:
            # Create interaction term
            X[f'{feat1}_{feat2}_interaction'] = X[feat1] * X[feat2]
            
            # Calculate correlation with target
            correlation = X[f'{feat1}_{feat2}_interaction'].corr(y)
            
            print(f"\nInteraction between {feat1} and {feat2}:")
            print(f"Correlation with target: {correlation:.3f}")
            
            # Remove interaction term
            X.drop(columns=[f'{feat1}_{feat2}_interaction'], inplace=True)

def analyze_outliers(X, numerical_cols, threshold=3):
    """Analyze outliers in numerical features using z-score method."""
    outliers = {}
    
    for col in numerical_cols:
        z_scores = np.abs((X[col] - X[col].mean()) / X[col].std())
        outlier_mask = z_scores > threshold
        n_outliers = outlier_mask.sum()
        
        if n_outliers > 0:
            outliers[col] = {
                'count': n_outliers,
                'percentage': (n_outliers / len(X)) * 100,
                'indices': X[outlier_mask].index.tolist()
            }
    
    if outliers:
        print("\nOutlier Analysis:")
        for col, info in outliers.items():
            print(f"\n{col}:")
            print(f"Number of outliers: {info['count']}")
            print(f"Percentage: {info['percentage']:.2f}%")
    
    return outliers

def analyze_feature_skewness(X, numerical_cols):
    """Analyze skewness of numerical features."""
    skewness = X[numerical_cols].skew()
    
    print("\nFeature Skewness:")
    print(skewness)
    
    highly_skewed = skewness[abs(skewness) > 1].index.tolist()
    if highly_skewed:
        print("\nHighly Skewed Features (|skewness| > 1):")
        print(highly_skewed)
    
    return skewness, highly_skewed

def balance_by_ethnicity(X, y, method='smote', random_state=42):
    """Balance the dataset by ethnicity using various methods.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
        method (str): Balancing method ('smote', 'undersample', or 'oversample')
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: (X_balanced, y_balanced)
    """
    if 'ethnicity' not in X.columns:
        print("Warning: ethnicity column not found in data")
        return X, y
    
    # Get unique ethnicities
    ethnicities = X['ethnicity'].unique()
    
    if method == 'smote':
        try:
            from imblearn.over_sampling import SMOTE
            from imblearn.under_sampling import RandomUnderSampler
            from imblearn.pipeline import Pipeline
            
            # Create SMOTE pipeline
            over = SMOTE(sampling_strategy=0.5, random_state=random_state)
            under = RandomUnderSampler(sampling_strategy=0.8, random_state=random_state)
            steps = [('o', over), ('u', under)]
            pipeline = Pipeline(steps=steps)
            
            # Fit and transform the data
            X_balanced, y_balanced = pipeline.fit_resample(X, y)
            
        except ImportError:
            print("Warning: imbalanced-learn not available. Using simple oversampling instead.")
            return balance_by_ethnicity(X, y, method='oversample', random_state=random_state)
    
    elif method == 'undersample':
        # Find the ethnicity with the smallest number of samples
        min_samples = float('inf')
        min_ethnicity = None
        for ethnicity in ethnicities:
            n_samples = len(X[X['ethnicity'] == ethnicity])
            if n_samples < min_samples:
                min_samples = n_samples
                min_ethnicity = ethnicity
        
        # Undersample all other ethnicities to match the smallest group
        X_balanced = pd.DataFrame()
        y_balanced = pd.Series()
        
        for ethnicity in ethnicities:
            mask = X['ethnicity'] == ethnicity
            X_group = X[mask]
            y_group = y[mask]
            
            if ethnicity == min_ethnicity:
                X_balanced = pd.concat([X_balanced, X_group])
                y_balanced = pd.concat([y_balanced, y_group])
            else:
                # Randomly sample to match the size of the smallest group
                indices = np.random.choice(len(X_group), min_samples, replace=False)
                X_balanced = pd.concat([X_balanced, X_group.iloc[indices]])
                y_balanced = pd.concat([y_balanced, y_group.iloc[indices]])
    
    elif method == 'oversample':
        # Find the ethnicity with the largest number of samples
        max_samples = 0
        max_ethnicity = None
        for ethnicity in ethnicities:
            n_samples = len(X[X['ethnicity'] == ethnicity])
            if n_samples > max_samples:
                max_samples = n_samples
                max_ethnicity = ethnicity
        
        # Oversample all other ethnicities to match the largest group
        X_balanced = pd.DataFrame()
        y_balanced = pd.Series()
        
        for ethnicity in ethnicities:
            mask = X['ethnicity'] == ethnicity
            X_group = X[mask]
            y_group = y[mask]
            
            if ethnicity == max_ethnicity:
                X_balanced = pd.concat([X_balanced, X_group])
                y_balanced = pd.concat([y_balanced, y_group])
            else:
                # Randomly sample with replacement to match the size of the largest group
                indices = np.random.choice(len(X_group), max_samples, replace=True)
                X_balanced = pd.concat([X_balanced, X_group.iloc[indices]])
                y_balanced = pd.concat([y_balanced, y_group.iloc[indices]])
    
    else:
        print(f"Warning: Unknown balancing method '{method}'. Returning original data.")
        return X, y
    
    # Shuffle the balanced dataset
    indices = np.random.permutation(len(X_balanced))
    X_balanced = X_balanced.iloc[indices]
    y_balanced = y_balanced.iloc[indices]
    
    return X_balanced, y_balanced
