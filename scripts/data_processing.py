import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

def load_data(data_dir='../data/'):
    description = pd.read_csv(os.path.join(data_dir, 'WiDS_Datathon_2020_Dictionary.csv'))
    description_dict = description.set_index('Variable Name').to_dict(orient='index')
    df = pd.read_csv(os.path.join(data_dir, 'training_v2.csv'))
    return df, description_dict

def get_description(col_name, description_dict):
    return description_dict.get(col_name)

def analyze_missing_values(df):
    missing_percentages = df.isnull().mean() * 100
    high_missing_cols = missing_percentages[missing_percentages > 20].index.tolist()
    return missing_percentages, high_missing_cols

def remove_high_missing_columns(df, threshold=20):
    df_initial = df.copy()
    missing_percentages = df.isnull().mean() * 100
    high_missing_cols = missing_percentages[missing_percentages > threshold].index.tolist()
    df_cleaned = df.drop(columns=high_missing_cols)
    print(f"Length of inefficient columns \t : {len(high_missing_cols)} / Before: {len(df_initial.columns)} After: {len(df_cleaned.columns)}")
    return df_cleaned, high_missing_cols

def remove_missing_samples(df):
    df_initial = df.copy()
    df_cleaned = df.dropna()
    print(f"Total number of samples after removing missing values: Now: {len(df_cleaned)} Before: {len(df_initial)}")
    print(f"Missing percentage of data: {(len(df_cleaned)/len(df_initial))*100:.2f} %")
    return df_cleaned

def split_features_target(df, target_col='hospital_death'):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def create_train_test_split(X, y, test_size=0.2, random_state=1):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Training set size: {X_train.shape}")
    print(f"Testing set size: {X_test.shape}")
    print(f"Mortality rate in training set: {y_train.mean():.2%}")
    print(f"Mortality rate in testing set: {y_test.mean():.2%}")
    return X_train, X_test, y_train, y_test

def identify_column_types(X):
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    print(f"Number of numerical features: {len(numerical_cols)}")
    print(f"Number of categorical features: {len(categorical_cols)}")
    return numerical_cols, categorical_cols

def create_preprocessing_pipeline(numerical_cols, categorical_cols, imputation_strategy='simple'):
    if imputation_strategy == 'simple':
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
    elif imputation_strategy == 'median':
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
    elif imputation_strategy == 'knn':
        numerical_transformer = Pipeline(steps=[
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', StandardScaler())
        ])
    elif imputation_strategy == 'iterative':
        numerical_transformer = Pipeline(steps=[
            ('imputer', IterativeImputer(max_iter=10, random_state=1)),
            ('scaler', StandardScaler())
        ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'
    )
    return preprocessor

def filter_high_missing_columns(X_train, X_test, threshold=80):
    missing_percentages = X_train.isnull().mean().sort_values(ascending=False) * 100
    high_missing_cols = missing_percentages[missing_percentages > threshold].index.tolist()
    print(f"Number of columns with >{threshold}% missing values: {len(high_missing_cols)}")
    X_train_filtered = X_train.drop(columns=high_missing_cols)
    X_test_filtered = X_test.drop(columns=high_missing_cols)
    return X_train_filtered, X_test_filtered, high_missing_cols

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
    
    print(f"Processed data saved to {filepath}")
    
    # Print some basic statistics about the data
    print("\nBasic statistics:")
    print(f"Number of training samples: {len(X_train)}")
    print(f"Number of testing samples: {len(X_test)}")
    print(f"Number of numerical features: {len(numerical_cols)}")
    print(f"Number of categorical features: {len(categorical_cols)}")
    print(f"Mortality rate in training set: {y_train.mean():.2%}")
    print(f"Mortality rate in testing set: {y_test.mean():.2%}")

def load_processed_data(filepath):
    """Load processed data from a pickle file."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data
