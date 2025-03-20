#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to run the preprocessing pipeline and save the processed data.
"""

import os
import pandas as pd
from data_processing import (
    load_data,
    analyze_missing_values,
    split_features_target,
    create_train_test_split,
    identify_column_types,
    filter_high_missing_columns,
    analyze_feature_distributions,
    analyze_feature_correlations,
    analyze_class_balance,
    analyze_outliers,
    analyze_feature_skewness,
    save_processed_data,
    balance_by_ethnicity
)
from visualization import (
    plot_missing_values,
    plot_target_distribution,
    plot_age_distribution_by_mortality,
    plot_physiological_vars_by_mortality,
    plot_categorical_features_by_mortality,
    plot_correlation_matrix_of_clinical_features
)
import config
import utils

def main():
    """Run the data preprocessing pipeline."""
    print("Starting data preprocessing pipeline...")
    
    # Create directories if they don't exist
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    data_path = os.path.join(config.DATA_DIR, 'icu_data.csv')
    df = load_data(data_path)
    if df is None:
        print("Failed to load data. Exiting...")
        return
    utils.print_dataset_info(df)
    
    # Analyze missing values
    print("\nAnalyzing missing values...")
    missing_percentages, high_missing_cols = analyze_missing_values(df)
    plot_missing_values(missing_percentages)
    
    # Split features and target
    print("\nSplitting features and target...")
    X, y = split_features_target(df, config.TARGET_COLUMN)
    
    # Create train-test split with stratification by ethnicity
    print("\nCreating train-test split...")
    X_train, X_test, y_train, y_test = create_train_test_split(
        X, y, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_STATE,
        stratify_by='ethnicity'
    )
    
    # Balance the training data by ethnicity
    print("\nBalancing training data by ethnicity...")
    X_train_balanced, y_train_balanced = balance_by_ethnicity(
        X_train, y_train,
        method='smote',  # You can change this to 'undersample' or 'oversample'
        random_state=config.RANDOM_STATE
    )
    
    # Print ethnicity distribution before and after balancing
    print("\nEthnicity distribution before balancing:")
    print(X_train['ethnicity'].value_counts())
    print("\nEthnicity distribution after balancing:")
    print(X_train_balanced['ethnicity'].value_counts())
    
    # Filter high missing columns
    print("\nFiltering high missing columns...")
    X_train_filtered, X_test_filtered = filter_high_missing_columns(
        X_train_balanced, X_test, threshold=config.MISSING_THRESHOLD_HIGH
    )
    
    # Identify column types
    print("\nIdentifying column types...")
    numerical_cols, categorical_cols = identify_column_types(X)
    
    # Analyze feature distributions
    print("\nAnalyzing feature distributions...")
    analyze_feature_distributions(X_train_filtered, numerical_cols, categorical_cols)
    
    # Plot target distribution
    print("\nPlotting target distribution...")
    plot_target_distribution(pd.concat([X_train_filtered, y_train_balanced], axis=1))
    
    # Plot age distribution by mortality
    print("\nPlotting age distribution by mortality...")
    plot_age_distribution_by_mortality(X_train_filtered, y_train_balanced)
    
    # Plot physiological variables by mortality
    print("\nPlotting physiological variables by mortality...")
    physiological_vars = [col for col in numerical_cols if any(var in col.lower() 
                        for var in ['heart_rate', 'blood_pressure', 'temperature', 'oxygen'])]
    plot_physiological_vars_by_mortality(X_train_filtered, y_train_balanced, physiological_vars)
    
    # Plot categorical features by mortality
    print("\nPlotting categorical features by mortality...")
    plot_categorical_features_by_mortality(X_train_filtered, y_train_balanced, categorical_cols)
    
    # Analyze feature correlations
    print("\nAnalyzing feature correlations...")
    correlation_matrix = analyze_feature_correlations(X_train_filtered, numerical_cols)
    plot_correlation_matrix_of_clinical_features(X_train_filtered, y_train_balanced)
    
    # Analyze class balance
    print("\nAnalyzing class balance...")
    analyze_class_balance(y_train_balanced)
    
    # Analyze outliers
    print("\nAnalyzing outliers...")
    analyze_outliers(X_train_filtered, numerical_cols, threshold=config.OUTLIER_THRESHOLD)
    
    # Analyze feature skewness
    print("\nAnalyzing feature skewness...")
    analyze_feature_skewness(X_train_filtered, numerical_cols)
    
    # Save processed data
    print("\nSaving processed data...")
    processed_data = {
        'X_train': X_train_filtered,
        'X_test': X_test_filtered,
        'y_train': y_train_balanced,
        'y_test': y_test,
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols
    }
    save_processed_data(
        processed_data,
        os.path.join(config.RESULTS_DIR, 'processed_data.pkl')
    )
    
    print("\nData preprocessing pipeline completed successfully!")
    
    # Print some basic statistics about the data
    print("\nBasic statistics:")
    print(f"Number of training samples: {len(X_train_filtered)}")
    print(f"Number of testing samples: {len(X_test_filtered)}")
    print(f"Number of numerical features: {len(numerical_cols)}")
    print(f"Number of categorical features: {len(categorical_cols)}")
    print(f"Mortality rate in training set: {y_train_balanced.mean():.2%}")
    print(f"Mortality rate in testing set: {y_test.mean():.2%}")

if __name__ == '__main__':
    main()
