"""
Script to run the preprocessing pipeline and save the processed data.
"""

import os
import pandas as pd
import pickle
import numpy as np
import data_processing as dp
import config
import visualization as viz
import matplotlib.pyplot as plt
import seaborn as sns
import utils

def main():
    """Run preprocessing pipeline."""
    # Create directories if they don't exist
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    print("Loading data...")
    df, description_dict = dp.load_data(config.DATA_DIR)
    utils.print_dataset_info(df)
    
    # Analyze missing values
    print("\nAnalyzing missing values...")
    missing_percentages, high_missing_cols = dp.analyze_missing_values(df)
    
    # Save missing value analysis plot
    plt.figure(figsize=(10, 6))
    missing_percentages.sort_values(ascending=False).head(20).plot(kind='bar')
    plt.title('Percentage of Missing Values in Top 20 Features')
    plt.xlabel('Features')
    plt.ylabel('Missing Values (%)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, 'missing_values.png'))
    plt.close()
    
    # Split data for modeling
    print("\nSplitting data for modeling...")
    X, y = dp.split_features_target(df, config.TARGET_COLUMN)
    X_train, X_test, y_train, y_test = dp.create_train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    
    # Filter high missing columns
    print("\nFiltering columns with high missing values...")
    X_train_filtered, X_test_filtered, high_missing_cols = dp.filter_high_missing_columns(
        X_train, X_test, threshold=config.MISSING_THRESHOLD_HIGH
    )
    
    # Identify column types
    print("\nIdentifying column types...")
    numerical_cols, categorical_cols = dp.identify_column_types(X_train_filtered)
    
    # Save the processed data
    print("\nSaving processed data...")
    processed_data = {
        'X_train': X_train_filtered,
        'X_test': X_test_filtered,
        'y_train': y_train,
        'y_test': y_test,
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols,
        'description_dict': description_dict
    }
    
    with open(os.path.join(config.RESULTS_DIR, 'processed_data.pkl'), 'wb') as f:
        pickle.dump(processed_data, f)
    
    print(f"Preprocessing completed. Results saved to {config.RESULTS_DIR}")
    
    # Print some basic statistics about the data
    print("\nBasic statistics:")
    print(f"Number of training samples: {len(X_train_filtered)}")
    print(f"Number of testing samples: {len(X_test_filtered)}")
    print(f"Number of numerical features: {len(numerical_cols)}")
    print(f"Number of categorical features: {len(categorical_cols)}")
    print(f"Mortality rate in training set: {y_train.mean():.2%}")
    print(f"Mortality rate in testing set: {y_test.mean():.2%}")

if __name__ == "__main__":
    main()
