#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to perform in-depth analysis on the trained models and results.
"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, precision_recall_curve

import data_processing as dp
import modeling as mdl
import visualization as viz
import utils
import config
from data_processing import load_processed_data

def main():
    """Run the analysis pipeline."""
    print("Starting analysis pipeline...")
    
    # Load processed data
    print("\nLoading processed data...")
    processed_data = load_processed_data(
        os.path.join(config.RESULTS_DIR, 'processed_data.pkl')
    )
    if processed_data is None:
        print("Failed to load processed data. Exiting...")
        return
    
    X_test = processed_data['X_test']
    y_test = processed_data['y_test']
    
    # Load best model
    print("\nLoading best model...")
    model_files = [f for f in os.listdir(config.MODELS_DIR) 
                  if f.startswith('best_model_')]
    if not model_files:
        print("No model files found. Exiting...")
        return
    
    best_model_file = model_files[0]  # Assuming first file is best model
    best_model_path = os.path.join(config.MODELS_DIR, best_model_file)
    
    with open(best_model_path, 'rb') as f:
        best_model = pickle.load(f)
    
    # Analyze classification thresholds
    print("\nAnalyzing classification thresholds...")
    threshold_results = mdl.analyze_classification_thresholds(
        best_model, X_test, y_test,
        threshold_range=config.THRESHOLD_RANGE,
        threshold_step=config.THRESHOLD_STEP
    )
    viz.plot_threshold_analysis_results(threshold_results)
    
    # Analyze performance by body system
    print("\nAnalyzing performance by body system...")
    body_system_results = mdl.analyze_performance_by_body_system(
        best_model, X_test, y_test,
        min_samples=config.MIN_SAMPLES_PER_SYSTEM,
        min_test_samples=config.MIN_TEST_SAMPLES_PER_SYSTEM
    )
    viz.plot_performance_by_body_system(body_system_results)
    
    # Analyze detailed body system analysis
    print("\nPerforming detailed body system analysis...")
    detailed_results = mdl.analyze_by_body_system(
        best_model, X_test, y_test
    )
    viz.plot_body_system_analysis_results(detailed_results)
    
    # Analyze fairness metrics by ethnicity
    print("\nAnalyzing fairness metrics by ethnicity...")
    fairness_results = mdl.analyze_fairness_by_ethnicity(
        best_model, X_test, y_test
    )
    viz.plot_fairness_metrics(fairness_results)
    
    # Analyze intersectional fairness
    print("\nAnalyzing intersectional fairness...")
    intersectional_results = mdl.analyze_intersectional_fairness(
        best_model, X_test, y_test,
        config.PROTECTED_ATTRIBUTES
    )
    viz.plot_intersectional_fairness(intersectional_results)
    
    # Perform SHAP analysis
    print("\nPerforming SHAP analysis...")
    shap_results = mdl.analyze_with_shap(
        best_model, X_test,
        n_samples=config.SHAP_N_SAMPLES,
        n_features=config.SHAP_N_FEATURES
    )
    viz.plot_shap_summary(shap_results)
    
    # Perform LIME analysis
    print("\nPerforming LIME analysis...")
    lime_results = mdl.explain_with_lime(
        best_model, X_test,
        n_features=config.LIME_N_FEATURES,
        n_samples=config.LIME_N_SAMPLES
    )
    viz.plot_lime_explanations(lime_results)
    
    # Analyze feature importance
    print("\nAnalyzing feature importance...")
    feature_names = mdl.get_feature_names(best_model.named_steps['preprocessor'])
    feature_importances = best_model.named_steps['classifier'].feature_importances_
    viz.plot_feature_importance(
        feature_names,
        feature_importances,
        top_n=config.TOP_N_FEATURES
    )
    
    # Analyze model calibration
    print("\nAnalyzing model calibration...")
    calibration_results = mdl.analyze_calibration(
        best_model, X_test, y_test,
        n_bins=config.N_CALIBRATION_BINS
    )
    viz.plot_calibration_curve(calibration_results)
    
    print("\nAnalysis pipeline completed successfully!")

if __name__ == '__main__':
    main()
