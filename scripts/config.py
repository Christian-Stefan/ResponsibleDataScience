# Path configurations
DATA_DIR = '../data/'
RESULTS_DIR = 'results/'
MODELS_DIR = 'models/'

# Data processing settings
MISSING_THRESHOLD_HIGH = 80  # Remove columns with missing values above this percentage
TARGET_COLUMN = 'hospital_death'

# Modeling settings
RANDOM_STATE = 1
TEST_SIZE = 0.2

# Imputation strategies to compare
IMPUTATION_STRATEGIES = ['simple', 'median']  # Kevin: Removed 'knn' and 'iterative' as they are slow, and maybe add something else?

# Random Forest parameters
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 10,
    'class_weight': 'balanced',
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# Grid search parameters
PARAM_GRID = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [10, 20],
    'classifier__min_samples_split': [5, 10]
}

# Cross-validation settings
CV_FOLDS = 3 # Kevin: Changed from 5 to 3 to speed up the process, can be 5 too
