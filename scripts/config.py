import os

# Directory paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Create necessary directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Data processing settings
TARGET_COLUMN = 'hospital_death'
TEST_SIZE = 0.2
RANDOM_STATE = 42
MISSING_THRESHOLD_HIGH = 80

# Imputation strategies
IMPUTATION_STRATEGIES = ['mean', 'median', 'most_frequent']

# Random Forest parameters
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'bootstrap': True,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'class_weight': 'balanced'
}

# Visualization settings
FIGURE_SIZE = (12, 8)
DPI = 300
PLOT_STYLE = 'seaborn-v0_8-whitegrid'
COLOR_PALETTE = 'husl'

# Model evaluation metrics
EVALUATION_METRICS = [
    'accuracy',
    'precision',
    'recall',
    'f1',
    'roc_auc',
    'pr_auc'
]

# Fairness analysis settings
PROTECTED_ATTRIBUTES = [
    'ethnicity',
    'gender',
    'age'
]

# Feature importance settings
TOP_N_FEATURES = 20
IMPORTANCE_THRESHOLD = 0.01

# SHAP analysis settings
SHAP_N_SAMPLES = 1000
SHAP_N_FEATURES = 20

# LIME analysis settings
LIME_N_FEATURES = 10
LIME_N_SAMPLES = 1000

# Body system analysis settings
MIN_SAMPLES_PER_SYSTEM = 100
MIN_TEST_SAMPLES_PER_SYSTEM = 20

# Threshold analysis settings
THRESHOLD_STEP = 0.05
THRESHOLD_RANGE = (0.1, 0.9)

# Calibration settings
N_CALIBRATION_BINS = 10

# Correlation analysis settings
CORRELATION_THRESHOLD = 0.8

# Outlier analysis settings
OUTLIER_THRESHOLD = 3

# Skewness analysis settings
SKEWNESS_THRESHOLD = 1.0
