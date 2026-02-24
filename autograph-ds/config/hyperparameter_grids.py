"""Hyperparameter search spaces for model tuning.

These grids define the parameter distributions for RandomizedSearchCV.
Values are chosen based on sklearn/XGBoost documentation and best practices.
"""

# RandomForest parameters
# Based on: https://scikit-learn.org/stable/modules/ensemble.html#random-forest-parameters
# These parameters control the complexity and behavior of the Random Forest ensemble
RF_PARAM_GRID = {
    # Number of trees in the forest
    # More trees generally improve performance but increase computation time
    # Range: 50-500 covers small to large ensembles
    'classifier__n_estimators': [50, 100, 200, 300, 500],
    
    # Maximum depth of trees
    # None = unlimited depth (can overfit), smaller values = more regularization
    # Range: 5-30 covers shallow to deep trees, None for unlimited
    'classifier__max_depth': [5, 10, 15, 20, 30, None],
    
    # Minimum samples required to split an internal node
    # Higher values reduce overfitting by requiring more samples per split
    # Range: 2-20 covers strict to lenient splitting criteria
    'classifier__min_samples_split': [2, 5, 10, 20],
    
    # Minimum samples required at a leaf node
    # Higher values create smoother decision boundaries
    # Range: 1-10 covers fine to coarse leaf nodes
    'classifier__min_samples_leaf': [1, 2, 5, 10],
    
    # Number of features to consider when looking for best split
    # 'sqrt' = sqrt(n_features), 'log2' = log2(n_features), None = all features
    # Different strategies balance variance and bias
    'classifier__max_features': ['sqrt', 'log2', None],
    
    # Whether to use bootstrap samples when building trees
    # True = bagging (reduces variance), False = use whole dataset
    'classifier__bootstrap': [True, False]
}

# XGBoost parameters
# Based on: https://xgboost.readthedocs.io/en/stable/parameter.html
# These parameters control the boosting process and tree structure
XGB_PARAM_GRID = {
    # Number of boosting rounds (sequential trees)
    # More rounds can improve performance but risk overfitting
    # Range: 50-500 covers small to large ensembles
    'classifier__n_estimators': [50, 100, 200, 300, 500],
    
    # Maximum depth of trees
    # XGBoost typically uses shallower trees than Random Forest
    # Range: 3-15 covers shallow to moderate depth
    'classifier__max_depth': [3, 5, 7, 10, 15],
    
    # Learning rate (shrinkage/eta)
    # Lower values make learning slower but often more robust
    # Range: 0.01-0.3 covers conservative to aggressive learning
    'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    
    # Subsample ratio of training instances
    # Lower values reduce overfitting by using fewer samples per tree
    # Range: 0.6-1.0 covers moderate to full sampling
    'classifier__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    
    # Subsample ratio of columns when constructing each tree
    # Lower values reduce overfitting by using fewer features per tree
    # Range: 0.6-1.0 covers moderate to full feature sampling
    'classifier__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    
    # Minimum loss reduction required for a split
    # Higher values make the model more conservative
    # Range: 0-0.5 covers no regularization to strong regularization
    'classifier__gamma': [0, 0.1, 0.2, 0.3, 0.5],
    
    # Minimum sum of instance weight needed in a child
    # Higher values prevent overfitting by requiring more weight per leaf
    # Range: 1-7 covers lenient to strict weight requirements
    'classifier__min_child_weight': [1, 3, 5, 7]
}

# IsolationForest parameters for anomaly detection
# Based on: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
IF_PARAM_GRID = {
    # Expected proportion of outliers in the dataset
    # 'auto' = automatic detection, numeric values = fixed proportion
    # Range: 0.05-0.2 covers 5%-20% expected anomalies
    'contamination': [0.05, 0.1, 0.15, 0.2, 'auto'],
    
    # Number of base estimators (isolation trees)
    # More trees improve stability but increase computation
    # Range: 50-200 covers small to moderate ensembles
    'n_estimators': [50, 100, 200],
    
    # Number of samples to draw for training each tree
    # Smaller values increase diversity, larger values increase stability
    # Range: 0.5-1.0 covers 50%-100% of training data
    'max_samples': [0.5, 0.7, 0.9, 1.0]
}

# Default parameters (current hardcoded values)
# These represent reasonable starting points for model training
RF_DEFAULTS = {
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'bootstrap': True,
    'class_weight': 'balanced',
    'random_state': 42
}

XGB_DEFAULTS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 1.0,
    'colsample_bytree': 1.0,
    'gamma': 0,
    'min_child_weight': 1,
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'random_state': 42
}

# Summary of parameter grid sizes
# These can be used for RandomizedSearchCV configuration
GRID_SUMMARY = {
    'RF_PARAM_GRID': {
        'total_combinations': 5 * 6 * 4 * 4 * 3 * 2,  # 2880 combinations
        'num_parameters': 6,
        'recommended_n_iter': 60  # For RandomizedSearchCV
    },
    'XGB_PARAM_GRID': {
        'total_combinations': 5 * 5 * 5 * 5 * 5 * 6 * 4,  # 75000 combinations
        'num_parameters': 7,
        'recommended_n_iter': 60  # For RandomizedSearchCV
    },
    'IF_PARAM_GRID': {
        'total_combinations': 5 * 3 * 4,  # 60 combinations
        'num_parameters': 3,
        'recommended_n_iter': 30  # For RandomizedSearchCV
    }
}
