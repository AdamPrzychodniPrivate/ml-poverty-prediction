import logging
from typing import Dict, Tuple, List

import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.metrics import r2_score, mean_squared_error


def generate_cv_splits(data: pd.DataFrame, model_options: Dict) -> List[Tuple]:
    """
    Generates cross-validation train-test splits.

    Args:
        data: DataFrame containing the features and target.
        model_options: Dictionary containing 'features', 'target', 'n_folds', and 'random_state'.

    Returns:
        A list of tuples, each containing X_train, X_test, y_train, y_test for each fold.
    """
    features = model_options["features"]
    target = model_options["target"]
    n_folds = model_options["n_folds"]
    random_state = model_options["random_state"]

    X = data[features]
    y = data[[target]]  

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    cv_data_splits = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        cv_data_splits.append((X_train, X_test, y_train, y_test))

    return cv_data_splits
    

from sklearn.ensemble import RandomForestRegressor as RF
import pandas as pd
from typing import Dict, List, Tuple

def train_independent_rf(X_train: pd.DataFrame, y_train: pd.DataFrame, model_options: Dict) -> Dict[str, RF]:
    """
    Trains independent Random Forest models for each outcome.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for the target variables.
        model_options: Dictionary containing 'n_trees', 'max_depth', and 'max_features'.

    Returns:
        Dictionary of trained Random Forest models for each outcome.
    """
    n_trees = model_options["n_trees"]
    max_depth = model_options["max_depth"]
    max_features = model_options["max_features"]

    models = {}
    for outcome in y_train.columns:
        model = RF(n_estimators=n_trees, max_depth=max_depth, max_features=max_features)
        model.fit(X_train, y_train[outcome])
        models[outcome] = model
    return models


def train_models_on_cv_folds(cv_data_splits: List[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]], model_options: Dict) -> Dict[str, Dict[str, RF]]:
    """
    Trains independent Random Forest models for each outcome on each fold of cross-validation splits.

    Args:
        cv_data_splits: List of tuples containing (X_train, X_test, y_train, y_test) for each fold.
        model_options: Dictionary containing 'n_trees', 'max_depth', and 'max_features'.

    Returns:
        Dictionary of dictionaries containing trained Random Forest models for each outcome, for each fold.
    """
    num_folds = len(cv_data_splits)
    print(f"Starting training of independent Random Forest models for {num_folds} folds...")

    target_names = ", ".join(cv_data_splits[0][2].columns)
    print(f"Training models for target: {target_names}")
    
    all_fold_models = {}
    for fold_index, (X_train, _, y_train, _) in enumerate(cv_data_splits):
        fold_key = f'fold_{fold_index + 1}'
        print(f"Training model for {fold_key}")

        fold_models = train_independent_rf(X_train, y_train, model_options)
        all_fold_models[fold_key] = fold_models
        print(f"Completed training model for {fold_key}")
    
    print("ok im done :)")
    return all_fold_models


def evaluate_model_performance(cv_data_splits: List[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]], 
                               trained_models: Dict[str, Dict[str, RF]]) -> Dict[str, Dict[str, float]]:
    """
    Evaluates the performance of trained models on each fold of cross-validation splits.

    Args:
        cv_data_splits: List of tuples containing (X_train, X_test, y_train, y_test) for each fold.
        trained_models: Dictionary of dictionaries containing trained models for each outcome, for each fold.

    Returns:
        A dictionary with evaluation metrics (R-squared and MSE) for each model on each fold.
    """
    performance_metrics = {}

    for fold_index, (_, X_test, _, y_test) in enumerate(cv_data_splits):
        fold_key = f'fold_{fold_index + 1}'
        models = trained_models[fold_key]
        fold_metrics = {}

        # for outcome, model in models.items():
        #     predictions = model.predict(X_test)
        #     r2 = r2_score(y_test, predictions)
        #     mse = mean_squared_error(y_test, predictions)
        #     fold_metrics[outcome] = {'R2': r2, 'MSE': mse}

        # performance_metrics[fold_key] = fold_metrics

        for outcome, model in models.items():
            predictions = model.predict(X_test)
            r2 = r2_score(y_test[outcome], predictions)
            mse = mean_squared_error(y_test[outcome], predictions)
            
            # Flatten the structure
            performance_metrics[f'{fold_key}_{outcome}_R2'] = r2
            performance_metrics[f'{fold_key}_{outcome}_MSE'] = mse

    return performance_metrics
