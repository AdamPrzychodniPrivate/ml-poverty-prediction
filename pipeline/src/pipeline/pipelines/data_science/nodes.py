import logging
from typing import Dict, Tuple, List

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import max_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold


def split_data(data: pd.DataFrame, parameters: Dict) -> Tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Split data.
    """
    X = data[parameters["features"]]
    y = data["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """Trains the linear regression model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.

    Returns:
        Trained model.
    """
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    return regressor


def evaluate_model(
    regressor: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, float]:
    """Calculates and logs the coefficient of determination.

    Args:
        regressor: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for price.
    """
    y_pred = regressor.predict(X_test)
    score = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    me = max_error(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("Model has a coefficient R^2 of %.3f on test data.", score)
    return {"r2_score": score, "mae": mae, "max_error": me}


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
