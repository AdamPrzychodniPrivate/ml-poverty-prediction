from typing import Dict, Tuple

import pandas as pd


def drop_rows_with_missing_values(data: pd.DataFrame, max_missing: int = None) -> pd.DataFrame:
    """
    Drop rows from a DataFrame based on the specified maximum number of allowed missing values.

    Parameters:
    - data (pd.DataFrame): The DataFrame from which rows are to be removed.
    - max_missing (int, optional): The maximum number of missing values allowed in a row. 
                                   Rows with more missing values than this number will be dropped. 
                                   If not specified or None, any row with at least one missing value will be dropped.

    Returns:
    - pd.DataFrame: A new DataFrame with rows dropped based on the specified criteria.

    Raises:
    - ValueError: If the max_missing is negative.
    - TypeError: If the provided dataframe is not a pandas DataFrame.
    """
    if max_missing is not None and (not isinstance(max_missing, int) or max_missing < 0):
        raise ValueError("max_missing must be a non-negative integer or None")
    if not isinstance(data, pd.DataFrame):
        raise TypeError("The first argument must be a pandas DataFrame")

    if max_missing is None:
        return data.dropna(axis=0)
    else:
        min_non_missing = data.shape[1] - max_missing
        return data.dropna(thresh=min_non_missing, axis=0)


def preprocess_raw_data(raw_data: pd.DataFrame, variables: Dict) -> pd.DataFrame:
    """
    Preprocesses the raw data from a research paper.

    This function performs necessary preprocessing steps on the dataset obtained from a research paper. 
    The specific preprocessing tasks can include cleaning the data, handling missing values, and converting 
    data types, but should be defined based on the actual requirements of the dataset.

    Args:
        raw_data: DataFrame containing the raw data from the research paper.

    Returns:
        pd.DataFrame: Preprocessed data, reflecting the necessary preprocessing steps applied to the raw data.
    """
    df = raw_data[variables]
    preprocessed_df = drop_rows_with_missing_values(df)

    return preprocessed_df
