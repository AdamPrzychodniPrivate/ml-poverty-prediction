from typing import Dict, Tuple

import pandas as pd


def _is_true(x: pd.Series) -> pd.Series:
    return x == "t"


def _parse_percentage(x: pd.Series) -> pd.Series:
    x = x.str.replace("%", "")
    x = x.astype(float) / 100
    return x


def _parse_money(x: pd.Series) -> pd.Series:
    x = x.str.replace("$", "").str.replace(",", "")
    x = x.astype(float)
    return x


def preprocess_companies(companies: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Preprocesses the data for companies.

    Args:
        companies: Raw data.
    Returns:
        Preprocessed data, with `company_rating` converted to a float and
        `iata_approved` converted to boolean.
    """
    companies["iata_approved"] = _is_true(companies["iata_approved"])
    companies["company_rating"] = _parse_percentage(companies["company_rating"])
    return companies, {"columns": companies.columns.tolist(), "data_type": "companies"}


def preprocess_shuttles(shuttles: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for shuttles.

    Args:
        shuttles: Raw data.
    Returns:
        Preprocessed data, with `price` converted to a float and `d_check_complete`,
        `moon_clearance_complete` converted to boolean.
    """
    shuttles["d_check_complete"] = _is_true(shuttles["d_check_complete"])
    shuttles["moon_clearance_complete"] = _is_true(shuttles["moon_clearance_complete"])
    shuttles["price"] = _parse_money(shuttles["price"])
    return shuttles


def create_model_input_table(
    shuttles: pd.DataFrame, companies: pd.DataFrame, reviews: pd.DataFrame
) -> pd.DataFrame:
    """Combines all data to create a model input table.

    Args:
        shuttles: Preprocessed data for shuttles.
        companies: Preprocessed data for companies.
        reviews: Raw data for reviews.
    Returns:
        Model input table.

    """
    rated_shuttles = shuttles.merge(reviews, left_on="id", right_on="shuttle_id")
    rated_shuttles = rated_shuttles.drop("id", axis=1)
    model_input_table = rated_shuttles.merge(
        companies, left_on="company_id", right_on="id"
    )
    model_input_table = model_input_table.dropna()
    return model_input_table


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
    print(variables)
    df = raw_data[variables]
    preprocessed_df = drop_rows_with_missing_values(raw_data)

    return preprocessed_df
