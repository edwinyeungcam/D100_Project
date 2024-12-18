from sklearn.model_selection import train_test_split
import pandas as pd

def split_data_randomly(df, train_ratio=0.8, random_state=24):
    """
    Randomly split a DataFrame into training and test sets with reproducibility.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to split.
    train_ratio : float, optional
        The proportion of data to include in the training set (default is 0.8).
    random_state : int, optional
        Random seed for reproducibility (default is 24).

    Returns
    -------
    pd.DataFrame
        A DataFrame with an additional 'split' column indicating 'train' or 'test'.
    """
    # Generate random split
    train_df, test_df = train_test_split(
        df, test_size=(1 - train_ratio), random_state=random_state
    )

    # Add 'split' column
    train_df["split"] = "train"
    test_df["split"] = "test"

    # Combine train and test DataFrames
    return pd.concat([train_df, test_df]).reset_index(drop=True)