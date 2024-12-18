import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def one_hot_encode_column(df, column_name):
    """
    One-hot encodes a specified categorical column in a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the column to one-hot encode.

    Returns:
        pd.DataFrame: A new DataFrame with the one-hot encoded column(s) added
                      and the original column dropped.
    """
    # Create an instance of OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False)  # Dense output for convenience

    # Fit and transform the specified column
    encoded_array = encoder.fit_transform(df[[column_name]])

    # Create a DataFrame from the encoded values
    encoded_df = pd.DataFrame(
        encoded_array, 
        columns=encoder.get_feature_names_out([column_name])
    )

    # Reset the index of the encoded DataFrame to align with the original DataFrame
    encoded_df.index = df.index

    # Concatenate the encoded columns with the original DataFrame
    df_encoded = pd.concat([df, encoded_df], axis=1)

    # Drop the original column
    df_encoded = df_encoded.drop(columns=[column_name])

    return df_encoded