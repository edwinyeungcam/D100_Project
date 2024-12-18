import pandas as pd


def check_binary_column(dataframe: pd.DataFrame, column: str) -> None:
    """
    Checks if a column in the DataFrame contains exactly two unique values.
    Subsequently, prints its unique values, value counts, and proportions.
    """
    unique_values = dataframe[column].unique()
    
    if len(unique_values) != 2:
        print(f"Column {column} does not contain exactly two unique values: {unique_values}")
    else:
        print(f"Column {column} contains exactly two unique values: {unique_values}.")
    
    # Value counts
    value_counts = dataframe[column].value_counts()
    total = len(dataframe[column])
    proportions = (value_counts / total).round(4)

    print(f"Counts and Proportions in {column}:")
    for value, count in value_counts.items():
        proportion = proportions[value]
        print(f"  {value}: {count} ({proportion * 100:.2f}%)")
        
    print()

def check_missing_values(df: pd.DataFrame, columns: list) -> dict:
    """
    Checks for missing values in the specified numerical columns of the DataFrame
    and returns a summary.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        columns (list): List of column names to check for missing values.

    Returns:
        dict: A dictionary summarizing missing values for each column.
    """
    # Ensure only numerical columns are checked
    numerical_columns = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]

    # Calculate missing values for each column
    missing_count = df[numerical_columns].isnull().sum()
    missing_percentage = (missing_count / len(df)) * 100

    # Create a descriptive summary for each column
    report = {}
    for column in numerical_columns:
        if missing_count[column] > 0:
            report[column] = f"Missing values ({missing_count[column]} = {missing_percentage[column]:.2f}%)"
        else:
            report[column] = "No missing values"

    return report

def check_outliers(df: pd.DataFrame, columns: list) -> bool:
    """
    Checks whether outliers are present in a DataFrame using the IQR method.
    """
    outlier_results = {}
    
    for column in columns:
        # Calculate IQR
        Q1 = df[column].quantile(0.25)  
        Q3 = df[column].quantile(0.75)  
        IQR = Q3 - Q1                   
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Check if outliers exist in the column
        has_outliers = ((df[column] < lower_bound) | (df[column] > upper_bound)).any()
        outlier_results[column] = has_outliers

    return outlier_results

def display_unique_values(data, columns):
    """
    Display unique values from specified columns in the DataFrame.

    Args:
        data (pd.DataFrame): The DataFrame containing the data.
        columns (list): List of column names to display unique values for.
    """
    for column in columns:
        print(f"{column}: {data[column].unique()}")