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

def check_missing_values(df: pd.DataFrame) -> dict:
    """
    Checks for missing values in each column of the DataFrame and returns a summary
    Importantly, for smoking history, missing values are represented as "No Info", therefore the function will also search for "No Info"
    """
    missing_placeholders = ["No Info"]

    df_copy = df.copy()

    # Replace custom "No Info" with NaN
    for placeholder in missing_placeholders:
        df_copy = df_copy.replace(placeholder, pd.NA)

    # Calculate missing values for each column
    missing_count = df_copy.isnull().sum()
    missing_percentage = (missing_count / len(df_copy)) * 100

    # Create a descriptive summary for each column
    report = {}
    for column in df_copy.columns:
        if missing_count[column] > 0:
            report[column] = f"Missing values ({missing_count[column]} = {missing_percentage[column]:.2f}%)"
        else:
            report[column] = "No missing values"

    return report