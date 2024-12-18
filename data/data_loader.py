import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

def load_csv(file_path: str, sample_size: int, output_file_path: str) -> pd.DataFrame:
    """
    Load the raw CSV file into a pandas DataFrame.
    Perform stratified sampling based on the 'loan_status' column to address class imbalance.
    Sample size is fixed (e.g., 5000) and distributed equally across 'loan_status' classes.
    Random state is set to 42 for reproducibility.
    Store the sampled DataFrame to a CSV file.
    """
    # Ensure the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Validate if the 'loan_status' column exists
    if 'loan_status' not in df.columns:
        raise ValueError("The dataset does not have a 'loan_status' column for stratification.")
    
    # Verify the sample size is a multiple of 2 (since we need equal samples for both classes)
    if sample_size % 2 != 0:
        raise ValueError("Sample size must be an even number for stratified sampling.")
    
    # Calculate the required sample size per class
    sample_size_per_class = sample_size // 2  # 2500 for each class if sample_size = 5000
    
    # Check if there are enough samples in each class
    loan_status_counts = df['loan_status'].value_counts()
    if (loan_status_counts[0] < sample_size_per_class) or (loan_status_counts[1] < sample_size_per_class):
        raise ValueError("Not enough data in one or more classes to perform stratified sampling.")
    
    # Randomly sample 2500 rows from each class
    sampled_0 = df[df['loan_status'] == 0].sample(n=sample_size_per_class, random_state=42)
    sampled_1 = df[df['loan_status'] == 1].sample(n=sample_size_per_class, random_state=42)
    
    # Combine the sampled data while shuffling the rows
    stratified_sample = pd.concat([sampled_0, sampled_1]).sample(frac=1, random_state=42)  
    
    # Save the sampled DataFrame to a CSV file
    stratified_sample.to_csv(output_file_path, index=False)
    print(f"Sampled data saved to {output_file_path}")
    
    return stratified_sample

def load_data(file_name: str) -> pd.DataFrame:
    """
    Load CSV file into a pandas Dataframe
    If there are abundant computational resources, one can load the raw csv file instead of the sampled_file
    """
    file_path = Path(__file__).parent.parent / "raw_data" / file_name
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")    
    df = pd.read_csv(file_path)
    return df

def write_to_parquet(df):

    path = Path(__file__).parent.parent / "raw_data" / "cleaned_stratified_diabetes_prediction_dataset.csv"
    df.to_parquet(path)