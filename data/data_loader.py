import pandas as pd
import os


def load_csv(file_path: str, sample_size: int, output_file_path: str) -> pd.DataFrame:
    """
    Load the raw CSV file into a pandas DataFrame
    Since there are more than 50000 observations in raw CSV file, only a fraction (e.g. 5000) of them will be randomly sampled due to computational constraint
    Random state is set to 42 for reproducibility
    Lastly, save the dataframe to a csv file. 
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    df = pd.read_csv(file_path)
    df = df.sample(n=sample_size, random_state=42)
    df.to_csv(output_file_path, index=False)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load CSV file into a pandas Dataframe
    If there are abundant computational resources, one can load the raw csv file instead of the sampled_file
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")    
    df = pd.read_csv(file_path)
    return df