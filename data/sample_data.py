# %%
# This script is intended to be run only once to generate a sampled CSV file.
# The stratified data will be saved in the "raw_data" directory, so there is no need to rerun this script once the file is created.
from pathlib import Path
from data_loader import load_csv

path = Path(__file__).parent.parent / "raw_data" / "diabetes_prediction_dataset.csv"
sample_size = 5000
output_path = Path(__file__).parent.parent / "raw_data" / "stratified_diabetes_prediction_dataset.csv"

load_csv(path, sample_size, output_path)
# %%
