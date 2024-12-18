# %%
import pandas as pd
from pathlib import Path
from data_split import split_data_randomly

# %%

path = Path(__file__).parent.parent / "raw_data" / "cleaned_stratified_diabetes_prediction_dataset.csv"
clean_df = pd.read_parquet(path)
clean_df.head()

# %%
split_data_by_id(clean_df)

# %%