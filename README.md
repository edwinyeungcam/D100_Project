# D100 Project

# This repository contains a task predicting diabetes. The project focuses exploratory data analysis (EDA) and model training. The repository is structured as a Python package and  includes all required configuration files, documentation, and scripts.

mamba env create
conda activate D100_project_env

pre-commit install

pip install --no-build-isolation -e .

# To run the EDA

jupyter notebook eda_cleaning.ipynb

# To run training model

python analyses/model_training.py

# To run the unit test

pytest analyses/unit_test_min_max.py
