# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dalex as dx
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, FunctionTransformer
from sklearn.model_selection import GridSearchCV
from data_split import split_data_randomly
from evaluation import main_evaluation, plot_predicted_vs_actual, plot_roc_auc, plot_pdp_for_top_features
from lightgbm import LGBMClassifier

# %%

path = Path(__file__).parent.parent / "raw_data" / "cleaned_stratified_diabetes_prediction_dataset.csv"
clean_df = pd.read_parquet(path)
clean_df.head()

# %%
train_test_df = split_data_randomly(clean_df)
train_test_df.head()

# %%
train_df = pd.DataFrame(train_test_df[train_test_df["split"] == "train"])
test_df = pd.DataFrame(train_test_df[train_test_df["split"] == "test"])

# Separate features (X) and target (y) for training set
X_train = pd.DataFrame(train_df.drop(columns=["diabetes", "split"]))  # Drop target and split columns
y_train = pd.DataFrame(train_df["diabetes"])  # Target column as DataFrame

# Separate features (X) and target (y) for testing set
X_test = pd.DataFrame(test_df.drop(columns=["diabetes", "split"]))  # Drop target and split columns
y_test = pd.DataFrame(test_df["diabetes"])  # Target column as DataFrame

# %%
preprocessor = ColumnTransformer(transformers=[
    # Step 1: Standardize age and bmi
    ('first_scaler', MinMaxScaler(), ['age','bmi']),
    
    # Step 2: Log-transform and standardize 'HbA1c_level' and 'blood_glucose_level'
    ('log_and_scale', Pipeline([
        ('log_transform', FunctionTransformer(np.log1p)),  # Apply log transformation
        ('scaler', MinMaxScaler())  # Standardize
    ]), ['HbA1c_level', 'blood_glucose_level']),
    
    # Step 3: Polynomial Features for 'bmi', 'HbA1c_level', 'blood_glucose_level' (After testing, no polynomial features lead to better accuracy)
    #('poly_features', PolynomialFeatures(degree=2, include_bias=False), ['age','bmi', 'HbA1c_level', 'blood_glucose_level']),
])

# %%
glm = LogisticRegression(penalty='elasticnet', solver='saga', max_iter=10000)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),          # Preprocessing step
    ('logistic', glm)             # Logistic Regression step
])
pipeline

#%%
# Define hyperparameter grid for Elastic Net Logistic Regression
param_grid_glm = {
    'logistic__C': [0.001, 0.01, 0.1, 1, 10],        # Regularization strength
    'logistic__l1_ratio': [0.01, 0.05, 0.1, 0.5, 1.0]  # Mix of L1 and L2 penalties
}

glm_grid_search = GridSearchCV(pipeline, param_grid_glm, cv=5, scoring='accuracy')

glm_grid_search.fit(X_train, y_train)

#%%
print(glm_grid_search.best_params_)

# %%
# Evaluation of the GLM Model

# Use the best model from GridSearchCV
glm_best_model = glm_grid_search.best_estimator_

# Perform evaluation
main_evaluation(glm_best_model, X_train, y_train, X_test, y_test, average='binary')

# %%
# Plot Predicted vs Actual 
plot_predicted_vs_actual(glm_best_model, X_test, y_test, decision_boundary=0.5)

# %%
# Plot ROCAUC Curve
plot_roc_auc(glm_best_model, X_test, y_test)

# %%
# Create a Dalex explainer for the GLM model
glm_explainer = dx.Explainer(glm_best_model, X_train, y_train, label="GLM Logistic Regression")

# Feature importance
glm_explainer.model_parts().plot()

#%%
# LGBM Model
# Define the hyperparameter grid for LightGBM
param_grid_lgbm = {
    'lgbm__learning_rate': [0.05, 0.01, 0.05, 0.1, 0.2],  
    'lgbm__n_estimators': [50, 100, 200, 500, 1000], 
    'lgbm__num_leaves': [7, 15, 31, 63, 127],  
    'lgbm__min_data_in_leaf': [5, 10, 20, 40, 60]  
}

lgbm = LGBMClassifier(random_state=24)

pipeline_lgbm_no_preprocessing = Pipeline(steps=[
    ('lgbm', lgbm)
])

pipeline

#%%
lgbm_grid_search = GridSearchCV(pipeline_lgbm_no_preprocessing, param_grid_lgbm, cv=5, scoring='accuracy')

lgbm_grid_search.fit(X_train, y_train)  # Ensure this step is executed


#%%
print(lgbm_grid_search.best_params_)

#%%
lgbm_best_model = lgbm_grid_search.best_estimator_

main_evaluation(lgbm_best_model, X_train, y_train, X_test, y_test, average='binary')

#%%
# Plot Predicted vs Actual 
plot_predicted_vs_actual(lgbm_best_model, X_test, y_test, decision_boundary=0.5)

# %%
# Plot ROCAUC Curve
plot_roc_auc(lgbm_best_model, X_test, y_test)

# %%
# Create a Dalex explainer for the LightGBM model
lgbm_explainer = dx.Explainer(lgbm_best_model, X_train, y_train, label="LightGBM Classifier")

# Feature importance
lgbm_explainer.model_parts().plot()


# %%
#PDP Plots
plot_pdp_for_top_features(lgbm_best_model, X_train, y_train, top_n=5)
