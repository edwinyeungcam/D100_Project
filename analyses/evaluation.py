from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import dalex as dx

def evaluate_model(model, X_train, y_train, X_test, y_test, average='binary'):
    """
    Evaluate a model's performance on both training and testing datasets.
    
    Parameters:
        model: Trained model to evaluate.
        X_train: Training feature set.
        y_train: True labels for the training set.
        X_test: Testing feature set.
        y_test: True labels for the testing set.
        average: Averaging method for precision and recall
                 
    Returns:
        A dictionary containing training and testing metrics.
    """
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_metrics = {
        'accuracy': accuracy_score(y_train, y_train_pred),
        'precision': precision_score(y_train, y_train_pred, average=average),
        'recall': recall_score(y_train, y_train_pred, average=average)
    }
    
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred, average=average),
        'recall': recall_score(y_test, y_test_pred, average=average)
    }
    
    return train_metrics, test_metrics

def display_metrics(metrics, dataset_type=""):
    """
    Display metrics in a formatted way.
    
    Parameters:
        metrics: Dictionary of metrics to display.
        dataset_type: Type of dataset (e.g., "Training" or "Testing").
    """
    print(f"{dataset_type} Dataset Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    print()

def main_evaluation(model, X_train, y_train, X_test, y_test, average='binary'):
    """
    Perform the evaluation of a model on training and testing datasets and display the results.
    
    Parameters:
        model: Trained model to evaluate.
        X_train: Training feature set.
        y_train: True labels for the training set.
        X_test: Testing feature set.
        y_test: True labels for the testing set.
        average: Averaging method for precision and recall (default: 'binary').
    """
    train_metrics, test_metrics = evaluate_model(model, X_train, y_train, X_test, y_test, average=average)
    display_metrics(train_metrics, dataset_type="Training")
    display_metrics(test_metrics, dataset_type="Testing")

def plot_predicted_vs_actual(model, X_test, y_test, decision_boundary=0.5):
    """
    Generate and plot the Predicted vs. Actual values for a given model and dataset.
    
    Parameters:
        model: Trained model used to predict probabilities.
        X_test: Test feature set.
        y_test: True labels for the test set.
        decision_boundary: Decision boundary for classification (default: 0.5).
    """
    # Predict probabilities for the positive class (1)
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Create a DataFrame with actual values and predicted probabilities
    pred_vs_actual_df = pd.DataFrame({
        'Actual': y_test.values.ravel(),
        'Predicted_Probability': y_test_pred_proba
    })
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=pred_vs_actual_df, x='Actual', y='Predicted_Probability', alpha=0.6)
    plt.title("Predicted vs. Actual Plot (GLM)")
    plt.xlabel("Actual Values (0 = Negative, 1 = Positive)")
    plt.ylabel("Predicted Probability of Positive Class")
    plt.axhline(decision_boundary, color='red', linestyle='--', label=f'Decision Boundary ({decision_boundary})')
    plt.legend()
    plt.show()

def plot_roc_auc(model, X_test, y_test):
    """
    Compute and plot the ROC-AUC curve for a given model and dataset.
    
    Parameters:
        model: Trained model used to predict probabilities.
        X_test: Test feature set.
        y_test: True labels for the test set.
    """
    # Predict probabilities for the positive class (1)
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Compute ROC curve and AUC score
    fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_proba)
    roc_auc = roc_auc_score(y_test, y_test_pred_proba)
    
    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label="Random Guess (AUC = 0.50)")
    plt.title("ROC-AUC Curve (GLM)")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.legend()
    plt.show()

def plot_pdp_for_top_features(model, X_train, y_train, top_n=5):
    """
    Generate and plot Partial Dependence Plots (PDP) for the top features based on feature importance.
    
    Parameters:
        model: Trained LightGBM pipeline model.
        X_train: Training feature set (Pandas DataFrame).
        y_train: True labels for the training set.
        top_n: Number of top features to generate PDPs for (default: 5).
    """
    # Extract feature importances from the LightGBM model
    feature_importances = model.named_steps['lgbm'].feature_importances_
    feature_names = X_train.columns

    # Combine feature importances with their names and sort them
    sorted_features = sorted(zip(feature_importances, feature_names), reverse=True, key=lambda x: x[0])

    # Select the top N most important features
    top_features = [feature for _, feature in sorted_features[:top_n]]

    print(f"Top {top_n} Features Based on LGBM Feature Importance:")
    print(top_features)

    # Create a Dalex explainer for the LightGBM model
    explainer = dx.Explainer(model, X_train, y_train, label="LightGBM Model")

    # Generate and plot Partial Dependence Plots for each feature
    for feature in top_features:
        print(f"Plotting PDP for feature: {feature}")
        pdp = explainer.model_profile(variables=[feature])  # Generate PDP for the feature
        
        # Plot the PDP
        plt.figure(figsize=(12, 8))  # Adjust figure size as needed
        pdp.plot(title=f"PDP for {feature}")
        plt.show()

def plot_roc_auc_compare_models(model1, model2, X_test, y_test, model1_name="Model 1", model2_name="Model 2"):
    """
    Compute and plot the ROC-AUC curves for two models on the same graph.
    
    Parameters:
        model1: First trained model used to predict probabilities.
        model2: Second trained model used to predict probabilities.
        X_test: Test feature set.
        y_test: True labels for the test set.
        model1_name: Name/label for the first model (used in the legend).
        model2_name: Name/label for the second model (used in the legend).
    """
    # Predict probabilities for the positive class (1) for both models
    y_test_pred_proba_1 = model1.predict_proba(X_test)[:, 1]
    y_test_pred_proba_2 = model2.predict_proba(X_test)[:, 1]
    
    # Compute ROC curve and AUC score for both models
    fpr1, tpr1, _ = roc_curve(y_test, y_test_pred_proba_1)
    roc_auc1 = roc_auc_score(y_test, y_test_pred_proba_1)
    
    fpr2, tpr2, _ = roc_curve(y_test, y_test_pred_proba_2)
    roc_auc2 = roc_auc_score(y_test, y_test_pred_proba_2)
    
    # Plot the ROC curves
    plt.figure(figsize=(8, 6))
    plt.plot(fpr1, tpr1, color='blue', label=f"{model1_name} (AUC = {roc_auc1:.2f})")
    plt.plot(fpr2, tpr2, color='green', label=f"{model2_name} (AUC = {roc_auc2:.2f})")
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label="Random Guess (AUC = 0.50)")
    
    # Add plot details
    plt.title("ROC-AUC Curve Comparison")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()