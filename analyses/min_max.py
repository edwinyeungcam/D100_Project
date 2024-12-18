import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


# This is a custom transformer that performs min max scaling to numerical variables

class CustomMinMaxScaler(BaseEstimator, TransformerMixin):
    def __init__(self, feature_range=(0, 1)):
        """
        Custom implementation of MinMaxScaler.
        """
        self.feature_range = feature_range

    def fit(self, X, y=None):
        """
        Compute the minimum and maximum to be used for scaling.
        
        Parameters:
            X (array-like): Input data to compute the scaling parameters.
            y (ignored): Not used, present for API consistency.
            
        Returns:
            self: Fitted scaler.
        """
        X = np.asarray(X)
        self.data_min_ = np.min(X, axis=0)  # Minimum value for each feature
        self.data_max_ = np.max(X, axis=0)  # Maximum value for each feature
        self.data_range_ = self.data_max_ - self.data_min_  # Range for each feature
        return self

    def transform(self, X):
        """
        Scale the data based on the computed min and max values.
        
        Parameters:
            X (array-like): Input data to scale.
            
        Returns:
            X_scaled: Scaled data.
        """
        check_is_fitted(self, attributes=["data_min_", "data_max_", "data_range_"])
        
        X = np.asarray(X)
        X_scaled = (X - self.data_min_) / self.data_range_  # Scale to [0, 1]
        X_scaled = X_scaled * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]  # Scale to [feature_range[0], feature_range[1]]
        return X_scaled

    def inverse_transform(self, X_scaled):
        """
        Reverse the scaling operation.
        
        Parameters:
            X_scaled (array-like): Scaled data to revert.
            
        Returns:
            X_original: Original data before scaling.
        """
        check_is_fitted(self, attributes=["data_min_", "data_max_", "data_range_"])
        
        X_scaled = np.asarray(X_scaled)
        X_original = (X_scaled - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0])  # Scale back to [0, 1]
        X_original = X_original * self.data_range_ + self.data_min_  # Scale back to original range
        return X_original