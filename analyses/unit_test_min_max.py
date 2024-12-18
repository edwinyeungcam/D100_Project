import numpy as np
import pytest
from .min_max import CustomMinMaxScaler

# To run the unit test, type "pytest analyses/unit_test_min_max.py" in the bash terminal

@pytest.mark.parametrize(
    "feature_range", [(0, 1)] # testing the (0,1) range
)
def test_scaler_transform(feature_range):
    """
    Test the transform method of CustomMinMaxScaler for various feature ranges.
    """
    # Generate a dataset
    X = np.random.normal(0, 1, (500, 10))  # 100 samples, 3 features

    # Fit and transform using the scaler
    scaler = CustomMinMaxScaler(feature_range=feature_range)
    X_transformed = scaler.fit_transform(X)

    # Check that the transformed data lies within the specified range
    assert X_transformed.max() <= feature_range[1], "Scaled values exceed the upper range."
    assert X_transformed.min() >= feature_range[0], "Scaled values fall below the lower range."

@pytest.mark.parametrize(
    "feature_range", [(0, 1)]
)
def test_scaler_inverse_transform(feature_range):
    """
    Test the inverse_transform method of CustomMinMaxScaler for consistency.
    """
    # Generate a dataset
    X = np.random.normal(0, 1, (500, 10))  # 100 samples, 3 features

    # Fit, transform, and inverse transform
    scaler = CustomMinMaxScaler(feature_range=feature_range)
    X_transformed = scaler.fit_transform(X)
    X_original = scaler.inverse_transform(X_transformed)

    # Check that the inverse transformation recovers the original data
    np.testing.assert_array_almost_equal(
        X, X_original, decimal=6, err_msg="Inverse transformation failed."
    )
