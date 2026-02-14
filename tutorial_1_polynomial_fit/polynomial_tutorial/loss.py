"""
loss.py
=======
Loss functions for training machine learning models.

This module implements common loss functions used in regression tasks.
For this tutorial, we focus on Mean Squared Error (MSE).

Author: ML Tutorial Series
Target Audience: PhD students and early career postdocs
"""

import torch


def mean_squared_error(y_pred, y_true):
    """
    Compute the Mean Squared Error (MSE) loss.
    
    MSE is defined as:
        MSE = (1/n) * Σ(y_pred - y_true)²
    
    This is the most common loss function for regression problems.
    It penalizes larger errors more heavily due to the squaring operation.
    
    Args:
        y_pred (torch.Tensor): Predicted values from the model
        y_true (torch.Tensor): True/observed values
    
    Returns:
        torch.Tensor: Scalar tensor containing the MSE loss
    
    Mathematical Properties:
        - Always non-negative
        - Convex (has a single global minimum)
        - Differentiable everywhere
        - Sensitive to outliers due to squaring
    """
    # Ensure both tensors have the same shape
    assert y_pred.shape == y_true.shape, \
        f"Shape mismatch: y_pred {y_pred.shape} vs y_true {y_true.shape}"
    
    # Compute squared differences
    squared_diff = (y_pred - y_true) ** 2
    
    # Take the mean over all samples
    mse = torch.mean(squared_diff)
    
    return mse


def root_mean_squared_error(y_pred, y_true):
    """
    Compute the Root Mean Squared Error (RMSE).
    
    RMSE = sqrt(MSE)
    
    RMSE is in the same units as the target variable, making it
    more interpretable than MSE.
    
    Args:
        y_pred (torch.Tensor): Predicted values
        y_true (torch.Tensor): True values
    
    Returns:
        torch.Tensor: Scalar tensor containing the RMSE
    """
    mse = mean_squared_error(y_pred, y_true)
    rmse = torch.sqrt(mse)
    return rmse


def mean_absolute_error(y_pred, y_true):
    """
    Compute the Mean Absolute Error (MAE).
    
    MAE = (1/n) * Σ|y_pred - y_true|
    
    MAE is less sensitive to outliers compared to MSE.
    
    Args:
        y_pred (torch.Tensor): Predicted values
        y_true (torch.Tensor): True values
    
    Returns:
        torch.Tensor: Scalar tensor containing the MAE
    """
    absolute_diff = torch.abs(y_pred - y_true)
    mae = torch.mean(absolute_diff)
    return mae
