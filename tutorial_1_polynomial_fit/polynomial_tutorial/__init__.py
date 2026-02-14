"""
Tutorial 1: Polynomial regression using PyTorch

This package aims to teach the basic pythonic structure of PyTorch
to students wanting to get into Machine Learning libraries
"""

__version__ = "0.1.0"

from .LinearRegressor import LinearRegressor,PolynomialRegressor
from .train import train_model, evaluate_model
from .utils import FeatureNormalizer 
from .loss import mean_squared_error, root_mean_squared_error, mean_absolute_error

__all__ = [
    "LinearRegressor",
    "PolynomialRegressor",
    "train_model",
    "evaluate_model",
    "FeatureNormalizer",
    "mean_squared_error",
    "root_mean_squared_error",
    "mean_absolute_error"
]
