"""
LinearRegressor.py (now supports Polynomial Regression)
========================================================
A polynomial regression model using PyTorch.

This module defines a polynomial model class that can fit
functions of the form: y = a₀ + a₁x + a₂x² + ... + aₙxⁿ

Author: ML Tutorial Series
Target Audience: PhD students and early career postdocs
"""

import torch
import torch.nn as nn


class PolynomialRegressor(nn.Module):
    """
    A polynomial regression model.
    
    This class implements a polynomial model of the form:
        y = a₀ + a₁x + a₂x² + a₃x³ + ... + aₙxⁿ
    
    where a₀, a₁, ..., aₙ are learnable parameters (coefficients).
    
    Attributes:
        order (int): The polynomial order (n)
        coeffs (nn.Parameter): Tensor of shape (order+1,) containing coefficients
    
    Key Pedagogical Concept:
        The polynomial order creates a bias-variance tradeoff:
        - order < data_order: UNDERFITTING (high bias, low variance)
        - order = data_order: IDEAL FIT (balanced bias-variance)
        - order > data_order: OVERFITTING (low bias, high variance)
    """
    
    def __init__(self, order=1):
        """
        Initialize the PolynomialRegressor with specified order.
        
        Args:
            order (int): Polynomial order. 
                        order=1 → Linear (y = a₀ + a₁x)
                        order=2 → Quadratic (y = a₀ + a₁x + a₂x²)
                        order=3 → Cubic (y = a₀ + a₁x + a₂x² + a₃x³)
                        etc.
        
        The coefficients are initialized with small random values to break
        symmetry and allow gradient descent to converge.
        """
        super(PolynomialRegressor, self).__init__()
        
        self.order = order
        
        # Initialize coefficients [a₀, a₁, a₂, ..., aₙ] with small random values
        # Shape: (order + 1,) to include the constant term a₀
        self.coeffs = nn.Parameter(torch.randn(order + 1) * 0.1)
    
    def forward(self, x):
        """
        Forward pass of the polynomial model.
        
        Computes: y = Σᵢ aᵢ · xⁱ for i = 0 to n
        
        Args:
            x (torch.Tensor): Input tensor of shape (n_samples,) or (n_samples, 1)
        
        Returns:
            torch.Tensor: Predicted output y
        
        Implementation Note:
            Uses explicit power computation for clarity.
            For very high orders (>5) or large x ranges, consider:
            - Feature normalization (e.g., x ∈ [-1, 1])
            - Orthogonal polynomials (Chebyshev, Legendre)
            - Gradient clipping
        """
        # Ensure x is the right shape
        if x.dim() == 1:
            x = x.unsqueeze(1)  # (n_samples,) → (n_samples, 1)
        
        # Create polynomial features: [1, x, x², x³, ..., xⁿ]
        # Using torch.pow for clarity (can also use x ** i)
        powers = torch.cat([x ** i for i in range(self.order + 1)], dim=1)
        # powers shape: (n_samples, order+1)
        
        # Compute y = a₀·1 + a₁·x + a₂·x² + ... + aₙ·xⁿ
        # Matrix multiplication: (n_samples, order+1) @ (order+1,) → (n_samples,)
        y_pred = powers @ self.coeffs
        
        return y_pred
    
    def get_parameters(self):
        """
        Get the current values of all polynomial coefficients.
        
        Returns:
            list: [a₀, a₁, a₂, ..., aₙ] - current coefficient values
        """
        return self.coeffs.detach().cpu().tolist()
    
    def get_parameter_dict(self):
        """
        Get coefficients as a dictionary for easier interpretation.
        
        Returns:
            dict: {0: a₀, 1: a₁, 2: a₂, ..., n: aₙ}
        """
        coeffs = self.get_parameters()
        return {i: coeff for i, coeff in enumerate(coeffs)}
    
    def __repr__(self):
        """
        String representation showing the polynomial equation.
        """
        coeffs = self.get_parameters()
        terms = []
        
        # Constant term
        terms.append(f"{coeffs[0]:.4f}")
        
        # Linear and higher order terms
        for i in range(1, len(coeffs)):
            sign = "+" if coeffs[i] >= 0 else "-"
            coeff_abs = abs(coeffs[i])
            if i == 1:
                terms.append(f"{sign} {coeff_abs:.4f}x")
            else:
                terms.append(f"{sign} {coeff_abs:.4f}x^{i}")
        
        equation = " ".join(terms)
        return f"PolynomialRegressor(order={self.order}, y = {equation})"


# Backward compatibility: LinearRegressor is a special case of PolynomialRegressor
class LinearRegressor(PolynomialRegressor):
    """
    Linear regression model (order=1 polynomial).
    
    This is a convenience class that defaults to order=1.
    Maintains backward compatibility with Tutorial 1.
    """
    def __init__(self):
        super(LinearRegressor, self).__init__(order=1)
    
    def get_parameters(self):
        """
        Get parameters in (a, b) format for backward compatibility.
        
        Returns:
            tuple: (a, b) where a is intercept and b is slope
        """
        coeffs = self.coeffs.detach().cpu().tolist()
        return coeffs[0], coeffs[1]
    
    def __repr__(self):
        """
        String representation for linear model.
        """
        a, b = self.get_parameters()
        return f"LinearRegressor(y = {a:.4f} + {b:.4f}x)"
