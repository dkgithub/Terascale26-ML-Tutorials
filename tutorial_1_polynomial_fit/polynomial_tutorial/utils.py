"""
utils.py
========
Utility functions for polynomial regression.

This module provides tools for feature normalization, which is critical
for numerical stability in polynomial regression.

Author: ML Tutorial Series
Target Audience: PhD students and early career postdocs
"""

import torch
import logging

logger = logging.getLogger(__name__)


class FeatureNormalizer:
    """
    Normalizes features to a standard range for numerical stability.
    
    For polynomial regression, normalizing x BEFORE computing powers
    is crucial to prevent numerical overflow/underflow.
    
    Example:
        If x ∈ [0, 10] and we compute x⁵:
        - Without normalization: x⁵ ∈ [0, 100,000] 🚨 HUGE RANGE
        - With normalization to [-1, 1]: x⁵ ∈ [-1, 1] ✅ STABLE
    
    This is especially important for polynomial orders > 2.
    """
    
    def __init__(self, method='standardize'):
        """
        Initialize the normalizer.
        
        Args:
            method (str): Normalization method
                - 'standardize': (x - mean) / std  → mean=0, std=1
                - 'minmax': (x - min) / (max - min)  → range [0, 1]
                - 'symmetric': 2*(x - min)/(max - min) - 1  → range [-1, 1]
        """
        self.method = method
        self.mean = None
        self.std = None
        self.min = None
        self.max = None
        self.fitted = False
    
    def fit(self, x):
        """
        Compute normalization parameters from data.
        
        Args:
            x (torch.Tensor): Input data to fit normalization parameters
        """
        if self.method == 'standardize':
            self.mean = x.mean()
            self.std = x.std()
            if self.std == 0:
                self.std = torch.tensor(1.0)  # Avoid division by zero
                logger.warning("Standard deviation is zero, setting to 1.0")
        
        elif self.method in ['minmax', 'symmetric']:
            self.min = x.min()
            self.max = x.max()
            if self.min == self.max:
                self.max = self.min + 1.0  # Avoid division by zero
                logger.warning("Min equals max, adjusting max to min+1")
        
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
        
        self.fitted = True
        logger.info(f"Fitted {self.method} normalizer: "
                   f"mean={self.mean}, std={self.std}, min={self.min}, max={self.max}")
    
    def transform(self, x):
        """
        Normalize data using fitted parameters.
        
        Args:
            x (torch.Tensor): Data to normalize
        
        Returns:
            torch.Tensor: Normalized data
        """
        if not self.fitted:
            raise RuntimeError("Normalizer must be fit before transform")
        
        if self.method == 'standardize':
            return (x - self.mean) / self.std
        
        elif self.method == 'minmax':
            return (x - self.min) / (self.max - self.min)
        
        elif self.method == 'symmetric':
            # Maps to [-1, 1]
            return 2 * (x - self.min) / (self.max - self.min) - 1
    
    def inverse_transform(self, x_normalized):
        """
        Denormalize data back to original scale.
        
        Args:
            x_normalized (torch.Tensor): Normalized data
        
        Returns:
            torch.Tensor: Original scale data
        """
        if not self.fitted:
            raise RuntimeError("Normalizer must be fit before inverse_transform")
        
        if self.method == 'standardize':
            return x_normalized * self.std + self.mean
        
        elif self.method == 'minmax':
            return x_normalized * (self.max - self.min) + self.min
        
        elif self.method == 'symmetric':
            # Reverse: x = ((x_norm + 1) / 2) * (max - min) + min
            return ((x_normalized + 1) / 2) * (self.max - self.min) + self.min
    
    def fit_transform(self, x):
        """
        Fit normalizer and transform data in one step.
        
        Args:
            x (torch.Tensor): Data to fit and normalize
        
        Returns:
            torch.Tensor: Normalized data
        """
        self.fit(x)
        return self.transform(x)


def demonstrate_normalization_effect():
    """
    Educational demonstration of why normalization is critical.
    
    Shows the range of polynomial features with and without normalization.
    """
    logger.info("\n" + "="*70)
    logger.info("DEMONSTRATION: Why Feature Normalization Matters")
    logger.info("="*70)
    
    # Create sample data
    x = torch.linspace(0, 10, 5)  # [0, 2.5, 5, 7.5, 10]
    
    logger.info(f"\nOriginal x: {x.tolist()}")
    logger.info(f"Range: [{x.min():.1f}, {x.max():.1f}]")
    
    # Show polynomial features WITHOUT normalization
    logger.info("\nWithout normalization:")
    for order in [1, 2, 3, 4, 5]:
        x_power = x ** order
        logger.info(f"  x^{order}: range [{x_power.min():.1f}, {x_power.max():.1f}] "
                   f"(span: {x_power.max() - x_power.min():.1f})")
    
    # Show polynomial features WITH normalization
    normalizer = FeatureNormalizer(method='symmetric')
    x_norm = normalizer.fit_transform(x)
    
    logger.info(f"\nNormalized x: {x_norm.tolist()}")
    logger.info(f"Range: [{x_norm.min():.1f}, {x_norm.max():.1f}]")
    
    logger.info("\nWith normalization to [-1, 1]:")
    for order in [1, 2, 3, 4, 5]:
        x_power = x_norm ** order
        logger.info(f"  x^{order}: range [{x_power.min():.1f}, {x_power.max():.1f}] "
                   f"(span: {x_power.max() - x_power.min():.1f})")
    
    logger.info("\n" + "="*70)
    logger.info("KEY INSIGHT: All polynomial features stay in [-1, 1] range!")
    logger.info("This prevents numerical overflow and enables stable training.")
    logger.info("="*70 + "\n")


if __name__ == "__main__":
    # Demo when run directly
    import logging
    from logger import configure_logging
    configure_logging()
    
    demonstrate_normalization_effect()
