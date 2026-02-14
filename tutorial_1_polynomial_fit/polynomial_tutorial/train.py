"""
train.py
========
Training loop implementation for PyTorch models.

This module contains the training logic for fitting a model to data
using gradient descent optimization.

Author: ML Tutorial Series
Target Audience: PhD students and early career postdocs
"""

import torch
import torch.optim as optim
from .loss import mean_squared_error
import logging

logger = logging.getLogger(__name__)


def train_model(model, x_train, y_train, num_epochs=1000, learning_rate=0.01, 
                print_every=100, verbose=True):
    """
    Train a PyTorch model using gradient descent.
    
    This function implements the standard training loop:
        1. Forward pass: compute predictions
        2. Compute loss
        3. Backward pass: compute gradients
        4. Update parameters using optimizer
    
    Args:
        model (nn.Module): The model to train (e.g., PolynomialRegressor)
        x_train (torch.Tensor): Training input data
        y_train (torch.Tensor): Training target data
        num_epochs (int): Number of training iterations
        learning_rate (float): Step size for gradient descent
        print_every (int): Print loss every N epochs
        verbose (bool): Whether to print training progress
    
    Returns:
        tuple: (trained_model, training_history)
            - trained_model: The model after training
            - training_history: Dict containing:
                - 'loss': List of loss values at each epoch
                - 'params': List of parameter snapshots at each epoch
    
    Training Algorithm:
        For each epoch:
            1. θ_new = θ_old - α * ∇L(θ)
        
        where:
            - θ: model parameters (coefficients)
            - α: learning rate
            - ∇L(θ): gradient of loss w.r.t. parameters
    """
    # Initialize optimizer (Stochastic Gradient Descent)
    # The optimizer will update model parameters based on their gradients
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # Store training history
    loss_history = []
    param_history = []  # Track parameter evolution
    
    # Get initial parameters for reporting
    if verbose:
        initial_params = model.get_parameters()
        logger.info("=" * 60)
        logger.info("Training Polynomial Regression Model")
        logger.info("=" * 60)
        logger.info(f"Model order: {model.order}")
        logger.info(f"Initial parameters: {initial_params}")
        logger.info(f"Learning rate: {learning_rate}")
        logger.info(f"Number of epochs: {num_epochs}")
        logger.info(f"Number of training samples: {len(x_train)}")
        logger.info("=" * 60)
    
    # Training loop
    for epoch in range(num_epochs):
        # 1. FORWARD PASS
        # Compute predictions using current model parameters
        y_pred = model(x_train)
        
        # 2. COMPUTE LOSS
        # Measure how far predictions are from true values
        loss = mean_squared_error(y_pred, y_train)
        
        # Store loss for later analysis
        loss_history.append(loss.item())
        
        # Store parameter snapshot
        # Detach and clone to avoid keeping computation graph in memory
        current_params = [p.detach().clone() for p in model.parameters()]
        param_history.append(current_params)
        
        # 3. BACKWARD PASS
        # Zero out gradients from previous iteration
        # (PyTorch accumulates gradients by default)
        optimizer.zero_grad()
        
        # Compute gradients of loss w.r.t. all parameters
        # This implements backpropagation: ∇L(θ)
        loss.backward()
        
        # 4. UPDATE PARAMETERS
        # Take a step in the direction of steepest descent
        # θ_new = θ_old - learning_rate * gradient
        optimizer.step()
        
        # Print progress
        if verbose and (epoch + 1) % print_every == 0:
            current_params_list = model.get_parameters()
            logger.info(f"Epoch [{epoch+1:4d}/{num_epochs}] | "
                  f"Loss: {loss.item():.6f} | "
                  f"Params: {[f'{p:.4f}' for p in current_params_list]}")
    
    # Print final results
    if verbose:
        logger.info("=" * 60)
        logger.info("Training completed!")
        final_params = model.get_parameters()
        logger.info(f"Final parameters: {final_params}")
        logger.info(f"Final loss: {loss_history[-1]:.6f}")
        logger.info("=" * 60)
    
    # Prepare training history dictionary
    training_history = {
        'loss': loss_history,
        'params': param_history
    }
    
    return model, training_history


def evaluate_model(model, x_test, y_test):
    """
    Evaluate a trained model on test data.
    
    Args:
        model (nn.Module): Trained model
        x_test (torch.Tensor): Test input data
        y_test (torch.Tensor): Test target data
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Set model to evaluation mode (important for models with dropout/batchnorm)
    model.eval()
    
    # Disable gradient computation for efficiency
    with torch.no_grad():
        y_pred = model(x_test)
        mse = mean_squared_error(y_pred, y_test)
        rmse = torch.sqrt(mse)
    
    # Set model back to training mode
    model.train()
    
    return {
        'mse': mse.item(),
        'rmse': rmse.item()
    }
