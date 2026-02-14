"""
=======
Main steering script for Linear Regression Tutorial.

This script demonstrates:
    1. Data generation from a known linear relationship
    2. Model definition and initialization
    3. Training loop and optimization
    4. Visualization of results

Author: Stephen Jiggins

Usage:
    python main.py
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from .LinearRegressor import LinearRegressor, PolynomialRegressor
from .train import train_model, evaluate_model
from .utils import FeatureNormalizer

# Good standard for logging information in python code
import logging
from .logger import configure_logging
configure_logging()
logger = logging.getLogger(__name__)

def generate_data(coeffs_true,
                  data_poly_order,
                  n_samples=100,
                  x_min=0, x_max=10,
                  noise_std=0.5):
    """
    Generate synthetic data from a polynomial relationship with noise.
    
    The true model is: 
    
       y = a₀ + a₁x + a₂x² + ... + aₙxⁿ + ε
    
    where ε ~ N(0, noise_std^2) is Gaussian noise
    
    Args:
        coeffs_true (list)    : True polynomial coefficients [a₀, a₁, a₂, ..., aₙ]
        data_poly_order (int) : Order of the true polynomial (n)
        n_samples (int)       : Number of data points to generate
        x_min (float)         : Minimum x value
        x_max (float)         : Maximum x value
        noise_std (float)     : Standard deviation of Gaussian noise
    
    Returns:
        tuple: (x_t, y_t) - PyTorch tensors containing input and output data
    
    Pedagogical Note:
        The data_poly_order determines the true complexity of the underlying
        function. This will be compared against regressor_poly_order to
        demonstrate the bias-variance tradeoff.
    """
    # Validate inputs
    assert len(coeffs_true) == data_poly_order + 1, \
        f"coeffs_true length ({len(coeffs_true)}) must equal data_poly_order + 1 ({data_poly_order + 1})"
    
    # Generate x values uniformly in [x_min, x_max]
    x = torch.linspace(x_min, x_max, n_samples)
    
    # Compute true y values: y = Σᵢ aᵢxⁱ
    y_true = torch.zeros(n_samples)
    for i, coeff in enumerate(coeffs_true):
        y_true += coeff * (x ** i)
    
    # Add Gaussian noise: ε ~ N(0, noise_std²)
    noise = torch.randn(n_samples) * noise_std
    y_noisy = y_true + noise
    
    return x, y_noisy


def plot_parameter_evolution(training_history, coeffs_true, model):
    """
    Visualize how parameters evolve during training.
    
    Creates a plot showing each coefficient's trajectory toward its optimal value.
    This helps students understand:
        1. Convergence dynamics
        2. Which parameters converge faster/slower
        3. The learning process in parameter space
    
    Args:
        training_history (dict): Contains 'loss' and 'params' from training
        coeffs_true (list): True coefficients for reference lines
        model: Trained model (for getting final parameters)
    """
    param_history = training_history['params']
    num_epochs = len(param_history)
    
    # Fix: Extract coefficients correctly from parameter history
    # Each epoch_params is a list with one element: the coeffs tensor of shape (n+1,)
    # We need to extract that tensor and convert to numpy
    param_array = np.array([epoch_params[0].detach().cpu().numpy() 
                            for epoch_params in param_history])
    # Shape: (num_epochs, num_params)
    
    num_params = param_array.shape[1]
    
    # Create figure with subplots for each parameter
    fig, axes = plt.subplots(1, num_params, figsize=(5 * num_params, 4))
    if num_params == 1:
        axes = [axes]  # Make it iterable
    
    epochs = np.arange(num_epochs)
    
    for i in range(num_params):
        ax = axes[i]
        
        # Plot parameter evolution
        ax.plot(epochs, param_array[:, i], 'b-', linewidth=2, 
                label=f'Estimated a_{i}')
        
        # Plot true value as reference (if available)
        if i < len(coeffs_true):
            ax.axhline(y=coeffs_true[i], color='g', linestyle='--', 
                      linewidth=2, label=f'True a_{i} = {coeffs_true[i]:.3f}')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(f'Coefficient a_{i}', fontsize=12)
        ax.set_title(f'Parameter a_{i} Evolution', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('parameter_evolution.png', dpi=150, bbox_inches='tight')
    logger.info("Parameter evolution plot saved as 'parameter_evolution.png'")
    plt.show()


def plot_results(x_t, y_t,
                 model,
                 training_history,
                 coeffs_true,
                 data_poly_order,
                 normalizer=None):
    """
    Visualize the training results.
    
    Creates a figure with two subplots:
        1. Data points and fitted curve
        2. Loss curve during training
    
    Args:
        x_t (torch.Tensor)      : Input data (in original or normalized scale)
        y_t (torch.Tensor)      : Observed output data
        model                   : Trained model
        training_history (dict) : Contains loss and parameter history
        coeffs_true (list)      : True coefficients (for reference)
        data_poly_order (int)   : Order of true data polynomial
        normalizer (FeatureNormalizer): Optional normalizer for denormalization
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Denormalize x for plotting if normalizer is provided
    if normalizer is not None:
        x_display = normalizer.inverse_transform(x_t).detach().numpy()
        x_plot_tensor = x_t  # Model expects normalized x
    else:
        x_display = x_t.detach().numpy()
        x_plot_tensor = x_t
    
    y_np = y_t.detach().numpy()
    
    # Get model predictions (model works on normalized scale)
    with torch.no_grad():
        y_pred = model(x_plot_tensor).numpy()
    
    # Compute true function (on original scale for display)
    if normalizer is not None:
        # Generate dense points for smooth curve in ORIGINAL scale
        x_dense_original = torch.linspace(x_display.min(), x_display.max(), 200)
        x_dense_normalized = normalizer.transform(x_dense_original)
        
        # True function on original scale
        y_true_dense = torch.zeros_like(x_dense_original)
        for i, coeff in enumerate(coeffs_true):
            y_true_dense += coeff * (x_dense_original ** i)
        
        # Model prediction on dense points
        with torch.no_grad():
            y_pred_dense = model(x_dense_normalized).numpy()
        
        x_plot = x_dense_original.numpy()
        y_true_plot = y_true_dense.numpy()
        y_pred_plot = y_pred_dense
    else:
        x_plot = x_display
        y_true_plot = torch.zeros_like(x_t)
        for i, coeff in enumerate(coeffs_true):
            y_true_plot += coeff * (x_t ** i)
        y_true_plot = y_true_plot.numpy()
        y_pred_plot = y_pred
    
    # Plot 1: Data and fitted curve
    axes[0].scatter(x_display, y_np, alpha=0.5, label='Observed data', s=30, c='blue')
    axes[0].plot(x_plot, y_pred_plot, 'r-', linewidth=2.5, 
                 label=f'Fitted (order={model.order})')
    axes[0].plot(x_plot, y_true_plot, 'g--', linewidth=2, 
                 label=f'True function (order={data_poly_order})')
    
    axes[0].set_xlabel('x', fontsize=12)
    axes[0].set_ylabel('y', fontsize=12)
    
    # Title indicates bias-variance scenario
    if model.order < data_poly_order:
        scenario = "UNDERFITTING (High Bias)"
    elif model.order == data_poly_order:
        scenario = "GOOD FIT (Balanced)"
    else:
        scenario = "POTENTIAL OVERFITTING (High Variance)"
    
    axes[0].set_title(f'Polynomial Regression: {scenario}', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Loss curve
    loss_history = training_history['loss']
    axes[1].plot(loss_history, linewidth=2, color='darkblue')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Mean Squared Error', fontsize=12)
    axes[1].set_title('Training Loss Curve', fontsize=14)
    axes[1].set_yscale('log')  # Log scale to see convergence better
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('polynomial_regression_results.png', dpi=150, bbox_inches='tight')
    logger.info("Main results plot saved as 'polynomial_regression_results.png'")
    plt.show()


def main():
    """
    Main function to run the polynomial regression tutorial.
    
    This tutorial demonstrates the bias-variance tradeoff by allowing
    different polynomial orders for data generation and model fitting.
    
    **IMPORTANT:** For polynomial orders > 2, we use feature normalization
    to prevent numerical instability (NaN values).
    """
    logger.info("\n" + "="*70)
    logger.info(" POLYNOMIAL REGRESSION & BIAS-VARIANCE TRADEOFF TUTORIAL")
    logger.info("="*70)
    
    # ========================================================================
    # CONFIGURATION: Experiment with these parameters!
    # ========================================================================

    #### Setup for showing the value of:
    ####  1. Feature normalisation
    ####  2. Regularisation
    ####  3. Learning rate and epoch interplay
    #### TRUE DATA GENERATION
    data_poly_order = 3  # Order of the TRUE underlying function
    #coeffs_true = [1.0, 0.5, -0.3, 0.1]  # True coefficients [a₀, a₁, a₂, a₃]
    #coeffs_true = [1.0, 0.5, -0.3]  # True coefficients [a₀, a₁]

    #coeffs_true = [1.0, 0.5, -0.1, 0.01, -0.001]  # True coefficients [a₀, a₁]
    #coeffs_true = [1.0, 0.5]  # True coefficients [a₀, a₁]
    coeffs_true = [1.0, 0.5, -0.1, 0.01]  # True coefficients [a₀, a₁, a2, a3]
    
    # MODEL CONFIGURATION
    regressor_poly_order = 3  # Order of the MODEL to fit
    
    # TRAINING CONFIGURATION
    n_samples     = 100
    noise_std     = 0.5
    num_epochs    = 1000000       # 2000000       # 200000
    learning_rate = 0.0000025     # 0.000001      # 0.00005  


    #### Default
    ## TRUE DATA GENERATION
    #data_poly_order = 3  # Order of the TRUE underlying function
    #coeffs_true = [1.0, 0.5, -0.3, 0.1]  # True coefficients [a₀, a₁, a₂, a₃]
    #
    ## MODEL CONFIGURATION
    #regressor_poly_order = 3  # Order of the MODEL to fit
    #
    ## TRAINING CONFIGURATION
    #n_samples = 100
    #noise_std = 0.5
    #num_epochs = 2000
    #learning_rate = 0.01  # Works well with normalized features
    
    # FEATURE NORMALIZATION (critical for high-order polynomials!)
    use_normalization = max(data_poly_order, regressor_poly_order) > 3
    normalization_method = 'symmetric'  # Maps to [-1, 1], best for polynomials
    
    # ========================================================================
    # STEP 1: Generate synthetic data
    # ========================================================================
    logger.info("\n[STEP 1] Generating synthetic polynomial data...")
    logger.info(f"Data polynomial order: {data_poly_order}")
    logger.info(f"True coefficients: {coeffs_true}")
    logger.info(f"Noise std: {noise_std}")
    
    # Generate data in ORIGINAL scale
    x_t, y_t = generate_data(
        coeffs_true=coeffs_true,
        data_poly_order=data_poly_order,
        n_samples=n_samples,
        x_min=0,
        x_max=10,
        noise_std=noise_std
    )
    
    logger.info(f"Generated {n_samples} data points")
    logger.info(f"Data shape: x_t = {x_t.shape}, y_t = {y_t.shape}")
    logger.info(f"x range: [{x_t.min():.2f}, {x_t.max():.2f}]")
    
    # ========================================================================
    # STEP 1.5: Feature Normalization (if needed)
    # ========================================================================
    normalizer = None
    if use_normalization:
        logger.info("\n[STEP 1.5] Applying feature normalization...")
        logger.info(f"⚠️  Polynomial order {max(data_poly_order, regressor_poly_order)} "
                   "requires normalization to prevent NaN!")
        logger.info(f"Method: {normalization_method} (maps x to [-1, 1])")
        
        normalizer = FeatureNormalizer(method=normalization_method)
        x_t_original = x_t.clone()  # Keep original for reference
        x_t = normalizer.fit_transform(x_t)
        
        logger.info(f"Before normalization: x ∈ [{x_t_original.min():.2f}, {x_t_original.max():.2f}]")
        logger.info(f"After normalization:  x ∈ [{x_t.min():.2f}, {x_t.max():.2f}]")
        logger.info("✓ All polynomial features will now stay in bounded range!")
    else:
        logger.info("\n[STEP 1.5] Skipping normalization (polynomial order ≤ 2)")
    
    # ========================================================================
    # STEP 2: Initialize the model
    # ========================================================================
    logger.info("\n[STEP 2] Initializing the polynomial regression model...")
    logger.info(f"Regressor polynomial order: {regressor_poly_order}")
    
    model = PolynomialRegressor(order=regressor_poly_order)
    logger.info(f"Model initialized with {regressor_poly_order + 1} parameters")
    
    # Analyze bias-variance scenario
    if regressor_poly_order < data_poly_order:
        logger.info("⚠️  UNDERFITTING: Model order < Data order (High Bias)")
    elif regressor_poly_order == data_poly_order:
        logger.info("✓ GOOD FIT: Model order = Data order (Balanced)")
    else:
        logger.info("⚠️  OVERFITTING RISK: Model order > Data order (High Variance)")
    
    # ========================================================================
    # STEP 3: Train the model
    # ========================================================================
    logger.info("\n[STEP 3] Training the model...")
    logger.info(f"Training on {'NORMALIZED' if use_normalization else 'ORIGINAL'} features")
    
    trained_model, training_history = train_model(
        model=model,
        x_train=x_t,  # Will be normalized if use_normalization=True
        y_train=y_t,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        print_every=400,
        verbose=True
    )
    
    # ========================================================================
    # STEP 4: Evaluate and compare results
    # ========================================================================
    logger.info("\n[STEP 4] Evaluating results...")
    
    fitted_params = trained_model.get_parameters()
    
    if use_normalization:
        logger.info("\n⚠️  Note: Fitted coefficients are for NORMALIZED features")
        logger.info("They cannot be directly compared to true coefficients")
        logger.info("(True coeffs are for original scale, fitted are for normalized scale)")
    
    logger.info(f"\nTrue coefficients (original scale):   {coeffs_true}")
    logger.info(f"Fitted coefficients (working scale):  {[f'{p:.4f}' for p in fitted_params]}")
    
    # Compute final metrics
    metrics = evaluate_model(trained_model, x_t, y_t)
    logger.info(f"\nFinal MSE:  {metrics['mse']:.6f}")
    logger.info(f"Final RMSE: {metrics['rmse']:.6f}")
    
    # ========================================================================
    # STEP 5: Visualize results
    # ========================================================================
    logger.info("\n[STEP 5] Creating visualizations...")
    logger.info("Plots will display in ORIGINAL scale for interpretability")
    
    # Plot 1: Parameter evolution during training
    plot_parameter_evolution(training_history, coeffs_true, trained_model)
    
    # Plot 2: Fitted curve vs true function (denormalized for display)
    plot_results(x_t, y_t, trained_model, training_history, 
                 coeffs_true, data_poly_order, normalizer=normalizer)
    
    logger.info("\n" + "="*70)
    logger.info(" TUTORIAL COMPLETED SUCCESSFULLY!")
    logger.info("="*70)
    
    # ========================================================================
    # PEDAGOGICAL SUMMARY
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info(" BIAS-VARIANCE TRADEOFF SUMMARY")
    logger.info("="*70)
    logger.info(f"""
    Data Order:       {data_poly_order}
    Regressor Order:  {regressor_poly_order}
    
    INTERPRETATION:
    """)
    
    if regressor_poly_order < data_poly_order:
        logger.info("""
    🔴 UNDERFITTING (High Bias, Low Variance)
    - Model is too simple to capture the true function
    - Cannot fit the training data well
    - Both training and test error will be high
    - Bias dominates the error
    
    Solution: Increase model complexity (higher polynomial order)
    """)
    elif regressor_poly_order == data_poly_order:
        logger.info("""
    🟢 GOOD FIT (Balanced Bias-Variance)
    - Model complexity matches data complexity
    - Can fit the training data well
    - Generalizes well to unseen data
    - Optimal tradeoff point
    
    This is the ideal scenario!
    """)
    else:
        logger.info("""
    🟡 OVERFITTING RISK (Low Bias, High Variance)
    - Model is too complex for the data
    - Fits training data very well (even noise)
    - May generalize poorly to new data
    - Variance dominates the error
    
    Solution: Reduce model complexity or add regularization
    """)
    
    # ========================================================================
    # EXERCISES FOR STUDENTS
    # ========================================================================
    logger.info("="*70)
    logger.info(" EXERCISES FOR STUDENTS")
    logger.info("="*70)
    logger.info("""
    BIAS-VARIANCE EXPERIMENTS:
    
    1. UNDERFITTING SCENARIO:
       Set data_poly_order = 3, regressor_poly_order = 1
       - Observe high training loss
       - See fitted line cannot capture curvature
       - Notice parameter evolution converges but to wrong function
    
    2. PERFECT FIT SCENARIO:
       Set data_poly_order = 2, regressor_poly_order = 2
       - Observe low training loss
       - See fitted curve matches true function closely
       - Compare true vs fitted coefficients
    
    3. OVERFITTING SCENARIO:
       Set data_poly_order = 2, regressor_poly_order = 6
       - Observe very low training loss
       - See fitted curve wiggles to fit noise
       - Notice extra coefficients have small but non-zero values
    
    4. NOISE SENSITIVITY:
       Vary noise_std from 0.1 to 2.0 with regressor_poly_order > data_poly_order
       - How does noise affect overfitting?
       - At what noise level is overfitting most problematic?
    
    5. SAMPLE SIZE EFFECT:
       Change n_samples: 20, 50, 100, 500
       - How does more data affect the bias-variance tradeoff?
       - Does more data help with overfitting?
    
    6. LEARNING RATE EXPLORATION:
       Try learning_rate: 0.0001, 0.001, 0.01, 0.1
       - How does learning rate affect convergence speed?
       - Can you find an optimal learning rate?
    
    7. PARAMETER EVOLUTION ANALYSIS:
       - Which parameters converge fastest?
       - Do higher-order terms converge slower?
       - How does learning rate affect convergence patterns?
    """)
    logger.info("="*70 + "\n")


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run the tutorial
    main()
