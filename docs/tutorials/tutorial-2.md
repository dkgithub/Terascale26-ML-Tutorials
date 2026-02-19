# Tutorial 2: From Perceptrons to Deep Neural Networks

Master the fundamentals of deep learning by understanding vanishing gradients, activation functions, and regularization techniques.

## Overview

This tutorial provides a comprehensive introduction to **Deep Neural Networks (DNNs)**, focusing on the critical challenges that arise when scaling from simple perceptrons to deep architectures. You'll learn why certain design choices matter and how to train stable, generalizable models.

**Duration:** 120-150 minutes

**Difficulty:** Intermediate

---

## Learning Objectives

By the end of this tutorial, you will:

- Understand the **vanishing gradient problem** and its impact on deep networks
- Compare activation functions (sigmoid, tanh, ReLU) and their effect on gradient flow
- Detect **overfitting** using train/validation/test splits
- Implement **regularization techniques** (L1, L2, Dropout) to prevent overfitting
- Visualize gradient distributions and training dynamics
- Build stable deep neural networks for regression tasks

---

## Prerequisites

**Required Knowledge:**

- Python programming
- Basic PyTorch (tensors, gradients, training loops)
- Neural networks (forward pass, backpropagation)
- Linear algebra (matrix multiplication, vectors)
- Calculus basics (derivatives, chain rule)

**Required Setup:**

- Complete [Getting Started Guide](../getting-started.md)
- Recommended: Complete [Tutorial 1: Polynomial Fitting](tutorial-1.md)

---

## Tutorial Structure

The tutorial consists of:

1. **Jupyter Notebook** - Interactive, step-by-step learning with visualizations
2. **Python Module** - Modular, reusable code (`perceptron_to_DNN_tutorial/`)
3. **Visualizations** - Gradient flow plots, training curves, and distribution analysis

---

## How to Run This Tutorial

You have **two options** for running this tutorial:

### Option 1: Jupyter Notebook (Recommended for Learning)

**Best for:** Interactive exploration, learning, experimentation

**Steps:**

1. Navigate to the tutorial directory:
   ```bash
   cd tutorial_2_perceptron-to-DNN
   ```

2. Activate your virtual environment:
   ```bash
   source .venv/bin/activate  # Linux/macOS
   .venv\Scripts\Activate.ps1  # Windows
   ```

3. Start Jupyter:
   ```bash
   jupyter lab
   # OR
   jupyter notebook
   ```

4. Open `tutorial_DNN.ipynb`

5. Select the "Tutorial Environment" kernel

6. Run the cells sequentially, reading the explanations in markdown cells

**Alternative: VSCode**

1. Open VSCode:
   ```bash
   code tutorial_2_perceptron-to-DNN
   ```

2. Open `tutorial_DNN.ipynb`

3. Select the "Tutorial Environment" kernel (or Python interpreter from `.venv`)

4. Run cells using the play button or `Shift+Enter`

5. View plots inline in the notebook

---

### Option 2: Terminal/CLI (For Automation)

**Best for:** Running complete experiments, batch processing, reproducibility

**Steps:**

1. Navigate to the tutorial directory:
   ```bash
   cd tutorial_2_perceptron-to-DNN
   ```

2. Activate your virtual environment:
   ```bash
   source .venv/bin/activate  # Linux/macOS
   .venv\Scripts\Activate.ps1  # Windows
   ```

3. Run the main script:
   ```bash
   python -m perceptron_to_DNN_tutorial.main
   ```

   **OR:**
   
   ```bash
   python perceptron_to_DNN_tutorial/main.py
   ```

4. Check outputs in the current directory (PNG files will be generated)

**Customize Configuration:**

Edit `perceptron_to_DNN_tutorial/main.py` to change:
- Network architecture (depth, width)
- Activation functions tested
- Number of training epochs
- Regularization parameters
- Dataset properties (polynomial order, noise level, sample count)

---

## What's Inside

### Python Package: `perceptron_to_DNN_tutorial/`

```
perceptron_to_DNN_tutorial/
├── __init__.py              # Package initialization
├── main.py                  # Entry point, full tutorial pipeline
├── MultiLayerPerceptron.py  # MLP implementation with multiple activations
├── train.py                 # Training functions with gradient tracking
├── loss.py                  # Loss functions and regularization
├── plotting.py              # Visualization utilities
├── utils.py                 # Data generation and normalization
└── logger.py                # Logging configuration
```

**Key Files:**

- **`MultiLayerPerceptron.py`**: Flexible MLP implementation
  - Support for sigmoid, tanh, ReLU activations
  - Configurable depth and width
  - Dropout support
  
- **`train.py`**: Advanced training functions
  - Gradient tracking for vanishing gradient analysis
  - Per-sample gradient distributions
  - Train/validation/test split support
  - Regularization (L1, L2, Elastic Net)

- **`plotting.py`**: Professional visualizations
  - Gradient flow over epochs
  - Signed gradient distributions
  - Train vs validation loss curves
  - Overfitting detection

- **`main.py`**: Complete tutorial workflow
  - Part 1: Vanishing gradient problem
  - Part 2: Overfitting detection
  - Part 3: Regularization techniques

---

## Notebook Structure

The Jupyter notebook is organized into three main parts:

### Part 1: The Vanishing Gradient Problem (40-50 minutes)

**What you'll learn:**
- Why deep networks fail with sigmoid/tanh activations
- How gradients shrink exponentially through layers
- Why ReLU activation solves the vanishing gradient problem

**Experiments:**
- Train 9-layer networks with different activations
- Visualize gradient flow through layers
- Analyze gradient magnitude distributions
- Compare final gradient ratios (first layer / last layer)

**Key visualizations:**
- Gradient flow over epochs (log scale)
- Signed gradient distributions at different training stages
- Layer-by-layer gradient comparison

### Part 2: Detecting Overfitting (30-40 minutes)

**What you'll learn:**
- Train/validation/test split methodology
- How to detect overfitting from loss curves
- The gap between training and validation loss

**Experiments:**
- Generate separate train/validation/test datasets
- Monitor both training and validation loss
- Identify when models memorize vs generalize
- Analyze the train-validation gap

**Key visualizations:**
- Train vs validation loss curves
- Overfitting gap over epochs
- Final model predictions vs ground truth

### Part 3: Regularization Techniques (40-50 minutes)

**What you'll learn:**
- L1 regularization (promotes sparsity)
- L2 regularization (weight decay)
- Dropout (random neuron deactivation)
- How to choose regularization strength

**Experiments:**
- Compare no regularization vs L2 vs dropout
- Observe reduced overfitting with regularization
- Analyze regularization penalty over time
- Compare final generalization performance

**Key visualizations:**
- Regularization comparison (4-panel plot)
- Train vs validation with/without regularization
- Overfitting gap reduction
- Regularization penalty curves

---

## Expected Outputs

When you run the tutorial, you'll generate:

**Visualization Files:**

- `vanishing_gradient_demo.png` - Gradient flow for different activations
- `gradient_distributions_epoch_first.png` - Gradient distributions at epoch 1
- `gradient_distributions_epoch_middle.png` - Gradient distributions at epoch 5000
- `gradient_distributions_epoch_last.png` - Gradient distributions at epoch 10000
- `deep_mlp_with_relu_results.png` - ReLU model training results
- `regularization_comparison.png` - Comparison of regularization techniques
- `deep_mlp_with_dropout_(p=0.3)_results.png` - Dropout model results

**Example Visualizations:**

### Gradient Flow Visualization

Shows how gradients flow through layers over time for different activations:
- **Sigmoid**: Exponential decay (vanishing!)
- **Tanh**: Moderate decay
- **ReLU**: Stable gradients throughout

### Gradient Distributions

Histograms showing the distribution of signed gradient values:
- **First epoch**: All activations show healthy gradients
- **Last epoch**: Sigmoid collapsed near zero, ReLU still healthy

### Regularization Effects

4-panel comparison showing:
1. Train vs validation loss curves
2. Overfitting gap over time
3. Regularization penalty
4. Final performance summary

---

## Key Concepts Covered

### 1. The Vanishing Gradient Problem

**The Problem:**

In deep networks with sigmoid/tanh activations, gradients shrink exponentially:

$$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial h_9} \cdot \frac{\partial h_9}{\partial h_8} \cdot \ldots \cdot \frac{\partial h_2}{\partial h_1} \cdot \frac{\partial h_1}{\partial W_1}$$

Each term can be < 1, causing the product to vanish.

**You'll learn:**
- Why sigmoid'(x) ∈ (0, 0.25] causes problems
- Why tanh'(x) ∈ (0, 1] is better but not perfect
- Why ReLU'(x) = 1 (for x > 0) solves vanishing gradients

**Hands-on:**
- Measure gradient ratios across layers
- Visualize gradient collapse in real-time
- Compare activation functions empirically

### 2. Overfitting and Generalization

**The Problem:**

Models can memorize training data without learning the underlying pattern:

```
Training Loss ↓↓↓ (keeps decreasing)
Validation Loss ↑ (starts increasing!)
```

**You'll learn:**
- Train/validation/test split best practices
- How to detect overfitting from loss curves
- The train-validation gap as an overfitting metric

**Hands-on:**
- Split data with different random seeds
- Track both train and validation loss
- Diagnose overfitting automatically

### 3. Regularization Techniques

**L1 Regularization (Lasso):**

$$\mathcal{L} = \mathcal{L}_{\text{data}} + \lambda_1 \sum_i |w_i|$$

Promotes sparse weights (many weights → 0)

**L2 Regularization (Ridge):**

$$\mathcal{L} = \mathcal{L}_{\text{data}} + \lambda_2 \sum_i w_i^2$$

Keeps weights small, prevents large values

**Dropout:**

Randomly deactivate neurons during training:
- Training: Each neuron has probability p of being dropped
- Testing: Use all neurons (scaled by 1-p)

**You'll implement:**
- Regularization penalties in loss computation
- Dropout layers in neural networks
- Hyperparameter tuning (λ values, dropout rate)

---

## Experiments to Try

After completing the tutorial, try these experiments:

### Easy (15-30 minutes each)

1. **Change network depth:**
   ```python
   architecture = [1, 128, 128, 128, 128, 128, 1]  # 5 layers instead of 9
   ```
   Does the vanishing gradient problem improve or worsen?

2. **Modify polynomial complexity:**
   ```python
   data_poly_order = 5  # Simpler polynomial
   noise_std = 1.0      # Less noise
   ```
   How does overfitting change?

3. **Adjust regularization strength:**
   ```python
   lambda_l2 = 0.1   # Stronger regularization
   dropout_rate = 0.5  # Higher dropout
   ```
   Observe the effect on train/validation gap

### Medium (30-60 minutes each)

4. **Compare all activation functions:**
   - Add 'leaky_relu' and 'elu' to the comparison
   - Measure gradient flow for each
   - Plot all on the same figure

5. **Regularization ablation study:**
   - Test L1, L2, and Elastic Net (L1 + L2)
   - Compare sparsity of learned weights
   - Plot weight distributions

6. **Learning rate experiments:**
   - Try learning rates: [0.001, 0.01, 0.1]
   - Observe training stability
   - Find the optimal value

### Advanced (1-2 hours each)

7. **Implement batch normalization:**
   - Add batch norm layers
   - Compare with/without batch norm
   - Analyze gradient flow improvements

8. **Early stopping:**
   - Implement validation-based early stopping
   - Save best model based on validation loss
   - Prevent overfitting automatically

9. **Cross-validation:**
   - Implement k-fold cross-validation
   - Average performance across folds
   - Get more robust performance estimates

---

## Common Issues & Solutions

### Vanishing gradients still occurring with ReLU

**Problem:** Gradients still very small even with ReLU

**Solutions:**
- Check for "dying ReLU" (all negative inputs → zero gradients)
- Try Leaky ReLU or ELU instead
- Reduce network depth
- Use better weight initialization (He initialization)

### Training loss not decreasing

**Problem:** Loss stays high or fluctuates

**Solutions:**
- Reduce learning rate (try 0.001 instead of 0.01)
- Check data normalization (features should be normalized)
- Increase number of epochs
- Verify model architecture (no bugs in forward pass)

### Severe overfitting despite regularization

**Problem:** Large train-validation gap even with regularization

**Solutions:**
- Increase regularization strength (larger λ)
- Increase dropout rate (try 0.5)
- Reduce model capacity (fewer/smaller layers)
- Get more training data
- Use early stopping

### Gradient distribution plots show unexpected patterns

**Problem:** Distributions don't match expected behavior

**Solutions:**
- Check if per-sample gradient tracking is enabled
- Verify epochs being tracked (first, middle, last)
- Ensure model is training (check loss decreases)
- Increase number of training samples for better statistics

---

## Theoretical Background

For deeper understanding, refer to these resources:

**Core Concepts:**

- **Vanishing Gradients**: [Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a.html) (Glorot & Bengio, 2010)
- **ReLU Activation**: [Deep Sparse Rectifier Neural Networks](http://proceedings.mlr.press/v15/glorot11a.html) (Glorot et al., 2011)
- **Dropout**: [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://jmlr.org/papers/v15/srivastava14a.html) (Srivastava et al., 2014)
- **Regularization**: [Deep Learning Book - Chapter 7: Regularization](https://www.deeplearningbook.org/contents/regularization.html) (Goodfellow et al., 2016)

**Additional Reading:**

- Batch Normalization: [Batch Normalization: Accelerating Deep Network Training](https://arxiv.org/abs/1502.03167)
- Weight Initialization: [Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852)
- Optimization: [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)

---

## Next Steps

After completing Tutorial 2:

- **Tutorial 3:** [From DNNs to Transformers](tutorial-3.md) - Learn attention mechanisms and modern architectures
- **Tutorial 4:** [Denoising Diffusion Probabilistic Models](tutorial-4.md) - Explore generative models
- Experiment with real datasets (MNIST, CIFAR-10)
- Implement other activation functions (Leaky ReLU, ELU, GELU)
- Study advanced regularization (spectral normalization, weight decay schedules)

---

## Quick Reference

**Start Jupyter:**
```bash
cd tutorial_2_perceptron-to-DNN
source .venv/bin/activate  # or .venv\Scripts\Activate.ps1 on Windows
jupyter lab
```

**Run CLI:**
```bash
cd tutorial_2_perceptron-to-DNN
source .venv/bin/activate
python -m perceptron_to_DNN_tutorial.main
```

**View outputs:**
```bash
ls *.png  # All visualization files
```

**Common imports:**
```python
from perceptron_to_DNN_tutorial.MultiLayerPerceptron import MultiLayerPerceptron
from perceptron_to_DNN_tutorial.train import train_model_with_gradient_tracking
from perceptron_to_DNN_tutorial.utils import create_toy_dataset, FeatureNormalizer
from perceptron_to_DNN_tutorial.plotting import plot_gradient_flow, plot_layer_gradient_norms
```

**Key configuration parameters:**
```python
# Network architecture
architecture_deep = [1, 128, 128, 128, 128, 128, 128, 128, 128, 128, 1]

# Data generation
data_poly_order = 9
n_train_samples = 200
noise_std = 2.5

# Training
num_epochs = 10000
learning_rate = 0.005

# Regularization
lambda_l2 = 0.01
dropout_rate = 0.3
```

---

## Need Help?

- Check [FAQ](../faq.md)
- See [Troubleshooting](../troubleshooting.md)
- Review [Getting Started Guide](../getting-started.md) for environment setup
- Open an issue on [GitHub](https://github.com/your-repo/issues)

---

!!! success "Ready to Start"
    Head to `tutorial_2_perceptron-to-DNN/tutorial_DNN.ipynb` and begin mastering deep neural networks!
