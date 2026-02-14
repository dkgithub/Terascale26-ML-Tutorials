# Tutorial 1: Denoising Diffusion Probabilistic Models (DDPM)

Learn the foundations of diffusion models by implementing DDPM from scratch.

## Overview

This tutorial provides a hands-on introduction to **Denoising Diffusion Probabilistic Models (DDPM)**, one of the most influential approaches in modern generative modeling.

**Duration:** 90-120 minutes

**Difficulty:** Intermediate

---

## Learning Objectives

By the end of this tutorial, you will:

- Understand the forward diffusion process (adding noise)
- Implement the reverse diffusion process (denoising)
- Train a neural network to predict noise
- Sample new data using iterative denoising
- Visualize how diffusion models work

---

## Prerequisites

**Required Knowledge:**

- Python programming
- Basic PyTorch (tensors, gradients, training loops)
- Neural networks (MLPs, backpropagation)
- Probability basics (Gaussian distributions)

**Required Setup:**

- Complete [Installation Guide](../installation.md)
- Complete [Getting Started Guide](../getting-started.md)

---

## Tutorial Structure

The tutorial consists of:

1. **Jupyter Notebook** - Interactive, step-by-step learning
2. **Python Scripts** - Modular, reusable code
3. **Visualizations** - Plots and animations of the diffusion process

---

## How to Run This Tutorial

You have **two options** for running this tutorial:

### Option 1: Jupyter Notebook (Recommended for Learning)

**Best for:** Interactive exploration, learning, experimentation

**Steps:**

1. Navigate to the tutorial directory:
   ```bash
   cd tutorial_1_ddpm
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

4. Open `tutorial_notebook_DDPM.ipynb`

5. Select the "Tutorial Environment" kernel

6. Run the cells sequentially

**Alternative: VSCode**

1. Open VSCode:
   ```bash
   code tutorial_1_ddpm
   ```

2. Open `tutorial_notebook_DDPM.ipynb`

3. Select the "Tutorial Environment" kernel

4. Run cells using the play button or `Shift+Enter`

---

### Option 2: Terminal/CLI (For Automation)

**Best for:** Running complete experiments, batch processing

**Steps:**

1. Navigate to the tutorial directory:
   ```bash
   cd tutorial_1_ddpm
   ```

2. Activate your virtual environment:
   ```bash
   source .venv/bin/activate  # Linux/macOS
   .venv\Scripts\Activate.ps1  # Windows
   ```

3. Run the main script:
   ```bash
   python -m ddpm_tutorial.main
   ```

   **OR:**
   
   ```bash
   python ddpm_tutorial/main.py
   ```

4. Check outputs in the `outputs/` directory

**Customize Configuration:**

Edit `ddpm_tutorial/main.py` to change:
- Dataset type (`moons`, `circles`, `swiss_roll`, etc.)
- Number of diffusion steps
- Training epochs
- Model architecture
- Sampling parameters

---

## What's Inside

### Python Package: `ddpm_tutorial/`

```
ddpm_tutorial/
├── __init__.py          # Package initialization
├── main.py              # Entry point, configuration
├── diffusion.py         # DDPM forward/reverse processes
├── models.py            # Neural network architectures
├── utils.py             # Data loading, helpers
└── visualization.py     # Plotting functions
```

**Key Files:**

- **`diffusion.py`**: Core DDPM implementation
  - Forward process (noise schedule)
  - Reverse process (sampling)
  - Loss computation

- **`models.py`**: Neural network models
  - Simple MLP denoiser
  - Time embedding
  - Sinusoidal positional encoding

- **`main.py`**: Complete training pipeline
  - Data loading
  - Model training
  - Sample generation
  - Visualization

---

## Notebook Structure

The Jupyter notebook is organized into sections:

### Part 1: Introduction
- Overview of diffusion models
- Mathematical framework
- Key concepts

### Part 2: Forward Diffusion Process
- Noise schedule (β values)
- Adding noise at each timestep
- Visualization of noising process

### Part 3: Reverse Diffusion Process
- Training objective
- Noise prediction network
- Sampling procedure

### Part 4: Implementation
- Building the model
- Training loop
- Loss computation

### Part 5: Experiments
- Training on 2D datasets
- Generating samples
- Comparing different configurations

### Part 6: Analysis
- Visualizing training progress
- Quality metrics
- Understanding what the model learned

---

## Expected Outputs

When you run the tutorial, you'll generate:

**In `outputs/` directory:**

- `training_curve.png` - Loss over time
- `forward_process.png` - How noise is added
- `reverse_process.png` - How samples are generated
- `sampling_animation.gif` - Animated denoising process
- `noise_schedule.png` - Beta and alpha values
- `real_vs_generated.png` - Quality comparison
- `marginal_distributions.png` - Statistical analysis

**Example visualizations you'll create:**

![Forward Process](https://via.placeholder.com/600x200?text=Forward+Diffusion+Process)
*How data is gradually corrupted with noise*

![Reverse Process](https://via.placeholder.com/600x200?text=Reverse+Diffusion+Process)
*How the model recovers data from noise*

---

## Key Concepts Covered

### 1. Forward Diffusion Process

The forward process gradually adds Gaussian noise to data:

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$$

**You'll learn:**
- How to design noise schedules
- Computing $\bar{\alpha}_t$ coefficients
- Sampling noisy data at any timestep

### 2. Reverse Diffusion Process

The reverse process learns to remove noise:

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

**You'll learn:**
- Training a denoiser network
- Predicting noise vs predicting mean
- Sampling new data

### 3. Training Objective

The model is trained to predict added noise:

$$\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon} \left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

**You'll implement:**
- Efficient training loss
- Time conditioning
- Gradient-based optimization

---

## Experiments to Try

After completing the tutorial, try these experiments:

### Easy (15-30 minutes each)

1. **Change the dataset:**
   ```python
   config["dataset_type"] = "circles"  # Try different shapes
   ```

2. **Modify noise schedule:**
   ```python
   config["beta_start"] = 0.0001
   config["beta_end"] = 0.02
   ```

3. **Adjust training:**
   ```python
   config["n_epochs"] = 200  # Train longer
   config["batch_size"] = 128  # Change batch size
   ```

### Medium (30-60 minutes each)

4. **Compare schedulers:**
   - Linear vs cosine noise schedules
   - Plot and compare sample quality

5. **Ablation study:**
   - Remove time embedding - what happens?
   - Use fewer/more diffusion steps

6. **Architecture experiments:**
   - Change hidden dimensions
   - Add/remove layers
   - Try different activation functions

### Advanced (1-2 hours each)

7. **Implement DDIM sampling:**
   - Faster sampling with fewer steps
   - Compare speed vs quality

8. **Conditional generation:**
   - Add class conditioning
   - Generate specific data types

9. **Scale to images:**
   - Use MNIST or CIFAR-10
   - Adapt architecture for images

---

## Common Issues & Solutions

### Training loss not decreasing

**Problem:** Loss stays high or increases

**Solutions:**
- Reduce learning rate
- Check noise schedule (beta values shouldn't be too large)
- Verify model architecture
- Ensure data is normalized

### Generated samples look like noise

**Problem:** Samples don't resemble training data

**Solutions:**
- Train longer (more epochs)
- Use more diffusion steps during sampling
- Check if model is learning (plot training loss)
- Verify noise schedule parameters

### Out of memory errors

**Problem:** CUDA/CPU out of memory

**Solutions:**
- Reduce batch size
- Use CPU instead of GPU
- Simplify model architecture
- Use fewer samples

---

## Theoretical Background

For deeper understanding, refer to these papers:

**Core Papers:**

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (Ho et al., 2020)
- [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) (Nichol & Dhariwal, 2021)

**Related Work:**

- [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) (Sohl-Dickstein et al., 2015)
- [Generative Modeling by Estimating Gradients](https://arxiv.org/abs/1907.05600) (Song & Ermon, 2019)

---

## Next Steps

After completing Tutorial 1:

- **Tutorial 2:** [Flow Matching](tutorial-2.md) - Learn ODE-based alternatives to SDEs
- **Tutorial 3:** [Advanced Topics](tutorial-3.md) - Conditional generation, guidance
- Experiment with your own datasets
- Read the original DDPM paper
- Explore score-based models

---

## Quick Reference

**Start Jupyter:**
```bash
cd tutorial_1_ddpm
source .venv/bin/activate
jupyter lab
```

**Run CLI:**
```bash
cd tutorial_1_ddpm
source .venv/bin/activate
python -m ddpm_tutorial.main
```

**View outputs:**
```bash
ls outputs/
```

**Common imports:**
```python
from ddpm_tutorial.diffusion import DDPM
from ddpm_tutorial.models import SimpleMLPDenoiser
from ddpm_tutorial.utils import create_toy_dataset
```

---

## Need Help?

- Check [FAQ](../faq.md)
- See [Troubleshooting](../troubleshooting.md)
- Open an issue on [GitHub](https://github.com/yourusername/generative-tutorials/issues)

---

!!! success "Ready to Start"
    Head to `tutorial_1_ddpm/tutorial_notebook_DDPM.ipynb` and begin learning!
