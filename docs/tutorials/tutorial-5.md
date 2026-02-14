# Tutorial 2: Flow Matching

Learn flow matching as a deterministic alternative to stochastic diffusion models.

## Overview

This tutorial provides a hands-on introduction to **Flow Matching**, an elegant ODE-based approach to generative modeling that offers an alternative to stochastic diffusion processes.

**Duration:** 90-120 minutes

**Difficulty:** Intermediate

---

## Learning Objectives

By the end of this tutorial, you will:

- Understand probability paths (linear vs variance preserving)
- Implement velocity field learning
- Master ODE-based sampling with Euler and RK45 solvers
- Compare flow matching with DDPM
- Visualize data flowing through probability space

---

## Prerequisites

**Required Knowledge:**

- Python programming
- Basic PyTorch (tensors, gradients, training loops)
- Neural networks (MLPs, backpropagation)
- Ordinary Differential Equations (basic understanding)

**Recommended:**

- Complete [Tutorial 1: DDPM](tutorial-1.md) first
- Understanding of probability distributions

**Required Setup:**

- Complete [Installation Guide](../installation.md)
- Complete [Getting Started Guide](../getting-started.md)

---

## Tutorial Structure

The tutorial consists of:

1. **Jupyter Notebook** - Interactive, step-by-step learning with mathematical explanations
2. **Python Scripts** - Modular, reusable code
3. **Visualizations** - Plots and animations showing flow processes

---

## How to Run This Tutorial

You have **two options** for running this tutorial:

### Option 1: Jupyter Notebook (Recommended for Learning)

**Best for:** Interactive exploration, understanding concepts, experimentation

**Steps:**

1. Navigate to the tutorial directory:
   ```bash
   cd tutorial_2_flow_matching
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

4. Open `tutorial_notebook_Flow.ipynb`

5. Select the "Tutorial Environment" kernel

6. Run the cells sequentially

**Alternative: VSCode**

1. Open VSCode:
   ```bash
   code tutorial_2_flow_matching
   ```

2. Open `tutorial_notebook_Flow.ipynb`

3. Select the "Tutorial Environment" kernel

4. Run cells using the play button or `Shift+Enter`

---

### Option 2: Terminal/CLI (For Automation)

**Best for:** Running complete experiments, batch processing, automation

**Steps:**

1. Navigate to the tutorial directory:
   ```bash
   cd tutorial_2_flow_matching
   ```

2. Activate your virtual environment:
   ```bash
   source .venv/bin/activate  # Linux/macOS
   .venv\Scripts\Activate.ps1  # Windows
   ```

3. Run the main script:
   ```bash
   python -m flow_matching_tutorial.main
   ```

   **OR:**
   
   ```bash
   python flow_matching_tutorial/main.py
   ```

4. Check outputs in the `outputs/` directory

**Customize Configuration:**

Edit `flow_matching_tutorial/main.py` to change:
- Dataset type (`moons`, `circles`, `swiss_roll`, etc.)
- Probability path type (`linear`, `variance_preserving`)
- Number of training epochs
- ODE solver (`euler`, `rk45`)
- Number of sampling steps
- Model architecture

---

## What's Inside

### Python Package: `flow_matching_tutorial/`

```
flow_matching_tutorial/
├── __init__.py          # Package initialization
├── main.py              # Entry point, configuration
├── flow.py              # Flow matching implementation (with TODOs)
├── flow_solutions.py    # Complete solutions
├── models.py            # Neural network architectures
├── utils.py             # Data loading, helpers
└── visualization.py     # Plotting functions
```

**Key Files:**

- **`flow.py`**: Core flow matching implementation
  - Probability paths (linear, variance preserving)
  - Velocity field computation
  - Flow matching loss
  - ODE solvers (Euler, RK45)
  - **Contains 3 TODOs for you to implement!**

- **`flow_solutions.py`**: Complete reference implementation
  - Use this to check your work
  - Contains fully working code

- **`models.py`**: Neural network models
  - Same architecture as Tutorial 1
  - Predicts **velocity** instead of **noise**

- **`main.py`**: Complete training pipeline
  - Configuration
  - Training loop
  - Sample generation
  - Comprehensive visualizations

---

## Notebook Structure

The Jupyter notebook is organized into sections:

### Part 1: Introduction
- Overview: SDE vs ODE
- Mathematical framework
- Key differences from DDPM

### Part 2: Probability Paths
- Linear interpolation
- Variance preserving paths
- Mathematical derivations
- Visual comparison

### Part 3: Velocity Fields
- Computing target velocities
- Training objective
- Flow matching loss

### Part 4: Implementation
- TODO #1: Probability paths
- TODO #2: Velocity fields  
- TODO #3: Flow matching loss
- Building and training the model

### Part 5: ODE Solvers
- Euler method (simple, fixed step)
- RK45 (adaptive, sophisticated)
- Comparison and trade-offs

### Part 6: Visualization
- Forward flow process (data → noise)
- Reverse flow process (noise → data)
- Velocity field visualization
- Animated sampling

### Part 7: Experiments
- Training on 2D datasets
- Comparing probability paths
- Evaluating sample quality

---

## Expected Outputs

When you run the tutorial, you'll generate:

**In `outputs/` directory:**

- `probability_paths.png` - Comparison of linear vs variance preserving
- `forward_flow_process.png` - Data flowing to noise at multiple timesteps
- `training_curve.png` - Loss over time
- `velocity_field.png` - Learned velocity field visualization
- `reverse_process_steps.png` - Stepwise denoising visualization
- `flow_sampling.gif` - Animated sampling process
- `ode_solver_comparison.png` - Euler vs RK45 comparison
- `real_vs_generated.png` - Quality comparison
- `marginal_distributions.png` - Statistical analysis

**Example visualizations you'll create:**

![Probability Paths](https://via.placeholder.com/600x200?text=Linear+vs+Variance+Preserving+Paths)
*Comparison of different probability paths*

![Flow Process](https://via.placeholder.com/600x200?text=Forward+and+Reverse+Flow)
*How data flows through probability space*

---

## Key Concepts Covered

### 1. Probability Paths

Different ways to interpolate between noise $x_0$ and data $x_1$:

**Linear Path:**
$$x_t = (1-t)x_0 + tx_1$$

- Straight line in Euclidean space
- Constant velocity
- Simple to compute

**Variance Preserving Path:**
$$x_t = \cos\left(\frac{\pi t}{2}\right)x_0 + \sin\left(\frac{\pi t}{2}\right)x_1$$

- Curved geodesic
- Preserves norm
- Often better sample quality

### 2. Velocity Fields

The time derivative of the probability path:

$$u_t = \frac{d}{dt}x_t$$

**Linear:** $u_t = x_1 - x_0$ (constant)

**Variance Preserving:** $u_t = -\frac{\pi}{2}\sin\left(\frac{\pi t}{2}\right)x_0 + \frac{\pi}{2}\cos\left(\frac{\pi t}{2}\right)x_1$ (time-varying)

### 3. Flow Matching Loss

Train a model to predict velocity at each point:

$$\mathcal{L} = \mathbb{E}_{t, x_0, x_1} \left[\|v_\theta(x_t, t) - u_t\|^2\right]$$

**Key difference from DDPM:**
- DDPM: Predict **noise** $\epsilon$
- Flow Matching: Predict **velocity** $v$

### 4. ODE Sampling

Sample by solving the ODE:

$$\frac{dx}{dt} = v_\theta(x, t), \quad x(0) \sim \mathcal{N}(0, I)$$

**Euler Method:**
$$x_{t+\Delta t} = x_t + v_\theta(x_t, t) \cdot \Delta t$$

**RK45:** Adaptive Runge-Kutta with automatic step size

---

## Experiments to Try

After completing the tutorial, try these experiments:

### Easy (15-30 minutes each)

1. **Change probability path:**
   ```python
   config["path_type"] = "linear"  # vs "variance_preserving"
   ```
   Compare training curves and sample quality

2. **Try different datasets:**
   ```python
   config["dataset_type"] = "swiss_roll"
   ```
   Which path type works better?

3. **Vary ODE steps:**
   ```python
   config["n_euler_steps"] = 50  # vs 100, 200
   ```
   How does this affect quality and speed?

4. **Visualize at different times:**
   ```python
   config["forward_timesteps"] = [0.0, 0.1, 0.5, 0.9, 1.0]
   ```

### Medium (30-60 minutes each)

5. **Compare path types:**
   - Train both linear and variance preserving
   - Compare convergence speed
   - Analyze final sample quality

6. **ODE solver comparison:**
   - Time Euler vs RK45
   - Compare number of function evaluations
   - Measure quality differences

7. **Architecture experiments:**
   - Change model size
   - Try different activation functions
   - Experiment with time embeddings

### Advanced (1-2 hours each)

8. **Design custom probability path:**
   - Implement your own path in `flow.py`
   - Try polynomial, exponential, or other interpolations
   - Compare with existing paths

9. **Implement different ODE solver:**
   - Add RK4 or other methods
   - Compare accuracy vs speed
   - Analyze error accumulation

10. **Extend to higher dimensions:**
    - Try 10D or 20D data
    - Compare path types in high dimensions
    - Analyze computational scaling

---

## Common Issues & Solutions

### Paths look identical in visualization

**Problem:** Linear and variance preserving paths look the same

**Solution:** This was a known bug (now fixed). The visualization uses asymmetric test points to show differences clearly.

### TypeError: dtype mismatch

**Problem:** `mat1 and mat2 must have the same dtype`

**Solution:** Fixed in latest version. Ensure you have `dtype=torch.float32` in tensor creation.

### AttributeError with numpy arrays

**Problem:** `'numpy.ndarray' object has no attribute 'cpu'`

**Solution:** Fixed in latest version. Visualization functions now handle both Tensors and numpy arrays.

### Training loss not decreasing

**Problem:** Loss stays high

**Solutions:**
- Check your TODO implementations
- Verify velocity field computation
- Ensure path computation is correct
- Compare with `flow_solutions.py`

---

## Comparison with Tutorial 1

| Aspect | DDPM (Tutorial 1) | Flow Matching (Tutorial 2) |
|--------|-------------------|----------------------------|
| **Process** | SDE (stochastic) | ODE (deterministic) |
| **Forward** | Add noise with schedule | Follow probability path |
| **Target** | Predict noise $\epsilon$ | Predict velocity $v$ |
| **Sampling** | Iterative denoising | ODE integration |
| **Speed** | Fixed steps | Can use adaptive solvers |
| **Model** | SimpleMLPDenoiser | **Same architecture!** |

---

## Theoretical Background

For deeper understanding, refer to these papers:

**Core Papers:**

- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) (Lipman et al., 2023)
- [Flow Straight and Fast](https://arxiv.org/abs/2209.03003) (Liu et al., 2022)

**Related Work:**

- [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366) (Chen et al., 2018)
- [FFJORD](https://arxiv.org/abs/1810.01367) (Grathwohl et al., 2018)
- [Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456) (Song et al., 2020)

---

## Next Steps

After completing Tutorial 2:

- **Tutorial 3:** [Advanced Topics](tutorial-3.md) - Coming Soon
- Compare your experience with Tutorial 1
- Read the Flow Matching paper
- Explore continuous normalizing flows
- Experiment with real image datasets

---

## Quick Reference

**Start Jupyter:**
```bash
cd tutorial_2_flow_matching
source .venv/bin/activate
jupyter lab
```

**Run CLI:**
```bash
cd tutorial_2_flow_matching
source .venv/bin/activate
python -m flow_matching_tutorial.main
```

**View outputs:**
```bash
ls outputs/
```

**Common imports:**
```python
from flow_matching_tutorial.flow import ConditionalFlowMatching
from flow_matching_tutorial.models import SimpleMLPDenoiser
from flow_matching_tutorial.utils import create_toy_dataset
```

**Check your solutions:**
```python
from flow_matching_tutorial.flow_solutions import ConditionalFlowMatching as SolutionFlow
```

---

## TODOs in the Tutorial

The tutorial includes 3 TODOs for you to implement:

**TODO #1:** Probability paths
- Implement `sample_probability_path()`
- Linear and variance preserving interpolation

**TODO #2:** Velocity fields
- Implement `compute_conditional_velocity()`
- Time derivatives of paths

**TODO #3:** Flow matching loss
- Implement `training_loss()`
- MSE between predicted and target velocity

Use `flow_solutions.py` to check your work!

---

## Need Help?

- Check [FAQ](../faq.md)
- See [Troubleshooting](../troubleshooting.md)
- Review `flow_solutions.py` for reference
- Open an issue on [GitHub](https://github.com/yourusername/generative-tutorials/issues)

---

!!! success "Ready to Start"
    Head to `tutorial_2_flow_matching/tutorial_notebook_Flow.ipynb` and begin learning!
