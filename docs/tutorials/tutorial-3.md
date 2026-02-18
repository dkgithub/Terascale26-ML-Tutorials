# Tutorial 3: From Deep Neural Networks to Transformers

Learn how to choose the right architecture for sequential data by implementing and comparing MLP, CNN, and Transformer models.

## Overview

This tutorial provides a hands-on introduction to **architecture selection for time-series forecasting**. You'll implement three fundamentally different architectures—MLPs, CNNs, and Transformers—and understand when each excels.

Through a carefully designed multi-scale wave dataset, you'll discover:
- **MLPs** struggle with temporal structure
- **CNNs** excel at local patterns but miss long-range dependencies
- **Transformers** capture all scales through global attention

This is the same architectural progression that powers modern AI: from simple feedforward networks to the attention mechanisms in GPT and BERT.

**Duration:** 90-120 minutes

**Difficulty:** Intermediate to Advanced

---

## Learning Objectives

By the end of this tutorial, you will:

- Understand how different architectures process sequential data
- Implement autoregressive time-series forecasting
- Compare MLP, CNN, and Transformer performance empirically
- Learn when CNNs excel (local patterns) vs when Transformers are needed (long-range)
- Discover why GPT uses Transformers, not CNNs
- Master architecture-task matching principles
- Analyze prediction errors and confidence bands

---

## Prerequisites

**Required Knowledge:**

- Python programming
- PyTorch basics (tensors, gradients, training loops)
- Neural networks (MLPs, CNNs, basic Transformers)
- Time-series concepts (autoregressive models, forecasting)
- Linear algebra (matrix operations, dot products)

**Recommended Background:**

- Tutorial 1: Linear Regression (polynomial fitting)
- Tutorial 2: Perceptron to DNN (feedforward networks)
- Familiarity with sequence modeling

**Required Setup:**

- Complete [Getting Started Guide](../getting-started.md)
- GPU recommended but not required (CPU optimizations included)

---

## Tutorial Structure

The tutorial consists of:

1. **Jupyter Notebook** - Interactive, step-by-step learning with visualizations
2. **Python Scripts** - Modular, reusable code for each architecture
3. **Multi-Scale Dataset** - Specially designed to show architecture strengths/weaknesses
4. **Visualizations** - Time-series plots, error analysis, and performance comparisons

**Key Components:**

- **Data Generator**: Creates synthetic time-series with multiple timescales
- **Three Architectures**: MLP, CNN (fixed for autoregressive use), Transformer
- **Training Pipeline**: Teacher forcing with autoregressive evaluation
- **Error Analysis**: Evolution plots, confidence bands, MSE tracking

---

## How to Run This Tutorial

You have **two options** for running this tutorial:

### Option 1: Jupyter Notebook (Recommended for Learning)

**Best for:** Interactive exploration, understanding concepts, experimentation

#### Using JupyterLab/Jupyter Notebook (CLI)

**Steps:**

1. Navigate to the tutorial directory:
   ```bash
   cd tutorial_3_from_DNNs_to_Transformers
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

4. Open `tutorial_DNN_to_Transformer.ipynb`

5. Select the "Tutorial Environment" kernel

6. Run the cells sequentially (`Shift+Enter` for each cell)

#### Using VSCode

**Steps:**

1. Open VSCode:
   ```bash
   code tutorial_3_from_DNNs_to_Transformers
   ```

2. Open `tutorial_DNN_to_Transformer.ipynb`

3. Select the "Tutorial Environment" kernel (Python 3.x from your venv)

4. Click the play button next to each cell or use `Shift+Enter`

5. View interactive plots inline

---

### Option 2: Terminal/CLI (For Automation)

**Best for:** Running complete experiments, batch processing, benchmarking

**Steps:**

1. Navigate to the tutorial directory:
   ```bash
   cd tutorial_3_from_DNNs_to_Transformers
   ```

2. Activate your virtual environment:
   ```bash
   source .venv/bin/activate  # Linux/macOS
   .venv\Scripts\Activate.ps1  # Windows
   ```

3. Run the main comparison script:
   ```bash
   python DNNs_to_Transformer_tutorial/main_multiscale_comparison_IMPROVED.py
   ```

4. Check outputs in the current directory

**Customize Configuration:**

Edit `main_multiscale_comparison_IMPROVED.py` to change:
- Forecast horizon (`H_forecast`)
- Training samples (`n_train`, `n_val`, `n_test`)
- Model sizes (hidden dimensions, channels)
- Training epochs and learning rate
- Dataset amplitudes (slow vs fast components)

---

## What's Inside

### Python Package: `DNNs_to_Transformer_tutorial/`

```
DNNs_to_Transformer_tutorial/
├── __init__.py                              # Package initialization
├── data_generator_multiscale.py             # Multi-scale wave dataset
├── MultiLayerPerceptron_AR.py               # MLP autoregressive model
├── CNN_AR_v2.py                             # CNN autoregressive (fixed)
├── Transformer_AR.py                        # Transformer autoregressive
├── train_autoregressive.py                  # Training utilities
├── main_multiscale_comparison_IMPROVED.py   # Complete pipeline
├── logger.py                                # Logging configuration
├── plotting.py                              # Visualization utilities
└── utils.py                                 # Helper functions
```

**Key Files:**

- **`data_generator_multiscale.py`**: Multi-scale wave system
  - Fast oscillations (period ~7 steps) - CNN-friendly
  - Slow trends (period ~100 steps) - Transformer needed
  - Regime transitions (every ~30 steps)
  - Combines all components with controlled amplitudes

- **`MultiLayerPerceptron_AR.py`**: Autoregressive MLP
  - Flattens temporal structure
  - Fully connected layers
  - Learns arbitrary position-specific correlations

- **`CNN_AR_v2.py`**: Autoregressive CNN (FIXED version)
  - 1D convolutions over time
  - No BatchNorm (critical for autoregressive!)
  - Attention-based temporal aggregation
  - Parameter sharing across time

- **`Transformer_AR.py`**: Autoregressive Transformer
  - Multi-head self-attention
  - Positional encoding
  - Global receptive field
  - GPT-style architecture for time-series

- **`main_multiscale_comparison_IMPROVED.py`**: Complete pipeline
  - Data generation and visualization
  - Model building (all three architectures)
  - Training with teacher forcing
  - Autoregressive evaluation
  - Error analysis with confidence bands

---

## Notebook Structure

The Jupyter notebook is organized into pedagogical sections:

### Part 1: Introduction & Motivation
- Why architecture choice matters
- Preview of results
- Multi-scale data visualization

### Part 2: Understanding the Data
- Multi-scale wave system explained
- Fast oscillations (period ~7)
- Slow trends (period ~100)
- Frequency spectrum analysis
- Why this tests architecture limits

### Part 3: Architecture Deep Dive

#### 3.1 Multi-Layer Perceptron (MLP)
- How MLPs see sequential data (flattening)
- Advantages: Universal approximation
- Disadvantages: No structure exploitation
- Parameter explosion for long sequences

#### 3.2 Convolutional Neural Network (CNN)
- How CNNs process time-series (sliding windows)
- 1D convolutions explained
- Receptive field calculation
- Parameter sharing benefits
- Critical bug: BatchNorm + Autoregressive = disaster
- Fixed aggregation strategies

#### 3.3 Transformer
- How Transformers handle sequences (attention)
- Self-attention mechanism
- Positional encoding
- Why Transformers excel at long-range dependencies
- Connection to GPT

### Part 4: Implementation
- Building each model from scratch
- Architecture specifications
- Parameter counts comparison
- Design decisions explained

### Part 5: Training
- Teacher forcing vs autoregressive training
- Why teacher forcing is faster
- Training loop walkthrough
- Early stopping strategy

### Part 6: Evaluation & Analysis
- Autoregressive generation
- Performance comparison
- Error evolution plots
- MSE accumulation with confidence bands
- Interpreting results

### Part 7: Experiments & Extensions
- Modify dataset amplitudes
- Change model sizes
- Try different aggregation strategies
- Explore failure modes

### Part 8: Key Takeaways
- When to use MLP, CNN, or Transformer
- Architecture-task matching principles
- Real-world applications

---

## Expected Outputs

When you run the tutorial, you'll generate:

**In the tutorial directory:**

- `01_multiscale_data_overview_v2.png` - Dataset visualization
  - Heatmap showing space-time evolution
  - Individual components (slow trend, fast wave) with **FULL time range**
  - Temporal evolution at selected positions
  - Frequency spectrum showing two timescales

- `multiscale_comparison_with_errors.png` - **Enhanced 9-panel comparison**
  - **Row 1**: Time-series predictions (all 3 models)
  - **Row 2**: Error evolution over forecast steps
  - **Row 3**: MSE accumulation with confidence bands

**Example visualizations you'll create:**

```
Data Overview:
┌─────────────┬─────────────────┐
│  Heatmap    │  Components     │
│  (space-    │  Fast + Slow    │
│   time)     │  (FULL range)   │
├─────────────┼─────────────────┤
│  Time       │  Frequency      │
│  Evolution  │  Spectrum       │
└─────────────┴─────────────────┘

Model Comparison (3×3 grid):
┌─────────┬─────────┬─────────────┐
│   MLP   │   CNN   │ Transformer │
│  Preds  │  Preds  │   Preds     │
├─────────┼─────────┼─────────────┤
│   MLP   │   CNN   │ Transformer │
│  Error  │  Error  │   Error     │
│  Evol   │  Evol   │   Evol      │
├─────────┼─────────┼─────────────┤
│   MLP   │   CNN   │ Transformer │
│  MSE +  │  MSE +  │   MSE +     │
│  Conf   │  Conf   │   Conf      │
└─────────┴─────────┴─────────────┘
```

**Performance Metrics:**

```
Expected Results:
-----------------
Transformer: MSE ≈ 0.10-0.13  (Best - captures all scales)
CNN:         MSE ≈ 0.11-0.14  (Good - captures fast wave)
MLP:         MSE ≈ 0.11-0.15  (Baseline - struggles with structure)

Note: Exact values depend on random seed and training
Clear hierarchy demonstrates architecture strengths!
```

---

## Key Concepts Covered

### 1. Multi-Scale Temporal Patterns

Understanding data with multiple timescales:

```python
y(x,t) = slow_trend(t) + fast_wave(x,t) + regime_shifts(t) + noise

Components:
  - Slow trend:     period ~100 steps (long-range)
  - Fast wave:      period ~7 steps   (local)
  - Regime shifts:  every ~30 steps   (medium)
```

**Why this matters:**
- Real-world data has multiple scales (weather, finance, sensors)
- Different architectures capture different scales
- Tests architecture limitations

### 2. Autoregressive Generation

Learn how models generate sequences iteratively:

```python
# Autoregressive forecasting (like GPT)
for step in range(forecast_horizon):
    y_next = model(history)
    history = cat([history[1:], y_next])  # Slide window
```

**Key insights:**
- Predictions feed back as input (autoregressive loop)
- Errors can accumulate over time
- This is how GPT generates text!

### 3. Architecture-Specific Data Processing

**MLP**: Flatten everything
```
[Time=150, Space=100] → Flatten → [15,000]
Learn: "Position X correlates with position Y"
```

**CNN**: Sliding windows with parameter sharing
```
[Time=150, Space=100] → Transpose → [Space=100, Time=150]
100 parallel timelines → 1D convolution over time
Learn: "This 7-step pattern repeats everywhere"
```

**Transformer**: Self-attention over all positions
```
[Time=150, Space=100] → Project → [Time=150, d_model=128]
Every position attends to all others
Learn: "Which past moments predict the future?"
```

### 4. Receptive Field Limitations

**CNN Receptive Field:**
```
Formula: RF = 1 + n_layers × (kernel_size - 1)

With 3 layers, kernel_size=7:
RF = 1 + 3×(7-1) = 19 time steps

Problem: Can't see period-100 slow trend!
Solution: Transformer (infinite receptive field)
```

### 5. Critical Bug: BatchNorm + Autoregressive

**The Problem:**
```
BatchNorm learns statistics from TRUE data (training)
But normalizes PREDICTIONS (generation)
→ Distribution mismatch → Feature collapse → Failure!

Solution: Use LayerNorm, InstanceNorm, or no normalization
```

**Why this matters:**
- This is why GPT uses LayerNorm, not BatchNorm
- Critical for any autoregressive model
- Students discover a real architectural constraint!

### 6. Error Analysis

**Three types of analysis:**

1. **Error Evolution**: Where do errors occur?
2. **MSE Accumulation**: Do errors compound over time?
3. **Confidence Bands**: How reliable are predictions?

---

## Experiments to Try

After completing the tutorial, try these experiments:

### Easy (15-30 minutes each)

1. **Modify data components:**
   ```python
   dataset = MultiScaleWaveDataset(
       slow_amplitude=0.8,   # Emphasize Transformer advantage
       fast_amplitude=0.3,   # De-emphasize CNN advantage
   )
   ```
   **Expected:** Transformer margin increases

2. **Change model sizes:**
   ```python
   # Make CNN bigger
   channels=[64, 128, 64]  # vs [32, 64, 32]
   # Does it close the gap with Transformer?
   ```

3. **Adjust forecast horizon:**
   ```python
   H_forecast = 5   # vs 10 or 20
   # How does shorter/longer horizon affect each model?
   ```

### Medium (30-60 minutes each)

4. **Compare CNN aggregation strategies:**
   ```python
   aggregation='adaptive_avg'   # vs 'attention' vs 'last_position'
   # Which works best? Why?
   ```

5. **Receptive field experiment:**
   ```python
   # Increase CNN receptive field
   kernel_sizes=[11, 11, 11]  # vs [7, 7, 7]
   # Can larger kernels capture slow trend?
   ```

6. **Training mode comparison:**
   ```python
   USE_DIRECT_TRAINING = True   # Fast but potentially less accurate
   USE_DIRECT_TRAINING = False  # Slower but proper teacher forcing
   # Compare results and training time
   ```

### Advanced (1-2 hours each)

7. **Implement hybrid architecture:**
   ```python
   # CNN extracts features → Transformer processes them
   # Combine local (CNN) and global (Transformer) strengths
   ```

8. **Add class conditioning:**
   ```python
   # Condition on regime type
   # "Generate fast oscillation pattern" vs "Generate slow trend"
   ```

9. **Real-world data:**
   - Try on actual time-series (stock prices, weather)
   - Which architecture works best?
   - Does multi-scale insight transfer?

---

## Common Issues & Solutions

### Models predict constant values

**Problem:** Predictions are flat, don't oscillate

**Cause:** Direct training mode used (too fast, models collapse to mean)

**Solution:**
```python
USE_DIRECT_TRAINING = False  # Use teacher forcing instead
```
Expected runtime: 10-15 minutes (still reasonable on CPU)

### CNN performs worse than MLP

**Problem:** CNN should beat MLP but doesn't

**Possible causes:**
1. **BatchNorm enabled** (critical bug for autoregressive!)
   ```python
   normalization='none'  # NOT 'batch'!
   ```

2. **Wrong aggregation**
   ```python
   aggregation='attention'  # NOT 'adaptive_avg' for this task
   ```

3. **Pattern period doesn't match kernel size**
   - If fast_wave period ≠ kernel_size, CNN struggles
   - Try adjusting kernel to match pattern

### Training too slow on CPU

**Problem:** 30+ minute training time

**Solutions:**
1. **Reduce forecast horizon:**
   ```python
   H_forecast = 10  # vs 20 or 50
   ```

2. **Smaller training set:**
   ```python
   n_train = 300  # vs 1000
   ```

3. **Smaller models:**
   ```python
   # Already optimized in provided code
   ```

Expected runtime after optimizations: 10-15 minutes

### Error bands look wrong

**Problem:** Confidence bands too wide or narrow

**Cause:** Using wrong metric

**Solution:**
- Bands are ±1 std of errors across spatial dimensions
- This is correct for showing prediction variability
- Wide bands = inconsistent predictions
- Narrow bands = consistent predictions

---

## Theoretical Background

### Core Papers

**Transformers:**
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- Original Transformer architecture

**Autoregressive Models:**
- [WaveNet](https://arxiv.org/abs/1609.03499) (van den Oord et al., 2016)
- Autoregressive CNNs for audio (no BatchNorm!)

**Architecture Comparisons:**
- [Temporal Convolutional Networks](https://arxiv.org/abs/1803.01271) (Bai et al., 2018)
- When CNNs match RNNs/Transformers

### Key Insights from Literature

1. **CNNs** excel when:
   - Pattern period ≈ receptive field
   - Local structure dominates
   - Parameter efficiency needed

2. **Transformers** excel when:
   - Long-range dependencies exist
   - Position relationships vary
   - Sufficient compute available

3. **Normalization** matters:
   - BatchNorm breaks autoregressive models
   - LayerNorm safe for sequences
   - GPT uses LayerNorm for this reason

---

## Next Steps

After completing Tutorial 3:

**Build on these concepts:**
- **Tutorial 4:** [Denoising Diffusion Models](tutorial-4.md) - Generative modeling with iterative refinement
- **Tutorial 5:** [Flow Matching](tutorial-5.md) - Continuous-time generative models

**Further exploration:**
- Implement attention visualization
- Try on real sequential datasets
- Explore Transformer variants (BERT, GPT architecture)
- Study positional encodings in depth

**Practical applications:**
- Time-series forecasting
- Natural language processing
- Audio generation
- Video prediction

---

## Quick Reference

**Start Jupyter:**
```bash
cd tutorial_3_from_DNNs_to_Transformers
source .venv/bin/activate
jupyter lab
# Open: tutorial_DNN_to_Transformer.ipynb
```

**Run CLI:**
```bash
cd tutorial_3_from_DNNs_to_Transformers
source .venv/bin/activate
python DNNs_to_Transformer_tutorial/main_multiscale_comparison_IMPROVED.py
```

**View outputs:**
```bash
ls *.png
# 01_multiscale_data_overview_v2.png
# multiscale_comparison_with_errors.png
```

**Common imports:**
```python
from DNNs_to_Transformer_tutorial.data_generator_multiscale import MultiScaleWaveDataset
from DNNs_to_Transformer_tutorial.MultiLayerPerceptron_AR import MultiLayerPerceptronAR
from DNNs_to_Transformer_tutorial.CNN_AR_v2 import CNN_AR_v2
from DNNs_to_Transformer_tutorial.Transformer_AR import TransformerAR
```

**Key hyperparameters:**
```python
T_history = 150        # History length
H_forecast = 10-20     # Forecast horizon
n_train = 300-1000     # Training samples
n_epochs = 30-50       # Training epochs
USE_DIRECT_TRAINING = False  # Use teacher forcing
```

---

## Need Help?

- Check [FAQ](../faq.md)
- See [Troubleshooting](../troubleshooting.md)
- Review [Getting Started](../getting-started.md)
- Open an issue on GitHub

---

## Summary

In this tutorial, you've learned:

✅ **Architecture selection** is critical for sequential data
✅ **MLPs** struggle with temporal structure (no inductive bias)
✅ **CNNs** excel at local patterns (parameter sharing, limited RF)
✅ **Transformers** capture all scales (global attention, flexible)
✅ **Autoregressive generation** works like GPT text generation
✅ **BatchNorm breaks** autoregressive models (distribution shift)
✅ **Error analysis** reveals model strengths/weaknesses

**The Big Picture:**
```
MLP  →  Can't exploit structure
CNN  →  Exploits local structure
Transformer  →  Exploits all structure

This progression mirrors the evolution of deep learning!
```

---

!!! success "Ready to Start"
    Head to `tutorial_3_from_DNNs_to_Transformers/tutorial_DNN_to_Transformer.ipynb` and begin learning how architecture choice shapes what models can learn!
