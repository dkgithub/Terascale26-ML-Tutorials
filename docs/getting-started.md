# Getting Started

This guide will walk you through cloning the repository and setting up your environment to run the tutorials.

---

## Step 1: Clone the Repository

First, clone the tutorial repository to your local machine.

=== "HTTPS"

    ```bash
    git clone https://github.com/sjiggins/Terascale26-ML-Tutorials.git
    cd Terascale26-ML-Tutorials
    ```

=== "SSH"

    ```bash
    git clone git@github.com:sjiggins/Terascale26-ML-Tutorials.git
    cd Terascale26-ML-Tutorials
    ```

=== "GitHub CLI"

    ```bash
	gh repo clone sjiggins/Terascale26-ML-Tutorials
    cd generative-tutorials
    ```

The relocate to the root directory of the repository:

```bash
cd Terascale26-ML-Tutorials
```

---

## Step 2: Choose Your Setup Method

You can run the tutorials using either:

1. **astral-uv** (Recommended) - Modern, fast package manager
2. **Traditional pip** - Classic Python package management

Choose the method that works best for you, however astrl-uv is significantly faster. It is likely that you do not have astral-uv installed on your laptop, or the NAF school account. To check this please run:

=== "Linux"
	```bash
	uv --version
	```

=== "macOS"
	```bash
	uv --version
	```
	
=== "Windows"
    ```powershell
    uv --version
    ```

If the command returns something like:

```bash
Command 'uv' not found, did you mean:
...
...
```

Then you will need to follow the installation process below.


### Install astral-uv

astral-uv is a fast Python package installer and resolver written in Rust. It's significantly faster than pip and conda.

=== "Linux"

    **Method 1: Using the install script (Recommended)**
    
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
    
    **Method 2: Using pip**
    
    ```bash
    pip install uv
    ```
    
    **Verify installation:**
    
    ```bash
    uv --version
    ```
    
    **Add to PATH (if needed):**
    
    The installer should automatically add uv to your PATH. If not, add this to your `~/.bashrc` or `~/.zshrc`:
    
    ```bash
    export PATH=""$HOME/.local/bin:$PATH"
    ```
    
    Then reload your shell:
    
    ```bash
    source ~/.bashrc  # or source ~/.zshrc
    ```

=== "macOS"

    **Method 1: Using the install script (Recommended)**
    
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
    
    **Method 2: Using Homebrew**
    
    ```bash
    brew install uv
    ```
    
    **Method 3: Using pip**
    
    ```bash
    pip install uv
    ```
    
    **Verify installation:**
    
    ```bash
    uv --version
    ```
    
    **Add to PATH (if needed):**
    
    The installer should automatically add uv to your PATH. If not, add this to your `~/.zshrc` or `~/.bash_profile`:
    
    ```bash
    export PATH=""$HOME/.local/bin:$PATH"
    ```
    
    Then reload your shell:
    
    ```bash
    source ~/.zshrc  # or source ~/.bash_profile
    ```

=== "Windows"

    **Method 1: Using PowerShell (Recommended)**
    
    Open PowerShell as Administrator and run:
    
    ```powershell
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```
    
    **Method 2: Using pip**
    
    ```powershell
    pip install uv
    ```
    
    **Method 3: Using winget**
    
    ```powershell
    winget install --id=astral-sh.uv -e
    ```
    
    **Verify installation:**
    
    ```powershell
    uv --version
    ```
    
    **Add to PATH (if needed):**
    
    The installer should automatically add uv to your PATH. If not:
    
    1. Press `Win + X` and select "System"
    2. Click "Advanced system settings"
    3. Click "Environment Variables"
    4. Under "User variables", find "Path" and click "Edit"
    5. Add: `C:\Users\<YourUsername>\.cargo\bin`
    6. Click OK and restart your terminal

!!! tip "Why astral-uv?"
    - **Fast**: 10-100x faster than pip
    - **Reliable**: Deterministic dependency resolution
    - **Compatible**: Works with existing pip ecosystem
    - **Modern**: Built with Rust for performance


## Method 1: Setup with astral-uv (Recommended)

### Create Virtual Environment
Inside the root directory of the tutorial repository exists a two files:
```bash

```

```bash
uv venv .venv
```

### Activate the Environment

=== "Linux / macOS"

    ```bash
    source .venv/bin/activate
    ```

=== "Windows (PowerShell)"

    ```powershell
    .venv\Scripts\Activate.ps1
    ```

=== "Windows (Command Prompt)"

    ```cmd
    .venv\Scripts\activate.bat
    ```

### Install Dependencies

Each tutorial has a `pyproject.toml` file that defines its dependencies. You can run the pip installation via two methods:

=== "astral-uv"

	```bash
	# Install from pyproject.toml
	uv pip install -e .
	```
=== "pip"

	```
	pip install -r requirements.txt
	```

### Register Jupyter Kernel
Now that you have a virtual environment setup for your 

```bash
uv pip install ipykernel
python -m ipykernel install --user --name=tutorial-env --display-name="Tutorial Environment"
```

### Verify Installation

```bash
python -c "import torch; print(torch.__version__)"
jupyter kernelspec list
```

---

## Method 2: Setup with Traditional pip

If you prefer traditional pip, you can use it instead.

### Navigate to a Tutorial

```bash
cd tutorial_1_ddpm        # For Tutorial 1
```

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate the Environment

=== "Linux / macOS"

    ```bash
    source .venv/bin/activate
    ```

=== "Windows (PowerShell)"

    ```powershell
    .venv\Scripts\Activate.ps1
    ```

=== "Windows (Command Prompt)"

    ```cmd
    .venv\Scripts\activate.bat
    ```

### Install Dependencies

```bash
# Install from pyproject.toml
pip install -e .

# OR install directly
pip install torch torchvision torchaudio numpy matplotlib tqdm
```

### Register Jupyter Kernel

```bash
pip install ipykernel
python -m ipykernel install --user --name=tutorial-env --display-name="Tutorial Environment"
```

### Verify Installation

```bash
python -c "import torch; print(torch.__version__)"
jupyter kernelspec list
```

---

## Step 3: Running the Tutorials

You have two options for running the tutorials:

### Option A: Run in Terminal (CLI)

Each tutorial can be run as a Python script:

```bash
# Make sure your virtual environment is activated
source .venv/bin/activate  # Linux/macOS
# OR
.venv\Scripts\Activate.ps1  # Windows

# Run the main script
python -m flow_matching_tutorial.main

# OR run directly
python flow_matching_tutorial/main.py
```

**Outputs will be saved to the `outputs/` directory.**

### Option B: Run in Jupyter Notebook/Lab

**Start Jupyter:**

```bash
# Start Jupyter Notebook
jupyter notebook

# OR start JupyterLab
jupyter lab
```

**Open the tutorial notebook:**

1. Navigate to the tutorial directory in Jupyter
2. Open `tutorial_notebook_Flow.ipynb` (or the appropriate notebook)
3. Select the "Tutorial Environment" kernel from the kernel menu
4. Run the cells!

### Option C: Run in VSCode

1. Open VSCode in the repository directory:
   ```bash
   code .
   ```

2. Open the notebook file (e.g., `tutorial_2_flow_matching/tutorial_notebook_Flow.ipynb`)

3. Click on the kernel selector in the top-right corner

4. Select "Tutorial Environment"

5. Run the cells using the run button or `Shift+Enter`

!!! tip "Recommended Workflow"
    - **Learning**: Use Jupyter Notebook/Lab for interactive exploration
    - **Experimentation**: Use VSCode for code editing and debugging
    - **Automation**: Use CLI scripts for batch processing

---

## Repository Structure

Here's an overview of the repository structure:

```
generative-tutorials/
├── docs/                      # Documentation (this website)
├── tutorial_1_ddpm/          # Tutorial 1: DDPM
│   ├── ddpm_tutorial/        # Python package
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── models.py
│   │   ├── diffusion.py
│   │   ├── utils.py
│   │   └── visualization.py
│   ├── outputs/              # Generated outputs
│   ├── pyproject.toml        # Dependencies
│   └── tutorial_notebook_DDPM.ipynb
│
├── tutorial_2_flow_matching/ # Tutorial 2: Flow Matching
│   ├── flow_matching_tutorial/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── models.py
│   │   ├── flow.py
│   │   ├── flow_solutions.py
│   │   ├── utils.py
│   │   └── visualization.py
│   ├── outputs/              # Generated outputs
│   ├── pyproject.toml
│   └── tutorial_notebook_Flow.ipynb
│
├── tutorial_3_advanced/      # Tutorial 3 (Coming Soon)
├── tutorial_4_score_based/   # Tutorial 4 (Coming Soon)
├── tutorial_5_applications/  # Tutorial 5 (Coming Soon)
├── mkdocs.yml                # Documentation config
├── README.md
└── LICENSE
```

---

## Running Multiple Tutorials

Each tutorial has its own virtual environment and kernel. To run multiple tutorials:

**Setup Tutorial 1:**

```bash
cd tutorial_1_ddpm
uv venv .venv
source .venv/bin/activate
uv pip install -e .
python -m ipykernel install --user --name=tutorial1 --display-name="Tutorial 1 - DDPM"
```

**Setup Tutorial 2:**

```bash
cd ../tutorial_2_flow_matching
uv venv .venv
source .venv/bin/activate
uv pip install -e .
python -m ipykernel install --user --name=tutorial2 --display-name="Tutorial 2 - Flow Matching"
```

**Now you can switch between kernels in Jupyter!**

---

## Common Workflows

### Daily Workflow

```bash
# Activate environment
cd tutorial_2_flow_matching
source .venv/bin/activate

# Pull latest changes
git pull

# Start Jupyter
jupyter lab

# Work on notebooks...

# Deactivate when done
deactivate
```

### Running Experiments

```bash
# Activate environment
source .venv/bin/activate

# Modify config in main.py
nano flow_matching_tutorial/main.py

# Run experiment
python -m flow_matching_tutorial.main

# Check outputs
ls outputs/
```

### Updating Dependencies

```bash
# Activate environment
source .venv/bin/activate

# Update with uv
uv pip install --upgrade torch numpy matplotlib

# OR update with pip
pip install --upgrade torch numpy matplotlib
```

---

## GPU Support (Optional)

If you have an NVIDIA GPU and want to use it:

### Check CUDA Availability

```bash
nvidia-smi
```

### Install PyTorch with CUDA Support

Visit [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) to get the right command for your system.

Example for CUDA 12.1:

```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Verify GPU

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
```

!!! note "CPU is Fine!"
    All tutorials work on CPU. GPU just makes them faster. The 2D toy datasets run quickly on CPU.

---

## Next Steps

Now that you're set up, choose a tutorial to start:

- [Tutorial 1: DDPM](tutorials/tutorial-1.md) - Start with diffusion models
- [Tutorial 2: Flow Matching](tutorials/tutorial-2.md) - Learn ODE-based generation

---

## Quick Reference

**Activate environment:**

```bash
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\Activate.ps1  # Windows
```

**Run notebook:**

```bash
jupyter lab
```

**Run CLI:**

```bash
python -m flow_matching_tutorial.main
```

**Update kernel:**

```bash
python -m ipykernel install --user --name=tutorial-env --display-name="Tutorial Environment"
```

---

## Troubleshooting

### Module not found

**Problem:** `ModuleNotFoundError: No module named 'flow_matching_tutorial'`

**Solution:** Install the package in editable mode:

```bash
source .venv/bin/activate
uv pip install -e .
```

### Kernel not showing in Jupyter

**Problem:** Can't see "Tutorial Environment" in kernel list

**Solution:** Reinstall the kernel:

```bash
python -m ipykernel install --user --name=tutorial-env --display-name="Tutorial Environment"
```

### Import errors in notebook

**Problem:** `ImportError: cannot import name 'something'`

**Solution:** Make sure you selected the correct kernel in Jupyter

### CUDA out of memory

**Problem:** GPU memory error

**Solution:** Use CPU instead or reduce batch size in config

For more help, see [Troubleshooting](troubleshooting.md).
