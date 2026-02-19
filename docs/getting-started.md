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

## Setting up Virtual Environment 

=== "Method 1: Astral-UV (Recommended)"

	### Create Virtual Environment
	To create the virtual environment with astral-uv, simply run:
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
	
	To install the dependencies needed for this tutorial, please use the following command:
	
	```bash
	# Install from pyproject.toml
	uv sync 
	```
	
	This will looking inside the central `pyproject.toml` file to find all dependencies and then install them in the `.venv` virtual environment.
	
	### Register Jupyter Kernel
	Now that you have a virtual environment setup, we need to declare the virtual environment to any jupyter(hub/lab/notebook) we start when using these tutorials. To do this, run the following:
	
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

=== "Method 2: Traditional pip"
	
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
	
	=== "pyproject.toml"
		```bash
		# Install from pyproject.toml
		pip install -e .
		```
		
	=== "requirements.txt"
		```bash
		# Install via requirements
		pip install -r requirements.txt
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


## Step 3: Running the Tutorials

You have two options for running the tutorials:

- Command Line (in terminal)
- Jupyter Notebook/Lab

In either of these two cases, you need to enter the sub-directory of each tutorial:

```bash
    ├── tutorial_1_polynomial_fit
    ├── tutorial_2_perceptron-to-DNN
    ├── tutorial_3_from_DNNs_to_Transformers
    ├── tutorial_4_ddpm
    ├── tutorial_5_flow_matching
```

Once inside this directory, you are ready to go! To start the tutorial please see the tutorial specific pages:

- [Tutorial 1: Polynomial Fit](tutorials/tutorial-1.md) - Start with a basic polnomial fit using PyTorch `nn.module` structure

For a complete overview of the tutorial see the repository structure below.

### Repository Structure

```
└── Terascale26-ML-Tutorials
    ├── docs      # markdown documentation that you are reading right now
    ├── LICENSE
    ├── mkdocs.yml
    ├── pyproject.toml
    ├── README.md
    ├── requirements.txt
    ├── tutorial_1_polynomial_fit
    │   ├── polynomial_tutorial
    │   │   ├── __init__.py
    │   │   ├── LinearRegressor.py
    │   │   ├── logger.py
    │   │   ├── loss.py
    │   │   ├── main.py
    │   │   ├── train.py
    │   │   └── utils.py
    │   ├── pyproject.toml
    │   └── tutorial_poly_fit.ipynb
    ├── tutorial_2_perceptron-to-DNN
    │   ├── perceptron_to_DNN_tutorial
    │   │   ├── __init__.py
    │   │   ├── logger.py
    │   │   ├── loss.py
    │   │   ├── main.py
    │   │   ├── MultiLayerPerceptron.py
    │   │   ├── plotting.py
    │   │   ├── train.py
    │   │   └── utils.py
    │   ├── pyproject.toml
    │   └── tutorial_DNN.ipynb
    ├── tutorial_3_from_DNNs_to_Transformers
    │   ├── DNNs_to_Transformer_tutorial
    │   │   ├── CNN_AR.py
    │   │   ├── CNN_AR_v2.py
    │   │   ├── data_generator_multiscale.py
    │   │   ├── data_generator.py
    │   │   ├── logger.py
    │   │   ├── loss.py
    │   │   ├── main.py
    │   │   ├── MultiLayerPerceptron_AR.py
    │   │   ├── plotting.py
    │   │   ├── Transformer_AR.py
    │   │   └── utils.py
    │   ├── pyproject.toml
    │   └── tutorial_DNN_to_Transformer.ipynb
    ├── tutorial_4_ddpm
    │   ├── ddpm_tutorial
    │   │   ├── diffusion.py
    │   │   ├── __init__.py
    │   │   ├── main.py
    │   │   ├── models.py
    │   │   ├── utils.py
    │   │   └── visualization.py
    │   ├── pyproject.toml
    │   └── tutorial_notebook.ipynb
    ├── tutorial_5_flow_matching
    │   ├── flow_matching_tutorial
    │   │   ├── flow-NAFjupytertests.py
    │   │   ├── flow.py
    │   │   ├── flow_solutions.py
    │   │   ├── __init__.py
    │   │   ├── main.py
    │   │   ├── models.py
    │   │   ├── utils.py
    │   │   └── visualization.py
    │   ├── pyproject.toml
    │   └── tutorial_notebook_Flow.ipynb
```

## Next Steps

Now that you're set up, choose a tutorial to start:

- [Tutorial 1: Polynomial Fit](tutorials/tutorial-1.md) - Start with a basic polnomial fit using PyTorch `nn.module` structure

---


<details markdown="1">
<summary>Optional Reading - see for quick reference, FAQs, etc...</summary>
## Common Workflows

### Daily Workflow

```bash
# Activate environment
source .venv/bin/activate
cd tutorial_1_polynomial_fit   # Or replace with another sub-directory

# Pull latest changes
git pull

# Start Jupyter
jupyter lab
# OR
jupyter notebook

# Work on notebooks...

# Deactivate when done
deactivate
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

## Quick Reference

**Activate environment:**

```bash
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\Activate.ps1  # Windows
```

**Run notebook:**

```bash
jupyter notebook
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

</details>
