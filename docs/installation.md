# Installation Guide

This guide will help you set up your development environment for the generative modeling tutorials.

## Overview

You'll need to install:

1. **astral-uv** - Modern Python package and project manager
2. **Jupyter Notebook** - Interactive computing environment
3. **ipykernel** - Jupyter kernel for Python virtual environments

---

## Step 1: Install astral-uv

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

---

## Step 2: Install Jupyter Notebook

Jupyter notebooks provide an interactive environment for running Python code with visualizations.

=== "Linux"

    **CLI Installation**
    
    Using uv (Recommended):
    
    ```bash
    # Create a tools virtual environment for Jupyter
    uv venv ~/.jupyter-env
    source ~/.jupyter-env/bin/activate
    uv pip install jupyter notebook jupyterlab
    ```
    
    Using pip:
    
    ```bash
    pip install jupyter notebook jupyterlab
    ```
    
    **Verify installation:**
    
    ```bash
    jupyter --version
    ```
    
    **Start Jupyter Notebook:**
    
    ```bash
    jupyter notebook
    ```
    
    **Start JupyterLab:**
    
    ```bash
    jupyter lab
    ```
    
    ---
    
    **VSCode Integration**
    
    1. Install VSCode from [https://code.visualstudio.com/](https://code.visualstudio.com/)
    
    2. Install the Python extension:
       ```bash
       code --install-extension ms-python.python
       ```
    
    3. Install the Jupyter extension:
       ```bash
       code --install-extension ms-toolsai.jupyter
       ```
    
    4. Open VSCode and you can now open `.ipynb` files directly

=== "macOS"

    **CLI Installation**
    
    Using uv (Recommended):
    
    ```bash
    # Create a tools virtual environment for Jupyter
    uv venv ~/.jupyter-env
    source ~/.jupyter-env/bin/activate
    uv pip install jupyter notebook jupyterlab
    ```
    
    Using pip:
    
    ```bash
    pip install jupyter notebook jupyterlab
    ```
    
    **Verify installation:**
    
    ```bash
    jupyter --version
    ```
    
    **Start Jupyter Notebook:**
    
    ```bash
    jupyter notebook
    ```
    
    **Start JupyterLab:**
    
    ```bash
    jupyter lab
    ```
    
    ---
    
    **VSCode Integration**
    
    1. Install VSCode from [https://code.visualstudio.com/](https://code.visualstudio.com/)
    
    2. Install the Python extension:
       ```bash
       code --install-extension ms-python.python
       ```
    
    3. Install the Jupyter extension:
       ```bash
       code --install-extension ms-toolsai.jupyter
       ```
    
    4. Open VSCode and you can now open `.ipynb` files directly

=== "Windows"

    **Installation**
    
    Using uv (Recommended):
    
    ```powershell
    # Create a tools virtual environment for Jupyter
    uv venv $HOME\.jupyter-env
    & $HOME\.jupyter-env\Scripts\Activate.ps1
    uv pip install jupyter notebook jupyterlab
    ```
    
    Using pip:
    
    ```powershell
    pip install jupyter notebook jupyterlab
    ```
    
    **Verify installation:**
    
    ```powershell
    jupyter --version
    ```
    
    **Start Jupyter Notebook:**
    
    ```powershell
    jupyter notebook
    ```
    
    **Start JupyterLab:**
    
    ```powershell
    jupyter lab
    ```
    
    ---
    
    **VSCode Integration**
    
    1. Install VSCode from [https://code.visualstudio.com/](https://code.visualstudio.com/)
    
    2. Install the Python extension:
       ```powershell
       code --install-extension ms-python.python
       ```
    
    3. Install the Jupyter extension:
       ```powershell
       code --install-extension ms-toolsai.jupyter
       ```
    
    4. Open VSCode and you can now open `.ipynb` files directly

!!! info "Jupyter Notebook vs JupyterLab"
    - **Jupyter Notebook**: Classic interface, simpler, lighter weight
    - **JupyterLab**: Modern interface, more features, IDE-like experience
    
    Both work with these tutorials - choose based on your preference!

---

## Step 3: Install and Configure ipykernel

ipykernel allows you to use your virtual environment as a Jupyter kernel.

### Install ipykernel

=== "Linux / macOS"

    **In your virtual environment:**
    
    ```bash
    # Activate your virtual environment first
    source .venv/bin/activate
    
    # Install ipykernel
    uv pip install ipykernel
    
    # Register the kernel with Jupyter
    python -m ipykernel install --user --name=tutorial-env --display-name="Tutorial Environment"
    ```
    
    **Verify the kernel is registered:**
    
    ```bash
    jupyter kernelspec list
    ```
    
    You should see your `tutorial-env` kernel listed.

=== "Windows"

    **In your virtual environment:**
    
    ```powershell
    # Activate your virtual environment first
    .venv\Scripts\Activate.ps1
    
    # Install ipykernel
    uv pip install ipykernel
    
    # Register the kernel with Jupyter
    python -m ipykernel install --user --name=tutorial-env --display-name="Tutorial Environment"
    ```
    
    **Verify the kernel is registered:**
    
    ```powershell
    jupyter kernelspec list
    ```
    
    You should see your `tutorial-env` kernel listed.

### Using the Kernel

**In Jupyter Notebook/Lab:**

1. Open your notebook
2. Click "Kernel" → "Change Kernel"
3. Select "Tutorial Environment"

**In VSCode:**

1. Open your `.ipynb` file
2. Click on the kernel selector in the top-right
3. Select "Tutorial Environment"

### Managing Kernels

**List all kernels:**

```bash
jupyter kernelspec list
```

**Remove a kernel:**

```bash
jupyter kernelspec uninstall tutorial-env
```

**Update a kernel:**

```bash
# Remove old kernel
jupyter kernelspec uninstall tutorial-env

# Activate environment and reinstall
source .venv/bin/activate  # or .venv\Scripts\Activate.ps1 on Windows
python -m ipykernel install --user --name=tutorial-env --display-name="Tutorial Environment"
```

!!! tip "Multiple Environments"
    You can register multiple virtual environments as different kernels:
    
    ```bash
    # Environment 1
    source .venv-tutorial1/bin/activate
    python -m ipykernel install --user --name=tutorial1 --display-name="Tutorial 1"
    
    # Environment 2
    source .venv-tutorial2/bin/activate
    python -m ipykernel install --user --name=tutorial2 --display-name="Tutorial 2"
    ```
    
    Then switch between them in Jupyter!

---

## Verification Checklist

Before proceeding, verify all installations:

- [ ] `uv --version` shows version 0.1.0 or higher
- [ ] `jupyter --version` shows Jupyter components
- [ ] `jupyter kernelspec list` shows at least one kernel
- [ ] Can open Jupyter Notebook/Lab in browser
- [ ] Can open `.ipynb` files in VSCode (if using VSCode)

!!! success "Installation Complete!"
    You're ready to proceed to the [Getting Started](getting-started.md) guide!

---

## Troubleshooting

### uv not found after installation

**Solution:** Add uv to your PATH:

=== "Linux / macOS"
    ```bash
    export PATH="$HOME/.cargo/bin:$PATH"
    source ~/.bashrc  # or ~/.zshrc
    ```

=== "Windows"
    Add `C:\Users\<YourUsername>\.cargo\bin` to your PATH environment variable.

### Jupyter kernel not showing in notebook

**Solution:** Reinstall the kernel:

```bash
source .venv/bin/activate
python -m ipykernel install --user --name=tutorial-env --display-name="Tutorial Environment"
```

### Permission denied errors

**Solution:** Don't use `sudo`. Install in user space:

```bash
pip install --user jupyter notebook
```

For more help, see [Troubleshooting](troubleshooting.md).
