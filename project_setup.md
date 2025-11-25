# Project Setup Guide

This document explains how to set up the development environment for the Reinforcement Learning Pong project.

## Prerequisites

- **Python 3.10+** (We used Python 3.13.5)
- A terminal/command prompt
- Git (for version control)

## Step-by-Step Setup

### 1. Verify Python Installation

First, check that Python is installed and meets the version requirement:

```bash
python --version
```

You should see Python 3.10 or higher. If not, download and install Python from [python.org](https://www.python.org/downloads/).

### 2. Create Virtual Environment

Navigate to the project directory and create a virtual environment:

```bash
python -m venv venv
```

This creates a `venv` folder containing an isolated Python environment.

### 3. Activate Virtual Environment

**On Windows (PowerShell):**
```bash
.\venv\Scripts\Activate.ps1
```

**On Windows (Command Prompt):**
```bash
venv\Scripts\activate.bat
```

**On macOS/Linux:**
```bash
source venv/bin/activate
```

You should see `(venv)` at the beginning of your command prompt when activated.

### 4. Install Required Dependencies

Install all required packages from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

This will install:
- `gymnasium[atari,accept-rom-license]` - Gymnasium with Atari environments and ROM license acceptance
- `ale-py` - Arcade Learning Environment (required for Atari games)
- `torch` - PyTorch for neural networks
- `numpy` - Numerical computing
- `matplotlib` - Plotting and visualization
- `opencv-python` - Image processing for preprocessing pipeline

**Note:** The `ale-py` package is crucial for Atari environments. If you encounter "Namespace ALE not found" errors, make sure `ale-py` is installed.

### 5. Verify Installation

Run the test script to verify everything is working:

```bash
python test_gymnasium.py
```

You should see output confirming:
- ✓ Environment created successfully
- ✓ Observation space: Box(0, 255, (210, 160, 3), uint8)
- ✓ Action space: Discrete(6)
- ✓ Reset and step operations work correctly

## Troubleshooting

### Issue: "Namespace ALE not found"

**Solution:** Make sure `ale-py` is installed:
```bash
pip install ale-py
```

Also ensure you import `ale_py` in your Python scripts before creating the environment:
```python
import gymnasium as gym
import ale_py  # This registers the ALE namespace
env = gym.make("ALE/Pong-v5")
```

### Issue: PowerShell execution policy error

If you get an execution policy error when activating the virtual environment in PowerShell, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue: pip not found

Make sure Python is properly installed and added to your PATH. You may need to use `python -m pip` instead of just `pip`.

## Project Structure

After setup, your project should have:
```
AI-Reinforcement-Learning/
├── venv/                    # Virtual environment (don't commit this)
├── .gitignore              # Git ignore file
├── requirements.txt        # Python dependencies
├── preprocessing.py        # Preprocessing functions
├── project_setup.md        # This file
├── todo.md                 # Project TODO list
├── README.md               # Project README
├── test_gymnasium.py       # Test script to verify setup
├── understand_gymnasium.py # Learn Gymnasium basics
├── test_preprocessing.py   # Test preprocessing pipeline
├── render_environment.py   # Render environment visually
└── visualize_preprocessing.py # Visualize preprocessing
```

## Next Steps

Once setup is complete, refer to `todo.md` for the next steps in the project:
- Understanding Gymnasium environments
- Implementing preprocessing
- Building the policy network
- Training the agent

## Additional Resources

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [ALE-py GitHub](https://github.com/mgbellemare/Arcade-Learning-Environment)

