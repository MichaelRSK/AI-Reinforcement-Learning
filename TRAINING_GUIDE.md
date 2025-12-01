# Complete Training and Testing Guide

This guide provides step-by-step instructions for training your Pong AI agent and testing its performance.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Training the AI](#training-the-ai)
4. [Monitoring Training Progress](#monitoring-training-progress)
5. [Testing the Trained AI](#testing-the-trained-ai)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Options](#advanced-options)

---

## 🔧 Prerequisites

### Required Software

1. **Python 3.8 or higher**
   - Check your version: `python --version` or `python3 --version`
   - Download from: https://www.python.org/downloads/

2. **pip** (Python package manager)
   - Usually comes with Python
   - Check: `pip --version`

3. **Git** (optional, for cloning repositories)
   - Download from: https://git-scm.com/downloads

### System Requirements

- **RAM**: At least 4GB (8GB+ recommended)
- **Storage**: ~2GB free space for dependencies and checkpoints
- **GPU** (optional but recommended): NVIDIA GPU with CUDA support for faster training
  - Training will work on CPU, but will be slower
  - Check GPU availability: The script will automatically detect and use GPU if available

---

## 🚀 Environment Setup

### Step 1: Navigate to Project Directory

Open your terminal/command prompt and navigate to the project folder:

```bash
cd "C:\Users\Michael\CSC 410 AI\AI-Reinforcement-Learning"
```

Or on Linux/Mac:
```bash
cd ~/path/to/AI-Reinforcement-Learning
```

### Step 2: Create Virtual Environment (Recommended)

**Why use a virtual environment?**
- Isolates project dependencies
- Prevents conflicts with other Python projects
- Makes it easier to manage packages

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` at the beginning of your command prompt, indicating the virtual environment is active.

### Step 3: Install Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

**What this installs:**
- `gymnasium[atari,accept-rom-license]` - Gymnasium library with Atari games support
- `ale-py` - Arcade Learning Environment for Atari games
- `torch` - PyTorch deep learning framework
- `numpy` - Numerical computing
- `matplotlib` - Plotting and visualization
- `opencv-python` - Image processing for preprocessing

**Expected output:**
- Installation may take 5-10 minutes
- You should see "Successfully installed..." messages
- If you see any errors, see the [Troubleshooting](#troubleshooting) section

### Step 4: Verify Installation

Test that everything is set up correctly:

```bash
python test_gymnasium.py
```

This should:
- Create the Pong environment
- Display environment properties
- Run a few test steps
- Print "✓ All tests passed!"

If you see errors, check the [Troubleshooting](#troubleshooting) section.

---

## 🎓 Training the AI

### Quick Start (Default Settings)

The simplest way to start training:

```bash
python train.py
```

This will:
- Train for **1000 episodes** (default)
- Save checkpoints every **100 episodes**
- Print progress every **10 episodes**
- Save checkpoints to the `checkpoints/` folder

### Understanding Training Output

When you run training, you'll see output like this:

```
======================================================================
STARTING REINFORCE TRAINING
======================================================================
Environment: ALE/Pong-v5
Number of actions: 6
Learning rate: 0.0001
Discount factor (gamma): 0.99
Normalize returns: True
Device: cuda
Number of episodes: 1000
======================================================================

Episode   10 | Reward:  -21.00 | Length:  500 | Avg Reward (10):  -20.50 | Loss: 0.012345
Episode   20 | Reward:  -19.00 | Length:  500 | Avg Reward (10):  -18.30 | Loss: 0.011234
...
Checkpoint saved: checkpoints/checkpoint_episode_100.pt
...
```

**What to look for:**
- **Reward**: Total reward for the episode (negative is normal early in training)
- **Length**: Number of steps in the episode
- **Avg Reward (10)**: Average reward over last 10 episodes (should increase over time)
- **Loss**: Policy gradient loss (should decrease over time)

### Customizing Training Parameters

You can modify training parameters by editing `train.py`. Look for the `main()` function at the bottom:

```python
def main():
    trainer = REINFORCETrainer(
        env_name="ALE/Pong-v5",
        learning_rate=1e-4,      # Learning rate (default: 0.0001)
        gamma=0.99,              # Discount factor (default: 0.99)
        normalize_returns=True,  # Normalize returns (default: True)
        checkpoint_dir="checkpoints"
    )
    
    trainer.train(
        num_episodes=1000,       # Number of episodes (default: 1000)
        save_frequency=100,      # Save checkpoint every N episodes (default: 100)
        print_frequency=10       # Print stats every N episodes (default: 10)
    )
```

**Common modifications:**

1. **Train for more episodes:**
   ```python
   trainer.train(num_episodes=5000)  # Train for 5000 episodes
   ```

2. **Save checkpoints more frequently:**
   ```python
   trainer.train(save_frequency=50)  # Save every 50 episodes
   ```

3. **Adjust learning rate:**
   ```python
   trainer = REINFORCETrainer(learning_rate=5e-5)  # Lower learning rate
   ```

4. **Change discount factor:**
   ```python
   trainer = REINFORCETrainer(gamma=0.95)  # Less emphasis on future rewards
   ```

### Training Time Estimates

**On CPU:**
- ~1-2 minutes per episode
- 1000 episodes: ~16-33 hours
- 5000 episodes: ~3-7 days

**On GPU (CUDA):**
- ~10-30 seconds per episode
- 1000 episodes: ~3-8 hours
- 5000 episodes: ~14-42 hours

**Note:** Training time varies based on:
- Hardware (CPU/GPU speed)
- Episode length (varies during training)
- System load

### Stopping and Resuming Training

**To stop training:**
- Press `Ctrl+C` in the terminal
- The current episode will complete, then training stops
- The last checkpoint will be saved

**To resume training:**
- Use the `load_checkpoint()` method (see [Advanced Options](#advanced-options))
- Or modify `train.py` to load a checkpoint before training

---

## 📊 Monitoring Training Progress

### Checkpoint Files

Checkpoints are saved in the `checkpoints/` folder with names like:
- `checkpoint_episode_100.pt`
- `checkpoint_episode_200.pt`
- `final_checkpoint.pt`

Each checkpoint contains:
- Model weights (neural network parameters)
- Optimizer state (for resuming training)
- Training statistics
- Hyperparameters

### Training Statistics File

After training completes, statistics are saved to:
```
checkpoints/training_stats.json
```

This file contains:
- Episode numbers
- Rewards per episode
- Episode lengths
- Running averages

**To view statistics:**
```python
import json
with open("checkpoints/training_stats.json", "r") as f:
    stats = json.load(f)
    print(f"Episodes: {len(stats['episode'])}")
    print(f"Best reward: {max(stats['reward'])}")
    print(f"Average reward (last 100): {sum(stats['reward'][-100:])/100}")
```

### Expected Training Progress

**Early Training (Episodes 1-100):**
- Rewards: -21 to -15 (losing most points)
- Episode length: ~500 steps
- Agent is exploring randomly

**Mid Training (Episodes 100-500):**
- Rewards: -15 to -5 (starting to win some points)
- Episode length: varies
- Agent is learning basic strategies

**Late Training (Episodes 500-1000+):**
- Rewards: -5 to +10 (winning more often)
- Episode length: longer (more rallies)
- Agent is playing strategically

**Note:** Training progress varies. Some runs may learn faster or slower.

---

## 🧪 Testing the Trained AI

### Quick Test (Visual)

Test your trained agent with visual rendering:

```bash
python test_agent.py --checkpoint checkpoints/final_checkpoint.pt --render
```

This will:
- Load the trained model
- Run the agent for a specified number of episodes
- Show the game visually (if `--render` is used)
- Display performance statistics

### Test Script Options

```bash
python test_agent.py --help
```

**Common options:**

1. **Test with specific checkpoint:**
   ```bash
   python test_agent.py --checkpoint checkpoints/checkpoint_episode_500.pt
   ```

2. **Test for multiple episodes:**
   ```bash
   python test_agent.py --checkpoint checkpoints/final_checkpoint.pt --episodes 10
   ```

3. **Test with rendering (see the game):**
   ```bash
   python test_agent.py --checkpoint checkpoints/final_checkpoint.pt --render
   ```

4. **Save gameplay video:**
   ```bash
   python test_agent.py --checkpoint checkpoints/final_checkpoint.pt --render --save-video output.mp4
   ```

### Understanding Test Output

When you run the test script, you'll see:

```
======================================================================
TESTING TRAINED AGENT
======================================================================
Loading checkpoint: checkpoints/final_checkpoint.pt
Checkpoint loaded: checkpoints/final_checkpoint.pt
Resuming from episode: 1000
======================================================================

Running 5 test episodes...
Episode 1: Reward = 8.00, Length = 1234
Episode 2: Reward = 12.00, Length = 1456
Episode 3: Reward = 5.00, Length = 987
Episode 4: Reward = 15.00, Length = 1678
Episode 5: Reward = 10.00, Length = 1345

======================================================================
TEST RESULTS
======================================================================
Average Reward: 10.00
Average Episode Length: 1340.0
Best Episode Reward: 15.00
Worst Episode Reward: 5.00
Win Rate: 100.0% (episodes with positive reward)
======================================================================
```

**What the metrics mean:**
- **Average Reward**: Mean reward across all test episodes (higher is better)
- **Average Episode Length**: Mean number of steps per episode
- **Best/Worst Episode Reward**: Best and worst performance
- **Win Rate**: Percentage of episodes with positive reward (agent won)

### Manual Testing (Python Script)

You can also test the agent programmatically:

```python
from train import REINFORCETrainer
import gymnasium as gym
import ale_py

# Create trainer and load checkpoint
trainer = REINFORCETrainer()
trainer.load_checkpoint("checkpoints/final_checkpoint.pt")

# Test for one episode
states, actions, rewards, log_probs = trainer.collect_episode()
total_reward = sum(rewards)
print(f"Episode reward: {total_reward}")
print(f"Episode length: {len(rewards)}")

trainer.close()
```

### Comparing Different Checkpoints

To compare performance across training:

```bash
# Test early checkpoint
python test_agent.py --checkpoint checkpoints/checkpoint_episode_100.pt --episodes 5

# Test mid checkpoint
python test_agent.py --checkpoint checkpoints/checkpoint_episode_500.pt --episodes 5

# Test final checkpoint
python test_agent.py --checkpoint checkpoints/final_checkpoint.pt --episodes 5
```

Compare the average rewards to see improvement over training.

---

## 🔍 Troubleshooting

### Common Issues

#### 1. "ModuleNotFoundError: No module named 'gymnasium'"

**Solution:**
```bash
pip install -r requirements.txt
```

Make sure your virtual environment is activated.

#### 2. "ALE/Pong-v5 not found"

**Solution:**
```bash
pip install gymnasium[atari,accept-rom-license]
pip install ale-py
```

You may need to accept the ROM license:
```bash
pip install gymnasium[accept-rom-license]
```

#### 3. "CUDA out of memory" or GPU errors

**Solution:**
- The script automatically falls back to CPU if GPU fails
- To force CPU usage, modify `train.py`:
  ```python
  trainer = REINFORCETrainer(device=torch.device("cpu"))
  ```

#### 4. Training is very slow

**Possible causes:**
- Running on CPU (expected to be slow)
- System is under heavy load
- Too many background processes

**Solutions:**
- Close other applications
- Use GPU if available (automatic)
- Reduce number of episodes for testing

#### 5. "FileNotFoundError: checkpoint file not found"

**Solution:**
- Make sure you've trained the model first
- Check that the checkpoint path is correct
- List available checkpoints:
  ```bash
  ls checkpoints/  # Linux/Mac
  dir checkpoints  # Windows
  ```

#### 6. Agent performance is poor

**Possible causes:**
- Not enough training (need 1000+ episodes)
- Learning rate too high/low
- Need to adjust hyperparameters

**Solutions:**
- Train for more episodes
- Try different learning rates (1e-5 to 1e-3)
- Check training statistics to see if learning is happening

#### 7. "Permission denied" errors

**Solution:**
- Make sure you have write permissions in the project directory
- On Windows, run terminal as Administrator if needed
- Check that `checkpoints/` folder can be created

### Getting Help

If you encounter other issues:

1. **Check error messages carefully** - they often indicate the problem
2. **Verify installation** - run `python test_gymnasium.py`
3. **Check Python version** - need 3.8+
4. **Check dependencies** - run `pip list` to see installed packages

---

## ⚙️ Advanced Options

### Resuming Training from Checkpoint

To resume training from a saved checkpoint:

```python
from train import REINFORCETrainer

# Create trainer
trainer = REINFORCETrainer()

# Load checkpoint
trainer.load_checkpoint("checkpoints/checkpoint_episode_500.pt")

# Continue training
trainer.train(num_episodes=2000, save_frequency=100)
```

### Training with Different Hyperparameters

Create a custom training script:

```python
from train import REINFORCETrainer

# Experiment with different settings
trainer = REINFORCETrainer(
    learning_rate=5e-5,      # Lower learning rate
    gamma=0.95,             # Different discount factor
    normalize_returns=True
)

trainer.train(num_episodes=2000)
```

### Batch Training (Multiple Episodes at Once)

The current implementation trains one episode at a time. For batch training, you would need to modify the code to collect multiple episodes before updating.

### Using TensorBoard (Optional)

To add TensorBoard logging, you would need to:
1. Install tensorboard: `pip install tensorboard`
2. Add logging code to `train.py`
3. View with: `tensorboard --logdir=logs`

---

## 📝 Summary

### Training Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Environment tested (`python test_gymnasium.py`)
- [ ] Training started (`python train.py`)
- [ ] Checkpoints being saved
- [ ] Training statistics monitored

### Testing Checklist

- [ ] Training completed (or checkpoint available)
- [ ] Test script run (`python test_agent.py`)
- [ ] Performance metrics reviewed
- [ ] Visual testing done (if desired)
- [ ] Results documented

### Next Steps After Training

1. **Evaluate performance** - Run test script to see how well agent plays
2. **Visualize learning** - Plot training curves from statistics
3. **Record gameplay** - Create video of agent playing
4. **Experiment** - Try different hyperparameters
5. **Document results** - Write up findings for your report

---

## 🎯 Quick Reference Commands

```bash
# Setup
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt

# Verify
python test_gymnasium.py

# Train
python train.py

# Test
python test_agent.py --checkpoint checkpoints/final_checkpoint.pt --render

# View checkpoints
dir checkpoints  # Windows
ls checkpoints   # Linux/Mac
```

---

**Good luck with your training! 🚀**

If you have questions or encounter issues, refer to the troubleshooting section or check the code comments in `train.py` and `test_agent.py`.

