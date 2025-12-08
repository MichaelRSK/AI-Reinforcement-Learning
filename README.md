# AI Reinforcement Learning - Pong Project

A reinforcement learning project that implements and compares two policy gradient algorithms—**REINFORCE** and **Proximal Policy Optimization (PPO)**—for training an AI agent to play Atari Pong.

## 🎮 Project Overview

This project implements and compares two reinforcement learning algorithms to train an agent to play the classic Atari game Pong:

- **Environment**: Gymnasium (Atari Pong-v5)
- **Algorithms**: 
  - **REINFORCE** (Monte Carlo Policy Gradient) - Foundational algorithm
  - **PPO** (Proximal Policy Optimization) - State-of-the-art algorithm
- **Framework**: PyTorch
- **Preprocessing**: Frame preprocessing (grayscale, downsampling, frame differencing)

### 🏆 Key Results

| Algorithm | Final Avg Reward | Win Rate | Best Episode | Status |
|-----------|------------------|----------|--------------|--------|
| **REINFORCE** | -20.13 | 2.07% | -16.00 | ✅ Complete |
| **PPO** | +3.66 | **58.71%** | +13.00 | ✅ Complete |
| **Improvement** | **+118.2%** | **+56.64%** | **+29 points** | - |

**Conclusion**: PPO demonstrates **dramatically superior performance**, achieving a **58.71% win rate** compared to REINFORCE's **2.07%**, representing a **28x improvement** in win rate and a complete reversal from losing (-20) to winning (+3.66) average rewards.

## 📋 Project Status

✅ **COMPLETE** - Both algorithms have been implemented, trained, and compared.

- ✅ REINFORCE implementation and training (1,000 episodes)
- ✅ PPO implementation and training (1,000 episodes)
- ✅ Comprehensive comparison and analysis
- ✅ Complete documentation

See `todo.md` for the complete checklist.

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd AI-Reinforcement-Learning
   ```

2. **Set up the environment**
   
   Follow the detailed setup instructions in [`project_setup.md`](project_setup.md), or:

   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate (Windows PowerShell)
   .\venv\Scripts\Activate.ps1
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python test_gymnasium.py
   ```

## 📁 Project Structure

```
AI-Reinforcement-Learning/
│
├── 🤖 REINFORCE Implementation
│   ├── train.py                    # REINFORCE training script
│   ├── policy_network.py           # Policy network architecture
│   └── checkpoints/                 # REINFORCE checkpoints
│       ├── checkpoint_episode_*.pt # Model checkpoints
│       └── training_stats.json     # Training statistics
│
├── 🚀 PPO Implementation
│   ├── train_ppo.py                # PPO training script
│   ├── ppo_network.py              # Actor-Critic network
│   └── checkpoints_ppo/             # PPO checkpoints
│       ├── ppo_checkpoint_*.pt      # Model checkpoints
│       └── ppo_training_stats.json  # Training statistics
│
├── 🔧 Shared Utilities
│   ├── preprocessing.py            # Frame preprocessing
│   ├── compare_models.py           # Model comparison tool
│   ├── test_agent.py               # Agent testing script
│   ├── test_gymnasium.py           # Environment testing
│   └── restart_training.py         # Training restart utility
│
├── 📊 Results & Analysis
│   ├── model_comparison.png         # Comparison visualization
│   ├── RESULTS_ANALYSIS.md         # REINFORCE detailed analysis
│   └── FINAL_SUMMARY.md            # Comprehensive project summary
│
└── 📋 Documentation
    ├── README.md                    # This file
    ├── PROJECT_JOURNEY.md           # Development journey and story
    ├── PROJECT_SUMMARY.md           # Project overview
    ├── TRAINING_GUIDE.md            # Training instructions
    ├── WATCH_AGENT_PLAY.md          # How to watch agent play
    ├── project_setup.md             # Setup instructions
    ├── requirements.txt             # Python dependencies
    └── todo.md                      # Project checklist
```

## 🛠️ Key Components

### REINFORCE Implementation

**Policy Network** (`policy_network.py`):
- CNN-based policy network
- Convolutional layers for feature extraction
- Fully connected layers for action prediction
- Outputs action probabilities using softmax
- ~550,000 parameters

**Training** (`train.py`):
- REINFORCE algorithm implementation
- Episode collection and experience storage
- Discounted return computation
- Policy gradient updates
- Checkpoint saving and loading

### PPO Implementation

**Actor-Critic Network** (`ppo_network.py`):
- Shared convolutional feature extractor
- Separate actor head (policy) and critic head (value)
- ~560,000 parameters
- Implements clipped surrogate objective

**Training** (`train_ppo.py`):
- PPO algorithm with all key components
- Rollout collection (2048 steps)
- Generalized Advantage Estimation (GAE)
- Multiple epochs per batch (4x sample efficiency)
- Clipped policy updates for stability

### Shared Utilities

**Preprocessing** (`preprocessing.py`):
- Convert RGB frames to grayscale
- Downsample frames (210×160 → 80×80)
- Compute frame differences to capture motion
- Normalize pixel values

**Testing & Evaluation**:
- `test_agent.py` - Test and evaluate trained agents
- `test_gymnasium.py` - Verify environment setup
- `compare_models.py` - Generate comparison visualizations

## 📚 Documentation

### Main Documentation
- **[FINAL_SUMMARY.md](FINAL_SUMMARY.md)** - **Comprehensive project summary with results** ⭐
- **[PROJECT_JOURNEY.md](PROJECT_JOURNEY.md)** - **Development journey and story** ⭐
- **[RESULTS_ANALYSIS.md](RESULTS_ANALYSIS.md)** - Detailed REINFORCE analysis
- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Complete training and testing guide
- **[WATCH_AGENT_PLAY.md](WATCH_AGENT_PLAY.md)** - How to watch trained agents play

### Setup & Reference
- **[project_setup.md](project_setup.md)** - Complete setup instructions
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Project overview
- **[todo.md](todo.md)** - Project checklist

## 🔧 Dependencies

See `requirements.txt` for the complete list. Key dependencies:
- `gymnasium[atari,accept-rom-license]` - RL environment
- `ale-py` - Arcade Learning Environment
- `torch` - PyTorch for neural networks
- `numpy` - Numerical computing
- `matplotlib` - Visualization
- `opencv-python` - Image processing

## 🎯 Quick Start - Training and Testing

### Training the Agents

**Train REINFORCE:**
```bash
# Install dependencies (if not already done)
pip install -r requirements.txt

# Train REINFORCE (1000 episodes, ~15 hours on CPU)
python train.py

# Results saved to: checkpoints/final_checkpoint.pt
```

**Train PPO:**
```bash
# Train PPO (1000 episodes, ~20 hours on CPU)
python train_ppo.py

# Results saved to: checkpoints_ppo/ppo_final_checkpoint.pt
```

### Testing Trained Agents

```bash
# Test REINFORCE agent
python test_agent.py --checkpoint checkpoints/final_checkpoint.pt

# Test PPO agent
python test_agent.py --checkpoint checkpoints_ppo/ppo_final_checkpoint.pt

# Test with visual rendering
python test_agent.py --checkpoint checkpoints_ppo/ppo_final_checkpoint.pt --render

# Test for more episodes
python test_agent.py --checkpoint checkpoints_ppo/ppo_final_checkpoint.pt --episodes 10
```

### Comparing Models

```bash
# Generate comparison plots and statistics
python compare_models.py

# Output: model_comparison.png (4-panel visualization)
```

For complete instructions, see **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)**.

## 🔬 Algorithm Details

### REINFORCE (Monte Carlo Policy Gradient)

- **Type**: On-policy, model-free, episodic
- **Key Features**: Simple implementation, high variance, sample inefficient
- **Hyperparameters**: Learning rate 3e-4, discount 0.99, gradient clipping 0.5
- **Results**: 2.07% win rate, -20.13 average reward

### PPO (Proximal Policy Optimization)

- **Type**: On-policy, model-free, actor-critic
- **Key Features**: Value function baseline, clipped updates, GAE, sample efficient
- **Hyperparameters**: Learning rate 3e-4, GAE λ=0.95, clip ε=0.2, 4 epochs
- **Results**: 58.71% win rate, +3.66 average reward

See **[FINAL_SUMMARY.md](FINAL_SUMMARY.md)** for detailed algorithm descriptions and hyperparameters.

## 📖 Learning Resources

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [OpenAI Spinning Up](https://spinningup.openai.com/) - Policy Gradient Methods
- [PPO Paper](https://arxiv.org/abs/1707.06347) - Schulman et al. (2017)

## 💡 Key Insights

1. **Variance Reduction Matters**: PPO's value function baseline dramatically reduces variance compared to REINFORCE
2. **Sample Efficiency**: PPO's multiple epochs per batch (4x) make it much more sample efficient
3. **Stability**: Clipped surrogate objective prevents destructive policy updates
4. **Modern Algorithms Win**: The 28x improvement demonstrates why PPO is widely used in practice

## 📊 Results Summary

- **REINFORCE**: Struggled with high variance, achieved only 2.07% win rate
- **PPO**: Achieved 58.71% win rate, demonstrating state-of-the-art performance
- **Improvement**: 28x better win rate, complete reversal from losing to winning

For detailed analysis, see **[FINAL_SUMMARY.md](FINAL_SUMMARY.md)** and **[PROJECT_JOURNEY.md](PROJECT_JOURNEY.md)**.

---

**Project Status**: ✅ **COMPLETE**

Both algorithms implemented, trained, and compared. Ready for academic submission.

*Last Updated: December 2025*
