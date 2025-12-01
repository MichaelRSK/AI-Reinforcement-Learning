# AI Reinforcement Learning - Pong Project

A reinforcement learning project that trains an AI agent to play Pong using Policy Gradient methods (REINFORCE) with PyTorch and Gymnasium.

## 🎮 Project Overview

This project implements a deep reinforcement learning agent that learns to play the classic Atari game Pong using:
- **Environment**: Gymnasium (Atari Pong-v5)
- **Algorithm**: Policy Gradient / REINFORCE
- **Framework**: PyTorch
- **Preprocessing**: Frame preprocessing (grayscale, downsampling, frame differencing)

## 📋 Current Progress

- ✅ **Step 1**: Project setup and environment configuration
- ✅ **Step 2**: Understanding Gymnasium environments
- ✅ **Step 3**: Environment preprocessing pipeline
- ✅ **Step 4**: Building the policy network
- ✅ **Step 5**: RL training loop (REINFORCE)
- ⏳ **Step 6**: Monitoring & evaluation
- ⏳ **Step 7**: Demonstration
- ⏳ **Step 8**: Project report

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
├── preprocessing.py              # Frame preprocessing functions
├── policy_network.py             # CNN policy network architecture
├── train.py                      # REINFORCE training loop
├── test_agent.py                 # Test and evaluate trained agent
├── test_gymnasium.py            # Test Gymnasium setup
├── understand_gymnasium.py      # Learn Gymnasium basics
├── test_preprocessing.py         # Test preprocessing pipeline
├── render_environment.py         # Render environment visually
├── visualize_preprocessing.py   # Visualize preprocessing results
├── requirements.txt             # Python dependencies
├── project_setup.md             # Detailed setup guide
├── TRAINING_GUIDE.md            # Complete training and testing guide
├── todo.md                      # Project TODO checklist
├── checkpoints/                 # Saved model checkpoints (created during training)
└── README.md                    # This file
```

## 🛠️ Key Components

### Policy Network

The `policy_network.py` module implements a CNN-based policy network:
- Convolutional layers for feature extraction
- Fully connected layers for action prediction
- Outputs action probabilities using softmax

### Training

The `train.py` module implements the REINFORCE algorithm:
- Episode collection and experience storage
- Discounted return computation
- Policy gradient updates
- Checkpoint saving and loading

### Preprocessing Pipeline

The `preprocessing.py` module provides functions to:
- Convert RGB frames to grayscale
- Downsample frames (210×160 → 80×80)
- Compute frame differences to capture motion
- Normalize pixel values

### Testing Scripts

- `test_gymnasium.py` - Verifies Gymnasium and Pong environment setup
- `test_preprocessing.py` - Tests all preprocessing functions
- `test_agent.py` - Test and evaluate trained agent performance
- `understand_gymnasium.py` - Educational script explaining Gymnasium concepts
- `render_environment.py` - Visual rendering of the game
- `visualize_preprocessing.py` - Visual comparison of original vs preprocessed frames

## 📚 Documentation

- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - **Complete guide for training and testing the AI** ⭐
- **[project_setup.md](project_setup.md)** - Complete setup instructions
- **[todo.md](todo.md)** - Project progress and TODO list

## 🔧 Dependencies

See `requirements.txt` for the complete list. Key dependencies:
- `gymnasium[atari,accept-rom-license]` - RL environment
- `ale-py` - Arcade Learning Environment
- `torch` - PyTorch for neural networks
- `numpy` - Numerical computing
- `matplotlib` - Visualization
- `opencv-python` - Image processing

## 🎯 Quick Start - Training and Testing

### Training the Agent

See **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** for detailed instructions, or:

```bash
# Install dependencies (if not already done)
pip install -r requirements.txt

# Start training
python train.py
```

### Testing the Trained Agent

```bash
# Test with default checkpoint
python test_agent.py --checkpoint checkpoints/final_checkpoint.pt

# Test with visual rendering
python test_agent.py --checkpoint checkpoints/final_checkpoint.pt --render

# Test for more episodes
python test_agent.py --checkpoint checkpoints/final_checkpoint.pt --episodes 10
```

For complete instructions, see **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)**.

## 📖 Learning Resources

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Policy Gradient Methods](https://spinningup.openai.com/en/latest/algorithms/vpg.html)

## 👥 Team

[Add your team members here]

## 📝 License

[Add your license here]

---

**Status**: Training Ready ✅

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for complete training and testing instructions.
