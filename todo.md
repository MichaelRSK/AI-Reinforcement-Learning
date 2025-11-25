# TODO — Reinforcement Learning Pong Project

This document outlines all major steps required to complete your Reinforcement Learning project using Gymnasium and Policy Gradient methods. Follow each section to guide your workflow.

---

## 🎮 1. Project Setup
- [ ] Install Python 3.10+
- [ ] Create virtual environment
- [ ] Install required libraries:
  - [ ] `gymnasium`
  - [ ] `gymnasium[atari]`
  - [ ] `gymnasium[accept-rom-license]`
  - [ ] `numpy`
  - [ ] `torch` (PyTorch)
  - [ ] `matplotlib` (for plots)
- [ ] Verify Pong environment loads correctly

---

## 🏟️ 2. Understand Gymnasium
- [ ] Read: how Gymnasium environments work (reset, step, observation_space, action_space)
- [ ] Run a simple script to interact with the Pong environment using random actions
- [ ] Understand the five outputs from `env.step()`

---

## 🔧 3. Environment & Preprocessing
- [ ] Convert 210×160×3 RGB frames to grayscale
- [ ] Downsample frames to smaller dimensions (e.g., 80×80)
- [ ] Implement frame differencing (current frame − previous frame) to capture motion
- [ ] Normalize pixel data
- [ ] Store preprocessing pipeline as a separate function

---

## 🧠 4. Build the Policy Network
- [ ] Create a PyTorch neural network following a simple CNN structure:
  - [ ] Conv2D → ReLU
  - [ ] Conv2D → ReLU
  - [ ] Flatten layer
  - [ ] Fully Connected → output probabilities for actions
- [ ] Verify forward pass works on dummy input
- [ ] Add softmax for action probabilities

---

## 🔁 5. RL Training Loop (Policy Gradient / REINFORCE)
- [ ] On each step:
  - [ ] Preprocess observation
  - [ ] Forward pass → action probabilities
  - [ ] Sample an action
  - [ ] Feed action into environment
  - [ ] Store: state, action, reward
- [ ] Detect end of episode → compute discounted rewards
- [ ] Normalize rewards
- [ ] Compute policy gradient loss
- [ ] Update network parameters
- [ ] Save model checkpoints

---

## 📊 6. Monitoring & Evaluation
- [ ] Plot average reward per episode
- [ ] Track episode length
- [ ] Track training stability (variance)
- [ ] Render agent gameplay periodically to observe learning quality

---

## 🎥 7. Demonstration
- [ ] Record a gameplay video of the trained agent
- [ ] Save frames or use Gymnasium rendering
- [ ] Export report-ready MP4

---

## 📝 8. Class Project Report
- [ ] Introduction to RL and Pong
- [ ] Explanation of policy gradient algorithm
- [ ] Description of preprocessing pipeline
- [ ] Architecture explanation (CNN model)
- [ ] Training process & challenges
- [ ] Learning curves and analysis
- [ ] Demonstration results
- [ ] Possible improvements (A2C, PPO, reward shaping, frame stacking)

---

## 📁 9. Final Project Packaging
- [ ] Clean up code into modules
- [ ] Create a `requirements.txt`
- [ ] Add README with instructions to run the project
- [ ] Attach TODO.md in your repo root
- [ ] Submit project to your class

---

## 🚀 Optional Stretch Goals
- [ ] Implement Advantage (A2C)
- [ ] Try PPO using Stable Baselines 3
- [ ] Add tensorboard logging
- [ ] Hyperparameter tuning experiments

---

**This TODO list covers your entire class project from environment setup to final report.**

