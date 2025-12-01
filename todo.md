# TODO — Reinforcement Learning Pong Project

This document outlines all major steps required to complete your Reinforcement Learning project using Gymnasium and Policy Gradient methods. Follow each section to guide your workflow.

---

## 🎮 1. Project Setup
Refer to `project_setup.md` for detailed setup instructions.

---

## 🏟️ 2. Understand Gymnasium
- [x] Read: how Gymnasium environments work (reset, step, observation_space, action_space)
- [x] Run a simple script to interact with the Pong environment using random actions
- [x] Understand the five outputs from `env.step()`

---

## 🔧 3. Environment & Preprocessing
- [x] Convert 210×160×3 RGB frames to grayscale
- [x] Downsample frames to smaller dimensions (e.g., 80×80)
- [x] Implement frame differencing (current frame − previous frame) to capture motion
- [x] Normalize pixel data
- [x] Store preprocessing pipeline as a separate function

---

## 🧠 4. Build the Policy Network
- [x] Create a PyTorch neural network following a simple CNN structure:
  - [x] Conv2D → ReLU
  - [x] Conv2D → ReLU
  - [x] Flatten layer
  - [x] Fully Connected → output probabilities for actions
- [x] Verify forward pass works on dummy input
- [x] Add softmax for action probabilities

---

## 🔁 5. RL Training Loop (Policy Gradient / REINFORCE)
- [x] On each step:
  - [x] Preprocess observation
  - [x] Forward pass → action probabilities
  - [x] Sample an action
  - [x] Feed action into environment
  - [x] Store: state, action, reward
- [x] Detect end of episode → compute discounted rewards
- [x] Normalize rewards
- [x] Compute policy gradient loss
- [x] Update network parameters
- [x] Save model checkpoints

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
- [x] Clean up code into modules
- [x] Create a `requirements.txt`
- [x] Add README with instructions to run the project
- [x] Attach TODO.md in your repo root
- [x] Create `.gitignore` file
- [ ] Submit project to your class

---

## 🚀 Optional Stretch Goals
- [ ] Implement Advantage (A2C)
- [ ] Try PPO using Stable Baselines 3
- [ ] Add tensorboard logging
- [ ] Hyperparameter tuning experiments

---

**This TODO list covers your entire class project from environment setup to final report.**

