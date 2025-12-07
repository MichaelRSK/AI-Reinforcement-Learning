# Watch Your Trained Agent Play Pong! 🎮

## Quick Start

### Watch PPO Agent (Recommended - Best Performance!)

```bash
python test_agent.py --checkpoint checkpoints_ppo/ppo_final_checkpoint.pt --render
```

This will:
- ✅ Load your trained PPO agent (58.71% win rate!)
- ✅ Open a window showing the game
- ✅ Run 5 episodes by default
- ✅ Show statistics after each episode

### Watch REINFORCE Agent

```bash
python test_agent.py --checkpoint checkpoints/final_checkpoint.pt --render
```

---

## Command Options

### Basic Usage

```bash
# Watch PPO agent play (5 episodes)
python test_agent.py --checkpoint checkpoints_ppo/ppo_final_checkpoint.pt --render

# Watch for more episodes
python test_agent.py --checkpoint checkpoints_ppo/ppo_final_checkpoint.pt --render --episodes 10

# Watch without rendering (just see stats)
python test_agent.py --checkpoint checkpoints_ppo/ppo_final_checkpoint.pt --episodes 10
```

### Available Checkpoints

**PPO Checkpoints:**
- `checkpoints_ppo/ppo_final_checkpoint.pt` - Final trained model (best!)
- `checkpoints_ppo/ppo_checkpoint_episode_1000.pt` - Episode 1000
- `checkpoints_ppo/ppo_checkpoint_episode_900.pt` - Episode 900
- `checkpoints_ppo/ppo_checkpoint_episode_800.pt` - Episode 800
- ... and so on

**REINFORCE Checkpoints:**
- `checkpoints/final_checkpoint.pt` - Final trained model
- `checkpoints/checkpoint_episode_1000.pt` - Episode 1000
- ... and so on

### Watch Different Training Stages

See how the agent improved over time:

```bash
# Early training (episode 100)
python test_agent.py --checkpoint checkpoints_ppo/ppo_checkpoint_episode_100.pt --render --episodes 3

# Mid training (episode 500)
python test_agent.py --checkpoint checkpoints_ppo/ppo_checkpoint_episode_500.pt --render --episodes 3

# Final trained agent (episode 1000)
python test_agent.py --checkpoint checkpoints_ppo/ppo_final_checkpoint.pt --render --episodes 3
```

---

## What You'll See

### Visual Window

When you use `--render`, a window will open showing:
- 🎮 The Pong game in real-time
- 🏓 Your agent's paddle (right side)
- 🏓 Opponent's paddle (left side)
- ⚪ The ball
- 📊 Score at the top

### Console Output

```
======================================================================
TESTING TRAINED AGENT
======================================================================
Detected agent type: PPO
Loading checkpoint: checkpoints_ppo/ppo_final_checkpoint.pt
Checkpoint loaded: checkpoints_ppo/ppo_final_checkpoint.pt
Resuming from episode: 1000
======================================================================

Running 5 test episode(s)...
(Close the render window or press Ctrl+C to stop early)

Episode 1: Reward = 5.00, Length = 1234
Episode 2: Reward = 8.00, Length = 1456
Episode 3: Reward = 3.00, Length = 987
Episode 4: Reward = 12.00, Length = 1678
Episode 5: Reward = 7.00, Length = 1345

======================================================================
TEST RESULTS
======================================================================
Average Reward: 7.00
Average Episode Length: 1340.0
Best Episode Reward: 12.00
Worst Episode Reward: 3.00
Win Rate: 100.0% (episodes with positive reward)
======================================================================
```

---

## Tips

### For Best Viewing Experience

1. **Use PPO Agent**: Much better gameplay (58% win rate vs 2%)
2. **Watch Multiple Episodes**: See consistency of performance
3. **Compare Stages**: Watch early vs late training to see improvement

### Performance Expectations

**PPO Agent:**
- ✅ Should win most games (positive rewards)
- ✅ Strategic paddle movement
- ✅ Good ball tracking
- ✅ Average reward: +3 to +5 per episode

**REINFORCE Agent:**
- ⚠️ Will lose most games (negative rewards)
- ⚠️ Less strategic play
- ⚠️ Average reward: -18 to -20 per episode

---

## Troubleshooting

### "Checkpoint file not found"

Make sure you're using the correct path:
```bash
# Check what checkpoints exist
ls checkpoints_ppo/*.pt
ls checkpoints/*.pt
```

### Window doesn't open / No rendering

- Make sure you included `--render` flag
- Some systems may need additional display libraries
- Try running without `--render` first to verify checkpoint loads

### "ModuleNotFoundError"

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Advanced: Save Gameplay Video

To record gameplay (requires additional setup):

```bash
python test_agent.py --checkpoint checkpoints_ppo/ppo_final_checkpoint.pt --render --save-video gameplay.mp4
```

---

## Enjoy Watching Your AI Play! 🎉

Your PPO agent should demonstrate:
- ✅ Smooth paddle movement
- ✅ Strategic positioning
- ✅ Good ball tracking
- ✅ Competitive gameplay

**Have fun watching your trained agent in action!** 🚀
