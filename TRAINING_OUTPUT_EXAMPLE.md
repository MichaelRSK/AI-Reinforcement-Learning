# Training Output Example

This document shows what you'll see during training with the enhanced output.

---

## Start of Training

```
======================================================================
  🎮 STARTING REINFORCE TRAINING FOR PONG
======================================================================
  Environment:          ALE/Pong-v5
  Device:               cuda (or cpu)
  Action Space:         6 actions
  Learning Rate:        0.0003
  Discount Factor:      0.99
  Normalize Returns:    True
  Total Episodes:       1000
  Save Every:           100 episodes
  Print Every:          10 episodes
======================================================================
  Reward Shaping Active:
    • Ball proximity:      +0.01
    • Potential ball hit:  +0.05
    • Paddle movement:     +0.001
    • Paddle-ball align:   +0.002-0.005
======================================================================
```

---

## Every 10 Episodes

```
======================================================================
Episode 10/1000 (1.0% complete)
======================================================================
  Game Score:         0 - 21  (Agent - Opponent)
  Episode Length:      978 steps
  Environment Reward:  -21.0
  Shaped Bonus:        +2.34
  Total Reward:       -18.66
  Avg Reward (10):    -18.45
  Loss:              0.3452
======================================================================
```

**What each line means:**
- **Game Score**: Actual Pong score (Agent points - Opponent points)
- **Episode Length**: How many frames the episode lasted
- **Environment Reward**: Total +1/-1 rewards from actual scoring
- **Shaped Bonus**: Total reward from shaping (proximity, hits, alignment)
- **Total Reward**: Environment + Shaped (what the agent optimizes)
- **Avg Reward (10)**: Average total reward over last 10 episodes
- **Loss**: Policy gradient loss value

---

## Every 10 Episodes (with trend - after episode 20)

```
======================================================================
Episode 50/1000 (5.0% complete)
======================================================================
  Game Score:         3 - 21  (Agent - Opponent)
  Episode Length:     1156 steps
  Environment Reward:  -18.0
  Shaped Bonus:        +2.87
  Total Reward:       -15.13
  Avg Reward (10):    -15.82
  Loss:              0.2891
  Trend:              📈 IMPROVING
======================================================================
```

**Trend Indicators:**
- 📈 IMPROVING: Recent 10 episodes are better than previous 10
- ➡️ STABLE: Performance is roughly the same
- 📉 DECLINING: Recent 10 episodes are worse (might need adjustment)

---

## Every 100 Episodes (Checkpoint Summary)

```
######################################################################
  CHECKPOINT #1 - Episode 100
######################################################################
  Last 100 Episodes Summary:
    Total Points Scored:  134
    Total Points Lost:    2100
    Win Rate:             6.0%
    Average Reward:       -17.23
    Best Reward:          -12.45
######################################################################


======================================================================
Episode 100/1000 (10.0% complete)
======================================================================
  Game Score:         2 - 21  (Agent - Opponent)
  Episode Length:     1089 steps
  Environment Reward:  -19.0
  Shaped Bonus:        +3.12
  Total Reward:       -15.88
  Avg Reward (10):    -15.34
  Loss:              0.2567
  Trend:              📈 IMPROVING
======================================================================
```

**Win Rate**: Percentage of points won (not games - Pong counts individual points)

---

## End of Training

```
======================================================================
  🎉 TRAINING COMPLETE! 🎉
======================================================================
  Total Episodes:           1000
  Total Training Steps:     1,045,234

  Performance (Last 100 Episodes):
    Average Reward:         +3.45
    Best Episode Reward:    +8.23
    Points Scored:          1834
    Points Lost:            1567
    Win Rate:               53.9%

  All-Time Best:
    Best Episode Reward:    +10.12

  Files Saved:
    Checkpoints:            checkpoints/
    Final Model:            final_checkpoint.pt
    Training Stats:         training_stats.json
======================================================================
```

---

## How to Interpret Progress

### Early Training (Episodes 1-100)
- **Expected**: Loss around -21, win rate <10%
- **Agent is**: Learning basic controls, mostly losing
- **Look for**: Shaped bonus increasing, occasional points scored

### Mid Training (Episodes 100-500)
- **Expected**: Loss around -15 to -10, win rate 10-30%
- **Agent is**: Learning to hit the ball back sometimes
- **Look for**: Increasing game scores, longer rallies

### Late Training (Episodes 500-1000)
- **Expected**: Loss around -5 to +5, win rate 30-50%
- **Agent is**: Becoming competitive, winning some games
- **Look for**: Win rate approaching 50%, positive avg rewards

### Signs of Success
✅ Trend shows 📈 IMPROVING consistently
✅ Win rate gradually increasing
✅ Shaped bonus staying 10-20% of total reward
✅ Game scores getting closer to 0

### Warning Signs
⚠️ Trend shows 📉 DECLINING for many episodes
⚠️ Win rate stuck at <5% after 200 episodes
⚠️ Shaped bonus >> environment rewards (gaming the system)
⚠️ Loss becomes NaN or explodes

---

## Quick Troubleshooting

**If agent isn't learning after 200 episodes:**
1. Check shaped bonus - if it's negative or very low, increase shaping
2. Check loss - if it's not decreasing, try higher learning rate
3. Check win rate - should be at least 5% by episode 200

**If training is unstable (wild swings):**
1. Reduce learning rate: 3e-4 → 1e-4
2. Reduce reward shaping by 50%
3. Check trend indicator - some oscillation is normal

**If agent is gaming the rewards:**
1. Reduce shaped bonus amounts
2. Make sure environment rewards dominate
3. Check points scored vs shaped bonus ratio
