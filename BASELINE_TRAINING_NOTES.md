# Baseline Training - Pure Environment Rewards

## What Changed

**Reward shaping has been DISABLED** to establish a clean baseline.

### Why?
The previous reward shaping had critical bugs:

1. **Ball detection was broken**: Detected ALL bright pixels (paddles, scores, center line), not just the ball
2. **Proximity reward was constant**: "Ball on agent's side" included the paddle itself, giving +0.01 almost every frame
3. **Alignment was self-referential**: Checking if paddle aligned with itself (detected as "ball")
4. **Result**: +10-16 shaped bonus per episode, drowning out the ±1 environment rewards

### Current State
- **Shaped Bonus should now be 0.0** every episode
- Agent learns purely from scoring points (+1) and losing points (-1)
- This is slower but gives us a clean baseline

---

## What to Expect

### Episode Output Changes
```
OLD (Broken Shaping):
  Environment Reward:  -21.0
  Shaped Bonus:       +13.15  ❌ WAY TOO HIGH!
  Total Reward:        -6.85

NEW (No Shaping):
  Environment Reward:  -21.0
  Shaped Bonus:        +0.00  ✅ Clean baseline
  Total Reward:       -21.00
```

### Performance Expectations

**Episodes 1-200 (Current Phase):**
- **Total Reward**: -21.0 to -19.0 (mostly losses)
- **Win Rate**: 0-5% (very few points scored)
- **Trend**: Likely ➡️ STABLE or slow 📈 IMPROVING
- **This is NORMAL and EXPECTED**

**Episodes 200-500:**
- **Total Reward**: -18.0 to -12.0 (starting to learn)
- **Win Rate**: 5-15% (occasional hits)
- **Trend**: 📈 IMPROVING gradually
- **Agent starts returning ball occasionally**

**Episodes 500-1000:**
- **Total Reward**: -10.0 to -5.0 (competitive)
- **Win Rate**: 15-30% (decent rallies)
- **Trend**: Continued 📈 IMPROVING
- **Agent can sustain short rallies**

**Episodes 1000+:**
- **Total Reward**: -5.0 to +5.0 (near parity)
- **Win Rate**: 30-50% (competitive play)
- **Agent becomes a worthy opponent**

---

## When to Stop Current Training

Since you're already at episode 210 with the broken shaping, you should:

### Option 1: Stop Now and Restart ⭐ RECOMMENDED
- The current model has learned bad habits (optimize for fake shaping)
- Start fresh from episode 1 with clean rewards
- This gives the cleanest learning curve

### Option 2: Continue and Compare
- Let it run to episode 300-400
- Compare win rate progress between old (210-400) and restart (1-200)
- More data but potentially wasted compute

**I recommend Option 1**: Stop the current training, delete the checkpoints, and start fresh.

---

## How to Restart Training

1. **Stop current training**: Press Ctrl+C in the terminal
2. **Delete old checkpoints** (optional but recommended):
   ```bash
   rm -rf checkpoints/
   mkdir checkpoints
   ```
3. **Start fresh**:
   ```bash
   python train.py
   ```

---

## What to Monitor

### Good Signs ✅
- Shaped bonus is exactly 0.0 every episode
- Win rate slowly increasing (even 1-2% improvement per 100 episodes is good)
- Trend showing 📈 IMPROVING over time
- Total reward gradually getting less negative

### Warning Signs ⚠️
- Shaped bonus is NOT 0.0 (shaping wasn't fully disabled)
- Win rate stuck at 0% after 300+ episodes
- Total reward staying at -21 constantly (no learning)
- Loss becoming NaN or exploding

### When Learning is Working
After 200-300 episodes you should see:
- **At least 1-3% win rate** (20-60 points scored per 100 episodes)
- **Occasional games with 2-3 points scored**
- **Total reward averaging -19 to -18** (better than -21)
- **Trend showing gradual improvement**

---

## Next Steps After Baseline

Once we confirm the agent CAN learn from pure rewards (even if slowly), we can:

### Phase 2: Add Back PROPER Reward Shaping
1. **Fix ball detection**: Isolate actual ball (small, moving, 2-6 pixels)
2. **Add smart shaping**:
   - +0.001 when ball approaches and paddle positioned correctly
   - +0.005 when ball is successfully returned
   - Keep total shaped bonus under +0.5 per episode
3. **Validate**: Compare learning speed with/without shaping

### Phase 3: Advanced Techniques
- Frame stacking (4 frames for velocity information)
- Advantage Actor-Critic (A2C/A3C)
- Proximal Policy Optimization (PPO)
- Learning rate scheduling

---

## Current Training Time Estimates

**Per Episode**: ~30-60 seconds (varies by episode length)
**100 Episodes**: ~1-1.5 hours
**1000 Episodes**: ~10-15 hours

**Recommendation**: 
- Run overnight for 500-1000 episodes
- Check progress in the morning
- If win rate is improving, continue
- If stuck at 0%, we need to investigate further

---

## Quick Troubleshooting

**Q: Shaped bonus is still not 0.0?**
A: The code wasn't updated properly. The `compute_shaped_reward` function should just return `float(env_reward)` with no additions.

**Q: Win rate still 0% after 500 episodes?**
A: Possible issues:
- Learning rate too low (try 5e-4 or 1e-3)
- Network might be too small/large
- Preprocessing might be removing too much information
- Gamma might be too high (try 0.95)

**Q: Should I use GPU?**
A: For this small network, CPU is fine. GPU would help if you batch multiple episodes or use larger networks.

---

## Bottom Line

**Current change**: Disabled broken reward shaping
**Expected outcome**: Slower but cleaner learning
**What to do**: Restart training from scratch
**Timeline**: Check back after 200-300 episodes (~4-6 hours)
**Success criteria**: Win rate > 2% by episode 300

Good luck! 🎮
