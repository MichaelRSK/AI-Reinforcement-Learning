# Hyperparameter Tuning Guide

## ⚠️ LATEST UPDATE: Reward Shaping DISABLED

**Reward shaping has been turned OFF** due to critical detection bugs. The agent now learns purely from environment rewards (±1 for scoring).

See `BASELINE_TRAINING_NOTES.md` for full details on what changed and what to expect.

---

## Changes Made

### 1. Learning Rate: 1e-4 → 3e-4
**Why**: REINFORCE can handle higher learning rates since it's on-policy. This will speed up initial learning.

**Alternatives to try**:
- `3e-4`: Faster learning (recommended starting point)
- `1e-4`: More stable, slower convergence (original)
- `5e-5`: Very stable, very slow (if training is unstable)
- `1e-3`: Aggressive (only if 3e-4 is too slow and stable)

**How to tune**: Watch your reward curve. If it's very noisy/unstable, decrease. If learning is too slow, increase.

---

## 2. Reward Shaping - REDUCED

### Original (Too High):
```python
Ball on agent's side: +0.04      # Was overwhelming the ±1 game rewards
Potential hit:        +0.15      # 15% of a goal - too much!
Paddle movement:      +0.002     # Only rewarded 2 of 3 movement actions
```

### New (Balanced):
```python
Ball on agent's side:     +0.01  # Encourages engagement
Potential hit:            +0.05  # Rewards successful returns
Paddle movement:          +0.001 # Encourages active play
Paddle-ball alignment:    +0.002-0.005  # NEW! Rewards good positioning
```

**Why the reduction?**
- With gamma=0.99 and ~1000-step episodes, small rewards compound heavily
- If ball is on agent's side for 200 frames at +0.04/frame = +8 reward (8x a goal!)
- Shaping should guide behavior, not replace the true objective
- Rule of thumb: Shaping rewards should be **10-100x smaller** than sparse rewards

---

## 3. New Reward: Paddle-Ball Alignment ⭐

**What it does**: Rewards the agent for keeping the paddle vertically aligned with the ball

**Benefits**:
- Teaches defensive positioning
- Encourages ball tracking
- More granular feedback than just "ball on your side"

**Amounts**:
- `+0.005`: Excellent alignment (< 20 pixels apart)
- `+0.002`: Good alignment (< 40 pixels apart)

---

## 4. Fixed: Action Space for Movement

**Original Bug**: Only rewarded actions 2 and 3 (claimed to be UP/RIGHT)

**Fixed**: Now rewards actions 2, 3, and 5 (vertical movements in Pong)
- Action 2: UP
- Action 3: DOWN (or DOWNFIRE)
- Action 5: DOWNFIRE

---

## Recommended Testing Strategy

### Phase 1: Current Settings (3e-4 learning rate)
```python
learning_rate=3e-4
# Reward shaping as configured above
```
Train for 500-1000 episodes and observe:
- Is the agent learning? (rewards trending up)
- Is training stable? (not too noisy)
- Are shaped rewards dominating? (check shaped_bonus in logs)

### Phase 2: If Training is Unstable
```python
learning_rate=1e-4  # Reduce
# OR reduce reward shaping by 50%
ball_proximity: 0.01 → 0.005
potential_hit: 0.05 → 0.025
```

### Phase 3: If Training is Too Slow
```python
learning_rate=5e-4  # Increase
# OR slightly increase reward shaping
```

### Phase 4: Advanced Tuning
Try these experiments:
1. **Remove alignment reward** - see if it helps or hurts
2. **Increase potential_hit to 0.1** - if agent isn't learning to return ball
3. **Add ball velocity rewards** - reward when ball is moving toward opponent
4. **Decay learning rate** - start at 5e-4, decay to 1e-4 over time

---

## What to Monitor

### Good Signs:
- Average reward trending upward over 100 episodes
- Points scored increasing
- Episode lengths staying relatively stable
- Shaped bonus staying < 50% of total reward

### Bad Signs:
- Reward plateaus immediately (learning rate too low OR shaping too weak)
- Wildly oscillating rewards (learning rate too high OR shaping too strong)
- Shaped bonus >> environment rewards (shaping is dominating)
- Agent learns to "game" the shaped rewards (e.g., just moving paddle randomly)

---

## Current Reward Breakdown

For a typical episode (~1000 steps):
```
Environment rewards:  -21 to +21  (game to 21 points)
Ball proximity:       ~100 frames × 0.01 = +1.0
Potential hits:       ~10 hits × 0.05 = +0.5
Paddle movement:      ~300 moves × 0.001 = +0.3
Alignment bonus:      ~200 frames × 0.003 = +0.6
                      -------------------------
Total shaped bonus:   ~+2.4 per episode
```

This means shaping adds ~2.4 reward to guide behavior, but the -21 to +21 game outcome still dominates! ✅

---

## Additional Reward Ideas to Try Later

### 1. Ball Velocity Reward
```python
# Reward when ball is moving away from agent (successful hit)
if ball_moving_left and on_agent_side:
    shaped_reward += 0.03
```

### 2. Consecutive Hits Bonus
```python
# Track rally length, reward longer rallies
if consecutive_hits > 3:
    shaped_reward += 0.01 * consecutive_hits
```

### 3. Defensive Positioning
```python
# Reward being in center when ball is far away
if ball_on_opponent_side and paddle_near_center:
    shaped_reward += 0.001
```

### 4. Penalize Long Inactivity
```python
# Small penalty for not moving when ball approaches
if ball_approaching and action == NOOP:
    shaped_reward -= 0.002
```

---

## Bottom Line

**Current configuration should work well!** The key improvements are:
1. ✅ Higher learning rate (3x faster)
2. ✅ Balanced reward shaping (not overwhelming)
3. ✅ Fixed action space bug
4. ✅ Added paddle-ball alignment reward
5. ✅ Reduced shaping amounts to ~10% of episode reward

Start training and monitor the first 100 episodes closely. Adjust from there!

---

## 🔧 How to Re-Enable Reward Shaping (FUTURE)

Once baseline training confirms the agent can learn, here's how to add back PROPER reward shaping:

### Step 1: Isolate Ball Detection
```python
def detect_ball_position(frame):
    """Detect actual ball position, not paddle/score."""
    gray = np.dot(frame[..., :3], [0.299, 0.587, 0.114])
    
    # Ball characteristics in Pong:
    # - Very bright (>236 pixel value)
    # - Small (2-6 pixels typically)
    # - In play area (rows 34-194, cols 16-143)
    
    play_area = gray[34:194, 16:143]
    bright_pixels = np.where(play_area > 236)
    
    if len(bright_pixels[0]) > 0:
        # Group pixels into objects
        # Ball is the smallest bright object (not paddle/score)
        # Return (y, x) position or None
        pass
    return None
```

### Step 2: Smart Proximity Reward
```python
# Only reward if ball is actually detected AND approaching
ball_pos = detect_ball_position(observation)
if ball_pos and ball_is_approaching(ball_pos, prev_ball_pos):
    shaped_reward += 0.0005  # Very small!
```

### Step 3: Hit Detection
```python
# Reward when ball velocity reverses (successful hit)
if ball_velocity_changed_direction() and abs(ball_y - paddle_y) < 20:
    shaped_reward += 0.005  # Reward successful return
```

### Step 4: Validate Shaping
After adding shaping back:
- **Shaped bonus should be <1.0 per episode**
- **Environment rewards should still dominate**
- **Compare learning curves: with vs without shaping**
- **Make sure agent isn't gaming the shaping**

### Golden Rules for Reward Shaping
1. **Sparse > Dense**: Less is more
2. **10-100x smaller**: Shaping should guide, not overwhelm
3. **Validate detection**: Print what you're detecting, make sure it's correct
4. **A/B test**: Always compare with/without shaping
5. **Can't hurt baseline**: Shaping should speed up learning, not change final policy
