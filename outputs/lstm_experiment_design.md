# ExpF: LSTM Policy Experiment Design
**Date:** 2026-07-27  
**Hypothesis:** A recurrent policy can integrate temporal velocity cues and learn to decelerate before high-speed scenarios become unrecoverable, breaking the ~20 mph structural ceiling.

---

## Why LSTM

The root cause of the speed ceiling is that a feedforward policy sees a single observation at each timestep and cannot distinguish "this sensor reading is high because I'm going fast" from "this sensor reading is high because an obstacle is genuinely close." Even with speed appended to the observation (ExpD, SAC), the policy has no memory of how quickly the reading is changing, so it cannot anticipate that it needs to brake now rather than steer later.

An LSTM-based policy maintains a hidden state across timesteps. In theory it can learn:
- Rate of change of sensor readings → "I'm closing fast"
- Speed trajectory → "I've been accelerating for 3 steps, I should brake"
- The joint (speed, sensor_trend) → decelerate-and-steer rather than steer-only

This is not guaranteed to work — if the CARLoS action space doesn't include meaningful braking authority, or if 500k steps is insufficient for the LSTM to learn temporally extended strategies, the ceiling will hold.

---

## SB3 Implementation: sb3-contrib RecurrentPPO

Stable Baselines 3 does not natively support LSTM policies, but `sb3-contrib` provides `RecurrentPPO`:

```bash
pip install sb3-contrib
```

```python
from sb3_contrib import RecurrentPPO

model = RecurrentPPO(
    "MlpLstmPolicy",
    env,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    learning_rate=3e-4,
    verbose=1,
    tensorboard_log="./lstm_tensorboard/",
)
```

Key differences from standard PPO:
- Policy is `MlpLstmPolicy` — MLP feature extractor feeding an LSTM layer, then actor/critic heads
- Hidden state (`lstm_states`) must be passed and reset at episode boundaries
- Rollout buffer stores LSTM states per step; `episode_start` flags are required
- Default LSTM hidden size is 256; configurable via `policy_kwargs={"lstm_hidden_size": 128}`

---

## Observation Space

No changes needed to the base observation. Use the standard 12D sensor reading:
- 12 distance sensors (range 120 ft, normalized to [0,1]) — the LSTM will learn to integrate these over time

Optionally, use the 300ft sensor variant from ExpE to give the LSTM more lookahead. However, start with 120ft to isolate whether recurrence alone helps.

Speed in the observation: include it (13D, same as ExpD/SAC). The LSTM can learn to correlate speed-over-time with sensor-change-over-time.

**Do NOT use SPS wrapper at training time** — let the LSTM learn its own speed-conditioned steering strategy.

---

## Training Curriculum

Same three-stage curriculum as Stage1–3, but adapted for RecurrentPPO:

| Stage | Scenario | Steps | Notes |
|---|---|---|---|
| Stage 1 Easy | Low speed (10-25 mph), wide lanes, few obstacles | 100k | Build basic driving capability |
| Stage 2 Medium | Medium speed (10-45 mph), mixed lanes/obstacles | 150k | Introduce speed variation |
| Stage 3 Hard | Full range (10-75 mph), narrow lanes, high obstacles | 250k | Target the high-speed regime |

Total: ~500k steps. Use biased sampling in Stage 3 to oversample 20–55 mph.

**Important:** RecurrentPPO requires `n_steps` to be large enough for the LSTM to encounter multi-step temporal patterns. Use `n_steps=2048` (default), which gives ~10–12 complete episodes per rollout at 200 steps/episode.

---

## SEMBAS Scanning with RecurrentPPO

The SEMBAS client (`src/carlos_sembas_client.py`) needs a small patch to handle LSTM state:

```python
# In the evaluation loop, initialize lstm state
lstm_states = None
episode_start = np.ones((1,), dtype=bool)

obs, _ = env.reset()
for step in range(200):
    action, lstm_states = model.predict(
        obs, state=lstm_states, episode_start=episode_start, deterministic=True
    )
    episode_start = np.zeros((1,), dtype=bool)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

The SEMBAS server protocol (socket, pass/fail label) is unchanged — only the client evaluation function needs updating.

**New flag to add:** `--algo recurrent_ppo` (or `--algo rppo`) in the client argparser.

---

## Concrete Next Steps (ExpF)

1. **Install sb3-contrib** in the rl_venv313 environment:
   ```bash
   source ~/rl_venv313/bin/activate
   pip install sb3-contrib
   ```

2. **Write training script** `rlagent/train_expf_lstm.py`:
   - Stage 1–3 curriculum with RecurrentPPO
   - Save final model to `models/ppo_carlos_expf_lstm.zip`

3. **Patch the SEMBAS client** to handle `lstm_states` in the eval loop

4. **Run training** (~500k steps, similar wall time to previous experiments)

5. **Run SEMBAS scan** with the new checkpoint

6. **Compare:** If ceiling > 20 mph, LSTM is the mechanism. If ceiling holds, the constraint is in braking authority and the next direction is to modify the action space to include explicit deceleration.

---

## Expected Outcomes

| Scenario | Interpretation |
|---|---|
| Ceiling > 22 mph | LSTM temporal integration breaks the feedforward limit → architecture change works |
| Ceiling 20–22 mph | Marginal improvement, recurrence helps partially → combine with braking action space |
| Ceiling ≤ 20 mph | Braking authority is the true constraint, not policy architecture |

---

## If ExpF Fails: Action Space Redesign

If LSTM doesn't move the ceiling, the last unexplored hypothesis is that the CARLoS action space doesn't give the policy enough braking authority to shed speed before a high-speed scenario becomes unrecoverable. Potential fix: modify CARLoS to add an explicit brake action, or increase the deceleration coefficient in the dynamics. This would be ExpG.
