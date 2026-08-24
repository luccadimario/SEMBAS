# ExpD SEMBAS Scan Analysis

**Date:** 2026-07-24  
**Checkpoint:** `ppo_carlos_expd_final.zip`  
**Flags:** `--sps --speed-aware` (13D obs, SPS wrapper, weight bridge from Stage3)

---

## Results

| Metric | ExpD | Stage3 Baseline | Delta |
|--------|------|----------------|-------|
| Pass rate | 42.5% | 40.3% | +2.2% |
| Speed ceiling | 17.1 mph | 19.8 mph | **−2.7 mph** |
| Hausdorff vs Stage3 | 0.1117 | — | (vs ~0.024 for PPO variants) |
| Boundary speed range | 10.2–17.5 mph | ~10–20 mph | shifted lower |
| Boundary points | 35 | 35 | — |
| Total queries | 212 | 212 | — |

---

## Interpretation

**This is Experiment 10: another null result, and arguably a slight regression.**

The speed ceiling did not advance past 20 mph — it dropped from 19.8 to 17.1 mph. All 35 boundary points sit between 10.2 and 17.5 mph, meaning the policy's pass/fail transition region is actually more conservative than the baseline.

The Hausdorff distance of 0.1117 is notably higher than the 0.023–0.037 range of all previous PPO variants, indicating the boundary shape shifted meaningfully but not in the direction we wanted. This is consistent with the weight surgery disrupting the Stage3 representation without the training recovering it fully within 500k steps.

**Root cause still holds.** The identifiability diagnosis from the step-level tracer remains the correct explanation: the policy cannot distinguish "close obstacle at 15 mph" from "close obstacle at 40 mph" based on sensor readings alone. Adding speed to the observation (ExpD) was necessary but not sufficient — the policy needed enough training signal to learn to modulate steering _and_ braking together at high speeds, and 500k steps from a biased checkpoint may not have been enough.

**SPS wrapper effect.** The SPS wrapper at evaluation time scales down steering proportionally to speed, but if the underlying policy was not trained to rely on SPS co-adaptation, this could actually impair performance at speeds the policy was previously navigating on the boundary.

---

## Recommended Next Step: Run ExpE

ExpE is the stronger architectural intervention:

- **Extended sensors: 120 ft → 300 ft** — gives 2.5× more reaction time at high speeds (51 steps lookahead at 40 mph vs 20 steps)
- **Normalised obs [0,1] + speed appended** — cleaner 13D input, avoids the raw-feet scale issue
- **Fresh curriculum from scratch** — no weight surgery disruption; builds the representation from the ground up for the extended sensor regime

```bash
cd /mnt/c/Users/lucca/Documents/SEMBAS-RL

# Train ExpE (~500k steps, may take several hours)
python /mnt/c/Users/lucca/Documents/CARLoS-Agents/rlagent/train_expe_extended_sensors.py

# Then scan
python3 src/sembas_server.py --output boundary_expe_final.json &
SERVER_PID=$!
sleep 2

python3 src/carlos_sembas_client.py \
  --checkpoint /mnt/c/Users/lucca/Documents/CARLoS-Agents/models/ppo_carlos_expe_final.zip \
  --algo ppo --sps --sensor-length 300

wait $SERVER_PID
```

If ExpE also returns a null, the conclusion for the abstract/paper is clear: the structural ODD boundary at ~17–20 mph is an inherent property of the 120-ft sensor architecture and the CARLoS dynamics, and boundary-aware scan evidence (SEMBAS) was the mechanism that distinguished this from a soft training artifact.
