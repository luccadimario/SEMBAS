# Boundary Shape Analysis — ExpD and ExpE
**Date:** 2026-07-27

## Speed Ceiling Summary Across All 10 Experiments

| Experiment | Speed Range (mph) | Mean | All below 20 mph |
|---|---|---|---|
| Stage3-282k | 10.2–20.1 | 14.4 | 33/35 (94%) |
| Stage4 +30k | 10.2–20.1 | 14.4 | 33/35 (94%) |
| Stage4b +150k | 10.2–20.1 | 14.6 | 32/35 (91%) |
| ExpA | 10.2–20.1 | 14.5 | 33/35 (94%) |
| ExpB | 10.2–20.1 | 14.4 | 33/35 (94%) |
| ExpC | 10.2–20.1 | 14.6 | 34/35 (97%) |
| SAC | 10.1–19.6 | 14.7 | 35/35 (100%) |
| SPS-PPO | 10.2–19.8 | 14.5 | 34/35 (97%) |
| **ExpD** | **10.2–17.5** | **13.8** | **35/35 (100%)** |
| **ExpE** | **10.2–19.6** | **13.5** | **35/35 (100%)** |

---

## ExpD Boundary Analysis

**Configuration:** 13D obs (speed appended), SPS wrapper, weight surgery from Stage3, 500k steps biased curriculum

- **35 boundary points, 90 PASS, 122 FAIL** (212 total queries)
- **Speed (x2):** min=10.2 mph, max=17.5 mph, mean=13.8 mph, std=2.3 mph
- **Lane width (x0):** mean=12.0 ft (full range 10–14 ft, std=0.8 ft) — evenly distributed
- **Obstacles (x1):** mean=6.9 (range 4–10, std=2.4) — evenly distributed
- **All 35 boundary points fall below 19.75 mph**

**Interpretation:** The ceiling actually regressed to 17.5 mph from the 20.1 mph baseline. The boundary is shifted uniformly lower across all lane width and obstacle configurations, indicating the weight surgery disrupted the underlying representation without the additional 500k steps being sufficient to recover. The obstacle and lane width distributions of boundary points are uniform, confirming that speed is the dominant axis and the regression is not localized to any particular sub-region of the space.

---

## ExpE Boundary Analysis

**Configuration:** 300ft sensors, normalized obs [0,1], speed appended (13D), fresh curriculum, 500k steps

- **35 boundary points, 78 PASS, 137 FAIL** (215 total queries)
- **Speed (x2):** min=10.2 mph, max=19.6 mph, mean=13.5 mph, std=2.2 mph
- **Lane width (x0):** mean=12.6 ft (std=0.8 ft) — **notably skewed toward wider lanes (min 11.8 ft)**
- **Obstacles (x1):** mean=7.9 (std=2.7) — evenly distributed
- **All 35 boundary points fall below 19.75 mph**

**Interpretation:** The ceiling held at 19.6 mph, nearly identical to the PPO baseline. However, the Hausdorff distance of 0.3382 (vs 0.023–0.037 for PPO variants) indicates the *shape* of the boundary changed substantially — specifically, the boundary is biased toward wider lanes. This makes physical sense: with 300ft sensors, the policy has more reaction time but still cannot steer hard enough at speed. In narrow lanes (10–12 ft), the policy fails even faster because obstacles are closer to the vehicle path. The wider-lane bias is a genuine architectural artifact of the extended sensor range interacting with the lane geometry.

---

## Key Finding: Speed is the Dominant Boundary Axis

Across all 10 experiments, the boundary point distribution in lane_width and obstacles dimensions is approximately uniform — boundary crossings occur at all lane widths and obstacle counts. The speed axis is what concentrates: virtually every boundary point sits below 20 mph, regardless of lane width or obstacle count. This means:

1. **The boundary is nearly speed-monotone:** there is no combination of lane width + obstacle count that pushes the ceiling above 20 mph.
2. **The constraint is truly physical:** more reaction time (ExpE), more speed awareness (SAC, ExpD, SPS-PPO), and different curricula all fail to change this.
3. **ExpE's lane-width bias** is the first experiment to show any spatial structure beyond pure speed — worth investigating whether wider-lane scenarios at the boundary are qualitatively different (closer obstacles? different steering demands?).

---

## Implications for ExpF (LSTM)

The boundary's speed-monotone character means a recurrent policy must learn to reduce speed *before* high-speed scenarios become unrecoverable. This requires:
- Temporal integration of velocity cues across multiple steps
- Learning a brake-first / decelerate-and-steer policy rather than steer-only
- The LSTM hidden state must carry the "I am going fast, I need to slow down" signal

If ExpF (LSTM) produces a ceiling > 20 mph, the hidden state will be the mechanism. If it doesn't, the constraint is in the action space (braking authority) not the policy architecture.
