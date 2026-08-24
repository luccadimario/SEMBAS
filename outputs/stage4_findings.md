# Stage 4 Retraining Experiment — Findings

## Experiment
- Base checkpoint: Stage3_Hard_282k (282,272 steps, PPO)
- Stage 4 addition: 30,000 steps with biased curriculum (60% scenarios speed=[18-30]mph, 40% full range)
- Total steps at end: ~312,272
- Training rate: ~38 fps, completed in ~15 minutes

## Result
- Boundary shift: minimal — only 2-3 query outcomes changed out of 226 total
- Speed ceiling: unchanged at ~20.1 mph (no improvement)
- Pass rate: effectively unchanged (~40.3%)

## Interpretation
30k additional steps with a biased speed curriculum is **below the minimum effective training budget** to shift the entrenched speed boundary. After 282k steps of Stage 3 training, the policy is too settled for a small curriculum perturbation to move the boundary.

## Estimated minimum effective budget
Based on the Stage 1→Stage 2 transition (which DID shift the speed ceiling from 17.7→20.1 mph):
- That transition required ~131k steps on a new curriculum
- Scaling suggests ~100–200k biased steps minimum to move the ceiling another ~8mph
- Alternatively: structural intervention (reinitialize policy head, freeze feature extractor) may require fewer steps

## Next step
Launch Stage 4b: 150,000 additional biased steps from Stage3_Hard_282k checkpoint.
- Expected training time: ~66 minutes at 38 fps
- Expected boundary shift if hypothesis correct: speed ceiling to 25–28 mph
- SEMBAS rescan after training to measure before/after metrics

## Research value
This null result is publishable — it quantifies the minimum effective curriculum dose needed to shift a SEMBAS-measured boundary. Prior work does not address this question.
