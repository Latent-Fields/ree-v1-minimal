# Experiment Run Summary

## Scenario
- suite: `baseline_explicit_cost`
- run_id: `2026-02-13T060000Z_baseline-explicit-cost_seed0`
- seed: `0`
- timestamp_utc: `2026-02-13T06:00:00Z`

## Outcome
- status: **PASS**
- steps_survived: 25
- total_harm: 0.000000
- final_residue: 0.000000

## Interpretation
- run passed known stop checks and did not trigger known signatures.

## MECH-056 Escalation Trace
- channel_escalation_order_observed: `trajectory_commit -> perceptual_sampling -> structural_consolidation`
- trigger_rationale_perceptual_sampling: activated after harm/collision cues (harm_events=0, hazard_events=0, collision_events=0).
- trigger_rationale_structural_consolidation: activated to consolidate persistent bias (final_residue=0.000000, structural_bias_rate=0.003200).
