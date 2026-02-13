# Experiment Run Summary

## Scenario
- suite: `claim_probe_mech_056`
- run_id: `2026-02-13T090000Z_example`
- seed: `7`
- timestamp_utc: `2026-02-13T09:00:00Z`
- claim_ids_tested: `MECH-056`
- evidence_class: `simulation`
- evidence_direction: `supports`

## Outcome
- status: **PASS**
- steps_survived: 42
- total_harm: 0.350000
- final_residue: 0.420000
- final_health: 0.710000
- final_energy: 0.440000

## MECH-056 Escalation Trace
- channel_escalation_order_observed: `trajectory_commit -> perceptual_sampling -> structural_consolidation`
- trigger_rationale_perceptual_sampling: activated after harm/collision cues (harm_events=2, hazard_events=1, collision_events=1).
- trigger_rationale_structural_consolidation: activated to consolidate persistent bias (final_residue=0.420000, structural_bias_rate=0.014762).

## Interpretation
- run passed known stop checks and did not trigger known signatures.
