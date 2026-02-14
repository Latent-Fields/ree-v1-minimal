# Weekly Handoff - ree-v1-minimal - 2026-02-09

## Metadata
- week_of_utc: `2026-02-09`
- producer_repo: `ree-v1-minimal`
- producer_commit: `7e5de4d660374cd829afe071e78a16b3c8a47fac`
- generated_utc: `2026-02-14T03:00:00Z`

## Contract Sync
- ree_assembly_repo: `https://github.com/Latent-Fields/REE_assembly`
- ree_assembly_commit: `b718e670817a313f42f6c80ad027656b366af81d`
- contract_lock_path: `contracts/ree_assembly_contract_lock.v1.json`
- contract_lock_hash: `b78f7df32e80e1279f15c92e1378d68f11ee8b131762a98b609a7cccad6aa4c0`
- schema_version_set: `experiment_pack/v1, experiment_pack_metrics/v1, jepa_adapter_signals/v1`

## CI Gates
| gate | status | evidence |
| --- | --- | --- |
| schema_validation | PASS | `EXPERIMENT_PACK_ROOT=runs/bridging_qualification_056_058_059_060 python scripts/validate_experiment_packs.py` |
| seed_determinism | PASS | `python scripts/check_bridging_seed_determinism.py --seeds 11,29 --timestamp-utc 2026-02-14T03:00:00Z` |
| hook_surface_coverage | N/A | `N/A for parity/backstop lane` |
| remote_export_import | N/A | `N/A for parity/backstop lane` |

## Run-Pack Inventory
| experiment_type | run_id | seed | condition_or_scenario | status | evidence_direction | claim_ids_tested | failure_signatures | execution_mode | compute_backend | runtime_minutes | pack_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| commit_dual_error_channels | 2026-02-14T030000Z_commit-dual-error-channels_seed11_pre_post_split_streams | 11 | pre_post_split_streams | PASS | supports | MECH-060 |  | local | local_cpu | N/A | runs/bridging_qualification_056_058_059_060/commit_dual_error_channels/runs/2026-02-14T030000Z_commit-dual-error-channels_seed11_pre_post_split_streams |
| commit_dual_error_channels | 2026-02-14T030000Z_commit-dual-error-channels_seed11_single_error_stream | 11 | single_error_stream | FAIL | weakens | MECH-060 | threshold:pre_commit_error_signal_to_noise, threshold:post_commit_error_attribution_gain, threshold:cross_channel_leakage_rate, threshold:commitment_reversal_rate | local | local_cpu | N/A | runs/bridging_qualification_056_058_059_060/commit_dual_error_channels/runs/2026-02-14T030000Z_commit-dual-error-channels_seed11_single_error_stream |
| commit_dual_error_channels | 2026-02-14T030000Z_commit-dual-error-channels_seed29_pre_post_split_streams | 29 | pre_post_split_streams | PASS | supports | MECH-060 |  | local | local_cpu | N/A | runs/bridging_qualification_056_058_059_060/commit_dual_error_channels/runs/2026-02-14T030000Z_commit-dual-error-channels_seed29_pre_post_split_streams |
| commit_dual_error_channels | 2026-02-14T030000Z_commit-dual-error-channels_seed29_single_error_stream | 29 | single_error_stream | FAIL | weakens | MECH-060 | threshold:pre_commit_error_signal_to_noise, threshold:post_commit_error_attribution_gain, threshold:cross_channel_leakage_rate, threshold:commitment_reversal_rate | local | local_cpu | N/A | runs/bridging_qualification_056_058_059_060/commit_dual_error_channels/runs/2026-02-14T030000Z_commit-dual-error-channels_seed29_single_error_stream |
| jepa_anchor_ablation | 2026-02-14T030000Z_jepa-anchor-ablation_seed11_ema_anchor_off | 11 | ema_anchor_off | FAIL | weakens | MECH-058 | threshold:latent_prediction_error_mean, threshold:latent_prediction_error_p95, threshold:latent_rollout_consistency_rate, threshold:e1_e2_timescale_separation_ratio, threshold:representation_drift_rate | local | local_cpu | N/A | runs/bridging_qualification_056_058_059_060/jepa_anchor_ablation/runs/2026-02-14T030000Z_jepa-anchor-ablation_seed11_ema_anchor_off |
| jepa_anchor_ablation | 2026-02-14T030000Z_jepa-anchor-ablation_seed11_ema_anchor_on | 11 | ema_anchor_on | PASS | supports | MECH-058 |  | local | local_cpu | N/A | runs/bridging_qualification_056_058_059_060/jepa_anchor_ablation/runs/2026-02-14T030000Z_jepa-anchor-ablation_seed11_ema_anchor_on |
| jepa_anchor_ablation | 2026-02-14T030000Z_jepa-anchor-ablation_seed29_ema_anchor_off | 29 | ema_anchor_off | FAIL | weakens | MECH-058 | threshold:latent_prediction_error_mean, threshold:latent_prediction_error_p95, threshold:latent_rollout_consistency_rate, threshold:e1_e2_timescale_separation_ratio, threshold:representation_drift_rate | local | local_cpu | N/A | runs/bridging_qualification_056_058_059_060/jepa_anchor_ablation/runs/2026-02-14T030000Z_jepa-anchor-ablation_seed29_ema_anchor_off |
| jepa_anchor_ablation | 2026-02-14T030000Z_jepa-anchor-ablation_seed29_ema_anchor_on | 29 | ema_anchor_on | PASS | supports | MECH-058 |  | local | local_cpu | N/A | runs/bridging_qualification_056_058_059_060/jepa_anchor_ablation/runs/2026-02-14T030000Z_jepa-anchor-ablation_seed29_ema_anchor_on |
| jepa_uncertainty_channels | 2026-02-14T030000Z_jepa-uncertainty-channels_seed11_deterministic_plus_dispersion | 11 | deterministic_plus_dispersion | FAIL | weakens | MECH-059 | threshold:latent_prediction_error_mean, threshold:latent_uncertainty_calibration_error, threshold:precision_input_completeness_rate, threshold:uncertainty_coverage_rate | local | local_cpu | N/A | runs/bridging_qualification_056_058_059_060/jepa_uncertainty_channels/runs/2026-02-14T030000Z_jepa-uncertainty-channels_seed11_deterministic_plus_dispersion |
| jepa_uncertainty_channels | 2026-02-14T030000Z_jepa-uncertainty-channels_seed11_explicit_uncertainty_head | 11 | explicit_uncertainty_head | PASS | supports | MECH-059 |  | local | local_cpu | N/A | runs/bridging_qualification_056_058_059_060/jepa_uncertainty_channels/runs/2026-02-14T030000Z_jepa-uncertainty-channels_seed11_explicit_uncertainty_head |
| jepa_uncertainty_channels | 2026-02-14T030000Z_jepa-uncertainty-channels_seed29_deterministic_plus_dispersion | 29 | deterministic_plus_dispersion | FAIL | weakens | MECH-059 | threshold:latent_prediction_error_mean, threshold:latent_uncertainty_calibration_error, threshold:precision_input_completeness_rate, threshold:uncertainty_coverage_rate | local | local_cpu | N/A | runs/bridging_qualification_056_058_059_060/jepa_uncertainty_channels/runs/2026-02-14T030000Z_jepa-uncertainty-channels_seed29_deterministic_plus_dispersion |
| jepa_uncertainty_channels | 2026-02-14T030000Z_jepa-uncertainty-channels_seed29_explicit_uncertainty_head | 29 | explicit_uncertainty_head | PASS | supports | MECH-059 |  | local | local_cpu | N/A | runs/bridging_qualification_056_058_059_060/jepa_uncertainty_channels/runs/2026-02-14T030000Z_jepa-uncertainty-channels_seed29_explicit_uncertainty_head |
| trajectory_integrity | 2026-02-14T030000Z_trajectory-integrity_seed11_trajectory_first_ablated | 11 | trajectory_first_ablated | FAIL | weakens | MECH-056 | threshold:ledger_edit_detected_count, threshold:explanation_policy_divergence_rate, threshold:domination_lock_in_events | local | local_cpu | N/A | runs/bridging_qualification_056_058_059_060/trajectory_integrity/runs/2026-02-14T030000Z_trajectory-integrity_seed11_trajectory_first_ablated |
| trajectory_integrity | 2026-02-14T030000Z_trajectory-integrity_seed11_trajectory_first_enabled | 11 | trajectory_first_enabled | PASS | supports | MECH-056 |  | local | local_cpu | N/A | runs/bridging_qualification_056_058_059_060/trajectory_integrity/runs/2026-02-14T030000Z_trajectory-integrity_seed11_trajectory_first_enabled |
| trajectory_integrity | 2026-02-14T030000Z_trajectory-integrity_seed29_trajectory_first_ablated | 29 | trajectory_first_ablated | FAIL | weakens | MECH-056 | threshold:ledger_edit_detected_count, threshold:explanation_policy_divergence_rate, threshold:domination_lock_in_events | local | local_cpu | N/A | runs/bridging_qualification_056_058_059_060/trajectory_integrity/runs/2026-02-14T030000Z_trajectory-integrity_seed29_trajectory_first_ablated |
| trajectory_integrity | 2026-02-14T030000Z_trajectory-integrity_seed29_trajectory_first_enabled | 29 | trajectory_first_enabled | PASS | supports | MECH-056 |  | local | local_cpu | N/A | runs/bridging_qualification_056_058_059_060/trajectory_integrity/runs/2026-02-14T030000Z_trajectory-integrity_seed29_trajectory_first_enabled |

## Claim Summary
| claim_id | runs_added | supports | weakens | mixed | unknown | recurring_failure_signatures |
| --- | --- | --- | --- | --- | --- | --- |
| MECH-056 | 4 | 2 | 2 | 0 | 0 | threshold:domination_lock_in_events, threshold:explanation_policy_divergence_rate, threshold:ledger_edit_detected_count |
| MECH-058 | 4 | 2 | 2 | 0 | 0 | threshold:e1_e2_timescale_separation_ratio, threshold:latent_prediction_error_mean, threshold:latent_prediction_error_p95, threshold:latent_rollout_consistency_rate, threshold:representation_drift_rate |
| MECH-059 | 4 | 2 | 2 | 0 | 0 | threshold:latent_prediction_error_mean, threshold:latent_uncertainty_calibration_error, threshold:precision_input_completeness_rate, threshold:uncertainty_coverage_rate |
| MECH-060 | 4 | 2 | 2 | 0 | 0 | threshold:commitment_reversal_rate, threshold:cross_channel_leakage_rate, threshold:post_commit_error_attribution_gain, threshold:pre_commit_error_signal_to_noise |

## Open Blockers
- Parity note vs latest ree-v2: disagreement on overlapping claims (MECH-056(ree-v1-minimal=mixed, ree-v2=supports), MECH-058(ree-v1-minimal=mixed, ree-v2=supports), MECH-059(ree-v1-minimal=mixed, ree-v2=supports), MECH-060(ree-v1-minimal=mixed, ree-v2=supports)); source=/Users/dgolden/Documents/GitHub/ree-v2/evidence/planning/weekly_handoff/latest.md.
- No additional blockers reported this cycle.

## Local Compute Options Watch
- local_options_last_updated_utc: `N/A`
- rolling_3mo_cloud_spend_eur: `N/A`
- local_blocked_sessions_this_week: `N/A`
- recommended_local_action: `N/A`
- rationale: `N/A for ree-v1-minimal parity/backstop lane unless explicitly in scope.`
