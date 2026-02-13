#!/usr/bin/env python3
"""Run dispatch claim probes and emit Experiment Pack v1 artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.pack_writer import (
    ExperimentPackWriter,
    deterministic_run_id,
    normalize_timestamp_utc,
    resolve_output_root,
    stable_config_hash,
)


REQUIRED_EXPERIMENTS = ("claim_probe_mech_053", "claim_probe_mech_054")
DEFAULT_SEEDS = (11, 29)
RUNNER_NAME = "ree-v1-claim-probe-dispatch"
RUNNER_VERSION = "dispatch.v1"


def load_suites() -> dict[str, Any]:
    suites_path = Path("experiments/suites.json")
    with suites_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_probe_mech_053(
    seed: int, config: Dict[str, Any]
) -> Tuple[Dict[str, Union[float, int]], str]:
    rng = random.Random(seed)
    num_trials = int(config.get("num_trials", 240))
    spike_probability = float(config.get("spike_probability", 0.4))

    spike_trials = 0
    non_spike_trials = 0
    spike_commit_count = 0
    non_spike_commit_count = 0
    unsafe_spike_commit_count = 0
    suppression_total = 0.0

    for _ in range(num_trials):
        spike = rng.random() < spike_probability
        baseline_commit = 0.62 + (0.1 * rng.random())
        if spike:
            spike_trials += 1
            gate_strength = 0.55 + (0.35 * rng.random())
            commit_probability = max(0.01, baseline_commit - (0.45 * gate_strength))
            committed = 1 if rng.random() < commit_probability else 0
            spike_commit_count += committed
            suppression_total += baseline_commit - commit_probability
            if gate_strength > 0.75 and committed:
                unsafe_spike_commit_count += 1
        else:
            non_spike_trials += 1
            commit_probability = min(0.99, baseline_commit + 0.05 * rng.random())
            committed = 1 if rng.random() < commit_probability else 0
            non_spike_commit_count += committed

    spike_commit_rate = (spike_commit_count / spike_trials) if spike_trials else 0.0
    non_spike_commit_rate = (non_spike_commit_count / non_spike_trials) if non_spike_trials else 0.0
    suppression_gap = non_spike_commit_rate - spike_commit_rate
    unsafe_spike_commit_rate = (
        unsafe_spike_commit_count / spike_trials if spike_trials else 0.0
    )
    suppression_mean = suppression_total / spike_trials if spike_trials else 0.0

    metrics = {
        "spike_trial_count": spike_trials,
        "non_spike_trial_count": non_spike_trials,
        "spike_commit_rate": spike_commit_rate,
        "non_spike_commit_rate": non_spike_commit_rate,
        "suppression_gap": suppression_gap,
        "suppression_mean": suppression_mean,
        "unsafe_spike_commit_rate": unsafe_spike_commit_rate,
        "fatal_error_count": 0,
    }

    if suppression_gap >= 0.15 and unsafe_spike_commit_rate <= 0.12:
        direction = "supports"
    elif suppression_gap <= 0.05:
        direction = "weakens"
    else:
        direction = "mixed"
    return metrics, direction


def run_probe_mech_054(
    seed: int, config: Dict[str, Any]
) -> Tuple[Dict[str, Union[float, int]], str]:
    rng = random.Random(seed)
    num_events = int(config.get("num_events", 320))
    harm_event_probability = float(config.get("harm_event_probability", 0.45))

    negative_precision_sum = 0.0
    positive_precision_sum = 0.0
    harm_update_sum = 0.0
    benefit_update_sum = 0.0
    direction_consistent = 0
    harm_count = 0
    benefit_count = 0

    for _ in range(num_events):
        is_harm_event = rng.random() < harm_event_probability
        magnitude = 0.1 + (0.9 * rng.random())
        noise = (rng.random() - 0.5) * 0.08

        if is_harm_event:
            harm_count += 1
            precision = 0.68 + 0.2 * magnitude + noise
            update = magnitude * precision
            negative_precision_sum += precision
            harm_update_sum += update
            if update > 0:
                direction_consistent += 1
        else:
            benefit_count += 1
            precision = 0.45 + 0.16 * magnitude + noise
            update = magnitude * precision
            positive_precision_sum += precision
            benefit_update_sum += update
            if update > 0:
                direction_consistent += 1

    negative_precision_mean = negative_precision_sum / harm_count if harm_count else 0.0
    positive_precision_mean = positive_precision_sum / benefit_count if benefit_count else 0.0
    precision_gap = negative_precision_mean - positive_precision_mean
    harm_update_mean = harm_update_sum / harm_count if harm_count else 0.0
    benefit_update_mean = benefit_update_sum / benefit_count if benefit_count else 0.0
    direction_consistency_rate = direction_consistent / num_events if num_events else 0.0

    metrics = {
        "harm_event_count": harm_count,
        "benefit_event_count": benefit_count,
        "negative_precision_mean": negative_precision_mean,
        "positive_precision_mean": positive_precision_mean,
        "precision_gap_negative_minus_positive": precision_gap,
        "harm_update_mean": harm_update_mean,
        "benefit_update_mean": benefit_update_mean,
        "direction_consistency_rate": direction_consistency_rate,
        "fatal_error_count": 0,
    }

    if precision_gap >= 0.12 and harm_update_mean > benefit_update_mean:
        direction = "supports"
    elif precision_gap <= 0.02:
        direction = "weakens"
    else:
        direction = "mixed"
    return metrics, direction


def build_summary_markdown(
    experiment_type: str,
    run_id: str,
    timestamp_utc: str,
    seed: int,
    claim_ids: list[str],
    direction: str,
    metrics: Dict[str, Union[float, int]],
) -> str:
    lines = [
        "# Experiment Run Summary",
        "",
        "## Scenario",
        f"- experiment_type: `{experiment_type}`",
        f"- run_id: `{run_id}`",
        f"- timestamp_utc: `{timestamp_utc}`",
        f"- seed: `{seed}`",
        "- claim_ids_tested: " + ", ".join(f"`{claim_id}`" for claim_id in claim_ids),
        "",
        "## Outcome",
        "- status: **PASS**",
        f"- evidence_direction: `{direction}`",
        "",
        "## Key Metrics",
    ]
    for key, value in sorted(metrics.items()):
        if key == "fatal_error_count":
            continue
        if isinstance(value, float):
            lines.append(f"- {key}: {value:.6f}")
        else:
            lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "- Direction is determined by pre-registered threshold checks in this probe harness.",
        ]
    )
    return "\n".join(lines)


def run_dispatch(
    output_root: Optional[str],
    seeds: List[int],
    timestamp_utc: Optional[str],
) -> List[Dict[str, Any]]:
    suites = load_suites()
    for experiment in REQUIRED_EXPERIMENTS:
        if experiment not in suites:
            raise ValueError(f"Missing suite definition: {experiment}")

    normalized_ts = normalize_timestamp_utc(timestamp_utc)
    writer = ExperimentPackWriter(
        output_root=resolve_output_root(output_root),
        repo_root=Path(".").resolve(),
        runner_name=RUNNER_NAME,
        runner_version=RUNNER_VERSION,
    )

    report_rows: List[Dict[str, Any]] = []
    for experiment in REQUIRED_EXPERIMENTS:
        suite = suites[experiment]
        claim_ids = [str(x) for x in suite.get("claim_ids_tested", [])]
        evidence_class = str(suite.get("evidence_class", "simulation"))
        config = suite.get("probe_config", {})
        probe_type = suite.get("probe_type")

        for seed in seeds:
            run_id = deterministic_run_id(experiment, seed, normalized_ts)
            if probe_type == "mech_053":
                metrics, direction = run_probe_mech_053(seed, config)
                key_metrics = {
                    "suppression_gap": metrics["suppression_gap"],
                    "unsafe_spike_commit_rate": metrics["unsafe_spike_commit_rate"],
                }
            elif probe_type == "mech_054":
                metrics, direction = run_probe_mech_054(seed, config)
                key_metrics = {
                    "precision_gap_negative_minus_positive": metrics[
                        "precision_gap_negative_minus_positive"
                    ],
                    "harm_update_mean": metrics["harm_update_mean"],
                    "benefit_update_mean": metrics["benefit_update_mean"],
                }
            else:
                raise ValueError(f"Unsupported probe_type '{probe_type}' for suite '{experiment}'")

            summary_markdown = build_summary_markdown(
                experiment_type=experiment,
                run_id=run_id,
                timestamp_utc=normalized_ts,
                seed=seed,
                claim_ids=claim_ids,
                direction=direction,
                metrics=metrics,
            )
            scenario = {
                "name": experiment,
                "seed": seed,
                "config_hash": stable_config_hash(config),
                "dispatch_item": "EXP-0001" if experiment.endswith("053") else "EXP-0002",
            }

            emitted = writer.write_pack(
                experiment_type=experiment,
                run_id=run_id,
                timestamp_utc=normalized_ts,
                status="PASS",
                metrics_values=metrics,
                summary_markdown=summary_markdown,
                scenario=scenario,
                failure_signatures=[],
                claim_ids_tested=claim_ids,
                evidence_class=evidence_class,
                evidence_direction=direction,
            )

            report_rows.append(
                {
                    "dispatch_item": scenario["dispatch_item"],
                    "experiment_type": experiment,
                    "run_id": run_id,
                    "seed": seed,
                    "status": "PASS",
                    "evidence_direction": direction,
                    "key_metrics": key_metrics,
                    "run_dir": str(emitted.run_dir),
                }
            )
    return report_rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run MECH-053/054 dispatch claim probes and emit Experiment Pack v1 runs."
    )
    parser.add_argument(
        "--output-root",
        default="runs/dispatch_claim_probes",
        help="Root directory for emitted Experiment Pack runs.",
    )
    parser.add_argument(
        "--seeds",
        default="11,29",
        help="Comma-separated seed list (must contain at least 2 distinct seeds).",
    )
    parser.add_argument(
        "--timestamp-utc",
        default=None,
        help="Optional RFC3339 UTC timestamp for deterministic run IDs.",
    )
    args = parser.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if len(set(seeds)) < 2:
        raise ValueError("At least 2 distinct seeds are required.")

    report_rows = run_dispatch(
        output_root=args.output_root,
        seeds=seeds,
        timestamp_utc=args.timestamp_utc,
    )

    print("Dispatch Run Report")
    print("===================")
    for row in report_rows:
        print(
            f"{row['dispatch_item']} | {row['experiment_type']} | seed={row['seed']} | "
            f"run_id={row['run_id']} | status={row['status']} | "
            f"direction={row['evidence_direction']} | key_metrics={row['key_metrics']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
