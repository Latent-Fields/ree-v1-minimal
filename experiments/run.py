import argparse
import json
from pathlib import Path
import random
import traceback
from typing import Optional

import torch

from experiments.metrics import compute_metrics_values, compute_summary
from experiments.pack_writer import (
    EVIDENCE_DIRECTIONS,
    ExperimentPackWriter,
    deterministic_run_id,
    resolve_output_root,
    stable_config_hash,
    normalize_timestamp_utc,
)
from ree_core import __version__ as REE_VERSION
from ree_core.agent import REEAgent
from ree_core.environment.grid_world import GridWorld

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency in some local envs
    np = None


def _clean_claim_ids(claim_ids: object) -> list[str]:
    if not isinstance(claim_ids, list):
        return []
    cleaned: list[str] = []
    for claim_id in claim_ids:
        value = str(claim_id).strip()
        if value and value not in cleaned:
            cleaned.append(value)
    return cleaned


def _resolve_claim_ids(suite_name: str, suite: dict, claim_ids_override: Optional[list[str]]) -> list[str]:
    raw_claim_ids = (
        claim_ids_override
        if claim_ids_override is not None
        else suite.get("claim_ids_tested", [])
    )
    claim_ids = _clean_claim_ids(raw_claim_ids)
    if claim_ids:
        return claim_ids
    raise ValueError(
        f"suite '{suite_name}' is missing claim_ids_tested; configure it in experiments/suites.json or pass --claim-id."
    )


def _resolve_evidence_class(suite: dict, evidence_class_override: Optional[str]) -> str:
    candidate = evidence_class_override
    if candidate is None:
        candidate = suite.get("evidence_class")
    if candidate is None:
        return "simulation"
    cleaned = str(candidate).strip()
    return cleaned or "simulation"


def _resolve_evidence_direction(
    suite: dict,
    status: str,
    evidence_direction_override: Optional[str],
) -> str:
    candidate = evidence_direction_override
    if candidate is None:
        suite_candidate = suite.get("evidence_direction")
        if suite_candidate is not None:
            candidate = str(suite_candidate)

    if candidate is None or not str(candidate).strip():
        return "supports" if status == "PASS" else "weakens"

    cleaned = str(candidate).strip().lower()
    if cleaned not in EVIDENCE_DIRECTIONS:
        expected = ", ".join(sorted(EVIDENCE_DIRECTIONS))
        raise ValueError(
            f"invalid evidence_direction '{candidate}' (expected one of: {expected})"
        )
    return cleaned


def load_suites() -> dict:
    suites_path = Path(__file__).resolve().parent / "suites.json"
    with suites_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def apply_overrides(agent: REEAgent, suite: dict) -> None:
    overrides = suite.get("overrides", {})
    for component_name, config_overrides in overrides.items():
        if not isinstance(config_overrides, dict):
            continue
        component = getattr(agent, component_name, None)
        if component is None:
            continue
        target = getattr(component, "config", component)
        for field_name, value in config_overrides.items():
            if hasattr(target, field_name):
                setattr(target, field_name, value)


def run_experiment_episode(agent: REEAgent, env: GridWorld, max_steps: int) -> dict:
    agent.reset()
    observation = env.reset()

    total_harm = 0.0
    total_reward = 0.0
    harm_event_count = 0
    hazard_event_count = 0
    collision_event_count = 0
    resource_event_count = 0

    done = False
    final_info = {"health": env.agent.health, "energy": env.agent.energy}
    steps = 0

    for _ in range(max_steps):
        action = agent.act(observation)
        observation, harm_signal, done, info = env.step(action)
        agent.update_residue(harm_signal)
        if agent.should_integrate():
            agent.offline_integration()

        if harm_signal < 0:
            total_harm += abs(float(harm_signal))
            harm_event_count += 1
        else:
            total_reward += float(harm_signal)

        event = info.get("event")
        if event == "hazard":
            hazard_event_count += 1
        elif event == "collision":
            collision_event_count += 1
        elif event == "resource":
            resource_event_count += 1

        final_info = info
        steps += 1
        if done:
            break

    residue_stats = agent.get_residue_statistics()
    return {
        "steps": steps,
        "max_steps": max_steps,
        "done": int(done),
        "total_harm": total_harm,
        "total_reward": total_reward,
        "final_residue": float(residue_stats["total_residue"].item()),
        "final_health": float(final_info.get("health", 0.0)),
        "final_energy": float(final_info.get("energy", 0.0)),
        "harm_event_count": harm_event_count,
        "hazard_event_count": hazard_event_count,
        "collision_event_count": collision_event_count,
        "resource_event_count": resource_event_count,
        "fatal_error_count": 0,
    }


def known_failure_signatures(result: dict) -> list[str]:
    signatures: list[str] = []
    if int(result.get("fatal_error_count", 0)) > 0:
        return ["fatal_error"]
    if float(result.get("final_health", 1.0)) <= 0.0:
        signatures.append("agent_health_depleted")
    if float(result.get("final_energy", 1.0)) <= 0.0:
        signatures.append("agent_energy_depleted")
    return signatures


def build_summary_markdown(
    suite_name: str,
    suite: dict,
    seed: int,
    run_id: str,
    timestamp_utc: str,
    status: str,
    claim_ids_tested: list[str],
    evidence_class: str,
    evidence_direction: str,
    result: dict,
    failure_signatures: list[str],
) -> str:
    summary = compute_summary(result)
    lines = [
        "# Experiment Run Summary",
        "",
        "## Scenario",
        f"- suite: `{suite_name}`",
        f"- run_id: `{run_id}`",
        f"- seed: `{seed}`",
        f"- timestamp_utc: `{timestamp_utc}`",
        f"- description: {suite.get('description', 'n/a')}",
        "- claim_ids_tested: " + ", ".join(f"`{claim_id}`" for claim_id in claim_ids_tested),
        f"- evidence_class: `{evidence_class}`",
        f"- evidence_direction: `{evidence_direction}`",
        "",
        "## Outcome",
        f"- status: **{status}**",
        f"- steps_survived: {summary['steps_survived']}",
        f"- total_harm: {summary['total_harm']:.6f}",
        f"- final_residue: {summary['final_residue']:.6f}",
        f"- final_health: {result.get('final_health', 0.0):.6f}",
        f"- final_energy: {result.get('final_energy', 0.0):.6f}",
        "",
        "## Interpretation",
    ]
    if failure_signatures:
        lines.append(
            "- run failed due to known failure signatures: "
            + ", ".join(f"`{sig}`" for sig in failure_signatures)
        )
    else:
        lines.append("- run passed known stop checks and did not trigger known signatures.")
    return "\n".join(lines)


def execute_experiment(
    suite_name: str,
    seed: int = 0,
    max_steps: int = 200,
    output_root: Optional[str] = None,
    run_id: Optional[str] = None,
    timestamp_utc: Optional[str] = None,
    claim_ids_tested: Optional[list[str]] = None,
    evidence_class: Optional[str] = None,
    evidence_direction: Optional[str] = None,
    runner_name: str = "ree-v1-minimal-harness",
    runner_version: str = REE_VERSION,
) -> Path:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)

    suites = load_suites()
    if suite_name not in suites:
        known = ", ".join(sorted(suites.keys()))
        raise ValueError(f"unknown suite '{suite_name}'. Available suites: {known}")
    suite = suites[suite_name]
    resolved_claim_ids = _resolve_claim_ids(suite_name, suite, claim_ids_tested)
    resolved_evidence_class = _resolve_evidence_class(suite, evidence_class)

    normalized_timestamp = normalize_timestamp_utc(timestamp_utc)
    resolved_run_id = run_id or deterministic_run_id(suite_name, seed, normalized_timestamp)

    scenario = {
        "name": suite_name,
        "seed": seed,
        "config_hash": stable_config_hash(suite),
        "max_steps": max_steps,
    }

    traces_dir = None
    trace_text = None
    try:
        env = GridWorld(seed=seed)
        agent = REEAgent.from_config(
            observation_dim=env.observation_dim,
            action_dim=env.action_dim,
            latent_dim=64,
        )
        apply_overrides(agent, suite)
        result = run_experiment_episode(agent, env, max_steps=max_steps)
    except Exception:
        traces_dir = "traces"
        trace_text = traceback.format_exc()
        result = {
            "steps": 0,
            "max_steps": max_steps,
            "done": 1,
            "total_harm": 0.0,
            "total_reward": 0.0,
            "final_residue": 0.0,
            "final_health": 0.0,
            "final_energy": 0.0,
            "harm_event_count": 0,
            "hazard_event_count": 0,
            "collision_event_count": 0,
            "resource_event_count": 0,
            "fatal_error_count": 1,
        }

    failure_signatures = known_failure_signatures(result)
    status = "FAIL" if failure_signatures else "PASS"
    resolved_evidence_direction = _resolve_evidence_direction(suite, status, evidence_direction)
    metrics_values = compute_metrics_values(result)

    summary_markdown = build_summary_markdown(
        suite_name=suite_name,
        suite=suite,
        seed=seed,
        run_id=resolved_run_id,
        timestamp_utc=normalized_timestamp,
        status=status,
        claim_ids_tested=resolved_claim_ids,
        evidence_class=resolved_evidence_class,
        evidence_direction=resolved_evidence_direction,
        result=result,
        failure_signatures=failure_signatures,
    )

    repo_root = Path(__file__).resolve().parents[1]
    writer = ExperimentPackWriter(
        output_root=resolve_output_root(output_root),
        repo_root=repo_root,
        runner_name=runner_name,
        runner_version=runner_version,
    )
    emitted = writer.write_pack(
        experiment_type=suite_name,
        run_id=resolved_run_id,
        timestamp_utc=normalized_timestamp,
        status=status,
        metrics_values=metrics_values,
        summary_markdown=summary_markdown,
        scenario=scenario,
        failure_signatures=failure_signatures,
        claim_ids_tested=resolved_claim_ids,
        evidence_class=resolved_evidence_class,
        evidence_direction=resolved_evidence_direction,
        traces_dir=traces_dir,
    )

    if trace_text and traces_dir:
        trace_path = emitted.run_dir / traces_dir / "fatal_error.txt"
        trace_path.write_text(trace_text, encoding="utf-8")

    return emitted.run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run REE experiment suite and emit Experiment Pack v1.")
    parser.add_argument("--suite", required=True, help="Experiment suite name from experiments/suites.json.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--max-steps", type=int, default=200, help="Maximum episode steps.")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Root output path. Defaults to REE_EXPERIMENT_OUTPUT_ROOT or runs.",
    )
    parser.add_argument("--run-id", default=None, help="Optional explicit run_id.")
    parser.add_argument(
        "--timestamp-utc",
        default=None,
        help="Optional RFC3339 timestamp used for manifest timestamp_utc and deterministic run_id.",
    )
    parser.add_argument(
        "--claim-id",
        action="append",
        dest="claim_ids",
        default=None,
        help="Claim ID tested by this run. Pass multiple times for multiple claim IDs.",
    )
    parser.add_argument(
        "--evidence-class",
        default=None,
        help="Evidence class for claim linkage (e.g., simulation, ablation, regression).",
    )
    parser.add_argument(
        "--evidence-direction",
        choices=sorted(EVIDENCE_DIRECTIONS),
        default=None,
        help="Evidence direction for claim linkage. Defaults to supports/weakens inferred from PASS/FAIL.",
    )
    parser.add_argument(
        "--runner-name",
        default="ree-v1-minimal-harness",
        help="Runner name written to manifest.runner.name.",
    )
    parser.add_argument(
        "--runner-version",
        default=REE_VERSION,
        help="Runner version written to manifest.runner.version.",
    )
    args = parser.parse_args()

    run_dir = execute_experiment(
        suite_name=args.suite,
        seed=args.seed,
        max_steps=args.max_steps,
        output_root=args.output_root,
        run_id=args.run_id,
        timestamp_utc=args.timestamp_utc,
        claim_ids_tested=args.claim_ids,
        evidence_class=args.evidence_class,
        evidence_direction=args.evidence_direction,
        runner_name=args.runner_name,
        runner_version=args.runner_version,
    )
    print(f"Run complete: {run_dir}")


if __name__ == "__main__":
    main()
