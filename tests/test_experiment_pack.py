"""Tests for Experiment Pack v1 emission."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import pytest

from experiments.pack_writer import (
    EVIDENCE_DIRECTIONS,
    ExperimentPackWriter,
    deterministic_run_id,
    normalize_timestamp_utc,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "tests" / "fixtures" / "schemas" / "v1"
FIXTURE_RUN_DIR = (
    ROOT
    / "tests"
    / "fixtures"
    / "experiment_pack_v1"
    / "baseline_explicit_cost"
    / "runs"
    / "2026-02-13T060000Z_baseline-explicit-cost_seed0"
)
SNAKE_CASE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
REQUIRED_ENVIRONMENT_FIELDS = (
    "env_id",
    "env_version",
    "dynamics_hash",
    "reward_hash",
    "observation_hash",
    "config_hash",
    "tier",
)
MECH056_REQUIRED_METRICS = (
    "trajectory_commit_channel_usage_count",
    "perceptual_sampling_channel_usage_count",
    "structural_consolidation_channel_usage_count",
    "precommit_semantic_overwrite_events",
    "structural_bias_magnitude",
    "structural_bias_rate",
    "environment_shortcut_leakage_events",
    "environment_unobservable_critical_state_rate",
    "environment_controllability_score",
    "environment_transition_consistency_rate",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_against_schema(document: Any, schema: dict[str, Any], path: str = "$") -> None:
    expected_type = schema.get("type")
    if expected_type is not None:
        _validate_type(document, expected_type, path)

    if "const" in schema:
        assert document == schema["const"], f"{path} must equal {schema['const']!r}"

    if "enum" in schema:
        assert document in schema["enum"], f"{path} must be one of {schema['enum']!r}"

    if isinstance(document, str) and "minLength" in schema:
        assert len(document) >= schema["minLength"], f"{path} must have minLength {schema['minLength']}"

    if isinstance(document, list):
        item_schema = schema.get("items")
        if item_schema:
            for idx, item in enumerate(document):
                _validate_against_schema(item, item_schema, f"{path}[{idx}]")
        if schema.get("uniqueItems"):
            assert len(document) == len(set(document)), f"{path} must contain unique items"

    if isinstance(document, dict):
        required = schema.get("required", [])
        for key in required:
            assert key in document, f"{path}.{key} is required"

        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)

        if additional is False:
            extra = set(document) - set(properties)
            assert not extra, f"{path} has unsupported properties: {sorted(extra)}"

        for key, value in document.items():
            if key in properties:
                _validate_against_schema(value, properties[key], f"{path}.{key}")
            elif isinstance(additional, dict):
                _validate_against_schema(value, additional, f"{path}.{key}")


def _validate_type(value: Any, expected_type: Any, path: str) -> None:
    if isinstance(expected_type, list):
        errors = []
        for one_type in expected_type:
            try:
                _validate_type(value, one_type, path)
                return
            except AssertionError as exc:
                errors.append(str(exc))
        raise AssertionError("; ".join(errors))

    if expected_type == "object":
        assert isinstance(value, dict), f"{path} must be an object"
    elif expected_type == "array":
        assert isinstance(value, list), f"{path} must be an array"
    elif expected_type == "string":
        assert isinstance(value, str), f"{path} must be a string"
    elif expected_type == "number":
        assert isinstance(value, (int, float)) and not isinstance(value, bool), (
            f"{path} must be a number"
        )
    elif expected_type == "integer":
        assert isinstance(value, int) and not isinstance(value, bool), (
            f"{path} must be an integer"
        )
    else:
        raise AssertionError(f"unsupported schema type in tests: {expected_type!r}")


def _schema(name: str) -> dict[str, Any]:
    return _load_json(SCHEMA_DIR / name)


def _assert_schema_valid_pack(run_dir: Path) -> None:
    manifest = _load_json(run_dir / "manifest.json")
    metrics = _load_json(run_dir / "metrics.json")

    _validate_against_schema(manifest, _schema("manifest.schema.json"))
    _validate_against_schema(metrics, _schema("metrics.schema.json"))

    values = metrics["values"]
    assert values, "metrics.values must not be empty"
    assert all(SNAKE_CASE_RE.match(k) for k in values), "metric keys must be stable snake_case"
    assert all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values.values())

    summary_text = (run_dir / "summary.md").read_text(encoding="utf-8").strip()
    assert summary_text, "summary.md must exist and contain text"

    assert "claim_ids_tested" in manifest
    assert isinstance(manifest["claim_ids_tested"], list)
    assert manifest["claim_ids_tested"], "claim_ids_tested must not be empty"
    assert all(isinstance(claim_id, str) for claim_id in manifest["claim_ids_tested"])

    assert "evidence_class" in manifest
    assert isinstance(manifest["evidence_class"], str)
    assert manifest["evidence_class"].strip()

    assert manifest.get("evidence_direction") in EVIDENCE_DIRECTIONS

    assert isinstance(manifest.get("producer_capabilities"), dict)
    assert manifest["producer_capabilities"], "producer_capabilities must not be empty"
    assert all(isinstance(v, bool) for v in manifest["producer_capabilities"].values())

    assert isinstance(manifest.get("environment"), dict)
    for field in REQUIRED_ENVIRONMENT_FIELDS:
        assert field in manifest["environment"]
        assert isinstance(manifest["environment"][field], str)
        assert manifest["environment"][field].strip()

    if "MECH-056" in manifest["claim_ids_tested"]:
        for key in MECH056_REQUIRED_METRICS:
            assert key in values, f"missing MECH-056 required metric {key}"
        assert "channel_escalation_order_observed" in summary_text
        if int(values["perceptual_sampling_channel_usage_count"]) > 0:
            assert "trigger_rationale_perceptual_sampling" in summary_text
        if int(values["structural_consolidation_channel_usage_count"]) > 0:
            assert "trigger_rationale_structural_consolidation" in summary_text


def test_deterministic_run_id() -> None:
    ts = normalize_timestamp_utc("2026-02-13T06:00:00Z")
    run_id = deterministic_run_id("baseline_explicit_cost", 7, ts)
    assert run_id == "2026-02-13T060000Z_baseline-explicit-cost_seed7"


def test_writer_emits_contract_shape(tmp_path: Path) -> None:
    writer = ExperimentPackWriter(
        output_root=tmp_path / "out",
        repo_root=ROOT,
        runner_name="ree-v1-minimal-harness",
        runner_version="0.1.0",
    )

    emitted = writer.write_pack(
        experiment_type="baseline_explicit_cost",
        run_id="2026-02-13T060000Z_baseline-explicit-cost_seed0",
        timestamp_utc="2026-02-13T06:00:00Z",
        status="PASS",
        metrics_values={
            "total_harm": 1.25,
            "final_residue": 0.42,
            "steps_survived": 12,
            "fatal_error_count": 0,
            "trajectory_commit_channel_usage_count": 12,
            "perceptual_sampling_channel_usage_count": 2,
            "structural_consolidation_channel_usage_count": 1,
            "precommit_semantic_overwrite_events": 0,
            "structural_bias_magnitude": 0.42,
            "structural_bias_rate": 0.035,
            "environment_shortcut_leakage_events": 0,
            "environment_unobservable_critical_state_rate": 0.0,
            "environment_controllability_score": 0.75,
            "environment_transition_consistency_rate": 0.98,
        },
        summary_markdown=(
            "# Summary\n\n"
            "All checks passed.\n\n"
            "## MECH-056 Escalation Trace\n"
            "- channel_escalation_order_observed: `trajectory_commit -> perceptual_sampling -> structural_consolidation`\n"
            "- trigger_rationale_perceptual_sampling: activated due to elevated risk.\n"
            "- trigger_rationale_structural_consolidation: activated due to persistent bias."
        ),
        scenario={"name": "baseline_explicit_cost", "seed": 0, "config_hash": "abc123"},
        failure_signatures=[],
        claim_ids_tested=["MECH-056", "Q-011"],
        evidence_class="simulation",
        evidence_direction="supports",
        producer_capabilities={
            "trajectory_integrity_channelized_bias": True,
            "mech056_dispatch_metric_set": True,
            "mech056_summary_escalation_trace": True,
        },
        environment={
            "env_id": "ree.grid_world",
            "env_version": "grid_world/v1",
            "dynamics_hash": "d1",
            "reward_hash": "r1",
            "observation_hash": "o1",
            "config_hash": "c1",
            "tier": "toy",
        },
    )

    assert emitted.run_dir.exists()
    assert emitted.manifest_path.exists()
    assert emitted.metrics_path.exists()
    assert emitted.summary_path.exists()
    _assert_schema_valid_pack(emitted.run_dir)


def test_writer_rejects_non_numeric_metrics(tmp_path: Path) -> None:
    writer = ExperimentPackWriter(
        output_root=tmp_path / "out",
        repo_root=ROOT,
        runner_name="ree-v1-minimal-harness",
        runner_version="0.1.0",
    )

    with pytest.raises(TypeError):
        writer.write_pack(
            experiment_type="baseline_explicit_cost",
            run_id="2026-02-13T060000Z_baseline-explicit-cost_seed0",
            timestamp_utc="2026-02-13T06:00:00Z",
            status="PASS",
            metrics_values={"fatal_error_count": False},
            summary_markdown="# Summary\n\nbad",
        )


def test_writer_rejects_invalid_evidence_direction(tmp_path: Path) -> None:
    writer = ExperimentPackWriter(
        output_root=tmp_path / "out",
        repo_root=ROOT,
        runner_name="ree-v1-minimal-harness",
        runner_version="0.1.0",
    )

    with pytest.raises(ValueError):
        writer.write_pack(
            experiment_type="baseline_explicit_cost",
            run_id="2026-02-13T060000Z_baseline-explicit-cost_seed0",
            timestamp_utc="2026-02-13T06:00:00Z",
            status="PASS",
            metrics_values={"fatal_error_count": 0},
            summary_markdown="# Summary\n\nbad",
            claim_ids_tested=["MECH-056"],
            evidence_class="simulation",
            evidence_direction="not_a_direction",
        )


def test_execute_experiment_emits_complete_pack(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from experiments.run import execute_experiment

    run_dir = execute_experiment(
        suite_name="baseline_explicit_cost",
        seed=3,
        max_steps=5,
        output_root=str(tmp_path / "evidence"),
        timestamp_utc="2026-02-13T06:00:00Z",
    )

    assert run_dir.exists()
    assert run_dir.parent.name == "runs"
    assert run_dir.parent.parent.name == "baseline_explicit_cost"
    _assert_schema_valid_pack(run_dir)

    manifest = _load_json(run_dir / "manifest.json")
    assert manifest["status"] in {"PASS", "FAIL"}
    assert manifest["claim_ids_tested"]
    assert manifest["evidence_class"]
    assert manifest["evidence_direction"] in EVIDENCE_DIRECTIONS
    assert isinstance(manifest.get("failure_signatures", []), list)


def test_fixture_pack_matches_schema_shape() -> None:
    _assert_schema_valid_pack(FIXTURE_RUN_DIR)
