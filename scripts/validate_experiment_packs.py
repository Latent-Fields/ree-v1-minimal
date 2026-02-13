#!/usr/bin/env python3
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any

MANIFEST_SCHEMA_VERSION = "experiment_pack/v1"
METRICS_SCHEMA_VERSION = "experiment_pack_metrics/v1"
ADAPTER_SCHEMA_VERSION = "jepa_adapter_signals/v1"
JEPA_RUNNER_NAME = "ree-v1-minimal-harness"
LOCK_FILE_PATH = Path("contracts/ree_assembly_contract_lock.v1.json")

SNAKE_CASE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
ALLOWED_STATUS = {"PASS", "FAIL"}
ALLOWED_UNCERTAINTY_ESTIMATORS = {"none", "dispersion", "ensemble", "head"}
ALLOWED_PE_LATENT_FIELDS = {"mean", "p95", "by_mask"}
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def rel(path: Path) -> str:
    return str(path).replace("\\", "/")


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def is_rfc3339_utc(value: Any) -> bool:
    if not isinstance(value, str) or not value.endswith("Z"):
        return False
    try:
        dt = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return False
    return dt.tzinfo is not None and dt.utcoffset() == timezone.utc.utcoffset(dt)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_contract_lock(errors: list[str]) -> int:
    if not LOCK_FILE_PATH.exists():
        errors.append(f"Missing contract lock file: {rel(LOCK_FILE_PATH)}")
        return 0

    lock_doc = load_json(LOCK_FILE_PATH)
    if not isinstance(lock_doc, dict):
        errors.append(f"{rel(LOCK_FILE_PATH)}: lock file must be a JSON object")
        return 0

    for field in (
        "ree_assembly_repo",
        "ree_assembly_commit",
        "contract_version",
        "schema_files",
        "last_synced_utc",
    ):
        if field not in lock_doc:
            errors.append(f"{rel(LOCK_FILE_PATH)}: missing required field '{field}'")

    if not is_nonempty_string(lock_doc.get("ree_assembly_repo")):
        errors.append(f"{rel(LOCK_FILE_PATH)}: ree_assembly_repo must be a non-empty string")
    if not is_nonempty_string(lock_doc.get("ree_assembly_commit")):
        errors.append(f"{rel(LOCK_FILE_PATH)}: ree_assembly_commit must be a non-empty string")
    if not is_nonempty_string(lock_doc.get("contract_version")):
        errors.append(f"{rel(LOCK_FILE_PATH)}: contract_version must be a non-empty string")
    if not is_rfc3339_utc(lock_doc.get("last_synced_utc")):
        errors.append(f"{rel(LOCK_FILE_PATH)}: last_synced_utc must be RFC3339 UTC")

    schema_files = lock_doc.get("schema_files")
    if not isinstance(schema_files, dict) or not schema_files:
        errors.append(f"{rel(LOCK_FILE_PATH)}: schema_files must be a non-empty object")
        return 0

    validated_count = 0
    for rel_path, expected_hash in schema_files.items():
        if not is_nonempty_string(rel_path):
            errors.append(f"{rel(LOCK_FILE_PATH)}: schema_files keys must be non-empty strings")
            continue
        if not is_nonempty_string(expected_hash) or not HEX64_RE.match(expected_hash):
            errors.append(
                f"{rel(LOCK_FILE_PATH)}: schema_files['{rel_path}'] must be a 64-char hex sha256"
            )
            continue

        schema_path = Path(rel_path)
        if not schema_path.exists():
            errors.append(
                f"{rel(LOCK_FILE_PATH)}: vendored schema missing ({rel(schema_path)})"
            )
            continue

        actual_hash = sha256_file(schema_path)
        if actual_hash != expected_hash:
            errors.append(
                f"{rel(LOCK_FILE_PATH)}: hash mismatch for {rel(schema_path)} "
                f"(expected {expected_hash}, got {actual_hash})"
            )
            continue
        validated_count += 1

    return validated_count


def check_manifest(manifest: Any, manifest_path: Path, errors: list[str]) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        errors.append(f"{rel(manifest_path)}: manifest must be a JSON object")
        return {}

    required = (
        "schema_version",
        "experiment_type",
        "run_id",
        "status",
        "timestamp_utc",
        "source_repo",
        "runner",
        "artifacts",
    )
    for field in required:
        if field not in manifest:
            errors.append(f"{rel(manifest_path)}: missing required field '{field}'")

    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        errors.append(
            f"{rel(manifest_path)}: schema_version must be '{MANIFEST_SCHEMA_VERSION}'"
        )

    experiment_type = manifest.get("experiment_type")
    run_id = manifest.get("run_id")
    status = manifest.get("status")
    timestamp_utc = manifest.get("timestamp_utc")

    if not is_nonempty_string(experiment_type):
        errors.append(f"{rel(manifest_path)}: experiment_type must be a non-empty string")
    if not is_nonempty_string(run_id):
        errors.append(f"{rel(manifest_path)}: run_id must be a non-empty string")
    if status not in ALLOWED_STATUS:
        errors.append(f"{rel(manifest_path)}: status must be PASS or FAIL")
    if not is_rfc3339_utc(timestamp_utc):
        errors.append(f"{rel(manifest_path)}: timestamp_utc must be RFC3339 UTC")

    source_repo = manifest.get("source_repo")
    if not isinstance(source_repo, dict):
        errors.append(f"{rel(manifest_path)}: source_repo must be an object")
    else:
        if not is_nonempty_string(source_repo.get("name")):
            errors.append(f"{rel(manifest_path)}: source_repo.name must be a non-empty string")
        if not is_nonempty_string(source_repo.get("commit")):
            errors.append(f"{rel(manifest_path)}: source_repo.commit must be a non-empty string")
        branch = source_repo.get("branch")
        if branch is not None and not is_nonempty_string(branch):
            errors.append(f"{rel(manifest_path)}: source_repo.branch must be a non-empty string")

    runner = manifest.get("runner")
    if not isinstance(runner, dict):
        errors.append(f"{rel(manifest_path)}: runner must be an object")
    else:
        if not is_nonempty_string(runner.get("name")):
            errors.append(f"{rel(manifest_path)}: runner.name must be a non-empty string")
        if not is_nonempty_string(runner.get("version")):
            errors.append(f"{rel(manifest_path)}: runner.version must be a non-empty string")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        errors.append(f"{rel(manifest_path)}: artifacts must be an object")
        return {}

    for field in ("metrics_path", "summary_path"):
        if not is_nonempty_string(artifacts.get(field)):
            errors.append(f"{rel(manifest_path)}: artifacts.{field} must be a non-empty string")

    optional_artifacts = ("adapter_signals_path", "traces_dir", "media_dir")
    for field in optional_artifacts:
        if field in artifacts and not is_nonempty_string(artifacts.get(field)):
            errors.append(f"{rel(manifest_path)}: artifacts.{field} must be a non-empty string")

    return artifacts


def check_metrics(metrics: Any, metrics_path: Path, errors: list[str]) -> None:
    if not isinstance(metrics, dict):
        errors.append(f"{rel(metrics_path)}: metrics must be a JSON object")
        return

    if metrics.get("schema_version") != METRICS_SCHEMA_VERSION:
        errors.append(f"{rel(metrics_path)}: schema_version must be '{METRICS_SCHEMA_VERSION}'")

    values = metrics.get("values")
    if not isinstance(values, dict):
        errors.append(f"{rel(metrics_path)}: values must be an object")
        return

    for key, value in values.items():
        if not is_nonempty_string(key):
            errors.append(f"{rel(metrics_path)}: metric keys must be non-empty strings")
            continue
        if not SNAKE_CASE_RE.match(key):
            errors.append(f"{rel(metrics_path)}: metric key '{key}' must be snake_case")
        if not is_number(value):
            errors.append(f"{rel(metrics_path)}: metric '{key}' must be numeric")


def check_adapter_signals(
    adapter: Any,
    adapter_path: Path,
    manifest: dict[str, Any],
    errors: list[str],
) -> None:
    if not isinstance(adapter, dict):
        errors.append(f"{rel(adapter_path)}: adapter signals must be a JSON object")
        return

    required = (
        "schema_version",
        "experiment_type",
        "run_id",
        "adapter",
        "stream_presence",
        "pe_latent_fields",
        "uncertainty_estimator",
        "signal_metrics",
    )
    for field in required:
        if field not in adapter:
            errors.append(f"{rel(adapter_path)}: missing required field '{field}'")

    if adapter.get("schema_version") != ADAPTER_SCHEMA_VERSION:
        errors.append(f"{rel(adapter_path)}: schema_version must be '{ADAPTER_SCHEMA_VERSION}'")

    if adapter.get("experiment_type") != manifest.get("experiment_type"):
        errors.append(
            f"{rel(adapter_path)}: experiment_type must match manifest ({manifest.get('experiment_type')})"
        )
    if adapter.get("run_id") != manifest.get("run_id"):
        errors.append(f"{rel(adapter_path)}: run_id must match manifest ({manifest.get('run_id')})")

    adapter_meta = adapter.get("adapter")
    if not isinstance(adapter_meta, dict):
        errors.append(f"{rel(adapter_path)}: adapter must be an object")
    else:
        if not is_nonempty_string(adapter_meta.get("name")):
            errors.append(f"{rel(adapter_path)}: adapter.name must be a non-empty string")
        if not is_nonempty_string(adapter_meta.get("version")):
            errors.append(f"{rel(adapter_path)}: adapter.version must be a non-empty string")

    stream_presence = adapter.get("stream_presence")
    if not isinstance(stream_presence, dict):
        errors.append(f"{rel(adapter_path)}: stream_presence must be an object")
        stream_presence = {}

    const_true_fields = ("z_t", "z_hat", "pe_latent", "trace_context_mask_ids")
    for field in const_true_fields:
        if stream_presence.get(field) is not True:
            errors.append(f"{rel(adapter_path)}: stream_presence.{field} must be true")

    for field in ("uncertainty_latent", "trace_action_token"):
        if not isinstance(stream_presence.get(field), bool):
            errors.append(f"{rel(adapter_path)}: stream_presence.{field} must be boolean")

    pe_latent_fields = adapter.get("pe_latent_fields")
    if not isinstance(pe_latent_fields, list):
        errors.append(f"{rel(adapter_path)}: pe_latent_fields must be an array")
        pe_latent_fields = []
    else:
        seen = set()
        for field in pe_latent_fields:
            if field in seen:
                errors.append(f"{rel(adapter_path)}: pe_latent_fields entries must be unique")
                break
            seen.add(field)
            if field not in ALLOWED_PE_LATENT_FIELDS:
                errors.append(f"{rel(adapter_path)}: invalid pe_latent_fields entry '{field}'")
        if "mean" not in pe_latent_fields or "p95" not in pe_latent_fields:
            errors.append(f"{rel(adapter_path)}: pe_latent_fields must include 'mean' and 'p95'")

    uncertainty_estimator = adapter.get("uncertainty_estimator")
    if uncertainty_estimator not in ALLOWED_UNCERTAINTY_ESTIMATORS:
        errors.append(
            f"{rel(adapter_path)}: uncertainty_estimator must be one of "
            "none|dispersion|ensemble|head"
        )

    signal_metrics = adapter.get("signal_metrics")
    if not isinstance(signal_metrics, dict):
        errors.append(f"{rel(adapter_path)}: signal_metrics must be an object")
        return

    required_signal_metrics = (
        "latent_prediction_error_mean",
        "latent_prediction_error_p95",
        "latent_residual_coverage_rate",
        "precision_input_completeness_rate",
    )
    for field in required_signal_metrics:
        if field not in signal_metrics:
            errors.append(f"{rel(adapter_path)}: signal_metrics missing '{field}'")

    for field in ("latent_prediction_error_mean", "latent_prediction_error_p95"):
        value = signal_metrics.get(field)
        if not is_number(value) or value < 0:
            errors.append(f"{rel(adapter_path)}: signal_metrics.{field} must be >= 0")

    for field in ("latent_residual_coverage_rate", "precision_input_completeness_rate"):
        value = signal_metrics.get(field)
        if not is_number(value) or value < 0 or value > 1:
            errors.append(f"{rel(adapter_path)}: signal_metrics.{field} must be in [0,1]")

    uncertainty_latent = stream_presence.get("uncertainty_latent") is True
    calibration_error = signal_metrics.get("latent_uncertainty_calibration_error")
    if uncertainty_latent:
        if uncertainty_estimator == "none":
            errors.append(
                f"{rel(adapter_path)}: uncertainty_estimator must not be 'none' when uncertainty_latent=true"
            )
        if not is_number(calibration_error) or calibration_error < 0:
            errors.append(
                f"{rel(adapter_path)}: signal_metrics.latent_uncertainty_calibration_error must be >= 0 "
                "when uncertainty_latent=true"
            )


def main() -> int:
    root = Path(os.environ.get("EXPERIMENT_PACK_ROOT", "evidence/experiments"))
    manifests = sorted(root.glob("*/runs/*/manifest.json"))
    errors: list[str] = []
    lock_validated = check_contract_lock(errors)
    adapter_count = 0

    if not manifests:
        errors.append(f"No manifest.json found under {rel(root)}")

    for manifest_path in manifests:
        run_dir = manifest_path.parent
        experiment_type_dir = manifest_path.parents[2].name
        run_id_dir = run_dir.name

        manifest = load_json(manifest_path)
        artifacts = check_manifest(manifest, manifest_path, errors)

        if manifest.get("experiment_type") != experiment_type_dir:
            errors.append(
                f"{rel(manifest_path)}: experiment_type must match directory ({experiment_type_dir})"
            )
        if manifest.get("run_id") != run_id_dir:
            errors.append(f"{rel(manifest_path)}: run_id must match directory ({run_id_dir})")

        metrics_path = run_dir / artifacts.get("metrics_path", "metrics.json")
        summary_path = run_dir / artifacts.get("summary_path", "summary.md")
        adapter_rel_path = artifacts.get("adapter_signals_path")

        if not metrics_path.exists():
            errors.append(f"{rel(manifest_path)}: metrics file missing ({rel(metrics_path)})")
        else:
            check_metrics(load_json(metrics_path), metrics_path, errors)

        if not summary_path.exists():
            errors.append(f"{rel(manifest_path)}: summary file missing ({rel(summary_path)})")

        runner = manifest.get("runner", {})
        runner_name = runner.get("name") if isinstance(runner, dict) else None
        if runner_name == JEPA_RUNNER_NAME and not adapter_rel_path:
            errors.append(
                f"{rel(manifest_path)}: JEPA-backed runs must declare artifacts.adapter_signals_path"
            )

        if adapter_rel_path:
            adapter_path = run_dir / adapter_rel_path
            if not adapter_path.exists():
                errors.append(
                    f"{rel(manifest_path)}: adapter signals file missing ({rel(adapter_path)})"
                )
            else:
                adapter_count += 1
                check_adapter_signals(load_json(adapter_path), adapter_path, manifest, errors)

    if errors:
        print("Experiment Pack contract validation failed:")
        for err in errors:
            print(f"- {err}")
        return 1

    print(
        "Experiment Pack contract validation passed for "
        f"{len(manifests)} run(s), adapter files validated: {adapter_count}, "
        f"lock schemas validated: {lock_validated}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
