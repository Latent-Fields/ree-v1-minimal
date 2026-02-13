#!/usr/bin/env python3
import json
import os
from pathlib import Path

from jsonschema import Draft202012Validator

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


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def rel(path: Path) -> str:
    return str(path).replace("\\", "/")


def main() -> int:
    root = Path(os.environ.get("EXPERIMENT_PACK_ROOT", "evidence/experiments"))
    schema_dir = Path("contracts/experiment_pack/v1")

    manifest_schema = load_json(schema_dir / "manifest.schema.json")
    metrics_schema = load_json(schema_dir / "metrics.schema.json")

    manifest_validator = Draft202012Validator(
        manifest_schema,
        format_checker=Draft202012Validator.FORMAT_CHECKER,
    )
    metrics_validator = Draft202012Validator(metrics_schema)

    manifests = sorted(root.glob("*/runs/*/manifest.json"))
    errors = []

    if not manifests:
        errors.append(f"No manifest.json found under {rel(root)}")

    for manifest_path in manifests:
        run_dir = manifest_path.parent
        experiment_type_dir = manifest_path.parents[2].name
        run_id_dir = run_dir.name

        manifest = load_json(manifest_path)
        for e in manifest_validator.iter_errors(manifest):
            errors.append(f"{rel(manifest_path)}: {e.message}")

        if manifest.get("experiment_type") != experiment_type_dir:
            errors.append(
                f"{rel(manifest_path)}: experiment_type must match directory "
                f"({experiment_type_dir})"
            )
        if manifest.get("run_id") != run_id_dir:
            errors.append(
                f"{rel(manifest_path)}: run_id must match directory ({run_id_dir})"
            )

        # Required for REE_assembly claim-evidence matrix linkage.
        for field in ("claim_ids_tested", "evidence_class", "evidence_direction"):
            if field not in manifest:
                errors.append(f"{rel(manifest_path)}: missing required linkage field '{field}'")
        if "producer_capabilities" not in manifest:
            errors.append(f"{rel(manifest_path)}: missing required field 'producer_capabilities'")
        elif not isinstance(manifest.get("producer_capabilities"), dict):
            errors.append(f"{rel(manifest_path)}: producer_capabilities must be an object")
        else:
            for key, value in manifest["producer_capabilities"].items():
                if not isinstance(value, bool):
                    errors.append(
                        f"{rel(manifest_path)}: producer_capabilities['{key}'] must be boolean"
                    )

        if "environment" not in manifest:
            errors.append(f"{rel(manifest_path)}: missing required field 'environment'")
        elif not isinstance(manifest.get("environment"), dict):
            errors.append(f"{rel(manifest_path)}: environment must be an object")
        else:
            env = manifest["environment"]
            for field in REQUIRED_ENVIRONMENT_FIELDS:
                value = env.get(field)
                if not isinstance(value, str) or not value.strip():
                    errors.append(
                        f"{rel(manifest_path)}: environment.{field} must be a non-empty string"
                    )

        artifacts = manifest.get("artifacts", {})
        metrics_path = run_dir / artifacts.get("metrics_path", "metrics.json")
        summary_path = run_dir / artifacts.get("summary_path", "summary.md")
        claim_ids = manifest.get("claim_ids_tested", [])

        if not metrics_path.exists():
            errors.append(f"{rel(manifest_path)}: metrics file missing ({rel(metrics_path)})")
        else:
            metrics = load_json(metrics_path)
            for e in metrics_validator.iter_errors(metrics):
                errors.append(f"{rel(metrics_path)}: {e.message}")
            values = metrics.get("values", {})
            if not isinstance(values, dict):
                errors.append(f"{rel(metrics_path)}: values must be an object")
            elif "MECH-056" in claim_ids:
                for key in MECH056_REQUIRED_METRICS:
                    if key not in values:
                        errors.append(
                            f"{rel(metrics_path)}: missing MECH-056 required metric '{key}'"
                        )
                    elif not isinstance(values[key], (int, float)) or isinstance(values[key], bool):
                        errors.append(
                            f"{rel(metrics_path)}: MECH-056 metric '{key}' must be numeric"
                        )

        if not summary_path.exists():
            errors.append(f"{rel(manifest_path)}: summary file missing ({rel(summary_path)})")
        elif "MECH-056" in claim_ids:
            summary_text = summary_path.read_text(encoding="utf-8")
            if "channel_escalation_order_observed" not in summary_text:
                errors.append(
                    f"{rel(summary_path)}: missing channel escalation order for MECH-056 run"
                )
            metrics_values = {}
            if metrics_path.exists():
                metrics_values = load_json(metrics_path).get("values", {})
            if int(metrics_values.get("perceptual_sampling_channel_usage_count", 0)) > 0:
                if "trigger_rationale_perceptual_sampling" not in summary_text:
                    errors.append(
                        f"{rel(summary_path)}: missing perceptual sampling trigger rationale"
                    )
            if int(metrics_values.get("structural_consolidation_channel_usage_count", 0)) > 0:
                if "trigger_rationale_structural_consolidation" not in summary_text:
                    errors.append(
                        f"{rel(summary_path)}: missing structural consolidation trigger rationale"
                    )

    if errors:
        print("Experiment Pack contract validation failed:")
        for err in errors:
            print(f"- {err}")
        return 1

    print(f"Experiment Pack contract validation passed for {len(manifests)} run(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
