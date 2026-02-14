#!/usr/bin/env python3
"""Generate weekly producer handoff report from run packs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def monday_of_current_week_utc() -> str:
    now = datetime.now(timezone.utc).date()
    monday = now - timedelta(days=now.weekday())
    return monday.isoformat()


def git_value(args: list[str], cwd: Path) -> str:
    try:
        out = subprocess.check_output(["git", *args], cwd=str(cwd), text=True)
        return out.strip()
    except Exception:
        return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate weekly producer handoff report.")
    parser.add_argument(
        "--run-root",
        default="runs/bridging_qualification_056_058_059_060",
        help="Experiment-pack root containing <experiment_type>/runs/<run_id>.",
    )
    parser.add_argument(
        "--output",
        default="evidence/planning/weekly_handoff_current.md",
        help="Output markdown path.",
    )
    parser.add_argument(
        "--week-of-utc",
        default=monday_of_current_week_utc(),
        help="Week start date (YYYY-MM-DD UTC).",
    )
    parser.add_argument(
        "--schema-validation-status",
        default="PASS",
        choices=["PASS", "FAIL"],
        help="CI gate status for schema validation.",
    )
    parser.add_argument(
        "--schema-validation-evidence",
        default="python scripts/validate_experiment_packs.py",
        help="Evidence string for schema validation gate.",
    )
    parser.add_argument(
        "--seed-determinism-status",
        default="PASS",
        choices=["PASS", "FAIL"],
        help="CI gate status for seed determinism.",
    )
    parser.add_argument(
        "--seed-determinism-evidence",
        default="python scripts/check_bridging_seed_determinism.py --seeds 11,29",
        help="Evidence string for seed determinism gate.",
    )
    parser.add_argument(
        "--hook-coverage-status",
        default="N/A",
        choices=["PASS", "FAIL", "N/A"],
        help="CI gate status for hook surface coverage.",
    )
    parser.add_argument(
        "--hook-coverage-evidence",
        default="N/A for parity/backstop lane",
        help="Evidence string for hook coverage gate.",
    )
    parser.add_argument(
        "--parity-note",
        default=(
            "Parity note: latest ree-v2 qualification outcomes were not available in this cycle; "
            "agree/disagree delta assessment deferred to next sync."
        ),
        help="Parity note inserted into Open Blockers.",
    )
    parser.add_argument(
        "--additional-blocker",
        action="append",
        default=[],
        help="Additional blocker bullet lines.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    run_root = repo_root / args.run_root
    output_path = repo_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lock_path = repo_root / "contracts/ree_assembly_contract_lock.v1.json"
    lock_doc = load_json(lock_path)
    lock_hash = sha256_file(lock_path)

    manifests = sorted(run_root.glob("*/runs/*/manifest.json"))
    if not manifests:
        print(f"No runs found under {run_root}")
        return 1

    rows: list[dict[str, Any]] = []
    claim_summary: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "runs_added": 0,
            "supports": 0,
            "weakens": 0,
            "mixed": 0,
            "unknown": 0,
            "failure_signatures": Counter(),
        }
    )
    schema_versions = {"experiment_pack/v1", "experiment_pack_metrics/v1"}

    for manifest_path in manifests:
        run_dir = manifest_path.parent
        manifest = load_json(manifest_path)
        metrics = load_json(run_dir / "metrics.json")
        schema_versions.add(str(manifest.get("schema_version", "")))
        schema_versions.add(str(metrics.get("schema_version", "")))

        artifacts = manifest.get("artifacts", {})
        if "adapter_signals_path" in artifacts:
            adapter = load_json(run_dir / artifacts["adapter_signals_path"])
            schema_versions.add(str(adapter.get("schema_version", "")))

        claim_ids = manifest.get("claim_ids_tested", [])
        claim_ids_str = ", ".join(claim_ids) if claim_ids else ""
        failure_signatures = manifest.get("failure_signatures", [])
        failure_signatures_str = ", ".join(failure_signatures) if failure_signatures else ""
        scenario = manifest.get("scenario", {})

        row = {
            "experiment_type": manifest["experiment_type"],
            "run_id": manifest["run_id"],
            "seed": scenario.get("seed", ""),
            "condition_or_scenario": scenario.get("condition", scenario.get("name", "")),
            "status": manifest["status"],
            "evidence_direction": manifest.get("evidence_direction", "unknown"),
            "claim_ids_tested": claim_ids_str,
            "failure_signatures": failure_signatures_str,
            "pack_path": str(run_dir.relative_to(repo_root)).replace("\\", "/"),
        }
        rows.append(row)

        for claim_id in claim_ids:
            summary = claim_summary[claim_id]
            summary["runs_added"] += 1
            direction = manifest.get("evidence_direction", "unknown")
            if direction in ("supports", "weakens", "mixed", "unknown"):
                summary[direction] += 1
            else:
                summary["unknown"] += 1
            for signature in failure_signatures:
                summary["failure_signatures"][signature] += 1

    schema_version_set = ", ".join(sorted(v for v in schema_versions if v))

    lines: list[str] = []
    producer_repo = repo_root.name
    producer_commit = git_value(["rev-parse", "HEAD"], repo_root)

    lines.append(f"# Weekly Handoff - {producer_repo} - {args.week_of_utc}")
    lines.append("")
    lines.append("## Metadata")
    lines.append(f"- week_of_utc: `{args.week_of_utc}`")
    lines.append(f"- producer_repo: `{producer_repo}`")
    lines.append(f"- producer_commit: `{producer_commit}`")
    lines.append(f"- generated_utc: `{now_utc()}`")
    lines.append("")

    lines.append("## Contract Sync")
    lines.append(f"- ree_assembly_repo: `{lock_doc.get('ree_assembly_repo', '')}`")
    lines.append(f"- ree_assembly_commit: `{lock_doc.get('ree_assembly_commit', '')}`")
    lines.append(f"- contract_lock_path: `contracts/ree_assembly_contract_lock.v1.json`")
    lines.append(f"- contract_lock_hash: `{lock_hash}`")
    lines.append(f"- schema_version_set: `{schema_version_set}`")
    lines.append("")

    lines.append("## CI Gates")
    lines.append("| gate | status | evidence |")
    lines.append("| --- | --- | --- |")
    lines.append(
        f"| schema_validation | {args.schema_validation_status} | `{args.schema_validation_evidence}` |"
    )
    lines.append(
        f"| seed_determinism | {args.seed_determinism_status} | `{args.seed_determinism_evidence}` |"
    )
    lines.append(
        f"| hook_surface_coverage | {args.hook_coverage_status} | `{args.hook_coverage_evidence}` |"
    )
    lines.append("")

    lines.append("## Run-Pack Inventory")
    lines.append(
        "| experiment_type | run_id | seed | condition_or_scenario | status | "
        "evidence_direction | claim_ids_tested | failure_signatures | pack_path |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        lines.append(
            "| {experiment_type} | {run_id} | {seed} | {condition_or_scenario} | {status} | "
            "{evidence_direction} | {claim_ids_tested} | {failure_signatures} | {pack_path} |".format(
                **row
            )
        )
    lines.append("")

    lines.append("## Claim Summary")
    lines.append(
        "| claim_id | runs_added | supports | weakens | mixed | unknown | recurring_failure_signatures |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for claim_id in sorted(claim_summary):
        summary = claim_summary[claim_id]
        recurring = [
            sig for sig, count in sorted(summary["failure_signatures"].items()) if count >= 2
        ]
        recurring_str = ", ".join(recurring)
        lines.append(
            f"| {claim_id} | {summary['runs_added']} | {summary['supports']} | "
            f"{summary['weakens']} | {summary['mixed']} | {summary['unknown']} | {recurring_str} |"
        )
    lines.append("")

    lines.append("## Open Blockers")
    lines.append(f"- {args.parity_note}")
    if args.additional_blocker:
        for blocker in args.additional_blocker:
            lines.append(f"- {blocker}")
    else:
        lines.append("- No additional blockers identified for this cycle.")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote weekly handoff: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
