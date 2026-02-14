#!/usr/bin/env python3
"""Generate deterministic weekly producer handoff markdown."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


REQUIRED_DIRECTIONS = ("supports", "weakens", "mixed", "unknown")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_rfc3339_utc(raw: str) -> datetime:
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def format_rfc3339_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def monday_of_week(value: date) -> date:
    return value - timedelta(days=value.weekday())


def git_value(repo_root: Path, *args: str) -> str:
    try:
        output = subprocess.check_output(["git", *args], cwd=str(repo_root), text=True)
    except Exception:
        return "unknown"
    return output.strip() or "unknown"


def seed_from_manifest(manifest: dict[str, Any]) -> str:
    scenario = manifest.get("scenario", {})
    if isinstance(scenario, dict):
        seed = scenario.get("seed")
        if seed is not None:
            return str(seed)
    run_id = str(manifest.get("run_id", ""))
    match = re.search(r"_seed(\d+)", run_id)
    return match.group(1) if match else "N/A"


def condition_from_manifest(manifest: dict[str, Any]) -> str:
    scenario = manifest.get("scenario", {})
    if isinstance(scenario, dict):
        for key in ("condition", "name"):
            value = scenario.get(key)
            if value:
                return str(value)
    return "N/A"


def parse_claim_summary_from_handoff(handoff_path: Path) -> dict[str, dict[str, int]]:
    text = handoff_path.read_text(encoding="utf-8")
    lines = [line.rstrip("\n") for line in text.splitlines()]

    try:
        start = lines.index("## Claim Summary")
    except ValueError:
        return {}

    end = len(lines)
    for idx in range(start + 1, len(lines)):
        if lines[idx].startswith("## "):
            end = idx
            break

    table_lines = [line.strip() for line in lines[start + 1 : end] if line.strip().startswith("|")]
    if len(table_lines) < 3:
        return {}

    headers = [cell.strip() for cell in table_lines[0].strip("|").split("|")]
    required = ["claim_id", "supports", "weakens", "mixed", "unknown"]
    index: dict[str, int] = {}
    for name in required:
        if name not in headers:
            return {}
        index[name] = headers.index(name)

    parsed: dict[str, dict[str, int]] = {}
    for row in table_lines[2:]:
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        if len(cells) < len(headers):
            continue
        claim_id = cells[index["claim_id"]]
        if not claim_id:
            continue
        counts: dict[str, int] = {}
        for direction in REQUIRED_DIRECTIONS:
            raw = cells[index[direction]]
            try:
                counts[direction] = int(raw)
            except ValueError:
                counts[direction] = 0
        parsed[claim_id] = counts
    return parsed


def dominant_direction(counts: dict[str, int]) -> str:
    best_value = max(counts.get(name, 0) for name in REQUIRED_DIRECTIONS)
    winners = [name for name in REQUIRED_DIRECTIONS if counts.get(name, 0) == best_value]
    if len(winners) != 1:
        return "mixed"
    return winners[0]


def build_parity_note(local_claims: dict[str, dict[str, Any]], ree_v2_handoff: Path) -> str:
    if not ree_v2_handoff.exists():
        return f"Parity note vs latest ree-v2: N/A (missing {ree_v2_handoff})."

    remote_claims = parse_claim_summary_from_handoff(ree_v2_handoff)
    if not remote_claims:
        return f"Parity note vs latest ree-v2: N/A (unable to parse claim summary in {ree_v2_handoff})."

    overlap = sorted(set(local_claims) & set(remote_claims))
    if not overlap:
        return f"Parity note vs latest ree-v2: N/A (no overlapping claims in {ree_v2_handoff})."

    agrees: list[str] = []
    disagrees: list[str] = []
    for claim_id in overlap:
        local_direction = dominant_direction(local_claims[claim_id])
        remote_direction = dominant_direction(remote_claims[claim_id])
        if local_direction == remote_direction:
            agrees.append(f"{claim_id}:{local_direction}")
        else:
            disagrees.append(
                f"{claim_id}(ree-v1-minimal={local_direction}, ree-v2={remote_direction})"
            )

    if not disagrees:
        return (
            "Parity note vs latest ree-v2: agrees on overlapping claims "
            f"({', '.join(agrees)}); source={ree_v2_handoff}."
        )
    if not agrees:
        return (
            "Parity note vs latest ree-v2: disagreement on overlapping claims "
            f"({', '.join(disagrees)}); source={ree_v2_handoff}."
        )
    return (
        "Parity note vs latest ree-v2: agrees on "
        f"({', '.join(agrees)}), disagrees on ({', '.join(disagrees)}); "
        f"source={ree_v2_handoff}."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate weekly handoff markdown from run packs.")
    parser.add_argument(
        "--run-root",
        default="runs/bridging_qualification_056_058_059_060",
        help="Path containing <experiment_type>/runs/<run_id>/manifest.json folders.",
    )
    parser.add_argument(
        "--output",
        default="evidence/planning/weekly_handoff/latest.md",
        help="Output markdown path.",
    )
    parser.add_argument(
        "--generated-utc",
        default="",
        help="Optional RFC3339 UTC timestamp. If unset, uses max run timestamp for deterministic output.",
    )
    parser.add_argument(
        "--week-of-utc",
        default="",
        help="Optional cycle start date (YYYY-MM-DD). If unset, derived from generated_utc.",
    )
    parser.add_argument(
        "--schema-validation-status",
        default="PASS",
        choices=["PASS", "FAIL"],
        help="schema_validation gate status.",
    )
    parser.add_argument(
        "--schema-validation-evidence",
        default=(
            "EXPERIMENT_PACK_ROOT=runs/bridging_qualification_056_058_059_060 "
            "python scripts/validate_experiment_packs.py"
        ),
        help="schema_validation gate evidence string.",
    )
    parser.add_argument(
        "--seed-determinism-status",
        default="PASS",
        choices=["PASS", "FAIL"],
        help="seed_determinism gate status.",
    )
    parser.add_argument(
        "--seed-determinism-evidence",
        default=(
            "python scripts/check_bridging_seed_determinism.py --seeds 11,29 "
            "--timestamp-utc 2026-02-14T03:00:00Z"
        ),
        help="seed_determinism gate evidence string.",
    )
    parser.add_argument(
        "--hook-surface-coverage-status",
        default="N/A",
        choices=["PASS", "FAIL", "N/A"],
        help="hook_surface_coverage gate status.",
    )
    parser.add_argument(
        "--hook-surface-coverage-evidence",
        default="N/A for parity/backstop lane",
        help="hook_surface_coverage gate evidence string.",
    )
    parser.add_argument(
        "--remote-export-import-status",
        default="N/A",
        choices=["PASS", "FAIL", "N/A"],
        help="remote_export_import gate status.",
    )
    parser.add_argument(
        "--remote-export-import-evidence",
        default="N/A for parity/backstop lane",
        help="remote_export_import gate evidence string.",
    )
    parser.add_argument(
        "--ree-v2-handoff",
        default="/Users/dgolden/Documents/GitHub/ree-v2/evidence/planning/weekly_handoff/latest.md",
        help="Path to latest ree-v2 handoff report used for parity note.",
    )
    parser.add_argument(
        "--parity-note",
        default="",
        help="Optional explicit parity note override.",
    )
    parser.add_argument(
        "--local-compute-options-watch",
        default="N/A",
        help="Set to N/A unless in scope.",
    )
    parser.add_argument(
        "--blocker",
        action="append",
        default=[],
        help="Additional blocker line; may be provided multiple times.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    run_root = (repo_root / args.run_root).resolve()
    output_path = (repo_root / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifests = sorted(run_root.glob("*/runs/*/manifest.json"))
    if not manifests:
        print(f"FAIL: no manifest.json files under {run_root}")
        return 1

    lock_path = repo_root / "contracts/ree_assembly_contract_lock.v1.json"
    if not lock_path.exists():
        print(f"FAIL: missing contract lock file at {lock_path}")
        return 1
    lock_doc = load_json(lock_path)
    lock_hash = sha256_file(lock_path)

    rows: list[dict[str, str]] = []
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
    observed_timestamps: list[datetime] = []
    schema_versions: set[str] = set()

    for manifest_path in manifests:
        manifest = load_json(manifest_path)
        run_dir = manifest_path.parent
        artifacts = manifest.get("artifacts", {})

        manifest_schema_version = str(manifest.get("schema_version", ""))
        if manifest_schema_version:
            schema_versions.add(manifest_schema_version)

        metrics_rel = str(artifacts.get("metrics_path", "metrics.json"))
        metrics_path = run_dir / metrics_rel
        metrics_schema_version = ""
        runtime_minutes = "N/A"
        if metrics_path.exists():
            metrics = load_json(metrics_path)
            metrics_schema_version = str(metrics.get("schema_version", ""))
            if metrics_schema_version:
                schema_versions.add(metrics_schema_version)
            values = metrics.get("values", {})
            if isinstance(values, dict):
                runtime = values.get("runtime_minutes")
                if isinstance(runtime, (int, float)):
                    runtime_minutes = f"{float(runtime):.3f}"

        adapter_rel = artifacts.get("adapter_signals_path")
        if isinstance(adapter_rel, str) and adapter_rel:
            adapter_path = run_dir / adapter_rel
            if adapter_path.exists():
                adapter_doc = load_json(adapter_path)
                adapter_version = str(adapter_doc.get("schema_version", ""))
                if adapter_version:
                    schema_versions.add(adapter_version)

        timestamp_raw = str(manifest.get("timestamp_utc", "")).strip()
        if timestamp_raw:
            try:
                observed_timestamps.append(parse_rfc3339_utc(timestamp_raw))
            except ValueError:
                pass

        run_id = str(manifest.get("run_id", run_dir.name))
        claim_ids = manifest.get("claim_ids_tested", [])
        if not isinstance(claim_ids, list):
            claim_ids = []
        failure_signatures = manifest.get("failure_signatures", [])
        if not isinstance(failure_signatures, list):
            failure_signatures = []

        execution_mode = "remote" if manifest.get("runner", {}).get("name") == "ree-remote-runner" else "local"
        compute_backend = "local_cpu" if execution_mode == "local" else "remote_unknown"

        row = {
            "experiment_type": str(manifest.get("experiment_type", run_dir.parents[1].name)),
            "run_id": run_id,
            "seed": seed_from_manifest(manifest),
            "condition_or_scenario": condition_from_manifest(manifest),
            "status": str(manifest.get("status", "FAIL")),
            "evidence_direction": str(manifest.get("evidence_direction", "unknown")),
            "claim_ids_tested": ", ".join(str(x) for x in claim_ids),
            "failure_signatures": ", ".join(str(x) for x in failure_signatures),
            "execution_mode": execution_mode,
            "compute_backend": compute_backend,
            "runtime_minutes": runtime_minutes,
            "pack_path": str(run_dir.relative_to(repo_root)).replace("\\", "/"),
        }
        rows.append(row)

        for claim_id in claim_ids:
            claim_key = str(claim_id)
            summary = claim_summary[claim_key]
            summary["runs_added"] += 1
            direction = row["evidence_direction"]
            if direction not in REQUIRED_DIRECTIONS:
                direction = "unknown"
            summary[direction] += 1
            for signature in failure_signatures:
                summary["failure_signatures"][str(signature)] += 1

    rows.sort(key=lambda row: (row["experiment_type"], row["run_id"]))

    if args.generated_utc:
        generated_dt = parse_rfc3339_utc(args.generated_utc)
    elif observed_timestamps:
        generated_dt = max(observed_timestamps)
    else:
        generated_dt = datetime.now(timezone.utc)
    generated_utc = format_rfc3339_utc(generated_dt)

    if args.week_of_utc:
        week_of_utc = args.week_of_utc
    else:
        week_of_utc = monday_of_week(generated_dt.date()).isoformat()

    producer_repo = repo_root.name
    producer_commit = git_value(repo_root, "rev-parse", "HEAD")
    schema_version_set = ", ".join(sorted(schema_versions))

    if args.parity_note:
        parity_note = args.parity_note
    else:
        parity_note = build_parity_note(claim_summary, Path(args.ree_v2_handoff))

    lines: list[str] = []
    lines.append(f"# Weekly Handoff - {producer_repo} - {week_of_utc}")
    lines.append("")
    lines.append("## Metadata")
    lines.append(f"- week_of_utc: `{week_of_utc}`")
    lines.append(f"- producer_repo: `{producer_repo}`")
    lines.append(f"- producer_commit: `{producer_commit}`")
    lines.append(f"- generated_utc: `{generated_utc}`")
    lines.append("")
    lines.append("## Contract Sync")
    lines.append(f"- ree_assembly_repo: `{lock_doc.get('ree_assembly_repo', 'N/A')}`")
    lines.append(f"- ree_assembly_commit: `{lock_doc.get('ree_assembly_commit', 'N/A')}`")
    lines.append("- contract_lock_path: `contracts/ree_assembly_contract_lock.v1.json`")
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
        "| hook_surface_coverage | "
        f"{args.hook_surface_coverage_status} | `{args.hook_surface_coverage_evidence}` |"
    )
    lines.append(
        "| remote_export_import | "
        f"{args.remote_export_import_status} | `{args.remote_export_import_evidence}` |"
    )
    lines.append("")
    lines.append("## Run-Pack Inventory")
    lines.append(
        "| experiment_type | run_id | seed | condition_or_scenario | status | "
        "evidence_direction | claim_ids_tested | failure_signatures | execution_mode | "
        "compute_backend | runtime_minutes | pack_path |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        lines.append(
            "| {experiment_type} | {run_id} | {seed} | {condition_or_scenario} | {status} | "
            "{evidence_direction} | {claim_ids_tested} | {failure_signatures} | "
            "{execution_mode} | {compute_backend} | {runtime_minutes} | {pack_path} |".format(**row)
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
            signature
            for signature, count in sorted(summary["failure_signatures"].items())
            if count >= 2
        ]
        recurring_joined = ", ".join(recurring)
        lines.append(
            f"| {claim_id} | {summary['runs_added']} | {summary['supports']} | "
            f"{summary['weakens']} | {summary['mixed']} | {summary['unknown']} | {recurring_joined} |"
        )
    lines.append("")
    lines.append("## Open Blockers")
    lines.append(f"- {parity_note}")
    for blocker in args.blocker:
        lines.append(f"- {blocker}")
    if not args.blocker:
        lines.append("- No additional blockers reported this cycle.")
    lines.append("")
    lines.append("## Local Compute Options Watch")
    lines.append("- local_options_last_updated_utc: `N/A`")
    lines.append("- rolling_3mo_cloud_spend_eur: `N/A`")
    lines.append("- local_blocked_sessions_this_week: `N/A`")
    lines.append(f"- recommended_local_action: `{args.local_compute_options_watch}`")
    lines.append("- rationale: `N/A for ree-v1-minimal parity/backstop lane unless explicitly in scope.`")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"PASS: generated weekly handoff at {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
