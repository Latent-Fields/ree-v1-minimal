#!/usr/bin/env python3
"""Validate weekly handoff markdown against shared template requirements."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys


REQUIRED_SECTIONS = [
    "## Metadata",
    "## Contract Sync",
    "## CI Gates",
    "## Run-Pack Inventory",
    "## Claim Summary",
    "## Open Blockers",
    "## Local Compute Options Watch",
]

REQUIRED_METADATA_KEYS = [
    "week_of_utc",
    "producer_repo",
    "producer_commit",
    "generated_utc",
]

REQUIRED_CONTRACT_KEYS = [
    "ree_assembly_repo",
    "ree_assembly_commit",
    "contract_lock_path",
    "contract_lock_hash",
    "schema_version_set",
]

REQUIRED_GATES = [
    "schema_validation",
    "seed_determinism",
    "hook_surface_coverage",
    "remote_export_import",
]

RUN_PACK_COLUMNS = [
    "experiment_type",
    "run_id",
    "seed",
    "condition_or_scenario",
    "status",
    "evidence_direction",
    "claim_ids_tested",
    "failure_signatures",
    "execution_mode",
    "compute_backend",
    "runtime_minutes",
    "pack_path",
]

CLAIM_SUMMARY_COLUMNS = [
    "claim_id",
    "runs_added",
    "supports",
    "weakens",
    "mixed",
    "unknown",
    "recurring_failure_signatures",
]

LOCAL_WATCH_KEYS = [
    "local_options_last_updated_utc",
    "rolling_3mo_cloud_spend_eur",
    "local_blocked_sessions_this_week",
    "recommended_local_action",
    "rationale",
]

VALID_DIRECTIONS = {"supports", "weakens", "mixed", "unknown"}
VALID_EXECUTION_MODE = {"local", "remote"}
VALID_RECOMMENDED_LOCAL_ACTION = {
    "hold_cloud_only",
    "upgrade_low",
    "upgrade_mid",
    "upgrade_high",
    "N/A",
}


def normalize_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def find_section(lines: list[str], name: str) -> int:
    try:
        return lines.index(name)
    except ValueError:
        return -1


def section_lines(lines: list[str], section: str) -> list[str]:
    start = find_section(lines, section)
    if start < 0:
        return []
    end = len(lines)
    for idx in range(start + 1, len(lines)):
        if lines[idx].startswith("## "):
            end = idx
            break
    return lines[start:end]


def check_bullet_key(lines: list[str], key: str, errors: list[str], section_name: str) -> None:
    pattern = re.compile(rf"^-\s+{re.escape(key)}:\s+`?.+`?\s*$")
    if not any(pattern.match(line.strip()) for line in lines):
        errors.append(f"{section_name}: missing key '{key}'")


def extract_table(lines: list[str]) -> list[list[str]]:
    table = [line.strip() for line in lines if line.strip().startswith("|")]
    return [normalize_row(line) for line in table]


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate weekly handoff markdown structure.")
    parser.add_argument(
        "--input",
        default="evidence/planning/weekly_handoff/latest.md",
        help="Path to weekly handoff markdown.",
    )
    args = parser.parse_args()

    report_path = Path(args.input)
    if not report_path.exists():
        print(f"FAIL: weekly handoff not found at {report_path}")
        return 1

    lines = report_path.read_text(encoding="utf-8").splitlines()
    errors: list[str] = []

    if not lines or not lines[0].startswith("# Weekly Handoff - "):
        errors.append("title: first line must start with '# Weekly Handoff - '")

    for section in REQUIRED_SECTIONS:
        if find_section(lines, section) < 0:
            errors.append(f"missing section: {section}")

    metadata = section_lines(lines, "## Metadata")
    for key in REQUIRED_METADATA_KEYS:
        check_bullet_key(metadata, key, errors, "Metadata")

    contract = section_lines(lines, "## Contract Sync")
    for key in REQUIRED_CONTRACT_KEYS:
        check_bullet_key(contract, key, errors, "Contract Sync")

    gates_section = section_lines(lines, "## CI Gates")
    gate_rows = extract_table(gates_section)
    if len(gate_rows) < 3:
        errors.append("CI Gates: missing table rows")
    else:
        if gate_rows[0] != ["gate", "status", "evidence"]:
            errors.append("CI Gates: required columns are gate|status|evidence")
        gate_names = {row[0] for row in gate_rows[2:] if len(row) >= 3}
        for gate in REQUIRED_GATES:
            if gate not in gate_names:
                errors.append(f"CI Gates: missing gate row '{gate}'")

    run_pack_section = section_lines(lines, "## Run-Pack Inventory")
    run_pack_rows = extract_table(run_pack_section)
    if len(run_pack_rows) < 3:
        errors.append("Run-Pack Inventory: missing inventory rows")
    else:
        if run_pack_rows[0] != RUN_PACK_COLUMNS:
            errors.append("Run-Pack Inventory: required columns missing or reordered")
        for index, row in enumerate(run_pack_rows[2:], start=1):
            if len(row) != len(RUN_PACK_COLUMNS):
                errors.append(f"Run-Pack Inventory: row {index} has wrong column count")
                continue
            data = dict(zip(RUN_PACK_COLUMNS, row))
            direction = data["evidence_direction"]
            if direction not in VALID_DIRECTIONS:
                errors.append(
                    f"Run-Pack Inventory: row {index} invalid evidence_direction '{direction}'"
                )
            if not data["pack_path"]:
                errors.append(f"Run-Pack Inventory: row {index} missing pack_path")
            execution_mode = data["execution_mode"]
            if execution_mode not in VALID_EXECUTION_MODE:
                errors.append(
                    f"Run-Pack Inventory: row {index} invalid execution_mode '{execution_mode}'"
                )
            if not data["runtime_minutes"]:
                errors.append(f"Run-Pack Inventory: row {index} missing runtime_minutes")

    claim_section = section_lines(lines, "## Claim Summary")
    claim_rows = extract_table(claim_section)
    if len(claim_rows) < 3:
        errors.append("Claim Summary: missing summary rows")
    else:
        if claim_rows[0] != CLAIM_SUMMARY_COLUMNS:
            errors.append("Claim Summary: required columns missing or reordered")

    blockers = section_lines(lines, "## Open Blockers")
    blocker_bullets = [line.strip() for line in blockers[1:] if line.strip().startswith("- ")]
    blocker_nonempty = [line.strip() for line in blockers[1:] if line.strip()]
    if not blocker_bullets and not any(line.lower() in {"none.", "none", "n/a"} for line in blocker_nonempty):
        errors.append("Open Blockers: expected blocker bullets or explicit 'None.'")

    parity_present = any("parity note" in line.lower() for line in claim_section[1:] + blockers[1:])
    if not parity_present:
        errors.append("Missing parity note (must appear in Claim Summary or Open Blockers)")

    local_watch = section_lines(lines, "## Local Compute Options Watch")
    for key in LOCAL_WATCH_KEYS:
        check_bullet_key(local_watch, key, errors, "Local Compute Options Watch")

    action_pattern = re.compile(r"^-\s+recommended_local_action:\s+`([^`]+)`\s*$")
    matched = None
    for line in local_watch:
        found = action_pattern.match(line.strip())
        if found:
            matched = found.group(1)
            break
    if matched is None:
        errors.append("Local Compute Options Watch: missing recommended_local_action value")
    elif matched not in VALID_RECOMMENDED_LOCAL_ACTION:
        errors.append(
            "Local Compute Options Watch: recommended_local_action must be "
            "hold_cloud_only|upgrade_low|upgrade_mid|upgrade_high|N/A"
        )

    if errors:
        print(f"FAIL: weekly handoff validation failed for {report_path}")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"PASS: weekly handoff validation passed for {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
