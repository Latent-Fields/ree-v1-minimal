#!/usr/bin/env python3
"""Validate weekly handoff report format against shared template contract."""

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

RUN_PACK_COLUMNS = [
    "experiment_type",
    "run_id",
    "seed",
    "condition_or_scenario",
    "status",
    "evidence_direction",
    "claim_ids_tested",
    "failure_signatures",
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

EVIDENCE_DIRECTIONS = {"supports", "weakens", "mixed", "unknown"}


def normalize_table_header(line: str) -> list[str]:
    parts = [p.strip() for p in line.strip().strip("|").split("|")]
    return parts


def find_section_index(lines: list[str], section: str) -> int:
    try:
        return lines.index(section)
    except ValueError:
        return -1


def section_slice(lines: list[str], section: str) -> list[str]:
    start = find_section_index(lines, section)
    if start < 0:
        return []
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].startswith("## "):
            end = i
            break
    return lines[start:end]


def ensure_bullet_key(section_lines: list[str], key: str, errors: list[str], section_name: str) -> None:
    pattern = re.compile(rf"^-\s+{re.escape(key)}:\s+`?.+`?\s*$")
    if not any(pattern.match(line.strip()) for line in section_lines):
        errors.append(f"{section_name}: missing key '{key}'")


def parse_table_rows(section_lines: list[str]) -> list[list[str]]:
    table_lines = [line.strip() for line in section_lines if line.strip().startswith("|")]
    return [normalize_table_header(line) for line in table_lines]


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate weekly handoff report template compliance.")
    parser.add_argument(
        "--report",
        default="evidence/planning/weekly_handoff_current.md",
        help="Path to weekly handoff markdown file.",
    )
    args = parser.parse_args()

    report_path = Path(args.report)
    errors: list[str] = []

    if not report_path.exists():
        print(f"Weekly handoff validation failed: report missing ({report_path})")
        return 1

    lines = [line.rstrip("\n") for line in report_path.read_text(encoding="utf-8").splitlines()]

    if not lines or not lines[0].startswith("# Weekly Handoff - "):
        errors.append("title: missing or invalid '# Weekly Handoff - ...' heading")

    for section in REQUIRED_SECTIONS:
        if find_section_index(lines, section) < 0:
            errors.append(f"missing section: {section}")

    metadata = section_slice(lines, "## Metadata")
    for key in REQUIRED_METADATA_KEYS:
        ensure_bullet_key(metadata, key, errors, "Metadata")

    contract = section_slice(lines, "## Contract Sync")
    for key in REQUIRED_CONTRACT_KEYS:
        ensure_bullet_key(contract, key, errors, "Contract Sync")

    ci_section = section_slice(lines, "## CI Gates")
    ci_rows = parse_table_rows(ci_section)
    if len(ci_rows) < 3:
        errors.append("CI Gates: missing gate table")
    else:
        if ci_rows[0] != ["gate", "status", "evidence"]:
            errors.append("CI Gates: required columns are gate|status|evidence")
        gate_names = {row[0] for row in ci_rows[2:] if len(row) >= 3}
        for gate in ("schema_validation", "seed_determinism", "hook_surface_coverage"):
            if gate not in gate_names:
                errors.append(f"CI Gates: missing gate row '{gate}'")

    run_pack_section = section_slice(lines, "## Run-Pack Inventory")
    run_pack_rows = parse_table_rows(run_pack_section)
    if len(run_pack_rows) < 3:
        errors.append("Run-Pack Inventory: missing inventory rows")
    else:
        if run_pack_rows[0] != RUN_PACK_COLUMNS:
            errors.append("Run-Pack Inventory: required columns missing or reordered")
        for idx, row in enumerate(run_pack_rows[2:], start=1):
            if len(row) != len(RUN_PACK_COLUMNS):
                errors.append(f"Run-Pack Inventory: row {idx} has wrong column count")
                continue
            data = dict(zip(RUN_PACK_COLUMNS, row))
            if not data["evidence_direction"]:
                errors.append(f"Run-Pack Inventory: row {idx} missing evidence_direction")
            elif data["evidence_direction"] not in EVIDENCE_DIRECTIONS:
                errors.append(
                    f"Run-Pack Inventory: row {idx} invalid evidence_direction '{data['evidence_direction']}'"
                )
            if not data["pack_path"]:
                errors.append(f"Run-Pack Inventory: row {idx} missing pack_path")

    claim_section = section_slice(lines, "## Claim Summary")
    claim_rows = parse_table_rows(claim_section)
    if len(claim_rows) < 3:
        errors.append("Claim Summary: missing summary rows")
    else:
        if claim_rows[0] != CLAIM_SUMMARY_COLUMNS:
            errors.append("Claim Summary: required columns missing or reordered")

    blocker_section = section_slice(lines, "## Open Blockers")
    if not any(line.strip().startswith("- ") for line in blocker_section[1:]):
        errors.append("Open Blockers: expected at least one bullet")

    if errors:
        print("Weekly handoff validation failed:")
        for err in errors:
            print(f"- {err}")
        return 1

    print(f"Weekly handoff validation passed: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
