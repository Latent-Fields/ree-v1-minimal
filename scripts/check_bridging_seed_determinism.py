#!/usr/bin/env python3
"""Determinism check for bridging qualification generators."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

SCRIPT_PATH = Path(__file__).resolve().with_name("run_mech_058_059_060_qualification.py")


def load_module():
    spec = importlib.util.spec_from_file_location("bridging_module", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    # Ensure decorators that inspect sys.modules during import can resolve module globals.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_seeds(raw: str) -> list[int]:
    seeds = sorted({int(x.strip()) for x in raw.split(",") if x.strip()})
    if not seeds:
        raise ValueError("at least one seed required")
    return seeds


def main() -> int:
    parser = argparse.ArgumentParser(description="Check deterministic metric generation for bridging runs.")
    parser.add_argument("--seeds", default="11,29", help="Comma-separated seed list.")
    parser.add_argument(
        "--timestamp-utc",
        default="2026-02-14T03:00:00Z",
        help="Timestamp used for deterministic run_id checks.",
    )
    args = parser.parse_args()

    module = load_module()
    seeds = parse_seeds(args.seeds)
    failures: list[str] = []
    checked = 0

    ts = module.normalize_timestamp_utc(args.timestamp_utc)

    for experiment_type, spec in module.EXPERIMENT_SPECS.items():
        for condition in spec["conditions"]:
            for seed in seeds:
                first = module.build_metrics(experiment_type, condition, seed)
                second = module.build_metrics(experiment_type, condition, seed)
                if first != second:
                    failures.append(
                        f"metrics mismatch: {experiment_type} condition={condition} seed={seed}"
                    )

                rid1 = module.deterministic_run_id(experiment_type, seed, ts)
                rid2 = module.deterministic_run_id(experiment_type, seed, ts)
                if rid1 != rid2:
                    failures.append(
                        f"run_id mismatch: {experiment_type} condition={condition} seed={seed}"
                    )

                checked += 1

    if failures:
        print("Seed determinism check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print(f"Seed determinism check passed for {checked} condition/seed pairs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
