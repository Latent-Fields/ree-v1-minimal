import argparse
import json
import os
import random
import datetime
import torch

from ree_core.agent import REEAgent
from experiments.metrics import compute_summary


def load_suites():
    with open("experiments/suites.json", "r") as f:
        return json.load(f)


def create_run_dir(suite_name):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"runs/{timestamp}_{suite_name}"
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def run_episode(agent, steps=200):
    total_harm = 0
    residue_over_time = []

    state = agent.reset()

    for _ in range(steps):
        action = agent.act(state)
        state, reward, done, info = agent.step(action)

        total_harm += info.get("harm", 0)
        residue_over_time.append(agent.residue_field.total_mass())

        if done:
            break

    return {
        "total_harm": total_harm,
        "residue_curve": residue_over_time,
        "steps": len(residue_over_time)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    suites = load_suites()
    suite = suites[args.suite]

    run_dir = create_run_dir(args.suite)

    agent = REEAgent()

    # Apply overrides
    if "overrides" in suite:
        e3_overrides = suite["overrides"].get("e3", {})
        for k, v in e3_overrides.items():
            setattr(agent.e3.config, k, v)

    result = run_episode(agent)

    summary = compute_summary(result)

    with open(f"{run_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Run complete:", summary)


if __name__ == "__main__":
    main()