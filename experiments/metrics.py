"""Metrics helpers for experiment harness runs."""


def compute_metrics_values(result):
    """Build stable numeric metrics for Experiment Pack v1."""
    return {
        "total_harm": float(result.get("total_harm", 0.0)),
        "total_reward": float(result.get("total_reward", 0.0)),
        "final_residue": float(result.get("final_residue", 0.0)),
        "steps_survived": int(result.get("steps", 0)),
        "max_steps": int(result.get("max_steps", 0)),
        "done": int(result.get("done", 0)),
        "final_health": float(result.get("final_health", 0.0)),
        "final_energy": float(result.get("final_energy", 0.0)),
        "harm_event_count": int(result.get("harm_event_count", 0)),
        "hazard_event_count": int(result.get("hazard_event_count", 0)),
        "collision_event_count": int(result.get("collision_event_count", 0)),
        "resource_event_count": int(result.get("resource_event_count", 0)),
        "fatal_error_count": int(result.get("fatal_error_count", 0)),
    }


def compute_summary(result):
    """Backward-compatible summary values for quick terminal output."""
    return {
        "total_harm": float(result.get("total_harm", 0.0)),
        "final_residue": float(result.get("final_residue", 0.0)),
        "steps_survived": int(result.get("steps", 0)),
    }
