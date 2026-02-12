def compute_summary(result):
    return {
        "total_harm": result["total_harm"],
        "final_residue": result["residue_curve"][-1] if result["residue_curve"] else 0,
        "steps_survived": result["steps"]
    }