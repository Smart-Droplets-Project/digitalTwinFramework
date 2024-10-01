def minimize_risk(info: dict):
    action = 0
    if info["Risk"] and info["Risk"][-1] > 0.05:
        action = 1.0
    return action
