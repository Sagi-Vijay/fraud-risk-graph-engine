def decision_from_score(score: float):
    if score < 0.3:
        return "APPROVE"
    elif score < 0.7:
        return "REVIEW"
    else:
        return "DECLINE"
