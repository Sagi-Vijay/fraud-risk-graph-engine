import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Basic transforms
    out["log_amount"] = (out["amount"] + 1).apply(lambda x: pd.np.log(x))

    # Velocity features
    out["high_velocity_1h"] = (out["velocity_1h"] >= 5).astype(int)
    out["high_velocity_24h"] = (out["velocity_24h"] >= 10).astype(int)

    # Night activity
    out["night_txn"] = out["event_hour"].isin([0,1,2,3,4,23]).astype(int)

    # Account maturity
    out["new_account"] = (out["account_age_days"] < 14).astype(int)

    # Simple risk aggregation
    out["rule_risk_score"] = (
        0.25 * out["high_velocity_1h"]
        + 0.2 * out["high_velocity_24h"]
        + 0.2 * out["geo_mismatch"]
        + 0.2 * out["email_domain_risk"]
        + 0.15 * out["night_txn"]
    )

    return out
