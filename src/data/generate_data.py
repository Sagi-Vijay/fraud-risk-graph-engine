from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass
class GeneratorConfig:
    n_users: int = 2000
    n_transactions: int = 12000
    fraud_ratio: float = 0.04
    seed: int = 42


def _choice(rng: random.Random, values: List[str]) -> str:
    return values[rng.randint(0, len(values) - 1)]


def generate_identities(cfg: GeneratorConfig) -> pd.DataFrame:
    rng = random.Random(cfg.seed)
    np.random.seed(cfg.seed)

    domains = ["gmail.com", "yahoo.com", "outlook.com", "proton.me"]
    suspicious_domains = ["mail-temp.net", "quickbox.cc", "temp-user.org"]
    cities = ["Chicago", "New York", "Dallas", "Phoenix", "Atlanta", "Seattle"]

    rows = []
    synthetic_user_ids = set(rng.sample(range(cfg.n_users), int(cfg.n_users * 0.05)))

    for i in range(cfg.n_users):
        user_id = f"U{i:06d}"
        base_age = int(np.clip(np.random.normal(35, 11), 18, 79))
        synthetic_flag = 1 if i in synthetic_user_ids else 0

        if synthetic_flag:
            email_domain = _choice(rng, suspicious_domains)
            device_id = f"D_SHARED_{rng.randint(1, 40):03d}"
            ip_address = f"10.0.{rng.randint(1, 8)}.{rng.randint(1, 20)}"
            email = f"user{rng.randint(10000,99999)}_{rng.randint(1,999)}@{email_domain}"
        else:
            email_domain = _choice(rng, domains)
            device_id = f"D{rng.randint(1, cfg.n_users * 2):07d}"
            ip_address = f"192.168.{rng.randint(1, 200)}.{rng.randint(1, 254)}"
            email = f"user{i}_{rng.randint(100,999)}@{email_domain}"

        phone = f"555{rng.randint(1000000, 9999999)}"
        city = _choice(rng, cities)
        account_age_days = int(np.clip(np.random.exponential(240), 1, 1800))
        rows.append(
            {
                "user_id": user_id,
                "email": email,
                "device_id": device_id,
                "ip_address": ip_address,
                "phone": phone,
                "city": city,
                "age": base_age,
                "account_age_days": account_age_days,
                "is_synthetic_identity": synthetic_flag,
            }
        )

    return pd.DataFrame(rows)


def generate_transactions(identities: pd.DataFrame, cfg: GeneratorConfig) -> pd.DataFrame:
    rng = random.Random(cfg.seed + 1)
    np.random.seed(cfg.seed + 1)

    merchant_categories = [
        "retail", "electronics", "grocery", "travel", "gaming", "fintech", "gift_cards"
    ]
    base_time = datetime(2025, 1, 1)

    fraud_count = int(cfg.n_transactions * cfg.fraud_ratio)
    fraud_indices = set(rng.sample(range(cfg.n_transactions), fraud_count))

    user_records = identities.to_dict("records")
    rows = []

    for i in range(cfg.n_transactions):
        person = user_records[rng.randint(0, len(user_records) - 1)]
        is_fraud = 1 if i in fraud_indices else 0
        synthetic_identity = int(person["is_synthetic_identity"])

        if is_fraud:
            amount = round(np.random.lognormal(mean=5.4, sigma=0.9), 2)
            category = _choice(rng, ["electronics", "gift_cards", "travel", "fintech"])
            hour = rng.choice([0, 1, 2, 3, 4, 23]) if hasattr(rng, 'choice') else [0,1,2,3,4,23][rng.randint(0,5)]
            device_id = person["device_id"] if rng.random() < 0.7 else f"D_SHARED_{rng.randint(1, 50):03d}"
            ip_address = person["ip_address"] if rng.random() < 0.6 else f"10.0.{rng.randint(1, 8)}.{rng.randint(1, 20)}"
            velocity_1h = rng.randint(4, 15)
            velocity_24h = velocity_1h + rng.randint(5, 30)
            geo_mismatch = rng.randint(0, 1)
            email_domain_risk = int(any(person["email"].endswith(x) for x in ["mail-temp.net", "quickbox.cc", "temp-user.org"]))
        else:
            amount = round(np.random.lognormal(mean=3.7, sigma=0.55), 2)
            category = _choice(rng, merchant_categories)
            hour = rng.randint(6, 22)
            device_id = person["device_id"]
            ip_address = person["ip_address"]
            velocity_1h = rng.randint(1, 3)
            velocity_24h = velocity_1h + rng.randint(0, 6)
            geo_mismatch = 1 if rng.random() < 0.03 else 0
            email_domain_risk = int(any(person["email"].endswith(x) for x in ["mail-temp.net", "quickbox.cc", "temp-user.org"]))

        event_time = base_time + timedelta(minutes=rng.randint(0, 60 * 24 * 90))
        rows.append(
            {
                "transaction_id": f"T{i:07d}",
                "user_id": person["user_id"],
                "email": person["email"],
                "device_id": device_id,
                "ip_address": ip_address,
                "city": person["city"],
                "merchant_category": category,
                "amount": amount,
                "event_timestamp": event_time.isoformat(),
                "event_hour": hour,
                "velocity_1h": velocity_1h,
                "velocity_24h": velocity_24h,
                "geo_mismatch": geo_mismatch,
                "account_age_days": person["account_age_days"],
                "email_domain_risk": email_domain_risk,
                "synthetic_identity_flag": synthetic_identity,
                "is_fraud": is_fraud,
            }
        )

    return pd.DataFrame(rows)


def save_datasets(output_dir: str | Path = "data/raw", cfg: GeneratorConfig | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cfg = cfg or GeneratorConfig()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    identities = generate_identities(cfg)
    transactions = generate_transactions(identities, cfg)

    identities.to_csv(output_path / "identities.csv", index=False)
    transactions.to_csv(output_path / "transactions.csv", index=False)
    return identities, transactions


if __name__ == "__main__":
    ids, tx = save_datasets()
    print(f"Saved identities: {ids.shape}, transactions: {tx.shape}")
