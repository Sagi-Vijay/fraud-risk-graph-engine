from pathlib import Path

import pandas as pd

from src.features.feature_engineering import build_features
from src.graph.build_graph import build_entity_graph, add_graph_features
from src.models.train_model import train


def main() -> None:
    data_path = Path("data/raw/transactions.csv")
    if not data_path.exists():
        raise FileNotFoundError(
            "Could not find data/raw/transactions.csv. Run `python src/data/generate_data.py` first."
        )

    Path("artifacts/models").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    df = build_features(df)

    graph = build_entity_graph(df)
    df = add_graph_features(df, graph)

    metrics = train(df)
    print("Training complete")
    print(metrics)


if __name__ == "__main__":
    main()
