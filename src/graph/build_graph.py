import networkx as nx
import pandas as pd


def build_entity_graph(df: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()

    for _, row in df.iterrows():
        u = f"user::{row['user_id']}"
        d = f"device::{row['device_id']}"
        ip = f"ip::{row['ip_address']}"
        e = f"email::{row['email']}"

        G.add_edge(u, d)
        G.add_edge(u, ip)
        G.add_edge(u, e)

    return G


def add_graph_features(df: pd.DataFrame, G: nx.Graph) -> pd.DataFrame:
    out = df.copy()

    def degree(node):
        return G.degree(node) if node in G else 0

    out["user_degree"] = out["user_id"].apply(lambda x: degree(f"user::{x}"))
    out["device_degree"] = out["device_id"].apply(lambda x: degree(f"device::{x}"))
    out["ip_degree"] = out["ip_address"].apply(lambda x: degree(f"ip::{x}"))

    return out
