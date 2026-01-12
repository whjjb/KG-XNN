import pandas as pd
from pathlib import Path

def load_kg_csv(kg_dir: str = "./kg"):
    kg_dir = Path(kg_dir)
    nodes = pd.read_csv(kg_dir / "nodes.csv")
    edges = pd.read_csv(kg_dir / "edges.csv")
    attr  = pd.read_csv(kg_dir / "attr.csv")
    return nodes, edges, attr