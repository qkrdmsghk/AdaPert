from typing import Dict

import pandas as pd
import torch

import src.constants as cs

TOP_K = 10  # STRING edges kept per target gene


class GSPGraph:
    """STRING v11.5 knowledge graph over the master gene set: for each target gene, the TOP_K edges of highest score.

    Args:
        gene2id: Graph node id of each gene (master gene set).

    Attributes:
        edge_index: [2, E] directed edges (regulator, target).
        num_nodes: Number of graph nodes N (genes of the master gene set).
    """

    def __init__(self, gene2id: Dict[str, int]):
        network = pd.read_parquet(cs.DATA_DIR / "graphs/string/v11.5.parquet")   # regulator, target, weight (score)
        network["regulator"] = network["regulator"].map(gene2id)
        network["target"] = network["target"].map(gene2id)
        network = network.dropna()
        network = (
            network.groupby("target")[["regulator", "target", "weight"]]   # selected: each group keeps "target"
            .apply(lambda x: x.nlargest(TOP_K, ["weight"]))
            .reset_index(drop=True)
        )
        self.edge_index = torch.tensor(network[["regulator", "target"]].to_numpy().T, dtype=torch.long)  # [2, E]
        self.num_nodes = len(gene2id)
