import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATv2Conv


class GNN(nn.Module):
    """GATv2 encoder over the knowledge graph: a learned embedding per node -> structural embeddings h_v [N, d].

    Each layer concatenates its multi-head output with its input (skip connection) and applies LeakyReLU and dropout;
    a linear layer maps the result to d.

    Args:
        num_nodes: Number of graph nodes N.
        out_dim: Output dimension d.
        num_layers: Number of GATv2 layers.
        hidden_dim: Size of the node embeddings and of each attention head.
        num_heads: Attention heads per layer.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        num_nodes: int,
        out_dim: int,
        num_layers: int = 4,
        hidden_dim: int = 128,
        num_heads: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.gnn_layers = nn.ModuleList()
        in_dim = hidden_dim
        for _ in range(num_layers):
            self.gnn_layers.append(GATv2Conv(in_dim, hidden_dim, heads=num_heads, concat=True, add_self_loops=True))
            in_dim = in_dim + num_heads * hidden_dim                    # [layer output, layer input]
        self.out_nn = nn.Linear(in_dim, out_dim)
        self.pert_embeddings = nn.Embedding(num_nodes, hidden_dim)      # input node features

    def forward(self, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.pert_embeddings(torch.arange(self.num_nodes, device=edge_index.device))   # [N, hidden]
        for layer in self.gnn_layers:
            x = torch.cat([layer(x, edge_index), x], dim=-1)            # [N, heads * hidden + in]
            x = F.leaky_relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.leaky_relu(self.out_nn(x))                                # [N, d]
        return F.dropout(x, p=self.dropout, training=self.training)
