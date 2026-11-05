"""
TxPert backbone (modified from TxPert, https://github.com/valence-labs/TxPert).

z_c encodes the control expression x_c (BasalMLP), h_p is the structural embedding of perturbation p from a GATv2 over
the knowledge graph (GNN), and s_p is the semantic (GenePT) embedding of p. AdaPert (src/models/adapert.py) adds the
context and replaces the decoder.
"""
import logging
from typing import Any, Dict, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from src.constants import CONTROL_LABEL
from src.data.graphmodule import GSPGraph
from src.models.basal_state_models.mlp import BasalMLP
from src.models.pert_models.basic_gnn import GNN

logger = logging.getLogger(__name__)

CTRL_IDX = -1  # graph node read for the control label


class TxPert(nn.Module):
    """Basal-state encoder, GNN over the knowledge graph, decoder and loss.

    Args:
        input_dim: Number of genes of the control expression x_c.
        p_emb_dim: Dimension of the semantic perturbation embedding s_p.
        adata_output_dim: Number of predicted genes G.
        graph: Knowledge graph.
        pert_model: GNN options (num_layers, hidden_dim, num_heads, dropout).
        hidden_dim: Decoder hidden size.
        latent_dim: Dimension d of z_c and h_p.
        dropout: Decoder dropout rate.
        non_deg_weight: Weight of the non-DEG loss.
    """

    def __init__(
        self,
        input_dim: int,
        p_emb_dim: int,
        adata_output_dim: int,
        graph: GSPGraph,
        pert_model: Dict[str, Any],
        hidden_dim: int = 512,
        latent_dim: int = 64,
        dropout: float = 0.2,
        non_deg_weight: float = 0.0,
    ):
        super().__init__()
        self.p_emb_dim = p_emb_dim
        self.adata_output_dim = adata_output_dim
        self.non_deg_weight = non_deg_weight
        self.num_nodes = int(graph.num_nodes)

        self.cntr_model = BasalMLP(input_dim, latent_dim)                                    # x_c -> z_c
        self.pert_model = GNN(num_nodes=self.num_nodes, out_dim=latent_dim, **pert_model)  # graph -> h_v
        self.decoder = self.build_decoder(2 * latent_dim + p_emb_dim, hidden_dim, dropout)  # [z_c, h_p, s_p] -> x̂
        self.register_buffer("edge_index", torch.as_tensor(graph.edge_index, dtype=torch.long), persistent=False)
        logger.info("Graph: %d nodes, %d edges; decoder input %d", self.num_nodes, self.edge_index.size(1),
                    self.decoder[0].in_features)

    def build_decoder(self, in_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
        """in_dim -> hidden_dim (BatchNorm, LeakyReLU, dropout) -> G."""
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.adata_output_dim),
        )

    def encode(
        self, cntr: torch.Tensor, pert_idxs: Sequence[Sequence]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Basal state and perturbation embeddings of a batch.

        Args:
            cntr: [B, input_dim] control expression x_c.
            pert_idxs: Graph node ids of the perturbations of each cell (CONTROL_LABEL for control cells).

        Returns:
            z_c [B, d], h [N, d], h_p [B, d] (sum of h over the perturbations of a cell), and the cell row and graph
            node of each perturbation ([P] each; the control label reads node CTRL_IDX).
        """
        z_c = self.cntr_model(cntr)                                                    # [B, d]
        h = self.pert_model(self.edge_index)                                           # [N, d]
        rows, nodes = [], []
        for i, perts in enumerate(pert_idxs):
            for p in perts:
                rows.append(i)
                nodes.append(CTRL_IDX if p == CONTROL_LABEL else int(p))
        rows_t = torch.tensor(rows, dtype=torch.long, device=z_c.device)
        nodes_t = torch.tensor(nodes, dtype=torch.long, device=z_c.device)
        h_p = torch.zeros(cntr.size(0), h.size(-1), device=z_c.device).index_add(0, rows_t, h[nodes_t])  # [B, d]
        return z_c, h, h_p, rows_t, nodes_t

    def loss(
        self, prediction: torch.Tensor, target: torch.Tensor, control: torch.Tensor, mask_degs: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """recon_loss + non_deg_weight * non_deg_loss.

        recon_loss is the mean squared error over all genes. non_deg_loss is the mean |x̂ - x_c| over the genes that are
        not DEGs of the cell's perturbation: it pulls the predicted change of these genes towards zero.

        Args:
            prediction: [B, G] predicted expression x̂.
            target: [B, G] observed expression x.
            control: [B, G] matched control expression x_c.
            mask_degs: [B, G] bool, True for the DEGs of each cell's perturbation (all True when unknown).
        """
        recon_loss = F.mse_loss(prediction, target)
        if mask_degs.any() and not mask_degs.all():
            non_deg_loss = (prediction - control)[~mask_degs].abs().mean()
        else:
            non_deg_loss = torch.zeros((), device=prediction.device)
        total_loss = recon_loss + self.non_deg_weight * non_deg_loss
        return total_loss, {"recon_loss": recon_loss, "non_deg_loss": non_deg_loss}
