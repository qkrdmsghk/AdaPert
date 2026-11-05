"""
AdaPert model: the TxPert backbone with a perturbation-conditioned context module.

The candidates of a perturbation p are the training perturbations within k hops of p in the knowledge graph (p itself
is never a candidate). A scorer conditioned on the semantic embedding s_p selects among them, and the weighted mean
r_p of the observed effects ΔX of the selected perturbations is the context of p. The decoder reads
[z_c, h_p, W_r r_p, s_p].
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from torch import nn

from src.constants import CONTROL_LABEL
from src.data.graphmodule import GSPGraph
from src.models.txpert import TxPert

logger = logging.getLogger(__name__)

DEFAULT_CONTEXT: Dict[str, Any] = {
    "k_hop": 3,               # candidates: training perturbations within k hops of p
    "hidden_dim": 64,         # scorer hidden size H
    "threshold": 0.1,         # T: keep a candidate if |C_p| * alpha > T, then renormalise
    "tau": 0.0,               # Gumbel-softmax temperature at training time (0 = no sampling)
    "align_weight": 0.0,      # weight of the alignment loss L_align
    "scorer_lr_scale": 1.0,   # scorer learning rate = lr * scorer_lr_scale
}


def build_hop_table(edge_index: torch.Tensor, num_nodes: int, targets: np.ndarray, k: int) -> np.ndarray:
    """Undirected shortest-path hop distance from every node to each target node, up to k.

    Args:
        edge_index: [2, E] directed edges.
        num_nodes: Number of graph nodes N.
        targets: [R] target node ids.
        k: Largest hop distance (>= 1).

    Returns:
        [N, R] uint8 distances in 1..k; 0 = farther than k hops, or the target itself.
    """
    if k < 1:
        raise ValueError(f"context.k_hop must be >= 1, got {k}")
    ei = np.asarray(edge_index.cpu() if torch.is_tensor(edge_index) else edge_index).astype(np.int64)
    adj = csr_matrix((np.ones(ei.shape[1], dtype=np.float32), (ei[0], ei[1])), shape=(num_nodes, num_nodes))
    adj = ((adj + adj.T) > 0).astype(np.float32).tocsr()
    targets = np.asarray(targets, dtype=np.int64)
    hop = np.zeros((num_nodes, len(targets)), dtype=np.uint8)
    walk = adj[:, targets]                                          # [N, R] walks of length 1
    for d in range(1, k + 1):
        if d > 1:
            walk = ((adj @ walk) > 0).astype(np.float32)            # walks of length d
        hop[(walk.toarray() > 0) & (hop == 0)] = d                  # first d with a walk = shortest distance
    hop[targets, np.arange(len(targets))] = 0                       # a node is never its own candidate
    return hop


class ContextScorer(nn.Module):
    """Perturbation-conditioned scorer over the candidate neighbours.

    The logit of candidate q for perturbation p is
        a_pq = w^T LeakyReLU(W_h h_q + E[d_pq] + W_s s_p) + b[d_pq],
    where d_pq in 1..k is the hop distance. The weights are a softmax over the candidates (with Gumbel noise at
    training time); candidates with |C_p| * alpha_pq <= T are dropped and the rest renormalised.

    Args:
        node_dim: Dimension d of the node embeddings h.
        sem_dim: Dimension of the semantic (GenePT) perturbation embedding s_p.
        hidden_dim: Hidden size H of the scoring MLP.
        k_hop: Largest hop distance k of the candidates.
        threshold: Selection threshold T in [0, 1) (0 keeps all candidates).
        tau: Gumbel-softmax temperature at training time (0 = no sampling).
    """

    def __init__(self, node_dim: int, sem_dim: int, hidden_dim: int, k_hop: int, threshold: float = 0.1,
                 tau: float = 0.0):
        super().__init__()
        if not 0.0 <= threshold < 1.0:
            raise ValueError(f"context.threshold must be in [0, 1), got {threshold}")
        if tau < 0:
            raise ValueError(f"context.tau must be >= 0, got {tau}")
        self.threshold, self.tau = float(threshold), float(tau)
        self.k = int(k_hop)
        self.W_h = nn.Linear(node_dim, hidden_dim)
        self.W_s = nn.Linear(sem_dim, hidden_dim, bias=False)
        self.E = nn.Embedding(self.k + 1, hidden_dim)                   # hop-distance embedding (row 0 unused)
        self.w = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.w.weight)
        nn.init.zeros_(self.w.bias)
        # hop bias b[d] = -2.5 (d - 1): at initialisation the nearer candidates have the larger weights
        self.b = nn.Parameter(torch.tensor([0.0] + [2.5 * (1 - d) for d in range(1, self.k + 1)]))  # [k + 1]

    def forward(
        self, h_q: torch.Tensor, s: torch.Tensor, hop: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Selection weights of each perturbation over the training perturbations.

        Args:
            h_q: [R, d] node embeddings of the training perturbations (effect-table rows).
            s: [U, sem_dim] semantic embedding of each perturbation.
            hop: [U, R] long hop distance to each training perturbation (0 = not a candidate).

        Returns:
            alpha [U, R] (rows without candidates are zero), has [U] bool, and a dict of detached statistics.
        """
        z = self.W_h(h_q).unsqueeze(0) + self.E(hop)                    # [U, R, H]
        z = z + self.W_s(s).unsqueeze(1)
        a = self.w(F.leaky_relu(z)).squeeze(-1) + self.b[hop]           # [U, R]
        mask = hop > 0
        if self.training and self.tau > 0:
            u = torch.rand_like(a).clamp_(1e-10, 1.0 - 1e-10)
            a = a - torch.log(-torch.log(u)) / self.tau                 # Gumbel noise
        has = mask.any(dim=1, keepdim=True)                             # [U, 1]
        # rows without candidates keep finite logits (zeroed below): softmax never sees an all -inf row
        a = a.masked_fill(~mask & has, float("-inf"))
        alpha = torch.softmax(a, dim=1) * mask                          # [U, R]
        if self.threshold > 0:
            n = mask.sum(1, keepdim=True).clamp(min=1)
            alpha = alpha * (alpha * n > self.threshold)
            alpha = alpha / alpha.sum(1, keepdim=True).clamp(min=1e-12)  # weighted mean over the kept candidates

        with torch.no_grad():
            hv = has.squeeze(1)
            zero = torch.zeros((), device=alpha.device)
            ent = -(alpha * alpha.clamp(min=1e-12).log()).sum(1)        # [U]
            stats = {
                "ctx/n_candidates": mask.sum(1).float().mean(),
                "ctx/n_selected": (alpha > 0).sum(1).float()[hv].mean() if hv.any() else zero,
                "ctx/frac_empty": (~hv).float().mean(),
                "ctx/eff": ent[hv].exp().mean() if hv.any() else zero,    # effective number of neighbours
            }
            for d in range(1, self.k + 1):
                stats[f"ctx/w_hop{d}"] = (alpha * (hop == d)).sum(1)[hv].mean() if hv.any() else zero
        return alpha, hv, stats


class AdaPert(TxPert):
    """TxPert backbone with a perturbation-conditioned context module.

    Args:
        context: Context options (see DEFAULT_CONTEXT).
        Other arguments: see TxPert.
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
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            input_dim=input_dim,
            p_emb_dim=p_emb_dim,
            adata_output_dim=adata_output_dim,
            graph=graph,
            pert_model=pert_model,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout,
            non_deg_weight=non_deg_weight,
        )
        unknown = set(context or {}) - set(DEFAULT_CONTEXT)
        if unknown:
            raise ValueError(f"Unknown context options: {sorted(unknown)}")

        self.ctx_cfg = {**DEFAULT_CONTEXT, **(context or {})}
        self.k_hop = int(self.ctx_cfg["k_hop"])
        self.align_weight = float(self.ctx_cfg["align_weight"])
        self.scorer_lr_scale = float(self.ctx_cfg["scorer_lr_scale"])    # own param group in configure_optimizers
        if self.scorer_lr_scale <= 0:
            raise ValueError(f"context.scorer_lr_scale must be > 0, got {self.scorer_lr_scale}")

        self.scorer = ContextScorer(
            node_dim=latent_dim,
            sem_dim=p_emb_dim,
            hidden_dim=int(self.ctx_cfg["hidden_dim"]),
            k_hop=self.k_hop,
            threshold=float(self.ctx_cfg["threshold"]),
            tau=float(self.ctx_cfg["tau"]),
        )
        logger.info("AdaPert context: %s", self.ctx_cfg)
        # hop distances to the effect-table rows: rebuilt with the table (set / load), not stored
        self.register_buffer("hop", torch.zeros(0, 0, dtype=torch.uint8), persistent=False)
        self.register_buffer("table_nodes", torch.zeros(0, dtype=torch.long), persistent=False)
        self.ctx_proj = nn.Linear(self.adata_output_dim, latent_dim, bias=False)       # W_r: ΔX -> d
        # effect table: pseudo-bulk ΔX of the training perturbations (training cells only), stored in the checkpoint
        self.register_buffer("effect_row", torch.full((self.num_nodes,), -1, dtype=torch.long))
        self.register_buffer("effect_table", torch.zeros(0, self.adata_output_dim))
        # Decoder input = [z_c, h_p, z_ctx, s_p]; the z_ctx columns start at zero -> the no-context model at init.
        self.decoder = self.build_decoder(3 * latent_dim + p_emb_dim, hidden_dim, dropout)
        with torch.no_grad():
            self.decoder[0].weight[:, 2 * latent_dim: 3 * latent_dim].zero_()
        self._effect, self._effect_nodes, self._effect_has = None, None, None
        self._ctx_stats: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------ effect table
    def set_effect_table(self, node_ids: torch.Tensor, Y: torch.Tensor) -> None:
        """Register the observed pseudo-bulk effects of the training perturbations.

        Args:
            node_ids: [P] graph node ids of the training perturbations.
            Y: [P, n_genes] pseudo-bulk ΔX from training cells only.
        """
        assert Y.dim() == 2 and Y.size(1) == self.adata_output_dim, f"bad effect table shape {tuple(Y.shape)}"
        assert len(node_ids) == Y.size(0), "node_ids / Y length mismatch"
        assert torch.isfinite(Y).all(), "non-finite effect table"
        dev = self.effect_row.device
        row = torch.full((self.num_nodes,), -1, dtype=torch.long, device=dev)
        row[node_ids.long().to(dev)] = torch.arange(len(node_ids), device=dev)
        self.effect_row = row
        self.effect_table = Y.float().to(dev)
        self._build_hop_table()

    def _build_hop_table(self) -> None:
        """Hop distance from every graph node to each effect-table row (training perturbation)."""
        row = self.effect_row.cpu()
        nodes = (row >= 0).nonzero().squeeze(1)
        R = self.effect_table.size(0)
        assert len(nodes) == R, f"effect table rows ({R}) and indexed nodes ({len(nodes)}) differ"
        table_nodes = torch.empty(R, dtype=torch.long)
        table_nodes[row[nodes]] = nodes                                           # [R] graph id of each row
        hop = torch.as_tensor(build_hop_table(self.edge_index, self.num_nodes, table_nodes.numpy(),
                                              self.k_hop))                        # [N, R] uint8
        dev = self.effect_row.device
        self.table_nodes, self.hop = table_nodes.to(dev), hop.to(dev)
        own = hop[table_nodes]                                                    # candidates of training perturbations
        logger.info("AdaPert context: %d training perturbations, %.1f candidates per training perturbation "
                    "(%s by hop)", R, (own > 0).sum(1).float().mean(),
                    " / ".join(f"{(own == d).sum(1).float().mean():.1f}" for d in range(1, self.k_hop + 1)))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # effect_table has a data-dependent number of rows: resize before the default copy
        key = prefix + "effect_table"
        if key in state_dict:
            self.effect_table = torch.empty_like(state_dict[key], device=self.effect_table.device)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        if self.effect_table.size(0) > 0:
            self._build_hop_table()

    # ------------------------------------------------------------------ optimizer groups
    def param_groups(self, lr: float) -> List[Dict[str, Any]]:
        """The scorer has its own learning rate lr * context.scorer_lr_scale."""
        return [
            {"params": [p for n, p in self.named_parameters() if not n.startswith("scorer.")], "lr": lr},
            {"params": list(self.scorer.parameters()), "lr": lr * self.scorer_lr_scale},
        ]

    # ------------------------------------------------------------------ forward
    def forward(self, cntr: torch.Tensor, pert_idxs: Sequence[Sequence], p_emb: torch.Tensor) -> torch.Tensor:
        """Predicted expression [B, G] (cntr [B, input_dim], p_emb [B, p_emb_dim] = s_p)."""
        z_c, h, h_p, rows_t, nodes_t = self.encode(cntr, pert_idxs)             # [B, d], [N, d], [B, d]
        dev = z_c.device
        z_ctx = torch.zeros(cntr.size(0), h.size(-1), device=dev)                # [B, d], zero for controls
        self._effect, self._effect_nodes, self._effect_has = None, None, None
        self._ctx_stats = {}
        is_pert = nodes_t >= 0
        if is_pert.any():
            assert self.effect_table.size(0) > 0, "effect table is empty: call set_effect_table first"
            p_rows, p_nodes = rows_t[is_pert], nodes_t[is_pert]
            uniq, inv = torch.unique(p_nodes, return_inverse=True)
            first = torch.zeros(len(uniq), dtype=torch.long, device=dev).scatter_(0, inv, p_rows)
            # the scorer reads h without gradient (it does not train the GNN)
            alpha, has_u, self._ctx_stats = self.scorer(
                h.detach()[self.table_nodes], p_emb[first], self.hop[uniq].long()
            )                                                                    # [U, R]
            r_u = alpha @ self.effect_table                                      # [U, G] r_p
            z_ctx = z_ctx.index_add(0, p_rows, self.ctx_proj(r_u)[inv])
            self._effect, self._effect_nodes, self._effect_has = r_u, uniq, has_u

        return self.decoder(torch.cat([z_c, h_p, z_ctx, p_emb], dim=-1))       # [B, G]

    # ------------------------------------------------------------------ loss
    def loss(
        self, prediction: torch.Tensor, target: torch.Tensor, control: torch.Tensor, mask_degs: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """TxPert loss + align_weight * L_align (see TxPert.loss for the arguments)."""
        total_loss, logs = super().loss(prediction, target, control, mask_degs)
        logs.update(self._ctx_stats)
        if self.align_weight > 0 and self.training and self._effect is not None:
            # L_align = 1 - Pearson(r_p, ΔX_p) over the training perturbations; p is never its own candidate
            row = self.effect_row[self._effect_nodes]
            ok = self._effect_has & (row >= 0)
            if ok.any():
                r = self._effect[ok] - self._effect[ok].mean(1, keepdim=True)                       # [n, G]
                y = self.effect_table[row[ok]] - self.effect_table[row[ok]].mean(1, keepdim=True)   # [n, G]
                pcc = (r * y).sum(1) / (r.norm(dim=1) * y.norm(dim=1) + 1e-8)
                align_loss = (1.0 - pcc).mean()
                total_loss = total_loss + self.align_weight * align_loss
                logs["align_loss"] = align_loss.detach()
        return total_loss, logs


@torch.no_grad()
def compute_effects(datamodule) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pseudo-bulk ΔX of every training perturbation, from training cells only.

    ΔX uses the same matched control as the training loss (dataset._get_control).

    Args:
        datamodule: Prepared PertDataModule after setup("fit").

    Returns:
        node_ids [P] (graph/pert ids) and Y [P, n_genes].
    """
    ds = datamodule.train_data
    X = ds.pert_data                                                   # [n_rows, G] (controls appended)
    ctrl_cache: Dict[Any, torch.Tensor] = {}                           # matched control of each batch
    sums: Dict[int, torch.Tensor] = {}
    counts: Dict[int, int] = {}
    for i in range(len(ds)):
        perts = ds.pert_conditions.iloc[i]
        if len(perts) != 1 or perts[0] == CONTROL_LABEL:
            continue
        node = int(perts[0])
        key = ds.treatment_cell_batches.iloc[i]
        if key not in ctrl_cache:
            ctrl_cache[key] = ds._get_control(i)
        d = X[i] - ctrl_cache[key]
        sums[node] = sums[node] + d if node in sums else d.clone()
        counts[node] = counts.get(node, 0) + 1

    node_ids = list(sums)
    Y = torch.stack([sums[n] / counts[n] for n in node_ids])
    assert torch.isfinite(Y).all(), "non-finite effects"
    logger.info("Effects: %d training perturbations x %d genes", Y.size(0), Y.size(1))
    return torch.tensor(node_ids, dtype=torch.long), Y
