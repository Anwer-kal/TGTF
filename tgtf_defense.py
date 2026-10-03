"""TGTF defense: temporal graph trust fusion."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple

from config import TGTFConfig
from models import get_flat_params, set_flat_params


class TGTFDefense:
    """Implements the TGTF defense from Algorithm 1 of the paper.

    The defense maintains per-client trust, a temporal trust EMA with
    asymmetric decay/recovery, and a persistent similarity matrix over
    client updates. It produces aggregation weights for each round.
    """

    def __init__(self, num_clients: int, config: TGTFConfig, device="cpu"):
        self.N = num_clients
        self.cfg = config
        self.device = device

        # Trust state
        self.T = torch.ones(num_clients, device=device)          # T_i^t
        # Persistent similarity matrix
        self.P = torch.zeros(num_clients, num_clients, device=device)
        # Previous global update (for direction consistency)
        self.prev_global_update = None

        # Diagnostics
        self.history = {
            "R": [],           # trust scores per round
            "A_bar": [],       # normalized anomaly per round
            "T": [],           # temporal trust per round
            "G": [],           # collusion risk per round
            "alpha": [],       # aggregation weights per round
        }

    # ------------------------------------------------------------------
    # Core math
    # ------------------------------------------------------------------
    @staticmethod
    def _robust_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Median/MAD robust normalization: |x - median| / (MAD + eps)."""
        med = x.median()
        mad = (x - med).abs().median()
        return (x - med).abs() / (mad + eps)

    def _compute_anomaly(self, deltas: torch.Tensor) -> torch.Tensor:
        """Compute normalized per-client anomaly scores A_bar_i^t in [0,1]."""
        N = deltas.shape[0]
        eps = self.cfg.eps_num

        # Magnitude
        norms = deltas.norm(dim=1)                     # [N]
        Z_N = self._robust_normalize(norms, eps)

        # Direction consistency vs previous global update
        if self.prev_global_update is None:
            # First round: direction signal undefined -> set C_i = 0
            C = torch.zeros(N, device=deltas.device)
            Z_C = torch.ones(N, device=deltas.device)
        else:
            g = self.prev_global_update.to(deltas.device)
            # cosine similarity between each delta and g
            deltas_n = F.normalize(deltas, dim=1, eps=eps)
            g_n = F.normalize(g.unsqueeze(0), dim=1, eps=eps)
            C = (deltas_n * g_n).sum(dim=1)            # [N]
            Z_C = self._robust_normalize(C, eps)

        # Combine and normalize
        A = self.cfg.alpha_A * Z_N + (1 - self.cfg.alpha_A) * Z_C
        A_bar = torch.clamp(A / self.cfg.tau_norm, 0.0, 1.0)
        return A_bar

    def _update_temporal_trust(self, A_bar: torch.Tensor) -> torch.Tensor:
        """Apply asymmetric trust update (Eq. 4)."""
        Q = 1.0 - A_bar
        # Where A_bar > tau_A: degrade (lambda_-)
        # Where A_bar <= tau_A: recover (lambda_+)
        mask_anom = (A_bar > self.cfg.tau_A).float()
        lam = mask_anom * self.cfg.lambda_minus + (1 - mask_anom) * self.cfg.lambda_plus
        self.T = lam * self.T + (1 - lam) * Q
        return self.T

    def _update_persistent_similarity(self, deltas: torch.Tensor):
        """Update persistent similarity matrix P (Eq. 5)."""
        eps = self.cfg.eps_num
        deltas_n = F.normalize(deltas, dim=1, eps=eps)
        S = deltas_n @ deltas_n.t()                    # cosine [N,N]
        # Zero out diagonal
        S.fill_diagonal_(0.0)
        # Clamp negatives so persistent similarities are non-negative
        S = torch.clamp(S, min=0.0)
        self.P = self.cfg.rho * self.P + (1 - self.cfg.rho) * S

    def _collusion_risk(self, A_bar: torch.Tensor) -> torch.Tensor:
        """Compute collusion risk G_i (Eq. 6)."""
        # Adjacency: persistent similarity above threshold
        adj = (self.P > self.cfg.tau_G).float()
        adj.fill_diagonal_(0.0)

        # Weighted sum of neighbor anomalies, weighted by P_ij
        weighted_anom = (self.P * adj) @ A_bar         # [N]
        denom = (self.P * adj).sum(dim=1) + self.cfg.eps_num
        G = weighted_anom / denom
        # For clients with no persistent neighbors, G = 0
        G[denom < 1e-6] = 0.0
        return G

    def _fuse_trust(self, A_bar, T, G):
        """Fuse the three evidence streams into R_i^t (Eq. 7)."""
        R = (self.cfg.omega1 * (1 - A_bar)
             + self.cfg.omega2 * T
             + self.cfg.omega3 * (1 - G))
        return torch.clamp(R, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def aggregate(self, deltas: List[torch.Tensor], sample_sizes: List[int],
                  round_idx: int) -> Tuple[torch.Tensor, Dict]:
        """Given client updates, return the aggregation weights and diagnostics.

        Args:
            deltas: list of N flattened update tensors (W_i^t - W^t).
            sample_sizes: list of N local dataset sizes.
            round_idx: 1-indexed communication round.

        Returns:
            alpha: [N] aggregation weights summing to 1.
            diag: dict of per-round diagnostics.
        """
        device = self.device
        deltas = torch.stack([d.to(device) for d in deltas], dim=0)  # [N, d]
        n = torch.tensor(sample_sizes, dtype=torch.float32, device=device)

        if round_idx <= self.cfg.warmup_rounds:
            # Warm-up: uniform trust
            R = torch.ones(self.N, device=device)
            A_bar = torch.zeros(self.N, device=device)
            G = torch.zeros(self.N, device=device)
        else:
            A_bar = self._compute_anomaly(deltas)
            self._update_temporal_trust(A_bar)
            self._update_persistent_similarity(deltas)
            G = self._collusion_risk(A_bar)
            R = self._fuse_trust(A_bar, self.T, G)

        # Trust-modulated weights (Eq. 8)
        q = n * (R ** self.cfg.kappa)
        alpha = q / q.sum()

        # Update global reference for the next round
        self.prev_global_update = (alpha.unsqueeze(1) * deltas).sum(dim=0).detach()

        # Diagnostics
        diag = {
            "A_bar": A_bar.detach().cpu(),
            "T": self.T.detach().cpu(),
            "G": G.detach().cpu(),
            "R": R.detach().cpu(),
            "alpha": alpha.detach().cpu(),
        }
        self.history["A_bar"].append(diag["A_bar"])
        self.history["T"].append(diag["T"])
        self.history["G"].append(diag["G"])
        self.history["R"].append(diag["R"])
        self.history["alpha"].append(diag["alpha"])
        return alpha, diag

    def reset(self):
        """Reset defense state between seeds."""
        self.T = torch.ones(self.N, device=self.device)
        self.P = torch.zeros(self.N, self.N, device=self.device)
        self.prev_global_update = None
        self.history = {"R": [], "A_bar": [], "T": [], "G": [], "alpha": []}

    # ------------------------------------------------------------------
    # Malicious mass bound (Theorem 1)
    # ------------------------------------------------------------------
    @staticmethod
    def malicious_mass_bound(R: torch.Tensor, malicious: List[int],
                             kappa: float, rho_mal: float) -> float:
        """Compute the per-round bound from Theorem 1 given observed trust."""
        r_M = R[malicious].max().item() if len(malicious) > 0 else 0.0
        honest = [i for i in range(R.shape[0]) if i not in set(malicious)]
        r_H = R[honest].min().item() if len(honest) > 0 else 1.0
        num = rho_mal * (r_M ** kappa)
        den = num + (1 - rho_mal) * (r_H ** kappa)
        return num / den if den > 0 else 0.0
