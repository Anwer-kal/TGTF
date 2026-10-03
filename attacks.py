"""Attack implementations: Gaussian, Label-Flip, Sign-Flip, Adaptive Backdoor,
and the TGTF-aware adaptive adversary."""

import torch
import torch.nn.functional as F
from typing import List, Dict

from config import AttackConfig, TGTFConfig
from tgtf_defense import TGTFDefense


class BaseAttack:
    def __init__(self, malicious_clients: List[int], cfg: AttackConfig):
        self.malicious = list(malicious_clients)
        self.cfg = cfg

    def perturb(self, deltas: List[torch.Tensor], round_idx: int,
                global_update: torch.Tensor = None) -> List[torch.Tensor]:
        """Return perturbed deltas. Default: no-op."""
        return deltas


class GaussianAttack(BaseAttack):
    """Untargeted Gaussian noise on the unit-normalized update."""

    def perturb(self, deltas, round_idx, global_update=None):
        out = list(deltas)
        for i in self.malicious:
            d = deltas[i]
            d_norm = d / (d.norm() + 1e-12)
            noise = torch.randn_like(d) * self.cfg.gaussian_sigma
            out[i] = d_norm + noise
        return out


class LabelFlipAttack(BaseAttack):
    """Handled at the data level; update perturbation is a no-op."""

    def perturb(self, deltas, round_idx, global_update=None):
        return list(deltas)


class SignFlipAttack(BaseAttack):
    """Colluding sign-flip: negate the update."""

    def perturb(self, deltas, round_idx, global_update=None):
        out = list(deltas)
        for i in self.malicious:
            out[i] = -deltas[i]
        return out


class AdaptiveBackdoorAttack(BaseAttack):
    """Mirror honest updates with prob p; inject backdoor updates otherwise.

    Because the backdoor is injected at the data level, the actual update
    perturbation here is a magnitude boost with prob (1-p) to simulate the
    stronger backdoor gradient. Honest updates are otherwise passed through.
    """

    def perturb(self, deltas, round_idx, global_update=None):
        out = list(deltas)
        for i in self.malicious:
            if torch.rand(1).item() < self.cfg.backdoor_prob:
                # Mirror: pass through
                out[i] = deltas[i]
            else:
                # Inject: boost magnitude in a fixed direction
                boost = 1.5
                out[i] = deltas[i] * boost
        return out


class TGTFAwareAttack(BaseAttack):
    """TGTF-aware adaptive attacker.

    Solves the Lagrangian of Eq. (10) with projected-gradient steps, using
    soft surrogates for median/MAD, the edge indicator, and clip.

    Requires a *copy* of the defense state at the current round to compute
    gradients through the trust pipeline.
    """

    def __init__(self, malicious_clients, cfg: AttackConfig,
                 defense: TGTFDefense, tgtf_cfg: TGTFConfig):
        super().__init__(malicious_clients, cfg)
        self.defense = defense
        self.tgtf_cfg = tgtf_cfg

    @staticmethod
    def _soft_median(x: torch.Tensor, T: float = 0.1) -> torch.Tensor:
        """Differentiable soft-median surrogate."""
        w = torch.softmax(x / T, dim=0)
        return (w * x).sum()

    @classmethod
    def _soft_mad(cls, x: torch.Tensor, med: torch.Tensor, T: float = 0.1):
        return cls._soft_median((x - med).abs(), T)

    def _soft_anomaly(self, deltas: torch.Tensor,
                      prev_global: torch.Tensor) -> torch.Tensor:
        """Differentiable surrogate for A_bar."""
        eps = self.tgtf_cfg.eps_num
        norms = deltas.norm(dim=1)
        med_N = self._soft_median(norms)
        mad_N = self._soft_mad(norms, med_N)
        Z_N = (norms - med_N).abs() / (mad_N + eps)

        if prev_global is None:
            Z_C = torch.ones_like(Z_N)
        else:
            dn = F.normalize(deltas, dim=1, eps=eps)
            gn = F.normalize(prev_global.unsqueeze(0), dim=1, eps=eps)
            C = (dn * gn).sum(dim=1)
            med_C = self._soft_median(C)
            mad_C = self._soft_mad(C, med_C)
            Z_C = (C - med_C).abs() / (mad_C + eps)

        A = self.tgtf_cfg.alpha_A * Z_N + (1 - self.tgtf_cfg.alpha_A) * Z_C
        return torch.clamp(A / self.tgtf_cfg.tau_norm, 0.0, 1.0)

    def _soft_collusion_risk(self, deltas: torch.Tensor,
                             A_bar: torch.Tensor,
                             P: torch.Tensor) -> torch.Tensor:
        """Differentiable surrogate for G using a sigmoid edge indicator."""
        T = 0.01
        adj = torch.sigmoid((P - self.tgtf_cfg.tau_G) / T)
        adj.fill_diagonal_(0.0)
        weighted = (P * adj) @ A_bar
        denom = (P * adj).sum(dim=1) + self.tgtf_cfg.eps_num
        return weighted / denom

    def perturb(self, deltas, round_idx, global_update=None):
        cfg = self.tgtf_cfg
        out = [d.clone() for d in deltas]
        N = len(deltas)
        d_dim = deltas[0].numel()

        # Snapshot defense state needed for gradients
        P_state = self.defense.P.detach().clone()
        T_state = self.defense.T.detach().clone()
        prev_g = self.defense.prev_global_update
        prev_g = prev_g.detach().clone() if prev_g is not None else None

        # Collect malicious updates as leaf tensors
        mal_tensors = {i: deltas[i].detach().clone().requires_grad_(True)
                       for i in self.malicious}

        # Build the full delta stack for the surrogate forward pass
        def build_stack():
            stack = []
            for i in range(N):
                if i in mal_tensors:
                    stack.append(mal_tensors[i])
                else:
                    stack.append(deltas[i].detach())
            return torch.stack(stack, dim=0)

        for _ in range(self.cfg.pgd_iters):
            stack = build_stack()
            A_bar = self._soft_anomaly(stack, prev_g)
            G = self._soft_collusion_risk(stack, A_bar, P_state)

            # Lagrangian objective: minimize (mean anomaly + gamma * mean G)
            # minus lambda * attack efficacy (approximated as -||delta||^2
            # since we don't have the true loss gradient at the server).
            stealth = A_bar[self.malicious].mean() + self.cfg.gamma * G[self.malicious].mean()
            # Efficacy proxy: magnitude of the malicious update
            efficacy = -torch.stack([mal_tensors[i].norm() for i in self.malicious]).mean()
            loss = stealth + self.cfg.lam * efficacy

            grads = torch.autograd.grad(loss, list(mal_tensors.values()),
                                        retain_graph=False, allow_unused=True)

            # Projected gradient step
            with torch.no_grad():
                for (i, t), g in zip(mal_tensors.items(), grads):
                    if g is None:
                        continue
                    t_new = t - self.cfg.pgd_step * g
                    # Projection onto ||delta|| <= eps_atk
                    n = t_new.norm()
                    if n > self.cfg.eps_atk:
                        t_new = t_new * (self.cfg.eps_atk / (n + 1e-12))
                    mal_tensors[i] = t_new.detach().requires_grad_(True)

        # Apply collusion constraint: project each malicious update towards
        # the group mean direction so that cos >= 1 - eta.
        with torch.no_grad():
            stack = build_stack()
            mal_stack = stack[self.malicious]
            mean_dir = F.normalize(mal_stack.mean(dim=0, keepdim=True), dim=1, eps=1e-8)
            for idx, i in enumerate(self.malicious):
                v = mal_tensors[i].detach()
                v_n = F.normalize(v.unsqueeze(0), dim=1, eps=1e-8)
                cos = (v_n * mean_dir).sum().item()
                if cos < 1 - self.cfg.eta:
                    # Blend with mean direction
                    blended = 0.5 * v_n.squeeze(0) + 0.5 * mean_dir.squeeze(0)
                    blended = F.normalize(blended.unsqueeze(0), dim=1, eps=1e-8).squeeze(0)
                    mal_tensors[i] = (blended * v.norm()).detach().requires_grad_(True)
                out[i] = mal_tensors[i].detach()

        return out
