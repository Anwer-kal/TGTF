"""Configuration for TGTF experiments."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TGTFConfig:
    """Hyperparameters for the TGTF defense."""

    # Current anomaly detection
    alpha_A: float = 0.4          # magnitude/direction balance
    tau_norm: float = 3.0         # anomaly saturation constant
    eps_num: float = 1e-8         # numerical stability

    # Asymmetric temporal trust
    lambda_minus: float = 0.30    # trust degradation (fast)
    lambda_plus: float = 0.90     # trust recovery (slow)
    tau_A: float = 0.65           # anomaly threshold

    # Persistent similarity graph
    rho: float = 0.85             # persistent similarity memory
    tau_G: float = 0.50           # persistent similarity threshold

    # Fusion
    omega1: float = 0.40          # weight for (1 - anomaly)
    omega2: float = 0.35          # weight for temporal trust
    omega3: float = 0.25          # weight for (1 - collusion risk)

    # Aggregation
    kappa: float = 3.0            # trust sensitivity exponent

    # Warm-up
    warmup_rounds: int = 5

    def __post_init__(self):
        total = self.omega1 + self.omega2 + self.omega3
        assert abs(total - 1.0) < 1e-6, f"Fusion weights must sum to 1, got {total}"
        assert self.lambda_plus > self.lambda_minus, "lambda_+ must exceed lambda_-"


@dataclass
class FLConfig:
    """Federated learning configuration."""

    num_clients: int = 100
    num_rounds: int = 100
    local_epochs: int = 5
    batch_size: int = 64
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4

    # Non-IID
    dirichlet_beta: float = 0.5

    # Attack
    malicious_fraction: float = 0.20
    attack_type: str = "adaptive_backdoor"  # gaussian, label_flip, sign_flip, adaptive_backdoor, tgtf_aware

    # Seeds
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1010])

    # Device
    device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"


@dataclass
class AttackConfig:
    """Attack-specific configuration."""

    # Gaussian
    gaussian_sigma: float = 0.5

    # Label flip
    num_classes: int = 10

    # Adaptive backdoor
    backdoor_prob: float = 0.7
    trigger_size: int = 3
    target_label: int = 0

    # TGTF-aware
    eps_atk: float = 0.5
    eta: float = 0.1              # collusion similarity slack
    gamma: float = 0.5            # stealth weight in Lagrangian
    lam: float = 1.0              # attack efficacy weight
    pgd_iters: int = 5
    pgd_step: float = 0.01
