# TGTF: Temporal Graph Trust Fusion

Reference implementation of **TGTF**, a persistence-aware temporal trust
framework for robust federated learning against adaptive colluding
poisoning.

## Installation

```bash
pip install -r requirements.txt
```

## Quick smoke test

```bash
python run_experiments.py --quick --dataset cifar10 --attack adaptive_backdoor
```

## Full experiment

```bash
python run_experiments.py \
    --dataset cifar10 \
    --attack adaptive_backdoor \
    --rounds 100 \
    --clients 100 \
    --mal-frac 0.20 \
    --seeds 42 123 456 789 1010 \
    --out results_cifar10_adaptive.json
```

## Supported attacks

- `gaussian`: untargeted additive Gaussian noise
- `label_flip`: systematic label-flip poisoning at the data level
- `sign_flip`: colluding sign-flip
- `adaptive_backdoor`: intermittent backdoor injection
- `tgtf_aware`: white-box attacker optimizing against TGTF with PGD

## Supported datasets

- `cifar10` (ResNet-18 adapted for 32x32, Dirichlet partitioning)
- `femnist` (2-conv CNN, Dirichlet partitioning; falls back to EMNIST
  or synthetic data if the LEAF FEMNIST files are unavailable)

## Key components

| File | Role |
|------|------|
| `config.py` | Hyperparameters (Table VIII of the paper) |
| `tgtf_defense.py` | Anomaly, temporal trust, persistent similarity, fusion, aggregation, bound |
| `attacks.py` | Five attack strategies including the TGTF-aware PGD attacker |
| `federated_learning.py` | FL loop with per-round evaluation |
| `models.py` | ResNet-18 (CIFAR) and 2-conv CNN (FEMNIST) |
| `data_utils.py` | Non-IID partitioning, backdoor/label-flip dataset wrappers |
| `metrics.py` | TCE, convergence round, t-based summaries |

## Notes on the TGTF-aware attacker

The attacker in `attacks.py` uses soft surrogates for the non-smooth
operations in TGTF (soft-median, sigmoid edge indicator, tanh clip) so
that gradients can flow through the defense. This mirrors the
differentiability discussion in Section IV-C of the paper. The
projected-gradient steps are followed by a projection onto the
collusion similarity constraint.

## Reproducibility

All randomness is seeded from `--seeds`. Results are written to JSON
with per-seed metrics and per-round histories, matching the
reproducibility note in the paper.
