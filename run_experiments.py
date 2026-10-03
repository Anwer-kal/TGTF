"""Run the TGTF experiments end to end."""

import argparse
import json
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import List

from config import FLConfig, TGTFConfig, AttackConfig
from models import get_model
from data_utils import (load_cifar10, load_femnist, dirichlet_partition,
                        make_client_loaders, BackdoorDataset)
from federated_learning import run_federated
from metrics import summarize, trust_calibration_error, convergence_round


def build_experiment(dataset: str, cfg: FLConfig, attack_cfg: AttackConfig,
                     seed: int, num_classes: int):
    """Build model, client loaders, test loader, and malicious client list."""
    if dataset.lower() == "cifar10":
        train, test = load_cifar10()
        attack_cfg.num_classes = 10
    elif dataset.lower() == "femnist":
        train, test = load_femnist()
        attack_cfg.num_classes = 62
    else:
        raise ValueError(dataset)

    # Extract targets for partitioning
    targets = np.array([train[i][1] for i in range(len(train))])
    client_indices = dirichlet_partition(targets, cfg.num_clients,
                                         cfg.dirichlet_beta, seed=seed)

    # Select malicious clients (deterministic given seed)
    rng = np.random.default_rng(seed)
    num_mal = int(cfg.malicious_fraction * cfg.num_clients)
    malicious = sorted(rng.choice(cfg.num_clients, size=num_mal,
                                  replace=False).tolist())

    # Build loaders
    client_loaders = make_client_loaders(train, client_indices, malicious,
                                         cfg, attack_cfg)

    test_loader = DataLoader(test, batch_size=256, shuffle=False, num_workers=0)

    sample_sizes = [len(idxs) for idxs in client_indices]

    model = get_model(dataset, num_classes=attack_cfg.num_classes)
    return model, client_loaders, sample_sizes, test_loader, malicious


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10",
                        choices=["cifar10", "femnist"])
    parser.add_argument("--attack", default="adaptive_backdoor",
                        choices=["gaussian", "label_flip", "sign_flip",
                                 "adaptive_backdoor", "tgtf_aware"])
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--clients", type=int, default=100)
    parser.add_argument("--mal-frac", type=float, default=0.20)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 123, 456, 789, 1010])
    parser.add_argument("--out", default="results.json")
    parser.add_argument("--quick", action="store_true",
                        help="Reduce rounds and clients for a smoke test")
    args = parser.parse_args()

    if args.quick:
        args.rounds = 10
        args.clients = 20
        args.seeds = [42]

    fl_cfg = FLConfig(
        num_clients=args.clients,
        num_rounds=args.rounds,
        malicious_fraction=args.mal_frac,
        attack_type=args.attack,
        dirichlet_beta=args.beta,
    )
    tgtf_cfg = TGTFConfig()
    attack_cfg = AttackConfig()

    seed_results = []
    for seed in args.seeds:
        print(f"\n=== seed {seed} | {args.dataset} | {args.attack} ===")
        model, client_loaders, sample_sizes, test_loader, malicious = \
            build_experiment(args.dataset, fl_cfg, attack_cfg, seed,
                             num_classes=10)
        result = run_federated(
            model, client_loaders, sample_sizes, test_loader,
            fl_cfg, tgtf_cfg, attack_cfg, malicious,
            num_classes=10, seed=seed, verbose=True,
        )
        # Compute TCE and CR
        R_hist = [r.numpy() for r in result["defense_history"]["R"]]
        tce = trust_calibration_error(R_hist, malicious)
        cr = convergence_round(result["history"]["ga"])
        result["tce"] = tce
        result["cr"] = cr
        result["seed"] = seed
        seed_results.append(result)
        print(f"  GA={result['final_ga']*100:.2f}  "
              f"ASR={result['final_asr']*100:.2f}  "
              f"FPR={result['final_fpr']*100:.2f}  "
              f"TPR={result['final_tpr']*100:.2f}  "
              f"TCE={tce:.3f}  CR={cr}")

    summary = summarize(seed_results)
    out = {
        "config": {
            "dataset": args.dataset,
            "attack": args.attack,
            "rounds": args.rounds,
            "clients": args.clients,
            "mal_frac": args.mal_frac,
            "beta": args.beta,
            "seeds": args.seeds,
        },
        "summary": summary,
        "per_seed": [
            {
                "seed": r["seed"],
                "final_ga": r["final_ga"],
                "final_asr": r["final_asr"],
                "final_fpr": r["final_fpr"],
                "final_tpr": r["final_tpr"],
                "tce": r["tce"],
                "cr": r["cr"],
                "history": r["history"],
            }
            for r in seed_results
        ],
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved results to {args.out}")
    print(json.dumps({k: v["mean"] for k, v in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
