"""Federated learning loop with TGTF defense."""

import copy
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Optional
from tqdm import tqdm

from config import FLConfig, TGTFConfig, AttackConfig
from models import get_model, get_flat_params, set_flat_params
from tgtf_defense import TGTFDefense
from attacks import (BaseAttack, GaussianAttack, LabelFlipAttack,
                     SignFlipAttack, AdaptiveBackdoorAttack, TGTFAwareAttack)


def build_attack(name: str, malicious: List[int], attack_cfg: AttackConfig,
                 defense: Optional[TGTFDefense] = None,
                 tgtf_cfg: Optional[TGTFConfig] = None) -> BaseAttack:
    name = name.lower()
    if name == "gaussian":
        return GaussianAttack(malicious, attack_cfg)
    if name == "label_flip":
        return LabelFlipAttack(malicious, attack_cfg)
    if name in ("sign_flip", "colluding", "colluding_sign_flip"):
        return SignFlipAttack(malicious, attack_cfg)
    if name == "adaptive_backdoor":
        return AdaptiveBackdoorAttack(malicious, attack_cfg)
    if name in ("tgtf_aware", "aware"):
        assert defense is not None and tgtf_cfg is not None
        return TGTFAwareAttack(malicious, attack_cfg, defense, tgtf_cfg)
    raise ValueError(f"Unknown attack: {name}")


def local_train(model: nn.Module, loader, cfg: FLConfig, device: str):
    """Run local SGD for `local_epochs` and return the update vector."""
    model = copy.deepcopy(model).to(device)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr,
                          momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()
    for _ in range(cfg.local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
    return get_flat_params(model).detach()


def evaluate(model: nn.Module, loader, device: str,
             backdoor: bool = False, target_label: int = 0,
             trigger_size: int = 3):
    """Return (clean_accuracy, attack_success_rate)."""
    model = model.to(device)
    model.eval()
    correct = total = 0
    asr_correct = asr_total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()

            if backdoor:
                # Apply trigger to a copy of the batch
                xb = x.clone()
                c, h, w = xb.shape[1:]
                xb[:, :, h - trigger_size:h, w - trigger_size:w] = 1.0
                predb = model(xb).argmax(dim=1)
                asr_correct += (predb == target_label).sum().item()
                asr_total += y.numel()
    acc = correct / max(total, 1)
    asr = asr_correct / max(asr_total, 1) if backdoor else 0.0
    return acc, asr


def run_federated(
    global_model: nn.Module,
    client_loaders: Dict[int, torch.utils.data.DataLoader],
    sample_sizes: List[int],
    test_loader,
    cfg: FLConfig,
    tgtf_cfg: TGTFConfig,
    attack_cfg: AttackConfig,
    malicious_clients: List[int],
    num_classes: int,
    seed: int = 0,
    verbose: bool = True,
):
    """Run one federated learning experiment with TGTF defense.

    Returns a dict of per-round and final metrics.
    """
    device = cfg.device
    torch.manual_seed(seed)
    global_model = global_model.to(device)

    defense = TGTFDefense(cfg.num_clients, tgtf_cfg, device=device)
    attack = build_attack(cfg.attack_type, malicious_clients, attack_cfg,
                          defense=defense, tgtf_cfg=tgtf_cfg)

    history = {
        "round": [],
        "ga": [],
        "asr": [],
        "mal_mass": [],
        "bound": [],
        "fpr": [],
        "tpr": [],
    }
    malicious_set = set(malicious_clients)

    iterator = tqdm(range(1, cfg.num_rounds + 1), desc=f"seed={seed}") if verbose \
        else range(1, cfg.num_rounds + 1)

    for t in iterator:
        # Broadcast and train locally
        client_updates = []
        for cid in range(cfg.num_clients):
            update = local_train(global_model, client_loaders[cid], cfg, device)
            client_updates.append(update)

        # Apply attack
        client_updates = attack.perturb(client_updates, t,
                                        global_update=defense.prev_global_update)

        # TGTF aggregation
        alpha, diag = defense.aggregate(client_updates, sample_sizes, t)

        # Apply the weighted average to the global model
        stacked = torch.stack(client_updates, dim=0)  # [N, d]
        new_flat = (alpha.unsqueeze(1) * stacked).sum(dim=0)
        set_flat_params(global_model, new_flat)

        # Evaluate
        ga, asr = evaluate(global_model, test_loader, device,
                           backdoor=(cfg.attack_type == "adaptive_backdoor"),
                           target_label=attack_cfg.target_label,
                           trigger_size=attack_cfg.trigger_size)

        # Malicious aggregation mass
        mal_mass = alpha[malicious_clients].sum().item() if malicious_clients else 0.0
        bound = TGTFDefense.malicious_mass_bound(
            diag["R"], malicious_clients, tgtf_cfg.kappa,
            len(malicious_clients) / cfg.num_clients)

        # FPR / TPR based on the anomaly threshold on R
        R = diag["R"]
        flagged = (R < 0.5).cpu().numpy()
        tp = sum(1 for i in malicious_clients if flagged[i])
        fp = sum(1 for i in range(cfg.num_clients)
                 if i not in malicious_set and flagged[i])
        tpr = tp / max(len(malicious_clients), 1)
        fpr = fp / max(cfg.num_clients - len(malicious_clients), 1)

        history["round"].append(t)
        history["ga"].append(ga)
        history["asr"].append(asr)
        history["mal_mass"].append(mal_mass)
        history["bound"].append(bound)
        history["fpr"].append(fpr)
        history["tpr"].append(tpr)

        if verbose:
            iterator.set_postfix({"GA": f"{ga*100:.1f}",
                                  "ASR": f"{asr*100:.1f}",
                                  "mal_mass": f"{mal_mass:.3f}",
                                  "bound": f"{bound:.3f}"})

    return {
        "history": history,
        "final_ga": history["ga"][-1],
        "final_asr": history["asr"][-1],
        "final_fpr": history["fpr"][-1],
        "final_tpr": history["tpr"][-1],
        "defense_history": defense.history,
    }
