"""Dataset loading and non-IID partitioning."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms


def load_cifar10(root: str = "./data"):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train = datasets.CIFAR10(root, train=True, download=True, transform=transform_train)
    test = datasets.CIFAR10(root, train=False, download=True, transform=transform_test)
    return train, test


def load_femnist(root: str = "./data"):
    """Load FEMNIST. Falls back to EMNIST if LEAF is not available."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    try:
        train = datasets.EMNIST(root, split="byclass", train=True, download=True, transform=transform)
        test = datasets.EMNIST(root, split="byclass", train=False, download=True, transform=transform)
    except Exception:
        # Synthetic fallback for environments without the dataset
        train = datasets.FakeData(size=5000, image_size=(1, 28, 28), num_classes=62,
                                  transform=transforms.ToTensor())
        test = datasets.FakeData(size=1000, image_size=(1, 28, 28), num_classes=62,
                                 transform=transforms.ToTensor())
    return train, test


def dirichlet_partition(targets, num_clients: int, beta: float, seed: int = 0):
    """Partition indices by a Dirichlet distribution over class labels.

    Returns a list of index lists, one per client.
    """
    rng = np.random.default_rng(seed)
    targets = np.asarray(targets)
    num_classes = int(targets.max()) + 1
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        idx_c = np.where(targets == c)[0]
        rng.shuffle(idx_c)
        proportions = rng.dirichlet([beta] * num_clients)
        # Allocate class-c samples to clients by proportion
        cuts = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        splits = np.split(idx_c, cuts)
        for i, split in enumerate(splits):
            client_indices[i].extend(split.tolist())

    # Shuffle each client's indices
    for i in range(num_clients):
        rng.shuffle(client_indices[i])
        if len(client_indices[i]) == 0:
            # Ensure every client has at least one sample
            client_indices[i] = [int(rng.integers(0, len(targets)))]

    return client_indices


class BackdoorDataset(Dataset):
    """Wrap a dataset and inject a backdoor trigger on a fraction of samples."""

    def __init__(self, base_dataset, indices, trigger_size=3, target_label=0,
                 poison_prob=0.7, apply_trigger=True):
        self.base = base_dataset
        self.indices = indices
        self.trigger_size = trigger_size
        self.target_label = target_label
        self.poison_prob = poison_prob
        self.apply_trigger = apply_trigger

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        x, y = self.base[idx]
        if self.apply_trigger and np.random.rand() < self.poison_prob:
            x = x.clone()
            c, h, w = x.shape
            s = self.trigger_size
            x[:, h - s:h, w - s:w] = 1.0  # white square
            y = self.target_label
        return x, y


class LabelFlipDataset(Dataset):
    """Wrap a dataset and flip labels by a fixed offset."""

    def __init__(self, base_dataset, indices, num_classes=10, offset=1):
        self.base = base_dataset
        self.indices = indices
        self.num_classes = num_classes
        self.offset = offset

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        x, y = self.base[idx]
        return x, (y + self.offset) % self.num_classes


def make_client_loaders(base_train, client_indices, malicious_clients, cfg, attack_cfg):
    """Build per-client DataLoaders, applying attack-specific dataset wrappers."""
    loaders = {}
    for cid, idxs in enumerate(client_indices):
        is_mal = cid in malicious_clients
        if is_mal and cfg.attack_type == "adaptive_backdoor":
            ds = BackdoorDataset(base_train, idxs,
                                 trigger_size=attack_cfg.trigger_size,
                                 target_label=attack_cfg.target_label,
                                 poison_prob=attack_cfg.backdoor_prob)
        elif is_mal and cfg.attack_type == "label_flip":
            ds = LabelFlipDataset(base_train, idxs,
                                  num_classes=attack_cfg.num_classes,
                                  offset=1)
        else:
            ds = Subset(base_train, idxs)
        loaders[cid] = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                                  num_workers=0, drop_last=False)
    return loaders
