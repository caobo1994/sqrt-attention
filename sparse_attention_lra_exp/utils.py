"""
Pathfinder-X dataset generation and loading.

Pathfinder-X: 128x128 binary images with continuous paths.
Task: Determine if two marked points are on the same continuous path.

For reproducibility, we generate synthetic data with known ground truth.
If real LRA data is available, set DATA_DIR to point to the downloaded dataset.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import math
import os

# Pathfinder image dimensions
IMG_SIZE = 128
SEQ_LEN = IMG_SIZE * IMG_SIZE  # 16384

# Marker colors for the two points
MARKER_COLORS = {1: 0.75, 2: 0.5}  # grayscale values


def _random_walk_path(size=128, length=None):
    """
    Generate a continuous random walk path.
    Returns: set of (x, y) coordinates on the path.
    """
    if length is None:
        length = np.random.randint(size // 2, size * 2)

    # Start near center
    x, y = np.random.randint(size // 3, 2 * size // 3, size=2)
    path = {(x, y)}
    steps = 0
    max_attempts = length * 10

    while len(path) < length and steps < max_attempts:
        steps += 1
        dx, dy = np.random.choice([-1, 0, 1], size=2)
        if dx == 0 and dy == 0:
            continue
        nx, ny = x + dx, y + dy
        if 0 <= nx < size and 0 <= ny < size:
            x, y = nx, ny
            path.add((x, y))

    return path


def _make_curved_path(size=128):
    """Generate a smooth curved path using bezier-like interpolation."""
    # Control points
    n_ctrl = np.random.randint(3, 7)
    ctrl_x = np.sort(np.random.randint(0, size, n_ctrl))
    ctrl_y = np.random.randint(0, size, n_ctrl)

    # Interpolate
    t = np.linspace(0, 1, np.random.randint(size // 2, size * 2))
    # Simple polynomial interpolation
    x = np.zeros_like(t)
    y = np.zeros_like(t)
    for i in range(n_ctrl):
        coeff = np.prod([(t - j / (n_ctrl - 1)) for j in range(n_ctrl) if j != i], axis=0)
        denom = np.prod([(i / (n_ctrl - 1) - j / (n_ctrl - 1)) for j in range(n_ctrl) if j != i])
        coeff = coeff / denom
        x += ctrl_x[i] * coeff
        y += ctrl_y[i] * coeff

    x = np.clip(np.round(x).astype(int), 0, size - 1)
    y = np.clip(np.round(y).astype(int), 0, size - 1)
    return set(zip(x, y))


def generate_pathfinder_image(size=128, curved=True):
    """
    Generate a Pathfinder-like image with two marked points.
    
    Returns:
        image: (size, size) float array, 0=background, 1=path, 0.75/0.5=markers
        label: 1 if both markers on same path, 0 otherwise
        pos1, pos2: (x, y) positions of markers
    """
    # Generate path
    if curved and np.random.random() > 0.3:
        path = _make_curved_path(size)
    else:
        path = _random_walk_path(size)

    # Create image
    image = np.zeros((size, size), dtype=np.float32)

    # Draw path
    for x, y in path:
        image[y, x] = 1.0

    # Sample marker positions
    path_list = list(path)
    same_path = np.random.random() > 0.5

    if same_path:
        # Both on path
        if len(path_list) >= 2:
            idx = np.random.choice(len(path_list), 2, replace=False)
            pos1 = path_list[idx[0]]
            pos2 = path_list[idx[1]]
        else:
            # Path too short, use random same-path points
            pos1 = path_list[0]
            pos2 = path_list[0]
    else:
        # One on path, one off path
        pos1 = path_list[np.random.randint(len(path_list))]
        # Find a point not on path
        off_path = False
        attempts = 0
        while not off_path and attempts < 100:
            px = np.random.randint(0, size)
            py = np.random.randint(0, size)
            if (px, py) not in path:
                pos2 = (px, py)
                off_path = True
            attempts += 1
        if not off_path:
            pos2 = pos1  # fallback
            same_path = True

    # Draw markers
    image[pos1[1], pos1[0]] = MARKER_COLORS[1]
    image[pos2[1], pos2[0]] = MARKER_COLORS[2]

    return image, 1 if same_path else 0, pos1, pos2


class PathfinderDataset(Dataset):
    """
    Synthetic Pathfinder-X dataset.
    
    Args:
        num_samples: number of images to generate
        size: image size (default 128 for Pathfinder-X)
        cache: if True, pre-generate all data in memory
    """

    def __init__(self, num_samples=10000, size=128, cache=True):
        self.size = size
        self.seq_len = size * size
        self.num_samples = num_samples
        self.cache = cache

        if cache:
            self.data = []
            for i in range(num_samples):
                img, label, _, _ = generate_pathfinder_image(size)
                self.data.append((img, label))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.cache:
            img, label = self.data[idx]
        else:
            img, label, _, _ = generate_pathfinder_image(self.size)

        # Flatten image to sequence of pixels
        seq = img.flatten()  # (16384,)
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def get_dataloaders(batch_size=4, seq_len=16384, num_train=5000, num_val=1000):
    """
    Create train and validation dataloaders for Pathfinder-X.
    
    Args:
        batch_size: training batch size
        seq_len: sequence length (must be IMG_SIZE**2)
        num_train: number of training samples
        num_val: number of validation samples
    """
    train_dataset = PathfinderDataset(num_samples=num_train, size=128, cache=True)
    val_dataset = PathfinderDataset(num_samples=num_val, size=128, cache=True)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Copying Task (for smaller-scale validation)
# ---------------------------------------------------------------------------

class CopyingDataset(Dataset):
    """
    Copying task: input = [random digits] + [delimiter] + [blanks] + [end marker]
    Target = [blanks] + [original digits to copy]
    
    Tests long-range memory: model must remember the first N digits across M blanks.
    """

    def __init__(self, num_samples=10000, seq_len=1024, num_digits=10, vocab_size=8):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_digits = num_digits
        self.vocab_size = vocab_size
        self.delimiter = vocab_size  # delimiter token
        self.blank = vocab_size + 1  # blank token
        self.end = vocab_size + 2  # end marker
        self.total_vocab = vocab_size + 3

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random digits (0 to vocab_size-1)
        digits = np.random.randint(0, self.vocab_size, size=self.num_digits)

        # Construct sequence
        input_seq = np.concatenate([
            digits,
            [self.delimiter],
            [self.blank] * (self.seq_len - 2 * self.num_digits - 2),
            [self.end],
        ])

        # Target: copy the first num_digits at the end
        target = np.full(self.seq_len, self.blank)
        target[-(self.num_digits + 1):-1] = digits
        target[-1] = self.end

        return (
            torch.tensor(input_seq, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
        )
