"""Test-time augmentation: усреднение вероятностей для original + horizontal flip."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from ..config import IMAGENET_MEAN, IMAGENET_STD


class TTADataset(Dataset):
    def __init__(self, samples: list[dict], image_size: int = 224, resize_pad: int = 32):
        self.samples = samples
        self.base = transforms.Compose([
            transforms.Resize(image_size + resize_pad),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self.flip = transforms.Compose([
            transforms.Resize(image_size + resize_pad),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image
        s = self.samples[idx]
        img = Image.open(s["path"]).convert("RGB")
        return self.base(img), self.flip(img), int(s["label"])


@torch.no_grad()
def tta_logits(model, samples: list[dict], image_size: int, batch_size: int,
               num_workers: int, device: str = "cuda") -> tuple[np.ndarray, np.ndarray]:
    ds = TTADataset(samples, image_size=image_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    model.eval()
    all_logits = []
    all_labels = []
    for base, flip, labels in loader:
        base = base.to(device, non_blocking=True)
        flip = flip.to(device, non_blocking=True)
        z1 = torch.softmax(model(base), dim=1)
        z2 = torch.softmax(model(flip), dim=1)
        all_logits.append(((z1 + z2) / 2.0).cpu())
        all_labels.append(labels)
    probs = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    return probs, labels
