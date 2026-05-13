from __future__ import annotations

import torch
from PIL import Image
from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def imagenet_eval_transform(image_size: int) -> T.Compose:
    crop = image_size
    resize = int(image_size * 1.14)
    return T.Compose([
        T.Resize(resize, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(crop),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def to_tensor_batch(img: Image.Image, image_size: int) -> torch.Tensor:
    t = imagenet_eval_transform(image_size)
    return t(img).unsqueeze(0)


def imagenet_denormalize(t: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(t.device)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(t.device)
    return (t * std + mean).clamp(0, 1)
