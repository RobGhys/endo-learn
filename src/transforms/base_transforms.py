import random
from typing import Dict, List

import torchvision.transforms.functional as F
from torch import nn
import torch
import numpy as np
from torchvision import transforms


class RandomRotation90(object):
    def __call__(self, img):
        angle = random.choice([0, 90, 180, 270])
        return F.rotate(img, angle)

    def __str__(self) -> str:
        return "RandomRotation 90°"


class RandomFlipAndMirror(object):
    def __init__(self, threshold: int = 0.5):
        self.threshold = threshold

    def __call__(self, img):
        # Flip (horizontal)
        if random.random() > self.threshold:
            img = F.hflip(img)

        # Mirror (vertical)
        if random.random() > self.threshold:
            img = F.vflip(img)

        return img

    def __str__(self) -> str:
        return "Random Flip And Mirror"


class RandomZoom(object):
    def __init__(self, min_scale=0.8, max_scale=1.2):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, img):
        scale = random.uniform(self.min_scale, self.max_scale)
        w, h = img.size
        new_w, new_h = int(w * scale), int(h * scale)
        img = F.resize(img, [new_h, new_w])
        return F.center_crop(img, [h, w])


class RandomShear(object):
    def __init__(self, shear_range=10):
        self.shear_range = shear_range

    def __call__(self, img):
        shear = random.uniform(-self.shear_range, self.shear_range)
        return F.affine(img, angle=0, translate=[0, 0], scale=1, shear=[shear, 0])  # Horizontal shear


class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img


class Resize(object):
    """Resize the image to a given size."""

    def __init__(self, output_size: List[int]):
        self.output_size = output_size

    def __call__(self, img: torch.Tensor):
        return F.resize(img=img, size=self.output_size, interpolation=F.InterpolationMode.BICUBIC)

    def __str__(self) -> str:
        return f"Resize to {self.output_size}"


class RandomCrop(object):
    """Randomly crop the image to a given size."""

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, img):
        width, height = img.size
        crop_width, crop_height = self.output_size

        # Ensure the crop size does not exceed the image size
        if width < crop_width or height < crop_height:
            raise ValueError("Crop size must be smaller than the image size.")

        top = random.randint(0, height - crop_height)
        left = random.randint(0, width - crop_width)

        return F.crop(img, top, left, crop_height, crop_width)

    def __str__(self) -> str:
        return f"RandomCrop to {self.output_size}"


def get_transforms(model_size=(224, 224), resize_size=(256, 256)) -> Dict[str, transforms.Compose]:
    """Utility function to get train and validation transforms."""
    results: Dict[str, transforms.Compose] = {
        'train': transforms.Compose([
            Resize(output_size=resize_size),  # Resize first
            RandomCrop(output_size=model_size),  # Then random crop
            RandomRotation90(),
            RandomFlipAndMirror(),
            RandomZoom(),
            RandomShear(),
            GaussianBlur(kernel_size=5),
            transforms.ToTensor()
        ]),
        'valid': transforms.Compose([
            Resize(output_size=model_size),
            transforms.ToTensor()
        ])
    }

    return results

if __name__ == "__main__":
    transforms: dict = get_transforms()
