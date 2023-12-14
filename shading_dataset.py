#@title Dataset

from pathlib import Path
from torchvision import transforms
from PIL import Image

import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch_ema import ExponentialMovingAverage

from torchvision.transforms.functional import rgb_to_grayscale


def collate_fn(examples):
    gscale_values = [example["input_pixels"] for example in examples]
    # rgb_values = [example["rgb_pixels"] for example in examples]
    controlnet_input_values = [example["controlnet_input_pixels"] for example in examples]

    controlnet_input_values = torch.stack(controlnet_input_values)
    controlnet_input_values = controlnet_input_values.to(memory_format=torch.contiguous_format).float()

    gscale_values = torch.stack(gscale_values)
    gscale_values = gscale_values.to(memory_format=torch.contiguous_format).float()

    batch = {
        "input_pixels": gscale_values,
        "controlnet_input_pixels": controlnet_input_values 
    }
    return batch


class ShadingDataset:
    def __init__(self, instance_data_root, device, H=512, W=512, size=100, batch_size=1):
        super().__init__()

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError(f"Instance {self.instance_data_root} images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.instance_images_path = self.instance_images_path[:min(len(self.instance_images_path), size)]
        self.num_instance_images = 32
        self._length = self.num_instance_images
        self.device = device
        self.H = H
        self.W = W
        self.size = size
        self.batch_size = batch_size
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize((H, W), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ]
        )
        self.controlnet_image_transforms = transforms.Compose(
            [
                transforms.Resize(size=(H, W)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )


    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        example["input_pixels"] = self.image_transforms(instance_image).type(torch.float16).to(self.device)
        example["controlnet_input_pixels"] = self.controlnet_image_transforms(instance_image).type(torch.float16).to(self.device)

        return example


    def dataloader(self):
        loader = DataLoader(self, batch_size=self.batch_size, collate_fn=collate_fn, shuffle=false, num_workers=0)
        return loader



class SingleImageDataset:
    def __init__(self, instance_data_path, device, size=1, H=512, W=512):
        super().__init__()

        self.instance_image_path = Path(instance_data_path)
        self.num_instance_images = 1
        self._length = self.num_instance_images
        self.device = device
        self.H = H
        self.W = W
        self.size = size
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize((H, W), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ]
        )
        self.controlnet_image_transforms = transforms.Compose(
            [
                transforms.Resize(size=(H, W)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def decolor(self, img):
      return rgb_to_grayscale(img, num_output_channels=3)


    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_image_path)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        example["input_pixels"] = self.image_transforms(instance_image).type(torch.float16).to(self.device)
        example["controlnet_input_pixels"] = self.controlnet_image_transforms(instance_image).type(torch.float16).to(self.device)
    
        return example


    def dataloader(self):
        loader = DataLoader(list(range(self.size)), batch_size=1, collate_fn=collate_fn, shuffle=false, num_workers=0)
        return loader