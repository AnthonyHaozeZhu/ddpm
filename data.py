# -*- coding: UTF-8 -*-
"""
@Project ：RES 
@File ：data.py
@Author ：AnthonyZ
@Date ：2022/11/20 17:57
"""


from datasets import load_dataset
from torchvision.transforms import transforms
import torchvision


class butterfly_128:
    def __init__(self, data_path, image_size):
        self.dataset = load_dataset(
            path="huggan/smithsonian_butterflies_subset",
            split="train",
            cache_dir=data_path
        )
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.dataset.set_transform(self._transform)

    def _transform(self, examples):
        images = [self.preprocess(image.convert("RGB")) for image in examples["image"]]
        return {"images": images}

    def get_dataset(self):
        return self.dataset


def LFWPeople(image_size, path):
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    train_data = torchvision.datasets.LFWPeople(
        root=path,
        split='train',
        download=True,
        transform=transform)  # 加载EMNIST数据集
    return train_data