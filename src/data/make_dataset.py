import logging
from pathlib import Path
from typing import Dict
import json
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from transformers import AutoFeatureExtractor

class FruitsDataset(Dataset):
    def __init__(
        self,
        input_filepath: str,
        data_type: str,
        feature_extractor: AutoFeatureExtractor,
    ) -> None:
        super(FruitsDataset, self).__init__()
        self.input_filepath = input_filepath
        self.data_type = data_type

        self.images = ImageFolder(self.input_filepath)
        self.feature_extractor = feature_extractor

        self.num_classes = len(self.images.classes)

        self.label2id = {label: i for i, label in enumerate(self.images.classes)}
        self.id2label = {i: label for i, label in enumerate(self.images.classes)}

    def __getitem__(self, idx: int) -> Dict:
        current = self.images[idx]
        img, label = current[0], current[1]

        pixel_values = self.feature_extractor(img, return_tensors="pt").pixel_values
        return {"pixel_values": pixel_values, "labels": torch.tensor(label)}

    def __len__(self) -> int:
        return len(self.images)
