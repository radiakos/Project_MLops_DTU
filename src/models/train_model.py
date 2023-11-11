import logging
import os
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoFeatureExtractor, get_scheduler
import torchvision
from transformers import ViTImageProcessor
import torchvision.transforms as transforms
import sys

from data.make_dataset import FruitsDataset
from data.data_cleaning import CleanData

from datasets import load_metric
from transformers import AutoModelForImageClassification

def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

if __name__ == "__main__":
    # 1. Load hyperparameters
    # 2. device = "cuda" if (gpu and torch.cuda.is_available()) else "cpu"
    # 3. Initialize wandb
    # 4. Load datasets

    # Create the FruitsDataset(s) and their DataLoaders
    model_name_or_path = 'google/vit-base-patch16-224-in21k'
    filepath = "../data/processed"
    processor = ViTImageProcessor.from_pretrained(model_name_or_path)
    
    data_cleaning = CleanData()
    data_cleaning.execute()

    train_dataset = FruitsDataset(filepath=filepath, feature_extractor=processor, data_type="train")
    val_dataset = FruitsDataset(filepath=filepath, feature_extractor=processor, data_type="valid")
    labels = train_dataset.get_labels()

    train_dataloader = DataLoader(train_dataset)
    valid_dataloader = DataLoader(val_dataset)

    metric = load_metric("accuracy")
    model = AutoModelForImageClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)}
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    epochs = 5

    num_training_steps = epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    device = 'cpu'
    model.to(device)

    for epoch in tqdm(range(epochs), desc="Training"):

        running_loss = 0.0
        accuracy = 0.0
        model.train()

        for batch in tqdm(train_dataloader, desc="Batch", leave=False):

            batch["pixel_values"] = torch.squeeze(batch["pixel_values"], 1)

            optimizer.zero_grad()

            batch = {k: v.to(device) for k, v in batch.items()}

            y_pred = model(**batch)

            class_pred = torch.argmax(torch.softmax(y_pred.logits, dim=1), dim=1)

            is_correct = (
                class_pred.detach().cpu().numpy() == np.array(batch["labels"].cpu())
            ).sum()

            accuracy += is_correct

            loss = y_pred.loss

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            lr_scheduler.step()

        running_loss /= len(train_dataloader)
        accuracy /= len(train_dataloader)

        print(f"Training Loss: {running_loss}, Training Accuracy: {accuracy}")

        model.eval()
        running_loss = 0.0
        accuracy = 0.0
        with torch.no_grad():
            for batch in tqdm(valid_dataloader, desc="Validation", leave=False):
                batch["pixel_values"] = torch.squeeze(batch["pixel_values"], 1)
                batch = {k: v.to(device) for k, v in batch.items()}

                y_pred = model(**batch)

                class_pred = torch.argmax(torch.softmax(y_pred.logits, dim=1), dim=1)

                is_correct = (
                    class_pred.detach().cpu().numpy() == np.array(batch["labels"].cpu())
                ).sum()

                accuracy += is_correct

                loss = y_pred.loss
                running_loss += loss.item()

            running_loss /= len(valid_dataloader)
            accuracy /= len(valid_dataloader)

            print(f"Validation Loss: {running_loss}, Validation Accuracy: {accuracy}")
