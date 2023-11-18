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
from sklearn.metrics import accuracy_score
from src.data.make_dataset import FruitsDataset
from src.data.data_cleaning import CleanData

from datasets import load_metric
from transformers import AutoModelForImageClassification


@hydra.main(config_path="../conf", config_name="models_config.yaml")
def main(cfg):
    # 1. Load hyperparameters
    dirs = cfg.dirs
    params = cfg.model

    train_batch_size = params.train_batch_size
    valid_batch_size = params.valid_batch_size
    test_batch_size = params.test_batch_size

    shuffle_flag = params.shuffle_flag
    num_workers = params.num_workers
    lr = params.lr
    gpu_flag = params.gpu_flag

    train_dir = dirs.train_dir
    valid_dir = dirs.valid_dir
    test_dir  = dirs.test_dir

    # 2. Initialize wandb

    # 3. Create the FruitsDataset(s) and their DataLoaders
    model_name_or_path = params.model_path
    processor = ViTImageProcessor.from_pretrained(model_name_or_path)

    train_dataset = FruitsDataset(input_filepath=train_dir, feature_extractor=processor, data_type="train")
    val_dataset = FruitsDataset(input_filepath=valid_dir, feature_extractor=processor, data_type="valid")
    test_dataset = FruitsDataset(input_filepath=test_dir, feature_extractor=processor, data_type="test")

    train_loader_options = {
        "shuffle": shuffle_flag,
        "batch_size": train_batch_size,
        "num_workers": num_workers,
    }

    valid_loader_options = {
        "shuffle": shuffle_flag,
        "batch_size": valid_batch_size,
        "num_workers": num_workers,
    }

    test_loader_options = {
        "shuffle": shuffle_flag,
        "batch_size": test_batch_size,
        "num_workers": num_workers,
    }

    train_dataloader = DataLoader(train_dataset, **train_loader_options)
    valid_dataloader = DataLoader(val_dataset, **valid_loader_options)
    test_dataloader = DataLoader(val_dataset, **test_loader_options)

    model = AutoModelForImageClassification.from_pretrained(
        model_name_or_path,
        num_labels=train_dataset.num_classes,
    )

    optimizer =  torch.optim.SGD(model.parameters(), lr=lr)
    epochs = params.epochs

    num_training_steps = epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=params.warmup_steps,
        num_training_steps=num_training_steps,
    )

    device = torch.device(
            "cuda" if (gpu_flag and  torch.cuda.is_available()) else "cpu"
        )
    print(f"Using device: {device}")

    model.to(device)

    # 4. Training loop
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

            accuracy += accuracy_score(batch["labels"].cpu().numpy(), class_pred.detach().cpu().numpy())

            loss = y_pred.loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            lr_scheduler.step()

        running_loss /= len(train_dataloader)
        accuracy /= len(train_dataloader)

        print(f"  Training Loss: {running_loss:.4f}, Training Accuracy: {accuracy:.4f}")

        # 5. Validation loop
        model.eval()
        running_loss = 0.0
        accuracy = 0.0

        with torch.no_grad():
            for batch in tqdm(valid_dataloader, desc="Validation", leave=False):
                batch["pixel_values"] = torch.squeeze(batch["pixel_values"], 1)
                batch = {k: v.to(device) for k, v in batch.items()}

                y_pred = model(**batch)
                class_pred = torch.argmax(torch.softmax(y_pred.logits, dim=1), dim=1)

                accuracy += accuracy_score(batch["labels"].cpu().numpy(), class_pred.detach().cpu().numpy())
                loss = y_pred.loss
                loss = y_pred.loss
                running_loss += loss.item()

            running_loss /= len(valid_dataloader)
            accuracy /= len(valid_dataloader)
            #print(f"correct: {accuracy} / length {len(valid_dataloader)}")

            print(f"  Validation Loss: {running_loss:.4f}, Validation Accuracy: {accuracy:.4f}")

        model.eval()  # Set the model to evaluation mode

    # 6. Test loop
    test_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing", leave=False):
            batch["pixel_values"] = torch.squeeze(batch["pixel_values"], 1)
            batch = {k: v.to(device) for k, v in batch.items()}

            y_pred = model(**batch)
            class_pred = torch.argmax(torch.softmax(y_pred.logits, dim=1), dim=1)

            accuracy += accuracy_score(batch["labels"].cpu().numpy(), class_pred.detach().cpu().numpy())
            loss = y_pred.loss
            loss = y_pred.loss
            test_loss += loss.item()


    # Calculate average loss and accuracy
    test_loss /= len(test_dataloader)
    accuracy /= len(test_dataloader)

    print(f"  Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
