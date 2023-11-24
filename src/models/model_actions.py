
import os
from pathlib import Path
from omegaconf import OmegaConf
import hydra
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from transformers import AutoModelForImageClassification, ViTImageProcessor,get_scheduler
from omegaconf import OmegaConf
import wandb
from src.data.make_dataset import FruitsDataset
import logging



class Model:
    def __init__(self, cfg):
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
        self.logger.info(f"Using device: {self.device}")
        self.dirs = cfg.dirs
        self.params = cfg.params
        self.model_path = cfg.model_path

    def log_metrics_to_wb(self,train_flag):
        if train_flag:
            wandb.log({"train_batch_size": self.params.train_batch_size})
            wandb.log({"valid_batch_size": self.params.valid_batch_size})
            wandb.log({"learning_rate": self.params.lr})
            wandb.log({"epochs": self.params.epochs})
        else:
            wandb.log({"test_batch_size": self.params.test_batch_size})
        if self.params.gpu_flag == True:
            wandb.log({"Device": "cuda"})
        else:
            wandb.log({"Device": "cpu"})
        return
        
    def log_image_to_wb(self, batch, class_pred, train_flag):
        if train_flag:
            table_name = "validation example"
        else:
            table_name = "test example"
        my_table = wandb.Table(columns=["image", "label", "prediction"])
        #find len of tensor class_pred
        for i in range(len(class_pred)):
            image = wandb.Image(batch['pixel_values'][i])
            prediction = class_pred[i]
            label= batch['labels'][i]
            my_table.add_data(image, label, prediction)
        wandb.log({table_name: my_table})
        return
    
    def log_histograms(self,class_pred,batch_labels):
        batch_labels=batch_labels.cpu().numpy()
        class_pred=class_pred.cpu().numpy()
        for i in range(len(np.unique(batch_labels))):
            pred_list=[]
            for j in range(len(class_pred)):
                if batch_labels[j]==i:
                    pred_list.append(class_pred[j])
            wandb.log({f"histogramm of class {i}": wandb.Histogram(pred_list)})
        return

    def split_dataset(self,flag):
        
        self.log_metrics_to_wb(train_flag=True)

        # Create the FruitsDataset(s) and their DataLoaders
        model_name_or_path = self.model_path
        processor = ViTImageProcessor.from_pretrained(model_name_or_path)

        if flag=="train":
            train_dir = self.dirs.train_dir
            valid_dir = self.dirs.valid_dir
            train_dataset = FruitsDataset(
                input_filepath=train_dir, feature_extractor=processor, data_type="train"
            )
            val_dataset = FruitsDataset(
                input_filepath=valid_dir, feature_extractor=processor, data_type="valid"
            )
            train_loader_options = {
            "shuffle": self.params.shuffle_flag,
            "batch_size": self.params.train_batch_size,
            "num_workers": self.params.num_workers,
            }

            valid_loader_options = {
                "shuffle": self.params.shuffle_flag,
                "batch_size": self.params.valid_batch_size,
                "num_workers": self.params.num_workers,
            }
            train_dataloader = DataLoader(train_dataset, **train_loader_options)
            valid_dataloader = DataLoader(val_dataset, **valid_loader_options)
            return train_dataset,train_dataloader,valid_dataloader


        if flag=="test":
            test_dir = self.dirs.test_dir
            test_dataset = FruitsDataset(
                input_filepath=test_dir, feature_extractor=processor, data_type="test"
            )
            test_loader_options = {
                "shuffle": self.params.shuffle_flag,
                "batch_size": self.params.test_batch_size,
                "num_workers": self.params.num_workers,
            }
            test_dataloader = DataLoader(test_dataset, **test_loader_options)
            return test_dataloader

    def train(self):
        model_name_or_path = self.model_path
        train_dataset,train_dataloader,valid_dataloader=self.split_dataset("train")

        model = AutoModelForImageClassification.from_pretrained(
            model_name_or_path,
            num_labels=train_dataset.num_classes,
        )

        optimizer = torch.optim.SGD(model.parameters(), lr=self.params.lr)
        epochs = self.params.epochs

        num_training_steps = epochs * len(train_dataloader)
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=self.params.warmup_steps,
            num_training_steps=num_training_steps,
        )

        self.logger.info(f"Using device: {self.device}")

        model.to(self.device)

        # Training loop
        for epoch in tqdm(range(epochs), desc="Training"):
            running_loss = 0.0
            accuracy = 0.0
            model.train()
            batch_list=[]
            class_pred_list=[]
            for batch in tqdm(train_dataloader, desc="Batch", leave=False):
                batch["pixel_values"] = torch.squeeze(batch["pixel_values"], 1)
                optimizer.zero_grad()
                batch = {k: v.to(self.device) for k, v in batch.items()}

                y_pred = model(**batch)
                class_pred = torch.argmax(torch.softmax(y_pred.logits, dim=1), dim=1)

                accuracy += accuracy_score(
                    batch["labels"].cpu().numpy(), class_pred.detach().cpu().numpy()
                )

                loss = y_pred.loss
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                lr_scheduler.step()
                batch_list.append(batch["labels"])
                class_pred_list.append(class_pred)

            running_loss /= len(train_dataloader)
            accuracy /= len(train_dataloader)


            wandb.log({"Training Loss": running_loss, "Training Accuracy": accuracy})
            self.logger.info(
                f"  Training Loss: {running_loss:.4f}, Training Accuracy: {accuracy:.4f}"
            )

            # 5. Validation loop
            model.eval()
            running_loss = 0.0
            accuracy = 0.0

            with torch.no_grad():
                for batch in tqdm(valid_dataloader, desc="Validation", leave=False):
                    batch["pixel_values"] = torch.squeeze(batch["pixel_values"], 1)
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                    y_pred = model(**batch)
                    class_pred = torch.argmax(torch.softmax(y_pred.logits, dim=1), dim=1)

                    accuracy += accuracy_score(
                        batch["labels"].cpu().numpy(), class_pred.detach().cpu().numpy()
                    )
                    loss = y_pred.loss
                    loss = y_pred.loss
                    running_loss += loss.item()
                
            self.log_image_to_wb(batch,class_pred,train_flag=True)
            running_loss /= len(valid_dataloader)
            accuracy /= len(valid_dataloader)

            wandb.log(
                {"Validation Loss": running_loss, "Validation Accuracy": accuracy}
            )
            self.logger.info(
                f"  Validation Loss: {running_loss:.4f}, Validation Accuracy: {accuracy:.4f}"
            )
            #save model and its weight to the model_dir and model_weight_dir
            name="model"+str(epoch)
            self.save_model(model,name)
        return model, name
    
    def save_model(self,model,name):
        model_dir = self.dirs.model_dir
        #find if there is such folder in the model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model.save_pretrained(model_dir+name)
        return
    
    def load_model(self,name=None):
        model_dir = self.dirs.model_dir
        if not os.path.exists(model_dir):
            raise Exception("No model found in the model_dir, please train a model first")
        if name is None:
            #find whatever model in the model_dir
            name=os.listdir(model_dir)[0]
            print("load model:",name)
        model = AutoModelForImageClassification.from_pretrained(model_dir+name)
        return model,name

    def test(self,model):
        model.to(self.device)
        model.eval()  # Set the model to evaluation model

        test_dataloader=self.split_dataset("test")
        # 6. Test loop
        test_loss = 0.0
        total_samples = 0
        accuracy=0.0
        log_table_counter=0
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Testing", leave=False):
                batch["pixel_values"] = torch.squeeze(batch["pixel_values"], 1)
                batch = {k: v.to(self.device) for k, v in batch.items()}

                y_pred = model(**batch)
                class_pred = torch.argmax(torch.softmax(y_pred.logits, dim=1), dim=1)

                accuracy += accuracy_score(
                    batch["labels"].cpu().numpy(), class_pred.detach().cpu().numpy()
                )
                
                loss = y_pred.loss
                loss = y_pred.loss
                test_loss += loss.item()
                if log_table_counter<10:
                    log_table_counter+=1
                    self.log_image_to_wb(batch,class_pred,train_flag=False)
        # Calculate average loss and accuracy
        test_loss /= len(test_dataloader)
        accuracy /= len(test_dataloader)

        wandb.log({"Test Loss": test_loss, "Accuracy": accuracy})
        self.logger.info(f"  Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")
        return

    def load_image(self):
        # we need to load one specific image from the image_path with name image_name (both defined in predict config)
        image_path = self.dirs.image_path
        if not os.path.exists(image_path):
            raise Exception("No image found in the image_path, please check the image_path")
        if self.params.image_name is None:
            #find whatever image in the image_path if name is not specified
            image_name=os.listdir(image_path)[0]
            print("load image:",image_name)
        else:
            image_name=self.params.image_name
        image=None
        image.to(self.device)
        return image, image_name

    def predict(self,model):
        model.to(self.device)
        model.eval()
        #load image
        image, image_name = self.load_image()
        #predict
        y_pred = model(**image)
        class_pred = torch.argmax(torch.softmax(y_pred.logits, dim=1), dim=1)
        return image_name, class_pred
