import os
import glob

import random
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
from PIL import Image
from torchvision import transforms
from google.cloud import storage


def upload_model_gcs(dir_path:str, bucket_name:str, blob_name:str, credentials_file:str):
    """Uploads a directory to a given Google Cloud Service bucket.
    Args:
        dir_path (str): Path of the directory to be uploaded.
        bucket_name (str): Name of the bucket to upload to.
        credentials_file (str): Name of the json file with the credentials.
        blob_name (str): folder name to be used for uploading.
    """
    # Initialize the Google Cloud Storage client with the credentials
    storage_client = storage.Client.from_service_account_json(credentials_file)

    # Get the target bucket
    bucket = storage_client.bucket(bucket_name)

    # Upload the file to the bucket
    rel_paths = glob.glob(dir_path + "/**", recursive=True)
    # for local_file in rel_paths:
    #     remote_path = f'{blob_name}/{"/".join(local_file.split(os.sep)[1:])}'
    #     if os.path.isfile(local_file):
    #         blob = bucket.blob(remote_path)
    #         blob.upload_from_filename(local_file)
    #     print(f"File {local_file} uploaded to {remote_path}.")
    for root, _, files in os.walk(dir_path):
        for local_file in files:
            local_file_path = os.path.join(root, local_file)
            relative_path = os.path.relpath(local_file_path, dir_path)
            remote_path = f"{blob_name}/{relative_path}"

            blob = bucket.blob(remote_path)
            blob.upload_from_filename(local_file_path)
            print(f"File {local_file_path} uploaded to {remote_path}.")


def download_model_gcs(dir_path:str, bucket_name:str, blob_name:str, credentials_file:str):
    """Downloads a directory from a given Google Cloud Service bucket.
    Args:
        dir_path (str): Path of the directory to be downloaded to.
        bucket_name (str): Name of the bucket to download from.
        credentials_file (str): Name of the json file with the credentials.
        blob_name (str): folder name to be used for downloading.
    """
    # Initialize the Google Cloud Storage client with the credentials
    storage_client = storage.Client.from_service_account_json(credentials_file)

    # Get the target bucket
    bucket = storage_client.bucket(bucket_name)

    # Download the file from the bucket
    blobs = bucket.list_blobs(prefix=blob_name)
    for blob in blobs:
        if blob.name[-1] == '/':
            continue
        remote_path = blob.name
        local_file_path = os.path.join(dir_path, remote_path)
        local_dir_path = os.path.dirname(local_file_path)
        if not os.path.exists(local_dir_path):
            os.makedirs(local_dir_path)
        blob.download_to_filename(local_file_path)
        print(f"File {remote_path} downloaded to {local_file_path}.")
    return

class Model:
    def __init__(self, cfg):
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
        self.logger.info(f"Using device: {self.device}")
        self.dirs = cfg.dirs
        self.params = cfg.params
        self.model_path = cfg.model_path
        self.gcs = cfg.gcs

    def log_metrics_to_wb(self,train_flag:bool):
        """Log metrics to wandb
        Args:
            train_flag (bool): True if we are in the training phase, False otherwise."""
        if train_flag:
            wandb.log({"train_batch_size": self.params.train_batch_size})
            wandb.log({"valid_batch_size": self.params.valid_batch_size})
            wandb.log({"learning_rate": self.params.lr})
            wandb.log({"epochs": self.params.epochs})
        else:
            wandb.log({"test_batch_size": self.params.test_batch_size})
        #wandb.log({"Device": self.device})
        return

    def log_image_to_wb(self, batch, class_pred, train_flag:bool):
        #batch is a dict
        #class_pred is a tensor
        """
        Log images to wandb
        
        Args:
            batch (dict): batch of images and labels
            class_pred (tensor): tensor of predictions
            train_flag (bool): True if we are in the training phase, False otherwise.
            """
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
        #class_pred and batch_labels are tensors
        """Log histograms to wandb
        
        Args:
            class_pred (tensor): tensor of predictions
            batch_labels (tensor): tensor of labels of the batch
            
            """
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
        #flag is either "train" or "test"
        """Split the dataset into train, valid and test sets.
        
        Args:
        
            flag (str): "train" if we are in the training phase, "test" otherwise.
            """
        
        #log metrics to wandb
        self.log_metrics_to_wb(train_flag=True)

        # Create the FruitsDataset(s) and their DataLoaders
        model_name_or_path = self.model_path
        processor = ViTImageProcessor.from_pretrained(model_name_or_path)

        #train and valid
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

        #test
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
        """Train the model.
        
        Returns:

            model (AutoModelForImageClassification): pretrained model to use
            
            """
        # 1. Create the FruitsDataset(s) and their DataLoaders
        model_name_or_path = self.model_path
        train_dataset,train_dataloader,valid_dataloader=self.split_dataset("train")
        
        # 2. Create the model, optimizer and scheduler
        model = AutoModelForImageClassification.from_pretrained(
            model_name_or_path,
            num_labels=train_dataset.num_classes,
        )

        optimizer = torch.optim.SGD(model.parameters(), lr=self.params.lr)
        epochs = self.params.epochs
        # 3. Create the learning rate scheduler
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
        """Save model to the model_dir.       
            """
        #save model to the model_dir
        model_dir = self.dirs.model_dir
        #find if there is such folder in the model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model.save_pretrained(model_dir+name)
        upload_model_gcs(os.path.join(model_dir, name), self.gcs.bucket_name, name, self.gcs.credentials_file)
        # upload_model_gcs(self.gcs.saved_dir, self.gcs.bucket_name, self.gcs.blob_name, self.gcs.credentials_file)       
        return
    
    def load_model(self,name=None):
        #load model from the model_dir
        model_dir = self.dirs.model_dir
        download_model_gcs(model_dir, self.gcs.bucket_name, name, self.gcs.credentials_file)
        if not os.path.exists(model_dir) or len(os.listdir(model_dir))==1:
            raise Exception("No model found in the model_dir, please correct path and name of the model from the bucket first")
        if name is None:
            #find first model in the model_dir
            name=os.listdir(model_dir)[0]
            if name=='.gitkeep':
                if len(os.listdir(model_dir))==1:
                    raise Exception("No model found in the model_dir, please correct path and name of the model from the bucket first")
                name=os.listdir(model_dir)[1]
            print("load model:",name)
        if not os.path.exists(model_dir+name):
            raise Exception("No model found in the model_dir, please correct path and name of the model from the bucket first")
        model = AutoModelForImageClassification.from_pretrained(model_dir+name)
        return model,name

    def test(self,model):
        """Test the model.

        Args:
            model (AutoModelForImageClassification): pretrained model to use

            """ 
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

    def choose_random_dir(self, dir_path):
        """Choose a random subdirectory of a directory given the directory path."""
        # this function returns a random subdirectory of a directory given the directory path -> used to select an image folder inside the Test dir (either fruit folder or quality folder) 
        subdirectories = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]

        if not subdirectories:
            raise Exception(f"No subdirectories found in the specified directory {dir_path}")
        # Choose a random subdirectory
        random_subdirectory = random.choice(subdirectories)
        return dir_path+'/'+random_subdirectory+'/'
    
    def choose_random_image(self, dir_path, sub_dir_path=None):
        """Choose a random image inside a folder given the folder path."""
        # this function return a random image inside a folder.
        if sub_dir_path == "":
            # if no subdirectory is specified, choose a random subdirectory
            image_files = [f for f in os.listdir(dir_path) if f.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG'))]
        else:
            image_files = [f for f in os.listdir(os.path.join(dir_path, sub_dir_path)) if f.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG'))]

        if not image_files:
            raise Exception(f"No image files found in the chosen subdirectory {sub_dir_path}")
        # Choose a random image file
        random_image_file = random.choice(image_files)
        return random_image_file

    def load_image(self):
        """Load an image from the test_dir.            
                Returns:    
                    image (PIL image): image to predict
                    image_name (str): name of the image to predict"""
        # we need to load one specific image from the test_dir with name image_name (both defined in predict config)
        image_dir = self.dirs.image_dir
        if "data/processed/test" in image_dir:
            test_image_folder = self.choose_random_dir(image_dir)
            if not os.path.exists(test_image_folder):
                raise Exception(f"No image found in the test_image_folder, please check the test_image_folder {test_image_folder}")
        else:
            #print(image_dir)
            #create empty path
            test_image_folder = ""
        if self.params.image_name == 'None':
            # find a random image in the test_image_folder if name is not specified
            image_name=self.choose_random_image(image_dir, test_image_folder)
            image_path=os.path.join(image_dir, test_image_folder, image_name)
        else:
            image_name=self.params.image_name
            image_path=os.path.join(image_dir,image_name)

        #print(f"Load image in the path : {image_path} ")
        if not os.path.exists(image_path):
            raise Exception(f"The image name is not correct, please check if the image exists inside the path {image_dir}")
        image = Image.open(image_path)
        return image, image_name

    def predict(self,model,fastapi_image=None):
        """Predict class of an image."""
        model.to(self.device)
        model.eval()
        #load image
        if fastapi_image is None:
            image, image_name = self.load_image()
        else:
            #this is used for fastapi
            image=fastapi_image
            image_name="fastapi_image"
        
        #predict
        if image.mode != "RGB":
            image = image.convert(mode="RGB")
        processor = ViTImageProcessor.from_pretrained(self.model_path)
        
        image_values = processor(images=image, return_tensors="pt").pixel_values
        image_values=image_values.to(self.device)
        y_pred = model(pixel_values=image_values)
        class_pred = torch.argmax(torch.softmax(y_pred.logits, dim=1), dim=1).item()
        return image_name, class_pred
