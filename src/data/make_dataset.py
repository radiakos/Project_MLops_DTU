import pandas as pd
import os
from torch.utils.data import Dataset
from PIL import Image
import torch

def load_image(image_path):
        return Image.open(image_path)

def check_and_transform_image(image):
    if image.mode == "L":
        # If it's a black and white image, convert it to RGB
        print(f" RGB image {image}")
        image = image.convert("RGB")
    return image


class FruitsDataset(Dataset):
    def __init__(
        self,
        filepath, 
        feature_extractor,
        data_type
    ) -> None:
        super(FruitsDataset, self).__init__()
        self.filepath = filepath
        self.feature_extractor = feature_extractor
        self.data_type = data_type

        print(f"{self.data_type} :  ")
        if self.data_type == "train":
            csv_name = "train_data.csv"
        elif self.data_type == "test":
            csv_name = "test_data.csv"
        else:
            csv_name = "validation_data.csv"
            
        data_csv = os.path.join(self.filepath, csv_name)
        self.dataframe = pd.read_csv(data_csv)

        self.process_df()

    def process_df(self):
        self.dataframe['image_path'] = self.dataframe['image']
        self.dataframe['image'] = self.dataframe['image_path'].apply(load_image)
        self.dataframe["label"].unique()
        mapping = {
            'Banana_Bad': 1,
            'Lemon_Mixed': 2,
            'Apple_Good' : 3, 
            'Guava_Mixed': 4,
            'Guava_Bad' : 5, 
            'Lime_Bad':6, 
            'Pomegranate_Good':7,
            'Guava_Good':8,
            'Lime_Good':9, 
            'Banana_Good':10, 
            'Apple_Bad':11, 
            'Pomegranate_Bad':12,
            'Orange_Good':13, 
            'Banana_Mixed':14, 
            'Orange_Bad':15, 
            'Pomegranate_Mixed':16,
            'Orange_Mixed':17, 
            'Apple_Mixed':18
        }
        self.dataframe['label'] = self.dataframe['label'].map(mapping)
        print(self.dataframe)
        self.dataframe["image"] = self.dataframe["image"].apply(check_and_transform_image)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image = self.dataframe.iloc[idx, 0]  # Assuming the "image" column contains actual images
        label = self.dataframe.iloc[idx, 1]  # Assuming the "label" column is at index 1 
        
        pixel_values = self.feature_extractor(image, return_tensors="pt").pixel_values
        print(f"image_path : {self.dataframe.iloc[idx, 2]}, shape: {pixel_values.shape}")
        return {"pixel_values": pixel_values, "labels": torch.tensor(label)}      
    
    def get_labels(self):
        labels = self.dataframe['label'].unique().tolist()
        return labels
