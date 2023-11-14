import os
import matplotlib.pyplot  as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image, ImageOps
import numpy as np
import zipfile
import random

zip_file_path = "data/external/archive.zip"
extracted_dir = "data/external/"

class CleanData():
    def __init__(
        self,
    ) -> None:
        super(CleanData, self).__init__()
        self.data_dir = "data/external/fruit_images"

    def rename_folder(self, target_folder, old_folder_name, new_folder_name):
        for folder_name in os.listdir(target_folder):
            if folder_name == old_folder_name:
                old_dir_path = os.path.join(target_folder, folder_name)
                new_dir_path = os.path.join(target_folder, new_folder_name)
                os.rename(old_dir_path, new_dir_path)

    def create_df(self):
        current_directory = os.getcwd()
        print("Current Directory:", current_directory)
        self.rename_folder(target_folder=extracted_dir, old_folder_name= 'Processed Images_Fruits', new_folder_name='fruit_images')

        fruit_images_dir = os.path.join(extracted_dir, "fruit_images")
        self.rename_folder(target_folder=fruit_images_dir, old_folder_name='Good Quality_Fruits', new_folder_name='good_quality_fruits')
        self.rename_folder(target_folder=fruit_images_dir, old_folder_name='Bad Quality_Fruits', new_folder_name='bad_quality_fruits')
        self.rename_folder(target_folder=fruit_images_dir, old_folder_name='Mixed Qualit_Fruits', new_folder_name='mixed_quality_fruits')

        bad_quality_path = self.data_dir + "/bad_quality_fruits"
        good_quality_path = self.data_dir + "/good_quality_fruits"
        mixed_quality_path = self.data_dir + "/mixed_quality_fruits"

        file_paths=[]
        labels=[]
        samples = 0
        sample_count = 20

        for fruit_quality in [bad_quality_path, good_quality_path, mixed_quality_path]:
            fruit_list = os.listdir(fruit_quality)
            for fruit in fruit_list:
                fruit_path = os.path.join(fruit_quality, fruit)
                image_list = os.listdir(fruit_path)
                for i, img in enumerate(image_list):
                    image_path = os.path.join(fruit_path, img)
                    if i < sample_count:
                        img = plt.imread(image_path)               
                        samples +=1
                    file_paths.append(image_path)
                    if fruit_quality == mixed_quality_path:
                        labels.append(fruit + '_Mixed')
                    else:
                        labels.append(fruit)
            
        fruit_series = pd.Series(file_paths, name='image')
        label_series = pd.Series(labels, name='label')
        df = pd.concat([fruit_series, label_series], axis=1)
        return df
    
    def df_information(self, df_: pd.DataFrame):    
        class_count = len(list(df_["label"].unique()))
        print(f"The dataset contains {df_.shape[0]} images.")
        print(f"The dataset contains the following {class_count} distinct classes. \n")

        for fruit_class in list(df_["label"].unique()):
            print(fruit_class)

        items_per_class = list(df_["label"].value_counts())
        print(f"\nEach of the above classses contains {items_per_class} images.")

    def split_df_to_train_and_test(self, df_:pd.DataFrame) :
        # Split the DataFrame into train (70%), validation (15%), and test (15%)
        train_df, test_and_valid_df = train_test_split(df_, test_size=0.2, random_state=42)
        valid_df, test_df = train_test_split(test_and_valid_df, test_size=0.5, random_state=42)

        # Check the lengths of the resulting DataFrames
        print("Train set length:", len(train_df))
        print("Validation set length:", len(valid_df))
        print("Test set length:", len(test_df))

        return train_df, valid_df, test_df
    
    def df_trim(self, df_: pd.DataFrame, desired_samples_per_class: int) -> pd.DataFrame:
        # Create an empty DataFrame to store the trimmed data
        trimmed_train_df = pd.DataFrame(columns=df_.columns)

        # Iterate through each class and select the first 200 samples
        for class_name in df_['label'].unique():
            class_samples = df_[df_['label'] == class_name].head(desired_samples_per_class)
            trimmed_train_df = pd.concat([trimmed_train_df, class_samples])

        # Reset the index of the trimmed DataFrame
        trimmed_train_df.reset_index(drop=True, inplace=True)

        return trimmed_train_df
    
    def classes_with_less_than_n_samples(self, df_: pd.DataFrame, desired_samples_per_class: int):
        class_counts = df_['label'].value_counts()
        classes_with_less_than_n_samples_list =  class_counts[class_counts < desired_samples_per_class].index.tolist()
        print(f"Classes with less than {desired_samples_per_class} samples : {classes_with_less_than_n_samples_list}")

        return classes_with_less_than_n_samples_list

    def augment_image(self, image_path, save_folder):
        os.makedirs(save_folder, exist_ok=True)
        image = Image.open(image_path)
        # Apply random rotation (you can customize the rotation angle)
        angle = np.random.randint(-15, 15)
        augmented_image = image.rotate(angle)

        # Apply horizontal flip with a 50% chance
        if np.random.choice([True, False]):
            augmented_image = ImageOps.mirror(augmented_image)

        if save_folder:
            # Ensure the save folder exists
            os.makedirs(save_folder, exist_ok=True)
            filename = f"AUG_IMG_{np.random.randint(1000)}_{angle}.jpg"
            file_path = os.path.join(save_folder, filename)
            augmented_image.save(file_path)
        
        return file_path
    
    def df_balance(self, df_: pd.DataFrame, desired_samples_per_class : int) -> pd.DataFrame:
        save_folder = os.path.join(self.data_dir, "augmented")
        # Iterate through the classes with fewer than 200 samples
        target_class = self.classes_with_less_than_n_samples(df_=df_, desired_samples_per_class=200)
        for class_name in target_class:
                class_samples = df_[df_["label"]==class_name].value_counts()
                class_df = df_[df_["label"] == class_name]

                images_to_augment = len(class_samples)
                while images_to_augment < desired_samples_per_class:
                    
                    # Choose an image from the class
                    random_index = random.randint(0, len(class_df) - 1)
                    # Get the image file path at the random index
                    random_image = class_df.iloc[random_index]["image"]

                    # Apply data augmentation to generate a new image
                    augmented_image = self.augment_image(image_path=random_image, save_folder=save_folder)

                    # Append the augmented image to your dataset
                    new_df = pd.DataFrame({'image': [augmented_image], 'label': [class_name]})
                    df_ = pd.concat([df_, new_df], ignore_index=True)
                    images_to_augment+=1
                print(f"For class [{class_name}] I have augmented [{desired_samples_per_class-len(class_samples)}] images.")
        return df_
    
    def resize_image(self, image_path, output_path, new_width : int, new_height: int):
        image = Image.open(image_path)
        resized_image = image.resize((new_width, new_height))
        resized_image.save(output_path)

    def resize_images_in_df(self, df_:pd.DataFrame):
        new_width = 224
        new_height = 224

        for index, row in df_.iterrows():
            image_path = row["image"]
            print(image_path, index)
            if os.path.exists(image_path):
                self.resize_image(image_path=image_path, output_path=image_path, new_width=new_width, new_height=new_height)
            else:
                print(f"File not found: {image_path}")

        print(df_["image"])
        return df_

    def showcase_df(self, df_:pd.DataFrame):
        # Create a figure to display the images
        distinct_labels = df_["label"].unique()
        plt.figure(figsize=(10, 10))
        for label in distinct_labels:
            # Get all images for the current label
            label_df = df_[df_["label"] == label]
            random_index = random.randint(0, len(label_df) - 1)
            sample_image = Image.open(label_df.iloc[random_index]["image"])
            # Load and display the image
            plt.subplot(5, 5, distinct_labels.tolist().index(label) + 1)
            plt.imshow(sample_image)
            plt.title(label, color='blue', fontsize=10)
            plt.axis("off")
        plt.show()


    def save_dfs(self, train_df, valid_df, test_df):
        processed_data_path = "data/processed"
        valid_df.to_csv(os.path.join(processed_data_path, "validation_data.csv"), index=False) 
        test_df.to_csv(os.path.join(processed_data_path, "test_data.csv"), index=False) 
        train_df.to_csv(os.path.join(processed_data_path, "train_data.csv"), index=False) 

    def execute(self):
        data = self.create_df()
        distinct_labels_df = data.drop_duplicates(subset="label")
        print(distinct_labels_df.head(18))

        self.df_information(df_=data)
        train_df, valid_df, test_df = self.split_df_to_train_and_test(df_=data)

        trimmed_train_df = self.df_trim(df_= train_df, desired_samples_per_class=200)
        self.df_information(df_= trimmed_train_df)

        df_balanced = self.df_balance(df_=trimmed_train_df, desired_samples_per_class=200)
        self.df_information(df_=df_balanced)

        df_balanced_resized = self.resize_images_in_df(df_=df_balanced)
        #data_cleaning.showcase_df(df_=df_balanced_resized)

        self.save_dfs(train_df=df_balanced_resized, valid_df=valid_df, test_df=test_df)
