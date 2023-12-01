import os
from fastapi import FastAPI, File, UploadFile, Query
from src.models.model_actions import Model
from omegaconf import OmegaConf
from PIL import Image
import wandb
import hydra
import torch
import cv2

app = FastAPI()

# Load config file
@app.post("/predict")
async def predict_upload(file: UploadFile = File(...)):
    cfg = OmegaConf.load('src/conf/predict_config.yaml')
    original_filename = file.filename
    with open(original_filename, 'wb') as image:
        content = await file.read()
        image.write(content)
        image.close()

    local_path = os.getcwd()
    cfg.dirs.image_dir = local_path+"\\data\\processed\\test\\"
    cfg.dirs.model_dir = local_path+"/src/models/saved_models/"
    directory = os.path.dirname(cfg.dirs.image_dir)
    image_path = os.path.join(directory, original_filename)
    image_name = cv2.imread(original_filename)
    image2 = Image.fromarray(image_name)
    # Image.open(image_path)
    # run = wandb.init(project="Project_MLOps")
    # Load model
    m_class = Model(cfg)
    model, model_name = m_class.load_model()
    image_name, class_name = m_class.predict(model, image2)

    return original_filename, class_name
