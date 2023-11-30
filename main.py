import os
from fastapi import FastAPI, File, UploadFile, Query
from src.models.model_actions import Model
from omegaconf import OmegaConf
import wandb
import hydra
import torch

app = FastAPI()
# Create FastAPI endpoint

#
@app.get("/predict")
async def main(fastapi_flag: bool = Query(default=None, description="FastAPI Flag")):
    cfg = OmegaConf.load('src/conf/predict_config.yaml')
    # Initialize wandb
    local_path=os.getcwd()
    cfg.dirs.image_dir=os.path.join(local_path, cfg.dirs.image_dir)
    cfg.dirs.model_dir=os.path.join(local_path, cfg.dirs.model_dir)
    #run = wandb.init(project="Project_MLOps")
    # Load model
    m_class = Model(cfg)
    model, model_name = m_class.load_model()
    # Determine whether to expect an image or not based on the query parameter
    # Predict the class of the image
    if fastapi_flag==True:
        image_name, class_name = m_class.predict(model, File(...))
    else:
        image_name, class_name = m_class.predict(model)
    
    return model_name


