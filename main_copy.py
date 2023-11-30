import os
from fastapi import FastAPI, File, UploadFile, Query
from src.models.model_actions import Model
from omegaconf import OmegaConf
import wandb
import hydra
import torch

app = FastAPI()

@hydra.main(config_path="src/conf", config_name="predict_config.yaml", version_base=None)
def main(cfg):
    print("Config:", OmegaConf.to_yaml(cfg))  # Print the entire configuration
    print("Is cuda available?", torch.cuda.is_available())
    # Initialize wandb
    run = wandb.init(project="Project_MLOps")
    # Load model
    m_class = Model(cfg)
    model, model_name = m_class.load_model()
    print("Predicting the class of the image")
    
    # Determine whether to expect an image or not based on the query parameter
    fastapi_flag = Query(default=cfg.params.fastapi_flag)
    # Predict the class of the image
    print(fastapi_flag)
    if fastapi_flag:
        image_name, class_name = m_class.predict(model, File(...))
    else:
        image_name, class_name = m_class.predict(model)
    print(class_name, image_name)
    
    # Keep only the value of class name
    wandb.log({"Test with model": model_name})
    wandb.log({"Image name": image_name})
    wandb.log({"Predicted class": class_name})
    print("The class of the image", image_name, "is", class_name)
    return class_name, image_name

# Create FastAPI endpoint
@app.get("/predict")
#file: UploadFile = File(...),
async def predict_upload( fastapi_flag: bool = Query(default=None, description="FastAPI Flag")):
    # Call the main function with the parsed configuration
    return {"image_name": fastapi_flag}


