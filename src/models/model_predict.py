import os
from model_actions import Model
from omegaconf import OmegaConf
import wandb
import hydra
import torch

# Load config file using hydra
@hydra.main(config_path="../conf", config_name="predict_config.yaml",version_base=None)
def main(cfg):
    print("Is cuda available?", torch.cuda.is_available())
    # Initialize wandb
    run = wandb.init(project=cfg.wandb.project)
    # Load model
    m_class=Model(cfg)
    model,model_name=m_class.load_model()
    print("Predicting the class of the image")
    # Predict the class of the image
    image_name, class_name=m_class.predict(model)
    #keep only value of class name
    wandb.log({"Test with model": model_name})
    wandb.log({"Image name": image_name})
    wandb.log({"Predicted class": class_name})
    print("The class of the image ", image_name, "is", class_name, "according to the model", model_name)
    return class_name

if __name__ == "__main__":
    main()
