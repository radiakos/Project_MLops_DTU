import os
from model_action import Model
from omegaconf import OmegaConf
import wandb
import hydra
import torch

# Load config file
@hydra.main(config_path="../conf", config_name="models_config.yaml",version_base=None)
def main(cfg):
    print("Is cuda available?", torch.cuda.is_available())
    # Initialize wandb
    run = wandb.init(project=cfg.wandb.project, config=OmegaConf.to_container(cfg))
    # Load hyperparameters
    m_class=Model(cfg)
    if cfg.params.train==True:
        print("Training and testing")
        model,model_name=m_class.train()
    else:
        print("Testing")
        model,model_name=m_class.load_model()
    wandb.log({"Test with model": model_name})
    m_class.test(model)

if __name__ == "__main__":
    main()
