import os
from model_actions import Model
from omegaconf import OmegaConf
import wandb
import hydra
import torch

# Load config file using hydra
@hydra.main(config_path="../conf", config_name="models_config.yaml",version_base=None)
def main(cfg):
    """Main function to run the model
    Args:
        cfg: Config file, with directory paths, hyperparameters and wandb and GCP settings"""
    
    # Check if cuda is available to use gpu, if not, use cpu
    print("Is cuda available?", torch.cuda.is_available())

    # Initialize wandb
    run = wandb.init(project=cfg.wandb.project)

    # Load model
    m_class=Model(cfg)
    if cfg.params.train==True:
        print("Training and testing")
        model,model_name=m_class.train()
    else:
        print("Testing")
        model,model_name=m_class.load_model()
    wandb.log({"Test with model": model_name})
    # Test the model
    m_class.test(model)

if __name__ == "__main__":
    main()
