import hydra
from omegaconf import DictConfig, OmegaConf
import os

@hydra.main(version_base=None, config_path="configs/experiment/setting", config_name="setting_strongreject.yaml")
def create_experiment(cfg: DictConfig):
    pass
if __name__ == "__main__":
    create_experiment()