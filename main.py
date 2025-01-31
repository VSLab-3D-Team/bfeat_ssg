from runners.trainer import BFeatVanillaTrainer
# import torch.multiprocessing as mp
from hydra import initialize, compose
import argparse

parser = argparse.ArgumentParser(description="Training BFeat Architecture")
parser.add_argument("--mode", type=str, default="train", choices=["train", "experiment"], help="Select mode for BFeat (train/evaluation)")
parser.add_argument("--config", type=str, default="baseline.yaml", help="Runtime configuration file path")
args = parser.parse_args()

def train(config):
    device = "cuda"
    trainer = BFeatVanillaTrainer(config, device)
    trainer.train()

def experiment(config):
    ## TODO: Implement for various experiments 
    
    pass

if __name__ == "__main__":
    # mp.set_start_method("spawn", force=True)
    with initialize(config_path="./config"):
        config = compose(config_name=args.config)
    
    runtime_mode = args.mode
    if runtime_mode == "train":
        train(config)
    elif runtime_mode == "experiment":
        experiment(config)
    