from runners.trainer import BFeatVanillaTrainer
from runners.trainer_skip_obj import BFeatSkipObjTrainer
from hydra import initialize, compose
import argparse

parser = argparse.ArgumentParser(description="Training BFeat Architecture")
parser.add_argument("--mode", type=str, default="train", choices=["train", "experiment"], help="Select mode for BFeat (train/evaluation)")
parser.add_argument("--runners", 
    type=str, default="vanilla", 
    choices=["vanilla", "skipobj", "fourierfeat"], 
    help="Select running model"
)
parser.add_argument("--config", type=str, default="baseline.yaml", help="Runtime configuration file path")
args = parser.parse_args()

def train(config):
    device = "cuda"
    if args.runners == "vanilla":
        trainer = BFeatVanillaTrainer(config, device)
        trainer.train()
    elif args.runners == "skipobj":
        trainer = BFeatSkipObjTrainer(config, device)
        trainer.train()

def experiment(config):
    ## TODO: Implement for various experiments e.g. ScanNet
    
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
    