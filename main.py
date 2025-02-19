# from runners.trainer import BFeatVanillaTrainer
# from runners.trainer_skip_obj import BFeatSkipObjTrainer
# from runners.trainer_jjam import BFeatJjamTongTrainer
from runners import *
from hydra import initialize, compose
import argparse

parser = argparse.ArgumentParser(description="Training BFeat Architecture")
parser.add_argument("--mode", type=str, default="train", choices=["train", "experiment"], help="Select mode for BFeat (train/evaluation)")
parser.add_argument("--runners", 
    type=str, default="vanilla", 
    choices=["vanilla", "skipobj", "jjamtong", "con_jjamtong", "con_relonly", "direct_gnn", "triplet_con"], 
    help="Select running model"
)
parser.add_argument("--config", type=str, default="baseline.yaml", help="Runtime configuration file path")
parser.add_argument("--exp_explain", type=str, default="default", help="Runtime configuration file path")
parser.add_argument("--resume", type=str, help="Resume training from checkpoint")
args = parser.parse_args()

def train(config):
    device = "cuda"
    if args.runners == "vanilla":
        trainer = BFeatVanillaTrainer(config, device)
    elif args.runners == "skipobj":
        trainer = BFeatSkipObjTrainer(config, device)
    elif args.runners == "jjamtong":
        trainer = BFeatJjamTongTrainer(config, device)
    elif args.runners == "con_jjamtong":
        trainer = BFeatRelSSLTrainer(config, device)
    elif args.runners == "con_relonly":
        trainer = BFeatRelOnlyContrasTrainer(config, device)
    elif args.runners == "direct_gnn":
        trainer = BFeatDirectGNNTrainer(config, device)
    elif args.runners == "triplet_con":
        trainer = BFeatTripletContrastiveTrainer(config, device)
    else:
        raise NotImplementedError
    trainer.train()

def experiment(config):
    ## TODO: Implement for various experiments e.g. ScanNet
    
    pass

if __name__ == "__main__":
    # mp.set_start_method("spawn", force=True)
    with initialize(config_path="./config"):
        override_list = [] if not args.resume else [f"+resume={args.resume}"]
        override_list.append(f"+exp_desc={args.exp_explain}")
        config = compose(config_name=args.config, overrides=override_list)
    
    runtime_mode = args.mode
    if runtime_mode == "train":
        train(config)
    elif runtime_mode == "experiment":
        experiment(config)
    