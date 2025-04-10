# from runners.trainer import BFeatVanillaTrainer
# from runners.trainer_skip_obj import BFeatSkipObjTrainer
# from runners.trainer_jjam import BFeatJjamTongTrainer
from runners import *
from experiments import *
from hydra import initialize, compose
import argparse

parser = argparse.ArgumentParser(description="Training BFeat Architecture")
parser.add_argument("--mode", type=str, default="train", choices=["train", "experiment"], help="Select mode for BFeat (train/evaluation)")
parser.add_argument("--runners", 
    type=str, default="vanilla", 
    choices=["vanilla", "geo_aux", "geo_mgat", "sgeo_mgat", "abl_jjamtong" ], # 
    help="Select running model"
) # "skipobj", "con_relonly", "triplet_con", "aux_con" , "triplet_gcn", "jjamtong_gcn", "sgg_point", "jjamtong_point"], "jjamtong", "full_scl", "direct_gnn", "finetune"
parser.add_argument("--config", type=str, default="baseline.yaml", help="Runtime configuration file path")
parser.add_argument("--exp_explain", type=str, default="default", help="Runtime configuration file path")
parser.add_argument("--ckp_path", type=str, help="Resume training from checkpoint")
args = parser.parse_args()

def train(config):
    device = "cuda"
    if args.runners == "vanilla":
        trainer = BFeatVanillaTrainer(config, device)
    elif args.runners == "geo_aux":
        trainer = BFeatGeoAuxTrainer(config, device)
    elif args.runners == "geo_mgat":
        trainer = BFeatGeoAuxMGATTrainer(config, device)
    elif args.runners == "sgeo_mgat":
        trainer = BFeatGeoSizeAuxMGATTrainer(config, device)
    elif args.runners == "abl_jjamtong":
        trainer = BFeatJjamTongTrainer(config, device)
    else:
        raise NotImplementedError
    trainer.train()

def experiment(config):
    ## TODO: Implement for various experiments e.g. ScanNet
    exp_runner = EntireExperimentRunners(args.runners, args.ckp_path, config, "cuda")
    exp_runner.validate()

if __name__ == "__main__":
    # mp.set_start_method("spawn", force=True)
    conf_name = args.config.split("/")[-1]
    conf_path = "/".join(args.config.split("/")[:-1])
    with initialize(config_path=conf_path):
        override_list = [] if not args.ckp_path else [f"+resume={args.ckp_path}"]
        override_list.append(f"+exp_desc={args.exp_explain}")
        config = compose(config_name=conf_name, overrides=override_list)
    
    runtime_mode = args.mode
    if runtime_mode == "train":
        train(config)
    elif runtime_mode == "experiment":
        experiment(config)
    