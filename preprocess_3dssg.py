"""
Preprocessing is NOTHING!!!!!!!!!!!!!!!!!
This code is not used in anywhere for now
"""

"""
Make preprocessor for later 
 - Train code for seraph/lab GPU server w. not enough RAM
"""

from dataset.preprocessor import Preprocessor3RScan
# import multiprocessing
# from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
from hydra import initialize, compose
import argparse

# lock = multiprocessing.Lock()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', "-c", type=str, default="baseline.yaml")
    args = argparser.parse_args()
    
    with initialize(config_path="./config"):
        config = compose(config_name=args.config)
    
    t_preprocessor = Preprocessor3RScan(config, "train_scans", "cuda")
    v_preprocessor = Preprocessor3RScan(config, "validation_scans", "cuda")
    
    print("Processing train dataset...")
    for i in tqdm(range(len(t_preprocessor))):
        t_preprocessor.process(i)
    
    print("Processing validation dataset...")
    for i in tqdm(range(len(v_preprocessor))):
        v_preprocessor.process(i)
    