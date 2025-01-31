"""
Preprocessing is NOTHING!!!!!!!!!!!!!!!!!
This code is not used in anywhere for now
"""

"""
Make preprocessor for later 
 - Train code for seraph/lab GPU server w. not enough RAM
"""

from config.define import *
from PIL.Image import Image
import numpy as np
import trimesh
import json
import hydra

import multiprocessing
from tqdm.contrib.concurrent import process_map
import argparse

lock = multiprocessing.Lock()




if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--parallel', action='store_true', help='parallel', required=False)
    argparser.add_argument('--num_worker', "-n", type=int, default=8, help='parallel', required=False)
    args = argparser.parse_args()
    
    
    