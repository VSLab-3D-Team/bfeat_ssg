#!/bin/bash

python3 -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
python3 -m pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.12.1+cu113.html
python3 -m pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.12.1+cu113.html
python3 -m pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.12.1+cu113.html
python3 -m pip install torch-geometric==2.2.0
python3 -m pip install git+https://github.com/openai/CLIP.git
python3 -m pip install hydra
python3 -m pip install hydra-core --upgrade --pre