from abc import ABC, abstractmethod
from typing import List
from dataset.dataloader import CustomDataLoader, collate_fn_bfeat
from dataset import build_dataset
from utils.logger import build_meters
from utils.eval_utils import *
import numpy as np
import torch
import torch.nn as nn
import wandb
from datetime import datetime
import os

class BaseTrainer(ABC):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        self.t_config = config.train
        self.d_config = config.dataset
        self.opt_config = config.optimizer
        self.t_dataset = build_dataset(self.d_config, split="train_scans", device=device)
        self.v_dataset = build_dataset(self.d_config, split="validation_scans", device=device)
        self.t_dataloader = CustomDataLoader(
            self.d_config, 
            self.t_dataset, 
            batch_size=self.t_config.batch_size,
            num_workers=self.t_config.workers,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn_bfeat
        )
        self.v_dataloader = CustomDataLoader(
            self.d_config, 
            self.v_dataset, 
            batch_size=self.t_config.batch_size,
            num_workers=self.t_config.workers,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn_bfeat
        )
        print("length of training data:", len(self.t_dataset))
        print("length of validation data:", len(self.v_dataset))
        
        self.num_obj_class = len(self.v_dataset.classNames)   
        self.num_rel_class = len(self.v_dataset.relationNames)
        self.obj_label_list = self.t_dataset.classNames
        self.rel_label_list = self.t_dataset.relationNames
    
        self.model: nn.Module | None = None
    
        # Wandb & Logger
        now = datetime.now()
        self.exp_name = f"{self.t_config.wandb_project}_{now.strftime('%Y-%m-%d_%H')}"
        self.__setup_checkpoint(self.exp_name)
        wandb.init(project="BetterFeat_3DSSG", name=self.exp_name)
        self.wandb_log = {}
        
        # Average & Max Meter
        self.meters = build_meters(self.t_config.meter)
    
    def reset_meters(self):
        for k in list(self.meters.keys()):
            self.meters[k].reset()
    
    def __setup_checkpoint(self, exp_name):
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
        if not os.path.exists('checkpoints/' + exp_name):
            os.makedirs('checkpoints/' + exp_name)
        if not os.path.exists('checkpoints/' + exp_name + '/' + 'models'):
            os.makedirs('checkpoints/' + exp_name + '/' + 'models')
    
    def to_device(self, *tensors) -> List[torch.Tensor]:
        c_tensor = [ t.to(self.device) for t in tensors ]
        return c_tensor
    
    @abstractmethod
    def train(self):
        """
        Implement training loop
        """
        pass
    
    @abstractmethod
    def evaluate_validation(self):
        """
        Implement evaluation loop
        """
        pass
    
    def write_wandb_log(self):
        for k in list(self.meters.keys()):
            if self.t_config.meter == "average":
                self.wandb_log[k] = self.meters[k].avg
            elif self.t_config.meter == "max":
                self.wandb_log[k] = self.meters[k].max_val
    
    def evaluate_train(self, obj_logits, gt_obj_cls, rel_logits, gt_rel_cls, edge_indices):
        top_k_obj = evaluate_topk_object(obj_logits, gt_obj_cls, topk=11)
        gt_edges = get_gt(gt_obj_cls, gt_rel_cls, edge_indices, self.d_config.multi_rel)
        top_k_rel = evaluate_topk_predicate(rel_logits.detach(), gt_edges, self.d_config.multi_rel, topk=6)
        obj_topk_list = [100 * (top_k_obj <= i).sum() / len(top_k_obj) for i in [1, 5, 10]]
        rel_topk_list = [100 * (top_k_rel <= i).sum() / len(top_k_rel) for i in [1, 3, 5]]
        self.meters["Train/Obj_R1"].update(obj_topk_list[0])
        self.meters["Train/Obj_R5"].update(obj_topk_list[1])
        self.meters["Train/Obj_R10"].update(obj_topk_list[2])
        self.meters["Train/Pred_R1"].update(rel_topk_list[0])
        self.meters["Train/Pred_R3"].update(rel_topk_list[1])
        self.meters["Train/Pred_R5"].update(rel_topk_list[2])
        log = [
            ("train/Obj_R1", obj_topk_list[0]),
            ("train/Obj_R5", obj_topk_list[1]),
            ("train/Obj_R10", obj_topk_list[2]),
            ("train/Pred_R1", rel_topk_list[0]),
            ("train/Pred_R3", rel_topk_list[1]),
            ("train/Pred_R5", rel_topk_list[2]),
        ]
        return log
    
    def compute_mean_predicate(self, cls_matrix_list, topk_pred_list):
        cls_dict = {}
        for i in range(26):
            cls_dict[i] = []
        
        for idx, j in enumerate(cls_matrix_list):
            if j[-1] != -1:
                cls_dict[j[-1]].append(topk_pred_list[idx])
        
        predicate_mean_1, predicate_mean_3, predicate_mean_5 = [], [], []
        for i in range(26):
            l = len(cls_dict[i])
            if l > 0:
                m_1 = (np.array(cls_dict[i]) <= 1).sum() / len(cls_dict[i])
                m_3 = (np.array(cls_dict[i]) <= 3).sum() / len(cls_dict[i])
                m_5 = (np.array(cls_dict[i]) <= 5).sum() / len(cls_dict[i])
                predicate_mean_1.append(m_1)
                predicate_mean_3.append(m_3)
                predicate_mean_5.append(m_5) 
           
        predicate_mean_1 = np.mean(predicate_mean_1)
        predicate_mean_3 = np.mean(predicate_mean_3)
        predicate_mean_5 = np.mean(predicate_mean_5)

        return predicate_mean_1 * 100, predicate_mean_3 * 100, predicate_mean_5 * 100
        
    def save_checkpoint(self, exp_name, file_name):
        save_file = os.path.join(f'checkpoints/{exp_name}/models/', file_name)
        torch.save(self.model.state_dict(), save_file)