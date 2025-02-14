from abc import ABC, abstractmethod
from typing import List
from dataset.dataloader import CustomDataLoader, collate_fn_bfeat, collate_fn_bfeat_mv
from dataset import build_dataset, build_dataset_multi_view
from utils.logger import build_meters, AverageMeter
from utils.eval_utils import *
import numpy as np
import torch
import torch.nn as nn
import wandb
import matplotlib.pyplot as plt
import clip
from datetime import datetime
import os

class BaseTrainer(ABC):
    def __init__(self, config, device, multi_view_ssl=False):
        super().__init__()
        self.config = config
        self.device = device
        self.t_config = config.train
        self.d_config = config.dataset
        self.opt_config = config.optimizer
        if not multi_view_ssl:
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
        else :
            self.t_dataset = build_dataset_multi_view(
                self.d_config, 
                split="train_scans", 
                device=device, 
                d_feats=self.config.model.dim_obj_feats
            )
            self.v_dataset = build_dataset_multi_view(
                self.d_config, 
                split="validation_scans", 
                device=device, 
                d_feats=self.config.model.dim_obj_feats
            )
            self.t_dataloader = CustomDataLoader(
                self.d_config, 
                self.t_dataset, 
                batch_size=self.t_config.batch_size,
                num_workers=self.t_config.workers,
                shuffle=True,
                drop_last=True,
                collate_fn=collate_fn_bfeat_mv
            )
            self.v_dataloader = CustomDataLoader(
                self.d_config, 
                self.v_dataset, 
                batch_size=self.t_config.batch_size,
                num_workers=self.t_config.workers,
                shuffle=True,
                drop_last=True,
                collate_fn=collate_fn_bfeat_mv
            )
        print("length of training data:", len(self.t_dataset))
        print("length of validation data:", len(self.v_dataset))
        
        self.num_obj_class = len(self.v_dataset.classNames)   
        self.num_rel_class = len(self.v_dataset.relationNames)
        self.obj_label_list = self.t_dataset.classNames
        self.rel_label_list = self.t_dataset.relationNames

        # Model Definition for encoder
        self.text_encoder, self.preprocessor = clip.load("ViT-B/32", device=device)
        self.text_encoder = self.text_encoder.eval()
        self.model: nn.Module | None = None
    
        # Wandb & Logger
        now = datetime.now()
        self.exp_name = f"{self.t_config.wandb_project}_{now.strftime('%Y-%m-%d_%H')}"
        self.__setup_checkpoint(self.exp_name)
        wandb.init(project="BetterFeat_3DSSG", name=self.exp_name)
        self.wandb_log = {}
        
        # Average & Max Meter
        self.meters = build_meters(self.t_config.meter)
    
    # Get text emebedding matrix for zero-shot classifier of closed-vocabulary
    @torch.no_grad()
    def build_text_classifier(self):
        obj_tokens = torch.cat([ clip.tokenize(f"A point cloud of a {obj}") for obj in self.obj_label_list ], dim=0).to(self.device)
        self.text_gt_matrix = self.text_encoder.encode_text(obj_tokens) # N_obj_cls X N_feat
        
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
    
    def add_meters(self, names):
        for m_name in names:
            self.meters[m_name] = AverageMeter(m_name)
    
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
        top_k_obj = evaluate_topk_object(obj_logits.detach(), gt_obj_cls, topk=11)
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
    
    def draw_graph(self,predicate_mean, topk_index):
        fig, ax1 = plt.subplots(figsize=(13,10))

        ax1.set_xlabel("Class")
        ax1.set_ylabel("Frequency", color="blue")
        ax1.plot([predicate_mean[i][0] for i in range(len(predicate_mean))], [predicate_mean[i][1] for i in range(len(predicate_mean))], marker="o", linestyle="-", color="blue", label="Frequency")
        ax1.tick_params(axis="y", labelcolor="blue")
        ax1.tick_params(axis="x", rotation=90)
        
        ax2 = ax1.twinx()
        ax2.set_ylabel("Acc", color="orange")
        ax2.bar([predicate_mean[i][0] for i in range(len(predicate_mean))], [predicate_mean[i][2][topk_index] for i in range(len(predicate_mean))], alpha=0.6, color="orange", label="Acc")
        ax2.tick_params(axis="y", labelcolor="orange")
        
        fig.tight_layout()
        # 그래프 제목
        tmp=[1,3,5]
        fig.suptitle(f"rel_acc_per_cls@{tmp[topk_index]}")
        plt.subplots_adjust(top=0.85)

        return fig
    
    def compute_predicate_acc_per_class(self, cls_matrix_list, topk_pred_list):
        cls_dict = {}
        for i in range(26):
            cls_dict[i] = []
        
        total_cnt=0
        for idx, j in enumerate(cls_matrix_list):
            if j[-1] != -1:
                cls_dict[j[-1]].append(topk_pred_list[idx])
                total_cnt+=1
        
        predicate_mean = []
        acc_freq=total_cnt
        for i in range(26):
            l = len(cls_dict[i])
            if l > 0:
                m_1 = (np.array(cls_dict[i]) <= 1).sum() / len(cls_dict[i])
                m_3 = (np.array(cls_dict[i]) <= 3).sum() / len(cls_dict[i])
                m_5 = (np.array(cls_dict[i]) <= 5).sum() / len(cls_dict[i])
                
                cls=self.rel_label_list[i]
                freq=len(cls_dict[i])
                predicate_mean.append([cls,freq,[m_1,m_3,m_5]])
        predicate_mean.sort(key=lambda x: x[1],reverse=True)
        for i in range(len(predicate_mean)):
            freq=predicate_mean[i][1]
            predicate_mean[i][1]=acc_freq/total_cnt
            acc_freq-=freq
        
        fig1=self.draw_graph(predicate_mean,0)
        fig2=self.draw_graph(predicate_mean,1)
        fig3=self.draw_graph(predicate_mean,2)
        self.wandb_log["rel_acc_per_cls"]=[wandb.Image(fig1),wandb.Image(fig2),wandb.Image(fig3)]
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)
        
    def save_checkpoint(self, exp_name, file_name):
        save_file = os.path.join(f'checkpoints/{exp_name}/models/', file_name)
        torch.save(self.model.state_dict(), save_file)