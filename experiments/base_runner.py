from abc import abstractmethod, ABC
from typing import List
from dataset.dataloader import CustomDataLoader, collate_fn_bfeat
from dataset import build_dataset
from utils.logger import build_meters
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class BaseExperimentRunner(ABC):
    def __init__(self, config, device):
        super().__init__()
        
        self.config = config
        self.device = device
        self.t_config = config.train
        self.d_config = config.dataset
        self.m_config = config.model
        self.opt_config = config.optimizer
        self.v_dataset = build_dataset(self.d_config, split="validation_scans", device=device)
        self.v_dataloader = CustomDataLoader(
            self.d_config, 
            self.v_dataset, 
            batch_size=self.t_config.batch_size,
            num_workers=self.t_config.workers,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn_bfeat
        )
        print("length of validation data:", len(self.v_dataset))
        
        self.num_obj_class = len(self.v_dataset.classNames)   
        self.num_rel_class = len(self.v_dataset.relationNames)
        self.obj_label_list = self.v_dataset.classNames
        self.rel_label_list = self.v_dataset.relationNames

        self.model: nn.Module | None = None

        # Average & Max Meter
        self.meters = build_meters(self.t_config.meter)
    
    def reset_meters(self):
        for k in list(self.meters.keys()):
            self.meters[k].reset()
    
    def to_device(self, *tensors) -> List[torch.Tensor]:
        c_tensor = [ t.to(self.device) for t in tensors ]
        return c_tensor
    
    @abstractmethod
    def validate(self):
        pass
    
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
            predicate_mean[i][1]
        
        fig1=self.draw_graph(predicate_mean,0)
        fig2=self.draw_graph(predicate_mean,1)
        fig3=self.draw_graph(predicate_mean,2)
        
        # self.wandb_log["rel_acc_per_cls"]=[wandb.Image(fig1),wandb.Image(fig2),wandb.Image(fig3)]
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)