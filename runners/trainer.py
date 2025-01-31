from typing import List
from dataset.dataloader import CustomDataLoader, collate_fn_bfeat
from dataset import build_dataset
from utils.eval_utils import *
from utils.logger import Progbar
from utils.contrastive_utils import ContrastiveSingleLabelSampler
from model.frontend.relextractor import *
from model.model import BFeatVanillaNet
from model.loss import TripletLoss, ContrastiveLoss
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import clip
import wandb
from datetime import datetime
import os

class BFeatVanillaTrainer():
    def __init__(self, config, device):
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
        
        # Contrastive positive/negative pair sampler  
        self.contrastive_sampler = ContrastiveSingleLabelSampler(config, device)
        
        # Model Definitions
        self.model = BFeatVanillaNet(self.config, self.num_obj_class, self.num_rel_class, device)
        self.text_encoder, self.text_preprocessor = clip.load("ViT-B/32", device=device)
        # Optimizer & Scheduler
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.opt_config.learning_rate, 
            weight_decay=self.opt_config.weight_decay
        )
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=self.t_config.epoch, eta_min=0, last_epoch=-1)
        # Loss function 
        self.c_criterion = TripletLoss(margin=0.3)
        
        # Wandb & Logger
        now = datetime.now()
        self.exp_name = f"{self.t_config.wandb_project}_{now.strftime('%Y-%m-%d_%H')}"
        self.__setup_checkpoint(self.exp_name)
        wandb.init(project="BetterFeat_3DSSG", name=self.exp_name)
    
    def __setup_checkpoint(self, exp_name):
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
        if not os.path.exists('checkpoints/' + exp_name):
            os.makedirs('checkpoints/' + exp_name)
        if not os.path.exists('checkpoints/' + exp_name + '/' + 'models'):
            os.makedirs('checkpoints/' + exp_name + '/' + 'models')
    
    def __to_device(self, *tensors) -> List[torch.Tensor]:
        c_tensor = [ t.to(self.device) for t in tensors ]
        return c_tensor
    
    def __dynamic_rel_weight(self, gt_rel_cls, ignore_none_rel=True):
        batch_mean = torch.sum(gt_rel_cls, dim=(0))
        zeros = (gt_rel_cls.sum(-1) ==0).sum().unsqueeze(0)
        batch_mean = torch.cat([zeros,batch_mean],dim=0)
        weight = torch.abs(1.0 / (torch.log(batch_mean+1)+1)) # +1 to prevent 1 /log(1) = inf                
        if self.t_config.none_ratio == 0:
            weight[0] = 0
            weight *= 1e-2 # reduce the weight from ScanNet
            # print('set weight of none to 0')
        else:
            weight[0] *= self.t_config.none_ratio

        weight[torch.where(weight==0)] = weight[0].clone() if not ignore_none_rel else 0
        weight = weight[1:]                
        return weight
    
    def train(self):
        
        self.model = self.model.train()
        n_iters = len(self.t_dataloader)
        val_metric = -987654321
        
        # Training Loop
        for e in range(self.t_config.epoch):
            self.wandb_log = {}
            progbar = Progbar(n_iters, width=20, stateful_metrics=['Misc/it'])
            self.model = self.model.train()
            loader = iter(self.t_dataloader)
            
            for idx, (
                obj_pts, 
                rel_pts, 
                descriptor,
                gt_rel_label,
                gt_obj_label,
                edge_indices,
                batch_ids
            ) in enumerate(loader):

                (
                    obj_pts, 
                    rel_pts, 
                    descriptor,
                    gt_rel_label,
                    gt_obj_label,
                    edge_indices,
                    batch_ids
                ) = self.__to_device(obj_pts, rel_pts, descriptor, gt_rel_label, gt_obj_label, edge_indices, batch_ids)
                
                self.optimizer.zero_grad()
                obj_pts = obj_pts.transpose(2, 1).contiguous()
                rel_pts = rel_pts.transpose(2, 1).contiguous()
                edge_feats, obj_pred, rel_pred = self.model(obj_pts, rel_pts, edge_indices.t().contiguous(), descriptor, batch_ids)
                rel_weight = self.__dynamic_rel_weight(gt_rel_label)
                c_obj_loss = F.cross_entropy(obj_pred, gt_obj_label)
                c_rel_loss = F.binary_cross_entropy(rel_pred, gt_rel_label, weight=rel_weight)
                
                pos_pair, neg_pair = self.contrastive_sampler.sample(gt_obj_label, gt_rel_label, edge_indices)
                contrastive_loss = self.c_criterion(edge_feats, pos_pair, neg_pair)
                
                # TODO: determine coefficient for each loss
                t_loss = c_obj_loss + c_rel_loss + contrastive_loss
                t_loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                self.wandb_log['Train/Total_Loss'] = t_loss
                self.wandb_log['Train/Obj_Cls_Loss'] = c_obj_loss
                self.wandb_log['Train/Rel_Cls_Loss'] = c_rel_loss
                logs = self.evaluate_train(obj_pred, gt_obj_label, rel_pred, gt_rel_label, edge_indices)
                t_log = [
                    ("train/rel_loss", c_rel_loss.detach().item()),
                    ("train/obj_loss", c_obj_loss.detach().item()),
                    ("train/contrastive_loss", contrastive_loss.detach().item()),
                    ("train/total_loss", t_loss.detach().item()),
                ] + logs
                t_log += [
                    ("Misc/epo", int(e)),
                    ("Misc/it", int(idx)),
                    ("lr", self.lr_scheduler.get_last_lr()[0])
                ]
                progbar.add(1, values=logs)
            
            if e % self.t_config.evaluation_interval == 0:
                mRecall_50 = self.evaluate_validation()
                if mRecall_50 >= val_metric:
                    self.save_checkpoint(self.exp_name, "best_model.pth")
                self.save_checkpoint(self.exp_name, 'ckpt_epoch_{epoch}.pth'.format(epoch=e))
            wandb.log(self.wandb_log)
    
    def evaluate_train(self, obj_logits, gt_obj_cls, rel_logits, gt_rel_cls, edge_indices):
        top_k_obj = evaluate_topk_object(obj_logits.detach(), gt_obj_cls, topk=11)
        gt_edges = get_gt(gt_obj_cls, gt_rel_cls, edge_indices, self.d_config.multi_rel)
        top_k_rel = evaluate_topk_predicate(rel_logits.detach(), gt_edges, self.d_config.multi_rel, topk=6)
        obj_topk_list = [100 * (top_k_obj <= i).sum() / len(top_k_obj) for i in [1, 5, 10]]
        rel_topk_list = [100 * (top_k_rel <= i).sum() / len(top_k_rel) for i in [1, 3, 5]]
        self.wandb_log["Train/Obj_R1"] = obj_topk_list[0]
        self.wandb_log["Train/Obj_R3"] = obj_topk_list[1]
        self.wandb_log["Train/Obj_R5"] = obj_topk_list[2]
        self.wandb_log["Train/Pred_R1"] = rel_topk_list[0]
        self.wandb_log["Train/Pred_R3"] = rel_topk_list[1]
        self.wandb_log["Train/Pred_R5"] = rel_topk_list[2]
        log = [
            ("train/Obj_R1", obj_topk_list[0]),
            ("train/Obj_R5", obj_topk_list[1]),
            ("train/Obj_R10", obj_topk_list[2]),
            ("train/Pred_R1", rel_topk_list[0]),
            ("train/Pred_R3", rel_topk_list[1]),
            ("train/Pred_R5", rel_topk_list[2]),
        ]
        return log
    
    def evaluate_validation(self):
        n_iters = len(self.v_dataloader)
        progbar = Progbar(n_iters, width=20, stateful_metrics=['Misc/it'])
        loader = iter(self.v_dataloader)
        
        topk_obj_list, topk_rel_list, topk_triplet_list, cls_matrix_list = np.array([]), np.array([]), np.array([]), []
        sub_scores_list, obj_scores_list, rel_scores_list = [], [], []
        logs = []
        
        with torch.no_grad():
            self.model = self.model.eval()
            for i, (
                obj_pts, 
                rel_pts, 
                descriptor,
                gt_rel_label,
                gt_obj_label,
                edge_indices,
                batch_ids
            ) in enumerate(loader):
                (
                    obj_pts, 
                    rel_pts, 
                    descriptor,
                    gt_rel_label,
                    gt_obj_label,
                    edge_indices,
                    batch_ids
                ) = self.__to_device(obj_pts, rel_pts, descriptor, gt_rel_label, gt_obj_label, edge_indices, batch_ids)
                
                self.optimizer.zero_grad()
                obj_pts = obj_pts.transpose(2, 1).contiguous()
                rel_pts = rel_pts.transpose(2, 1).contiguous()
                _, obj_pred, rel_pred = self.model(obj_pts, rel_pts, edge_indices.t().contiguous(), descriptor, batch_ids)
                top_k_obj = evaluate_topk_object(obj_pred.detach().cpu(), gt_obj_label, topk=11)
                gt_edges = get_gt(gt_obj_label, gt_rel_label, edge_indices, self.d_config.multi_rel)
                top_k_rel = evaluate_topk_predicate(rel_pred.detach().cpu(), gt_edges, self.d_config.multi_rel, topk=6)
                top_k_triplet, cls_matrix, sub_scores, obj_scores, rel_scores = \
                    evaluate_triplet_topk(
                        obj_pred.detach().cpu(), rel_pred.detach().cpu(), 
                        gt_edges, edge_indices, self.d_config.multi_rel, 
                        topk=101, use_clip=True, obj_topk=top_k_obj
                    )
                
                topk_obj_list = np.concatenate((topk_obj_list, top_k_obj))
                topk_rel_list = np.concatenate((topk_rel_list, top_k_rel))
                topk_triplet_list = np.concatenate((topk_triplet_list, top_k_triplet))
                if cls_matrix is not None:
                    cls_matrix_list.extend(cls_matrix)
                    sub_scores_list.extend(sub_scores)
                    obj_scores_list.extend(obj_scores)
                    rel_scores_list.extend(rel_scores)
                
                logs += [
                    ("Acc@1/obj_cls_acc", (topk_obj_list <= 1).sum() * 100 / len(topk_obj_list)),
                    ("Acc@5/obj_cls_acc", (topk_obj_list <= 5).sum() * 100 / len(topk_obj_list)),
                    ("Acc@10/obj_cls_acc", (topk_obj_list <= 10).sum() * 100 / len(topk_obj_list)),
                    ("Acc@1/rel_cls_acc", (topk_rel_list <= 1).sum() * 100 / len(topk_rel_list)),
                    ("Acc@3/rel_cls_acc", (topk_rel_list <= 3).sum() * 100 / len(topk_rel_list)),
                    ("Acc@5/rel_cls_acc", (topk_rel_list <= 5).sum() * 100 / len(topk_rel_list)),
                    ("Acc@50/triplet_acc", (topk_triplet_list <= 50).sum() * 100 / len(topk_triplet_list)),
                    ("Acc@100/triplet_acc", (topk_triplet_list <= 100).sum() * 100 / len(topk_triplet_list))
                ]

                progbar.add(1, values=logs)
            
            cls_matrix_list = np.stack(cls_matrix_list)
            sub_scores_list = np.stack(sub_scores_list)
            obj_scores_list = np.stack(obj_scores_list)
            rel_scores_list = np.stack(rel_scores_list)
            mean_recall = get_mean_recall(topk_triplet_list, cls_matrix_list)
            
            obj_acc_1 = (topk_obj_list <= 1).sum() * 100 / len(topk_obj_list)
            obj_acc_5 = (topk_obj_list <= 5).sum() * 100 / len(topk_obj_list)
            obj_acc_10 = (topk_obj_list <= 10).sum() * 100 / len(topk_obj_list)
            rel_acc_1 = (topk_rel_list <= 1).sum() * 100 / len(topk_rel_list)
            rel_acc_3 = (topk_rel_list <= 3).sum() * 100 / len(topk_rel_list)
            rel_acc_5 = (topk_rel_list <= 5).sum() * 100 / len(topk_rel_list)
            triplet_acc_50 = (topk_triplet_list <= 50).sum() * 100 / len(topk_triplet_list)
            triplet_acc_100 = (topk_triplet_list <= 100).sum() * 100 / len(topk_triplet_list)
            
            rel_acc_mean_1, rel_acc_mean_3, rel_acc_mean_5 = self.__compute_mean_predicate(cls_matrix_list, topk_rel_list)
            logs += [
                ("Acc@1/obj_cls_acc", obj_acc_1),
                ("Acc@5/obj_cls_acc", obj_acc_5),
                ("Acc@10/obj_cls_acc", obj_acc_10),
                ("Acc@1/rel_cls_acc", rel_acc_1),
                ("Acc@1/rel_cls_acc_mean", rel_acc_mean_1),
                ("Acc@3/rel_cls_acc", rel_acc_3),
                ("Acc@3/rel_cls_acc_mean", rel_acc_mean_3),
                ("Acc@5/rel_cls_acc", rel_acc_5),
                ("Acc@5/rel_cls_acc_mean", rel_acc_mean_5),
                ("Acc@50/triplet_acc", triplet_acc_50),
                ("Acc@100/triplet_acc", triplet_acc_100),
                ("mean_recall@50", mean_recall[0]),
                ("mean_recall@100", mean_recall[1])
            ]
            self.wandb_log["Validation/Acc@1/obj_cls"] = obj_acc_1
            self.wandb_log["Validation/Acc@5/obj_cls"] = obj_acc_5
            self.wandb_log["Validation/Acc@10/obj_cls"] = obj_acc_10
            self.wandb_log["Validation/Acc@1/rel_cls_acc"] = rel_acc_1
            self.wandb_log["Validation/Acc@1/rel_cls_acc_mean"] = rel_acc_mean_1
            self.wandb_log["Validation/Acc@3/rel_cls_acc"] = rel_acc_3
            self.wandb_log["Validation/Acc@3/rel_cls_acc_mean"] = rel_acc_mean_3
            self.wandb_log["Validation/Acc@5/rel_cls_acc"] = rel_acc_5
            self.wandb_log["Validation/Acc@5/rel_cls_acc_mean"] = rel_acc_mean_5
            self.wandb_log["Validation/Acc@50/triplet_acc"] = triplet_acc_50
            self.wandb_log["Validation/Acc@100/triplet_acc"] = triplet_acc_100
            self.wandb_log["Validation/mRecall@50"] = mean_recall[0]
            self.wandb_log["Validation/mRecall@100"] = mean_recall[1]        
        return mean_recall[0]
            
            
    def __compute_mean_predicate(self, cls_matrix_list, topk_pred_list):
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
    

# print("Obj pts Shape:", obj_pts.shape)
# print("Rel pts Shape:", rel_pts.shape)
# print("Obj desc. Shape:", descriptor.shape)
# print("Rel label Shape:", gt_rel_label.shape)
# print("Obj label Shape:", gt_obj_label.shape)
# print("Edge index Shape:", edge_indices.shape)
# print("Batch idx Shape:", batch_ids.shape)