from utils.eval_utils import *
from utils.logger import Progbar
from runners.base_trainer import BaseTrainer
from utils.model_utils import TFIDFMaskLayer, TFIDFTripletWeight
from model.frontend.relextractor import *
from model.models.model_geo_aux_mgat import BFeatGeoAuxMGATNet
from model.loss import MultiLabelInfoNCELoss, WeightedFocalLoss
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, CyclicLR
import wandb

class BFeatGeoAuxMGATTrainer(BaseTrainer):
    def __init__(self, config, device):
        super().__init__(config, device, geo_aux=True)
        
        # Model Definitions
        self.m_config = config.model
        self.model = BFeatGeoAuxMGATNet(
            self.config, 
            self.num_obj_class, 
            self.num_rel_class, 
            device
        ).to(device)
        
        # Optimizer & Scheduler
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.opt_config.learning_rate, 
            weight_decay=self.opt_config.weight_decay
        )
        if self.t_config.scheduler == "cosine":
            self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=self.t_config.epoch, eta_min=0, last_epoch=-1)
        elif self.t_config.scheduler == 'cyclic':
            self.lr_scheduler = CyclicLR(
                self.optimizer, base_lr=self.opt_config.learning_rate / 10, 
                step_size_up=self.t_config.epoch, max_lr=self.opt_config.learning_rate * 5, 
                gamma=0.8, mode='exp_range', cycle_momentum=False
            )
        else:
            raise NotImplementedError
        # Loss function 
        self.f_criterion = WeightedFocalLoss()
        self.c_criterion = MultiLabelInfoNCELoss(device=self.device, temperature=self.t_config.loss_temperature).to(self.device)
        self.tfidf = TFIDFMaskLayer(self.num_obj_class, self.device)
        self.w_edge = TFIDFTripletWeight(self.num_obj_class, self.num_rel_class, self.device)
        
        self.add_meters([
            "Train/Geo_Aux_Loss",
            "Train/Edge_CLIP_Aux_Loss",
        ])
        self.del_meters([
            "Train/Contrastive_Loss"
        ])
        
        # Resume training if ckp path is provided.
        if 'resume' in self.config:
            self.resume_from_checkpoint(self.config.resume)
    
    def cosine_loss(self, A, B, t=1):
        return torch.clamp(t - F.cosine_similarity(A, B, dim=-1), min=0).mean()
    
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
    
    def __dynamic_obj_weight(self, gt_obj_cls, alpha=0.5):
        num_classes = len(self.obj_label_list)
        class_counts = torch.bincount(gt_obj_cls, minlength=num_classes).float()
        class_counts = class_counts + 1e-6  
        weights = 1.0 / (class_counts ** alpha)
        weights = weights / weights.sum()
        return weights
    
    def train(self):
        
        self.model = self.model.train()
        n_iters = len(self.t_dataloader)
        val_metric = -987654321
        
        # Training Loop
        for e in range(self.t_config.epoch):
            self.wandb_log = {}
            progbar = Progbar(n_iters, width=40, stateful_metrics=['Misc/epo', 'Misc/it'])
            self.model = self.model.train()
            loader = iter(self.t_dataloader)
            
            for idx, (
                obj_pts, 
                rel_pts, 
                descriptor,
                edge_2d_feats,
                gt_rel_label,
                gt_obj_label,
                edge_indices,
                edge_feat_mask,
                batch_ids
            ) in enumerate(loader):

                (
                    obj_pts, 
                    rel_pts, 
                    descriptor,
                    edge_2d_feats,
                    gt_rel_label,
                    gt_obj_label,
                    edge_indices,
                    edge_feat_mask,
                    batch_ids
                ) = self.to_device(
                    obj_pts, rel_pts, descriptor, edge_2d_feats, 
                    gt_rel_label, gt_obj_label, edge_indices, 
                    edge_feat_mask, batch_ids
                )
                
                self.optimizer.zero_grad()
                obj_pts = obj_pts.transpose(2, 1).contiguous()
                rel_pts = rel_pts.transpose(2, 1).contiguous()
                
                # TF-IDF Attention Mask Generation
                attn_tfidf_weight = None # self.w_edge.get_mask(gt_obj_label, gt_rel_label, edge_indices, batch_ids)
                
                _, obj_pred, rel_pred, pred_edge_clip, pred_geo_desc, edge_desc = \
                    self.model(obj_pts, rel_pts, edge_indices.t().contiguous(), descriptor, edge_feat_mask, batch_ids, attn_tfidf_weight)
                rel_weight = self.__dynamic_rel_weight(gt_rel_label)
                obj_weight = self.__dynamic_obj_weight(gt_obj_label).to(self.device)
                c_obj_loss = F.cross_entropy(obj_pred, gt_obj_label, weight=obj_weight)
                c_rel_loss = F.binary_cross_entropy(rel_pred, gt_rel_label, weight=rel_weight)
                
                # pos_pair, neg_pair, rel_indices = self.contrastive_sampler.sample(gt_obj_label, gt_rel_label, edge_indices)
                # contrastive_loss = self.c_criterion(edge_feats, pos_pair, neg_pair, rel_indices)
                
                geo_aux_loss = F.l1_loss(pred_geo_desc, edge_desc)
                edge_clip_aux_loss = self.cosine_loss(pred_edge_clip, edge_2d_feats)
                
                # TODO: determine coefficient for each loss
                lambda_o = self.t_config.lambda_obj # 0.1
                lambda_r = self.t_config.lambda_rel
                lambda_g = self.t_config.lambda_geo
                lambda_v = self.t_config.lambda_view
                    
                # Geo Aux: 0.3 or 1.0
                t_loss = lambda_o * c_obj_loss \
                    + lambda_r * c_rel_loss \
                    + lambda_g * geo_aux_loss \
                    + lambda_v * edge_clip_aux_loss
                t_loss.backward()
                self.optimizer.step()
                self.meters['Train/Total_Loss'].update(t_loss.detach().item())
                self.meters['Train/Obj_Cls_Loss'].update(c_obj_loss.detach().item())
                self.meters['Train/Rel_Cls_Loss'].update(c_rel_loss.detach().item()) 
                self.meters['Train/Geo_Aux_Loss'].update(geo_aux_loss.detach().item()) 
                self.meters['Train/Edge_CLIP_Aux_Loss'].update(edge_clip_aux_loss.detach().item()) 
                t_log = [
                    ("train/rel_loss", c_rel_loss.detach().item()),
                    ("train/obj_loss", c_obj_loss.detach().item()),
                    ("train/total_loss", t_loss.detach().item()),
                    ("Misc/epo", int(e)),
                    ("Misc/it", int(idx)),
                    ("lr", self.lr_scheduler.get_last_lr()[0])
                ]
                if e % self.t_config.log_interval == 0:
                    logs = self.evaluate_train(obj_pred, gt_obj_label, rel_pred, gt_rel_label, edge_indices)
                    t_log += logs
                progbar.add(1, values=t_log)
                
            self.lr_scheduler.step()
            if e % self.t_config.evaluation_interval == 0:
                mRecall_50 = self.evaluate_validation()
                if mRecall_50 >= val_metric:
                    self.save_checkpoint(self.exp_name, "best_model.pth")
                    val_metric = mRecall_50
                if e % self.t_config.save_interval == 0:
                    self.save_checkpoint(self.exp_name, 'ckpt_epoch_{epoch}.pth'.format(epoch=e))
            
            self.wandb_log["Train/learning_rate"] = self.lr_scheduler.get_last_lr()[0]
            self.write_wandb_log()
            wandb.log(self.wandb_log)
    
    def evaluate_validation(self):
        n_iters = len(self.v_dataloader)
        progbar = Progbar(n_iters, width=40, stateful_metrics=['Misc/it'])
        loader = iter(self.v_dataloader)
        
        topk_obj_list, topk_rel_list, topk_triplet_list, gt_obj_list, cls_matrix_list = np.array([]), np.array([]), np.array([]), np.array([]), []
        sub_scores_list, obj_scores_list, rel_scores_list = [], [], []
        sgcls_recall_list, predcls_recall_list  = [],[]
        logs = []
        
        with torch.no_grad():
            self.model = self.model.eval()
            for idx, (
                obj_pts, 
                rel_pts, 
                descriptor,
                edge_2d_feats,
                gt_rel_label,
                gt_obj_label,
                edge_indices,
                edge_feat_mask,
                batch_ids
            ) in enumerate(loader):

                (
                    obj_pts, 
                    rel_pts, 
                    descriptor,
                    edge_2d_feats,
                    gt_rel_label,
                    gt_obj_label,
                    edge_indices,
                    edge_feat_mask,
                    batch_ids
                ) = self.to_device(
                    obj_pts, rel_pts, descriptor, edge_2d_feats, 
                    gt_rel_label, gt_obj_label, edge_indices, 
                    edge_feat_mask, batch_ids
                )
                
                obj_pts = obj_pts.transpose(2, 1).contiguous()
                rel_pts = rel_pts.transpose(2, 1).contiguous()
                # tfidf_class = self.tfidf.get_mask(gt_obj_label, batch_ids)
                # attn_tfidf_weight = tfidf_class[gt_obj_label.long()] # N_obj X 1 
                
                _, obj_pred, rel_pred, _, _, _ = self.model(obj_pts, rel_pts, edge_indices.t().contiguous(), descriptor, edge_feat_mask, batch_ids) # attn_tfidf_weight
                top_k_obj = evaluate_topk_object(obj_pred.detach(), gt_obj_label, topk=11)
                gt_edges = get_gt(gt_obj_label, gt_rel_label, edge_indices, self.d_config.multi_rel)
                top_k_rel = evaluate_topk_predicate(rel_pred.detach(), gt_edges, self.d_config.multi_rel, topk=6)
                top_k_triplet, cls_matrix, sub_scores, obj_scores, rel_scores = \
                    evaluate_triplet_topk(
                        obj_pred.detach(), rel_pred.detach(), 
                        gt_edges, edge_indices, self.d_config.multi_rel, 
                        topk=101, use_clip=True, obj_topk=top_k_obj
                    )
                
                sgcls_recall = evaluate_triplet_recallk(obj_pred.detach(), rel_pred.detach(), gt_edges, edge_indices, self.d_config.multi_rel, [20,50,100], 100, use_clip=True, evaluate='triplet')
                predcls_recall = evaluate_triplet_recallk(obj_pred.detach(), rel_pred.detach(), gt_edges, edge_indices, self.d_config.multi_rel, [20,50,100], 100, use_clip=True, evaluate='rels')
                
                sgcls_recall_list.append(sgcls_recall)
                predcls_recall_list.append(predcls_recall)
                
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
            
            sgcls_recall_list=np.array(sgcls_recall_list) # N_graph X [correct@20,correct@50,correct@100]
            predcls_recall_list=np.array(predcls_recall_list) # N_graph X [correct@20,correct@50,correct@100]
            
            sgcls_recall=np.mean(sgcls_recall_list,axis=0)
            predcls_recall=np.mean(predcls_recall_list,axis=0)
            
            rel_acc_mean_1, rel_acc_mean_3, rel_acc_mean_5 = self.compute_mean_predicate(cls_matrix_list, topk_rel_list)
            self.compute_predicate_acc_per_class(cls_matrix_list, topk_rel_list)
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
                ("mean_recall@100", mean_recall[1]),
                
                ("SGcls@20", sgcls_recall[0]),
                ("SGcls@50", sgcls_recall[1]),
                ("SGcls@100", sgcls_recall[2]),
                ("Predcls@20", predcls_recall[0]),
                ("Predcls@50", predcls_recall[1]),
                ("Predcls@100", predcls_recall[2]),
            ]
            progbar.add(1, values=logs)
            
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
            
            self.wandb_log["Validation/SGcls@20"] = sgcls_recall[0]    
            self.wandb_log["Validation/SGcls@50"] = sgcls_recall[1]    
            self.wandb_log["Validation/SGcls@100"] = sgcls_recall[2]    
            self.wandb_log["Validation/Predcls@20"] = predcls_recall[0]
            self.wandb_log["Validation/Predcls@50"] = predcls_recall[1]
            self.wandb_log["Validation/Predcls@100"] = predcls_recall[2]       
        return (obj_acc_1 + rel_acc_1 + rel_acc_mean_1 + mean_recall[0] + triplet_acc_50) / 5 
    

# print("Obj pts Shape:", obj_pts.shape)
# print("Rel pts Shape:", rel_pts.shape)
# print("Obj desc. Shape:", descriptor.shape)
# print("Rel label Shape:", gt_rel_label.shape)
# print("Obj label Shape:", gt_obj_label.shape)
# print("Edge index Shape:", edge_indices.shape)
# print("Batch idx Shape:", batch_ids.shape)