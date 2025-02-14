from utils.eval_utils import *
from utils.logger import Progbar
from utils.contrastive_utils import ContrastiveFreqWeightedSampler, ContrastiveHybridTripletSampler
from utils.model_utils import rotation_matrix
from runners.base_trainer import BaseTrainer
from model.frontend.relextractor import *
from model.models.model_rel_only import BFeatRelObjConNet
from model.backend.classifier import RelCosineClassifier
from model.loss import MultiLabelInfoNCELoss, IntraModalBarlowTwinLoss, SupervisedCrossModalInfoNCE
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, CyclicLR
import wandb
import clip

## TODO: Relationship Feature Extractor Contrastive learning only
class BFeatRelSSLTrainer(BaseTrainer):
    def __init__(self, config, device):
        super().__init__(config, device, multi_view_ssl=True)
        
        self.m_config = config.model
        # Contrastive positive/negative pair sampler  
        if self.t_config.sampler == "triplet":
            self.contrastive_sampler = ContrastiveHybridTripletSampler(config, device)
        else:
            self.contrastive_sampler = ContrastiveFreqWeightedSampler(config, device)
        # Model Definitions
        self.build_text_classifier()
        self.model = BFeatRelObjConNet(self.config, self.text_gt_matrix, device, num_layers=9).to(device)
        ## Contrastive loss only for Relationship Feature extractor
        self.rel_classifier = RelCosineClassifier(
            self.rel_label_list, 
            self.obj_label_list, 
            self.device, d_feats=self.m_config.dim_edge_feats
        )
        
        # Optimizer & Scheduler
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.opt_config.learning_rate, 
            weight_decay=self.opt_config.weight_decay
        )
        if self.t_config.scheduler == "consine":
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
        self.c_criterion = MultiLabelInfoNCELoss(device=self.device, temperature=self.t_config.loss_temperature).to(self.device)
        self.intra_criterion = IntraModalBarlowTwinLoss().to(self.device)
        self.cm_visual_criterion = SupervisedCrossModalInfoNCE(self.device, temperature=0.07) 
        self.cm_text_criterion = SupervisedCrossModalInfoNCE(self.device, temperature=0.07) 
        
        # Add trace meters
        self.add_meters([
            "Train/IMBT_Obj_Loss",  # Intra-Modal Object point cloud contrastive loss
            "Train/CM_Visual_Loss", # Cross-Modal 3D-2D contrastive loss
            "Train/CM_Text_Loss"    # Cross-Modal 3D-Text contrastive loss
        ])
        # Remove trace meters
        self.del_meters([
            "Train/Obj_Cls_Loss",
            "Train/Rel_Cls_Loss"
        ])
        
        # Resume training if ckp path is provided.
        if 'resume' in self.config:
            self.resume_from_checkpoint(self.config.resume)
    
    def __data_augmentation(
        self, 
        points: torch.Tensor # Shape: B X N_pts X N_dim
    ):
        # random rotate
        matrix= np.eye(3)
        matrix[0:3,0:3] = rotation_matrix([0, 0, 1], np.random.uniform(0, 2*np.pi, 1))
        matrix = torch.from_numpy(matrix).to(self.device).float()
        
        _, N, _ = points.shape
        centroid = points[:, :, :3].mean(1)
        points[:, :, :3] -= centroid.unsqueeze(1).repeat(1, N, 1)
        points_rot = torch.einsum('bnc,ca->bna', points[..., :3], matrix.T)
        points[...,:3] = points_rot
        if self.m_config.use_normal:
            ofset = 3
            if self.m_config.use_rgb:
                ofset += 3
            points_rot_feat = torch.einsum('bnc,ca->bna', points[..., ofset: 3 + ofset], matrix.T)
            points[..., ofset: 3 + ofset] = points_rot_feat
        return points
    
    @torch.no_grad()
    def __get_text_feat(self, gt_obj: torch.Tensor):
        return self.text_gt_matrix[gt_obj.long()]
    
    def train(self):
        self.model = self.model.train()
        n_iters = len(self.t_dataloader)
        val_metric = -987654321
        
        # Training Loop
        for e in range(self.t_config.epoch):
            self.wandb_log = {}
            progbar = Progbar(n_iters, width=20, stateful_metrics=['Misc/epo', 'Misc/it'])
            self.model = self.model.train()
            loader = iter(self.t_dataloader)
            
            for idx, (
                obj_pts, 
                rgb_feats,
                rel_pts, 
                descriptor,
                gt_rel_label,
                gt_obj_label,
                zero_mask,
                edge_indices,
                batch_ids
            ) in enumerate(loader):

                (
                    obj_pts, 
                    rgb_feats,
                    rel_pts, 
                    descriptor,
                    gt_rel_label,
                    gt_obj_label,
                    zero_mask,
                    edge_indices,
                    batch_ids
                ) = self.to_device(
                    obj_pts, rgb_feats, rel_pts, 
                    descriptor, gt_rel_label, gt_obj_label, 
                    zero_mask, edge_indices, batch_ids
                )
                
                self.optimizer.zero_grad()
                obj_aug_1 = self.__data_augmentation(obj_pts)
                obj_aug_2 = self.__data_augmentation(obj_pts)
                obj_pts_data = torch.cat([ obj_aug_1, obj_aug_2 ], dim=0)
                obj_pts = obj_pts_data.transpose(2, 1).contiguous()
                obj_feats, edge_feats, obj_pred, obj_t1_feats, obj_t2_feats = self.model(obj_pts, edge_indices.t().contiguous(), descriptor)

                # Object Encoder Contrastive loss
                text_feat = self.__get_text_feat(gt_obj_label)
                loss_imbt = self.intra_criterion(obj_t1_feats, obj_t2_feats)  
                loss_cm_visual = self.cm_visual_criterion(obj_feats, rgb_feats, gt_obj_label, zero_mask)
                loss_cm_text = self.cm_text_criterion(obj_feats, text_feat, gt_obj_label)
                obj_loss = 0.1 * loss_imbt + loss_cm_visual + loss_cm_text
                
                pos_pair, neg_pair, rel_indices = self.contrastive_sampler.sample(gt_obj_label, gt_rel_label, edge_indices)
                contrastive_loss = self.c_criterion(edge_feats, pos_pair, neg_pair, rel_indices)
                
                # TODO: determine coefficient for each loss
                lambda_c = self.t_config.lambda_con
                lambda_oc = self.t_config.lambda_obj_con
                t_loss = lambda_c * contrastive_loss + \
                    lambda_oc * obj_loss
                t_loss.backward()
                self.optimizer.step()
                self.meters['Train/Total_Loss'].update(t_loss.detach().item())
                self.meters['Train/Contrastive_Loss'].update(contrastive_loss.detach().item()) 
                self.meters['Train/IMBT_Obj_Loss'].update(loss_imbt.detach().item()) 
                self.meters['Train/CM_Visual_Loss'].update(loss_cm_visual.detach().item()) 
                self.meters['Train/CM_Text_Loss'].update(loss_cm_text.detach().item()) 
                t_log = [
                    ("train/contrastive_loss", contrastive_loss.detach().item()),
                    ("train/obj_feat_loss", obj_loss.detach().item()),
                    ("train/total_loss", t_loss.detach().item()),
                    ("Misc/epo", int(e)),
                    ("Misc/it", int(idx)),
                    ("lr", self.lr_scheduler.get_last_lr()[0])
                ]
                if e % self.t_config.log_interval == 0:
                    rel_pred = self.rel_classifier(edge_feats, obj_pred, edge_indices)
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
        progbar = Progbar(n_iters, width=20, stateful_metrics=['Misc/it'])
        loader = iter(self.v_dataloader)
        
        topk_obj_list, topk_rel_list, topk_triplet_list, cls_matrix_list = np.array([]), np.array([]), np.array([]), []
        sub_scores_list, obj_scores_list, rel_scores_list = [], [], []
        logs = []
        
        with torch.no_grad():
            self.model = self.model.eval()
            for idx, (
                obj_pts, 
                rgb_feats,
                rel_pts, 
                descriptor,
                gt_rel_label,
                gt_obj_label,
                zero_mask,
                edge_indices,
                batch_ids
            ) in enumerate(loader):

                (
                    obj_pts, 
                    rgb_feats,
                    rel_pts, 
                    descriptor,
                    gt_rel_label,
                    gt_obj_label,
                    zero_mask,
                    edge_indices,
                    batch_ids
                ) = self.to_device(
                    obj_pts, rgb_feats, rel_pts, 
                    descriptor, gt_rel_label, gt_obj_label, 
                    zero_mask, edge_indices, batch_ids
                )
                
                obj_pts = obj_pts.transpose(2, 1).contiguous()
                rel_pts = rel_pts.transpose(2, 1).contiguous()
                obj_pred, edge_feats = self.model(obj_pts, edge_indices.t().contiguous(), descriptor, is_train=False)
                rel_pred = self.rel_classifier(edge_feats, obj_pred, edge_indices)
                
                top_k_obj = evaluate_topk_object(obj_pred.detach(), gt_obj_label, topk=11)
                gt_edges = get_gt(gt_obj_label, gt_rel_label, edge_indices, self.d_config.multi_rel)
                top_k_rel = evaluate_topk_predicate(rel_pred.detach(), gt_edges, self.d_config.multi_rel, topk=6)
                top_k_triplet, cls_matrix, sub_scores, obj_scores, rel_scores = \
                    evaluate_triplet_topk(
                        obj_pred.detach(), rel_pred.detach(), 
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
        return (obj_acc_1 + rel_acc_1 + rel_acc_mean_1 + mean_recall[0] + triplet_acc_50) / 5 
    
    