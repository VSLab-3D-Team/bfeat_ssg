from utils.eval_utils import *
from utils.logger import Progbar
from utils.model_utils import rotation_matrix
from runners.base_trainer import BaseTrainer
from model.frontend.relextractor import *
from model.models.model_encoder import BFeatEncoderPretrainNet
from model.backend.classifier import RelCosineClassifier, consine_classification_obj
from model.loss import MultiLabelInfoNCELoss, IntraModalBarlowTwinLoss, SupervisedCrossModalInfoNCE
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, CyclicLR
import wandb

## TODO: Relationship Feature Extractor Contrastive learning only
class BFeatRelSSLTrainer(BaseTrainer):
    def __init__(self, config, device):
        super().__init__(config, device, multi_view_ssl=True)
        
        self.m_config = config.model
        # Model Definitions
        self.build_text_classifier()
        self.model = BFeatEncoderPretrainNet(self.config, device).to(device)
        ## Contrastive loss only for Relationship Feature extractor
        self.rel_classifier = RelCosineClassifier(
            self.embedding_vector_loader,
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
    
    def __pcd_augmentation(self, _pcd: torch.Tensor, is_obj=True):
        if is_obj:
            pcd_aug_1 = self.__data_augmentation(_pcd)
        else:
            pcd_aug_1 = _pcd
        pcd_aug_2 = self.__data_augmentation(_pcd)
        pts_data = torch.cat([ pcd_aug_1, pcd_aug_2 ], dim=0)
        pts_data = pts_data.transpose(2, 1).contiguous()
        return pts_data
    
    def __get_loss(self, obj_feats, obj_t1_feats, obj_t2_feats, rgb_feats, gt_obj_label, zero_mask):
        text_feat = self.__get_text_feat(gt_obj_label)
        loss_imbt = self.intra_criterion(obj_t1_feats, obj_t2_feats)  
        loss_cm_visual = self.cm_visual_criterion(obj_feats, rgb_feats, gt_obj_label, zero_mask)
        loss_cm_text = self.cm_text_criterion(obj_feats, text_feat, gt_obj_label)
        pcd_loss = 0.1 * loss_imbt + loss_cm_visual + loss_cm_text
        return pcd_loss, loss_imbt, loss_cm_visual, loss_cm_text
    
    def train(self):
        self.model = self.model.train()
        n_iters = len(self.t_dataloader)
        val_metric = 987654321
        
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
                obj_pts = self.__pcd_augmentation(obj_pts)
                rel_pts = self.__pcd_augmentation(rel_pts)
                obj_feats, edge_feats, obj_t1_feats, obj_t2_feats, edge_t1_feats, edge_t2_feats = \
                    self.model(obj_pts, rel_pts, edge_indices.t().contiguous())

                # Object Encoder Contrastive loss
                obj_loss, loss_imbt, loss_cm_visual, loss_cm_text = \
                    self.__get_loss(obj_feats, obj_t1_feats, obj_t2_feats, rgb_feats, gt_obj_label, zero_mask)
                
                rel_loss, loss_rel_imbt, loss_rel_cm_visual, loss_rel_cm_text = \
                    self.__get_loss(edge_feats, edge_t1_feats, edge_t2_feats, rgb_feats, gt_rel_label, zero_mask)
                
                # TODO: determine coefficient for each loss
                lambda_c = self.t_config.lambda_con
                lambda_oc = self.t_config.lambda_obj_con
                t_loss = lambda_c * rel_loss + \
                    lambda_oc * obj_loss
                t_loss.backward()
                self.optimizer.step()
                self.meters['Train/Total_Loss'].update(t_loss.detach().item())
                self.meters['Train/IMBT_Obj_Loss'].update(loss_imbt.detach().item()) 
                self.meters['Train/CM_Visual_Loss'].update(loss_cm_visual.detach().item()) 
                self.meters['Train/CM_Text_Loss'].update(loss_cm_text.detach().item()) 
                t_log = [
                    ("train/contrastive_loss", rel_loss.detach().item()),
                    ("train/obj_feat_loss", obj_loss.detach().item()),
                    ("train/total_loss", t_loss.detach().item()),
                    ("Misc/epo", int(e)),
                    ("Misc/it", int(idx)),
                    ("lr", self.lr_scheduler.get_last_lr()[0])
                ]
                progbar.add(1, values=t_log)
            
            self.lr_scheduler.step()
            if e % self.t_config.evaluation_interval == 0:
                val_loss = self.evaluate_validation()
                if val_loss < val_metric:
                    self.save_checkpoint(self.exp_name, "best_model.pth")
                    val_metric = val_loss
                if e % self.t_config.save_interval == 0:
                    self.save_checkpoint(self.exp_name, 'ckpt_epoch_{epoch}.pth'.format(epoch=e))
            
            self.wandb_log["Train/learning_rate"] = self.lr_scheduler.get_last_lr()[0]
            self.write_wandb_log()
            wandb.log(self.wandb_log)
    
    def evaluate_validation(self):
        n_iters = len(self.v_dataloader)
        progbar = Progbar(n_iters, width=20, stateful_metrics=['Misc/it'])
        loader = iter(self.v_dataloader)
        
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
                
                obj_pts = self.__pcd_augmentation(obj_pts)
                rel_pts = self.__pcd_augmentation(rel_pts)
                obj_feats, edge_feats, obj_t1_feats, obj_t2_feats, edge_t1_feats, edge_t2_feats = \
                    self.model(obj_pts, rel_pts, edge_indices.t().contiguous())
                
                # Object Encoder Contrastive loss
                obj_loss, _, _, _ = \
                    self.__get_loss(obj_feats, obj_t1_feats, obj_t2_feats, rgb_feats, gt_obj_label, zero_mask)
                
                rel_loss, _, _, _ = \
                    self.__get_loss(edge_feats, edge_t1_feats, edge_t2_feats, rgb_feats, gt_rel_label, zero_mask)
                
                lambda_c = self.t_config.lambda_con
                lambda_oc = self.t_config.lambda_obj_con
                t_loss = lambda_c * rel_loss + \
                    lambda_oc * obj_loss
                
                self.meters['Validation/Total_Loss'].update(t_loss.detach().item())
                t_log = [
                    ("val/contrastive_loss", rel_loss.detach().item()),
                    ("val/obj_feat_loss", obj_loss.detach().item()),
                    ("val/total_loss", t_loss.detach().item()),
                ]
                progbar.add(1, values=t_log)
                
            self.wandb_log["Validation/SGcls@100"] = self.meters['Validation/Total_Loss'].avg
        return self.meters['Validation/Total_Loss'].avg
    
    