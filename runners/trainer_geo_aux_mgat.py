from utils.eval_utils import *
from utils.logger import Progbar
from runners.base_trainer import BaseTrainer
from utils.model_utils import TFIDFMaskLayer, TFIDFTripletWeight
from model.frontend.relextractor import *
from model.models.model_geo_aux import BFeatGeoAuxNet
from model.models.model_geo_aux_mgat import BFeatGeoAuxMGATNet
from model.loss import MultiLabelInfoNCELoss, ContrastiveSafeLoss, WeightedFocalLoss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, CyclicLR
import wandb

class CLIPTextEncoder(nn.Module):
    def __init__(self, clip_model_name="ViT-B/32", device="cuda"):
        super().__init__()
        try:
            import clip
            self.model, _ = clip.load(clip_model_name, device=device)
            self.text_encoder = self.model.encode_text
            for param in self.parameters():
                param.requires_grad = False
            self.clip_available = True
        except Exception as e:
            print(f"CLIP initialization failed: {e}")
            self.clip_available = False
    
    def forward(self, text):
        if not hasattr(self, 'clip_available') or not self.clip_available:
            return torch.ones(len(text), 512, device=next(self.parameters()).device)
        
        try:
            import clip
            text_tokens = clip.tokenize(text).to(next(self.parameters()).device)
            return self.text_encoder(text_tokens)
        except Exception as e:
            print(f"CLIP inference failed: {e}")
            return torch.ones(len(text), 512, device=next(self.parameters()).device)

class TripletProjector(nn.Module):
    def __init__(self, node_dim, edge_dim, output_dim=512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim)
        )
    
    def forward(self, subj_feat, obj_feat, rel_feat):
        combined = torch.cat([subj_feat, rel_feat, obj_feat], dim=1)
        return self.proj(combined)

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
        
        try:
            self.clip_text_encoder = CLIPTextEncoder(device=device).to(device)
            
            node_dim = self.m_config.dim_obj_feats
            edge_dim = self.m_config.dim_edge_feats
            
            self.triplet_projector = TripletProjector(node_dim, edge_dim).to(device)
            
            self.edge_projector = nn.Sequential(
                nn.Linear(edge_dim, edge_dim),
                nn.ReLU(),
                nn.Linear(edge_dim, 512)
            ).to(device)

            self.rel_encoder_projector = nn.Sequential(
                nn.Linear(edge_dim, edge_dim),
                nn.ReLU(),
                nn.Linear(edge_dim, 512)
            ).to(device)
            
            self._text_embeddings_cache = {}
            
            self.lambda_triplet = getattr(self.t_config, 'lambda_triplet', 1.0)
            self.lambda_edge = getattr(self.t_config, 'lambda_edge', 1.0)
            self.lambda_text_aux = getattr(self.t_config, 'lambda_text_aux', 1.0)
            
            self.use_triplet_loss = True
            self.use_edge_loss = True
            self.use_text_aux_loss = True
            print(f"Triplet loss initialization success. (Weight: {self.lambda_triplet})")
            print(f"Edge loss initialization success. (Weight: {self.lambda_edge})")
            print(f"Edge text aux loss initialization success. (Weight: {self.lambda_text_aux})")
            
            self.add_meters([
                "Train/Triplet_Loss",
                "Train/Edge_Text_Loss",
                "Train/Edge_Text_Aux_Loss"
            ])
        
        except Exception as e:
            print(f"Triplet/Edge loss initialization failed: {e}")
            self.use_triplet_loss = False
            self.use_edge_loss = False
            self.use_text_aux_loss = False
        
        self.add_meters([
            "Train/Geo_Aux_Loss",
            "Train/Edge_CLIP_Aux_Loss",
        ])
        self.del_meters([
            "Train/Contrastive_Loss"
        ])

        self._initialize_replay_buffer()
        
        self.lambda_aug = getattr(self.t_config, 'lambda_aug', 1.0)
        print(f"Augmentation loss initialization success. (Weight: {self.lambda_aug})")
        
        # Resume training if ckp path is provided.
        if 'resume' in self.config:
            self.resume_from_checkpoint(self.config.resume)
    
    def _get_text_embedding(self, text_template, fill_values):
        if not hasattr(self, 'use_triplet_loss') or not self.use_triplet_loss:
            return torch.ones(1, 512, device=self.device)
            
        text = text_template.format(**fill_values)
        
        if text in self._text_embeddings_cache:
            return self._text_embeddings_cache[text]
        
        with torch.no_grad():
            embedding = self.clip_text_encoder([text])
            embedding = F.normalize(embedding, p=2, dim=1)
            self._text_embeddings_cache[text] = embedding
            
        return embedding
    
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
    
    def _initialize_replay_buffer(self):

        import time
        import random
        
        self.replay_buffer = {
            'buffers': {i: [] for i in range(self.num_rel_class)},
            'update_class_difficulty': self._update_buffer_class_difficulty,
            'update_class_frequency': self._update_buffer_class_frequency
        }
        
        self.replay_buffers = self.replay_buffer['buffers']
        
        self.total_buffer_size = 2000 # 조정
        self.min_samples_per_class = 8 # 조정
        
        self.class_difficulty = torch.ones(self.num_rel_class, device=self.device)
        self.class_frequency = torch.zeros(self.num_rel_class, device=self.device)
        
        self.class_centroids = torch.zeros((self.num_rel_class, self.m_config.dim_edge_feats), device=self.device)
        self.class_counts = torch.zeros(self.num_rel_class, device=self.device)
        
        self.augmentation_strength = torch.ones(self.num_rel_class, device=self.device) * 0.2
        
        self.confusion_matrix = torch.zeros((self.num_rel_class, self.num_rel_class), device=self.device)
        
        self.aug_batch_ratio = 0.3  # 배치 내 증강 샘플 초기 비율
        self.aug_warmup_epochs = 1  # 증강 시작 에포크
        
        self.aug_strategy_weights = {
            'centroid': 0.4,
            'interpolation': 0.3,
            'contrastive': 0.3
        }
        
        self.add_meters([
            "Train/Augmentation_Loss",
            "Train/Original_Samples",
            "Train/Augmented_Samples",
            "Train/Class_Accuracy_Mean",
            "Train/Class_Accuracy_Min"
        ])
        
        print("Replay buffer initialized with total size:", self.total_buffer_size)
        print("Min samples per class:", self.min_samples_per_class)

    def _update_buffer_class_difficulty(self, class_accuracy):
 
        clamped_accuracy = torch.clamp(class_accuracy, min=0.0, max=1.0)
        self.class_difficulty = 1.0 - clamped_accuracy

    def _update_buffer_class_frequency(self, class_counts):

        self.class_frequency = class_counts

    def _update_class_statistics(self, pred, target):

        with torch.no_grad():
            if self.d_config.multi_rel:
                pred_cls = (pred > 0.5).float()
                correct = (pred_cls == target).float()
                
                class_correct = torch.sum(correct * target, dim=0)
                class_total = torch.sum(target, dim=0)
                class_total = torch.max(class_total, torch.ones_like(class_total))  # 0으로 나누기 방지
                new_accuracy = class_correct / class_total
                
                if not hasattr(self, 'class_accuracy'):
                    self.class_accuracy = torch.zeros_like(new_accuracy)
                self.class_accuracy = 0.9 * self.class_accuracy + 0.1 * new_accuracy
                
                for i in range(target.size(0)):
                    true_classes = torch.where(target[i] > 0.5)[0]
                    pred_classes = torch.where(pred[i] > 0.5)[0]
                    
                    for true_cls in true_classes:
                        for pred_cls in pred_classes:
                            if true_cls != pred_cls:
                                self.confusion_matrix[true_cls.item(), pred_cls.item()] += 1
            else:
                pred_cls = torch.argmax(pred, dim=1)
                target_cls = torch.argmax(target, dim=1)
                
                for c in range(self.num_rel_class):
                    class_mask = (target_cls == c)
                    if torch.sum(class_mask) > 0:
                        class_correct = torch.sum((pred_cls == target_cls) & class_mask).float()
                        class_total = torch.sum(class_mask).float()
                        new_accuracy = class_correct / class_total
                        
                        if not hasattr(self, 'class_accuracy'):
                            self.class_accuracy = torch.zeros(self.num_rel_class, device=self.device)
                        self.class_accuracy[c] = 0.9 * self.class_accuracy[c] + 0.1 * new_accuracy.item()
                
                for true_cls, pred_cls in zip(target_cls, pred_cls):
                    if true_cls.item() != pred_cls.item():
                        self.confusion_matrix[true_cls.item(), pred_cls.item()] += 1
            
            if hasattr(self, 'meters') and 'Train/Class_Accuracy_Mean' in self.meters:
                self.meters['Train/Class_Accuracy_Mean'].update(torch.mean(self.class_accuracy).item())
                self.meters['Train/Class_Accuracy_Min'].update(torch.min(self.class_accuracy).item())

    def _compute_difficulty_score(self, pred, target):
        
        with torch.no_grad():
            if self.d_config.multi_rel:
                pred_prob = pred.clone()
                correct_class_prob = torch.sum(pred_prob * target, dim=1)
                
                difficulty = 1.0 - correct_class_prob
            else:
                pred_prob = F.softmax(pred, dim=1)
                target_indices = torch.argmax(target, dim=1)
                correct_class_prob = pred_prob[torch.arange(pred.size(0)), target_indices]
                
                difficulty = 1.0 - correct_class_prob
            
            difficulty = torch.clamp(difficulty, min=0.0, max=1.0)
            
            return difficulty

    def _allocate_buffer_space(self):
        
        remaining_space = self.total_buffer_size - (self.min_samples_per_class * self.num_rel_class)
        if remaining_space < 0:
            remaining_space = 0
            self.min_samples_per_class = self.total_buffer_size // self.num_rel_class
        
        freq_factor = torch.log(torch.sum(self.class_frequency) / (self.class_frequency + 1))
        combined_score = self.class_difficulty * freq_factor
        
        if torch.sum(combined_score) > 0:
            normalized_score = combined_score / torch.sum(combined_score)
        else:
            normalized_score = torch.ones_like(combined_score) / self.num_rel_class
        
        additional_space = (normalized_score * remaining_space).int().cpu().numpy()
        
        buffer_sizes = {i: self.min_samples_per_class + additional_space[i] for i in range(self.num_rel_class)}
        return buffer_sizes

    def _add_to_replay_buffer(self, features, class_indices, difficulty_scores):
      
        import time
        
        buffer_sizes = self._allocate_buffer_space()
        
        for i, (feature, cls_idx, diff_score) in enumerate(zip(features, class_indices, difficulty_scores)):
            cls = cls_idx.item()
            
            sample = {
                'feature': feature.detach().cpu(),
                'difficulty': diff_score.item(),
                'timestamp': time.time()
            }
            
            if len(self.replay_buffers[cls]) >= buffer_sizes[cls] and buffer_sizes[cls] > 0:
                self.replay_buffers[cls].sort(key=lambda x: (x['difficulty'], -x['timestamp']))
                self.replay_buffers[cls].pop(0)
            
            if buffer_sizes[cls] > 0:
                self.replay_buffers[cls].append(sample)

    def _update_centroids(self, features, class_indices):
        with torch.no_grad():
            for feat, cls_idx in zip(features, class_indices):
                cls = cls_idx.item()
                
                old_count = self.class_counts[cls].item()
                new_count = old_count + 1
                
                if old_count > 0:
                    old_centroid = self.class_centroids[cls].clone()
                    new_centroid = (old_centroid * old_count + feat.detach()) / new_count
                    self.class_centroids[cls] = new_centroid
                else:
                    self.class_centroids[cls] = feat.detach().clone()
                    
                self.class_counts[cls] = torch.tensor(new_count, device=self.class_counts.device)
            
    def _update_augmentation_strength(self):
       
        clamped_accuracy = torch.clamp(self.class_accuracy, min=0.0, max=1.0)
    
        self.augmentation_strength = 0.1 + 0.2 * (1.0 - clamped_accuracy)

    def _select_augmentation_strategy(self):
        
        import random
        
        strategies = list(self.aug_strategy_weights.keys())
        weights = list(self.aug_strategy_weights.values())
        
        return random.choices(strategies, weights=weights, k=1)[0]

    def _sample_from_replay_buffer(self, batch_size):
        
        import random
        
        batch_size = min(batch_size, 8)
        
        non_empty_classes = [cls for cls in range(self.num_rel_class) if len(self.replay_buffers[cls]) > 0]
        
        if len(non_empty_classes) == 0:
            return None, None
        
        weights = torch.tensor([max(0.0, self.class_difficulty[cls].item()) for cls in non_empty_classes], device=self.device)
        
        if torch.sum(weights) <= 0:
            weights = torch.ones(len(non_empty_classes), device=self.device)
        
        weights = F.softmax(weights, dim=0)
        
        actual_samples = min(batch_size, len(non_empty_classes))
        
        if actual_samples == 0:
            return None, None
        
        class_indices = torch.multinomial(weights, actual_samples, replacement=True)
        
        class_sample_counts = {}
        for idx in class_indices:
            cls = non_empty_classes[idx.item()]
            class_sample_counts[cls] = class_sample_counts.get(cls, 0) + 1
        
        remaining_samples = batch_size - actual_samples
        if remaining_samples > 0 and len(class_sample_counts) > 0:
            selected_classes = list(class_sample_counts.keys())
            selected_counts = list(class_sample_counts.values())
            total_selected = sum(selected_counts)
            
            for i in range(remaining_samples):
                probs = [count / total_selected for count in selected_counts]
                selected_idx = random.choices(range(len(selected_classes)), weights=probs)[0]
                selected_cls = selected_classes[selected_idx]
                class_sample_counts[selected_cls] = class_sample_counts.get(selected_cls, 0) + 1
                selected_counts[selected_idx] += 1
                total_selected += 1
        
        sampled_features = []
        sampled_classes = []
        
        for cls, count in class_sample_counts.items():
            if count > 0 and len(self.replay_buffers[cls]) > 0:
                difficulties = torch.tensor([max(0.0, sample['difficulty']) for sample in self.replay_buffers[cls]])
                
                if torch.sum(difficulties) <= 0:
                    sample_weights = torch.ones(len(self.replay_buffers[cls])) / len(self.replay_buffers[cls])
                else:
                    sample_weights = difficulties / torch.sum(difficulties)
                
                sample_weights = torch.clamp(sample_weights, min=0.0)
                if torch.sum(sample_weights) <= 0:
                    sample_weights = torch.ones_like(sample_weights) / len(sample_weights)
                
                try:
                    sample_indices = torch.multinomial(
                        sample_weights, 
                        min(count, len(self.replay_buffers[cls])), 
                        replacement=True
                    )
                    
                    for idx in sample_indices:
                        sampled_features.append(self.replay_buffers[cls][idx.item()]['feature'].to(self.device))
                        sampled_classes.append(cls)
                except RuntimeError as e:
                    print(f"Error during sampling from class {cls}: {e}")
                    print(f"sample_weights stats: min={sample_weights.min().item()}, max={sample_weights.max().item()}, sum={sample_weights.sum().item()}")
                    if len(self.replay_buffers[cls]) > 0:
                        random_indices = torch.randint(0, len(self.replay_buffers[cls]), (min(count, len(self.replay_buffers[cls])),))
                        for idx in random_indices:
                            sampled_features.append(self.replay_buffers[cls][idx.item()]['feature'].to(self.device))
                            sampled_classes.append(cls)
        
        if not sampled_features:
            return None, None
        
        return torch.stack(sampled_features), torch.tensor(sampled_classes, device=self.device)

    def _apply_augmentation(self, features, class_indices):
   
        strategy = self._select_augmentation_strategy()
        
        if strategy == 'centroid':
            return self._centroid_based_augmentation(features, class_indices)
        elif strategy == 'interpolation':
            return self._interpolation_based_augmentation(features, class_indices)
        elif strategy == 'contrastive':
            return self._contrastive_augmentation(features, class_indices)
        else:
            return features

    def _centroid_based_augmentation(self, features, class_indices):
        device = features.device
        augmented_features = []
        
        centroids = self.class_centroids.to(device)
        aug_strength = self.augmentation_strength.to(device)
        
        for i, (feat, cls_idx) in enumerate(zip(features, class_indices)):
            cls = cls_idx.item()
            
            if self.class_counts[cls] > 0:
                direction = centroids[cls] - feat
                offset = aug_strength[cls] * direction
                aug_feat = feat + offset
                
                aug_feat = F.normalize(aug_feat, p=2, dim=0)
                augmented_features.append(aug_feat)
            else:
                augmented_features.append(feat.clone())
        
        return torch.stack(augmented_features)

    def _interpolation_based_augmentation(self, features, class_indices):
    
        import random
        
        class_features = {i: [] for i in range(self.num_rel_class)}
        for feat, cls_idx in zip(features, class_indices):
            cls = cls_idx.item()
            class_features[cls].append(feat)
        
        augmented_features = []
        
        for feat, cls_idx in zip(features, class_indices):
            cls = cls_idx.item()
            
            if len(class_features[cls]) > 1:
                other_feats = [f for f in class_features[cls] if not torch.equal(f, feat)]
                if other_feats:
                    other_feat = random.choice(other_feats)
                    
                    lambda_val = torch.rand(1, device=self.device).item() * self.augmentation_strength[cls]
                    aug_feat = feat + lambda_val * (other_feat - feat)
                    
                    aug_feat = F.normalize(aug_feat, p=2, dim=0)
                    augmented_features.append(aug_feat)
                else:
                    augmented_features.append(feat)
            else:
                if self.class_counts[cls] > 0:
                    alpha = self.augmentation_strength[cls]
                    aug_feat = feat + alpha * (self.class_centroids[cls] - feat)
                    aug_feat = F.normalize(aug_feat, p=2, dim=0)
                    augmented_features.append(aug_feat)
                else:
                    augmented_features.append(feat)
        
        return torch.stack(augmented_features)

    def _contrastive_augmentation(self, features, class_indices):

        augmented_features = []
        
        for feat, cls_idx in zip(features, class_indices):
            cls = cls_idx.item()
            
            if hasattr(self, 'confusion_matrix'):
                confusion_row = self.confusion_matrix[cls].clone()
                confusion_row[cls] = 0
                most_confused_cls = torch.argmax(confusion_row).item()
                
                if self.class_counts[cls] > 0 and self.class_counts[most_confused_cls] > 0:
                    beta = self.augmentation_strength[cls] * 1.5 
                    centroid_diff = self.class_centroids[cls] - self.class_centroids[most_confused_cls]
                    aug_feat = feat + beta * centroid_diff
                    
                    aug_feat = F.normalize(aug_feat, p=2, dim=0)
                    augmented_features.append(aug_feat)
                else:
                    if self.class_counts[cls] > 0:
                        alpha = self.augmentation_strength[cls]
                        aug_feat = feat + alpha * (self.class_centroids[cls] - feat)
                        aug_feat = F.normalize(aug_feat, p=2, dim=0)
                        augmented_features.append(aug_feat)
                    else:
                        augmented_features.append(feat)
            else:
                if self.class_counts[cls] > 0:
                    alpha = self.augmentation_strength[cls]
                    aug_feat = feat + alpha * (self.class_centroids[cls] - feat)
                    aug_feat = F.normalize(aug_feat, p=2, dim=0)
                    augmented_features.append(aug_feat)
                else:
                    augmented_features.append(feat)
        
        return torch.stack(augmented_features)
    
    def train(self):

        self.model = self.model.train()
        n_iters = len(self.t_dataloader)
        val_metric = -987654321

        class_counts = torch.zeros(self.num_rel_class, device=self.device)
        
        # Training Loop
        for e in range(self.t_config.epoch):
            torch.autograd.set_detect_anomaly(True)
            self.wandb_log = {}
            progbar = Progbar(n_iters, width=40, stateful_metrics=['Misc/epo', 'Misc/it'])
            self.model = self.model.train()
            loader = iter(self.t_dataloader)

            if e >= self.aug_warmup_epochs:
                progress = min(1.0, (e - self.aug_warmup_epochs) / (self.t_config.epoch - self.aug_warmup_epochs))
                self.aug_batch_ratio = 0.5 + 0.5 * progress
            
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
                
                edge_feats, obj_pred, rel_pred, pred_edge_clip, pred_geo_desc, edge_desc = \
                    self.model(
                        obj_pts, rel_pts, edge_indices.t().contiguous(), 
                        descriptor, edge_feat_mask, batch_ids, attn_tfidf_weight,
                        edge_2d_feats
                    )
                
                if self.d_config.multi_rel:
                    batch_class_counts = torch.sum(gt_rel_label, dim=0)
                else:
                    batch_class_counts = torch.bincount(
                        torch.argmax(gt_rel_label, dim=1), 
                        minlength=self.num_rel_class
                    )
                class_counts += batch_class_counts
                
                self._update_class_statistics(rel_pred, gt_rel_label)
                
                difficulty_scores = self._compute_difficulty_score(rel_pred, gt_rel_label)
                
                if self.d_config.multi_rel:
                    rel_classes = torch.argmax(gt_rel_label, dim=1)
                else:
                    rel_classes = torch.argmax(gt_rel_label, dim=1)
                
                self._add_to_replay_buffer(edge_feats, rel_classes, difficulty_scores)
                
                self._update_centroids(edge_feats, rel_classes)
                
                self._update_augmentation_strength()

                rel_weight = self.__dynamic_rel_weight(gt_rel_label)
                obj_weight = self.__dynamic_obj_weight(gt_obj_label).to(self.device)
                c_obj_loss = F.cross_entropy(obj_pred, gt_obj_label, weight=obj_weight)
                c_rel_loss = F.binary_cross_entropy(rel_pred, gt_rel_label, weight=rel_weight)
                
                # pos_pair, neg_pair, rel_indices = self.contrastive_sampler.sample(gt_obj_label, gt_rel_label, edge_indices)
                # contrastive_loss = self.c_criterion(edge_feats, pos_pair, neg_pair, rel_indices)
                
                geo_aux_loss = F.l1_loss(pred_geo_desc, edge_desc)
                edge_clip_aux_loss = self.cosine_loss(pred_edge_clip, edge_2d_feats)
                
                triplet_loss = torch.tensor(0.0, device=self.device)
                edge_text_loss = torch.tensor(0.0, device=self.device)
                edge_text_aux_loss = torch.tensor(0.0, device=self.device)
                if hasattr(self, 'use_triplet_loss') and self.use_triplet_loss:
                    try:
                        with torch.no_grad():
                            _obj_feats, _, _ = self.model.point_encoder(obj_pts)
                        node_features = _obj_feats.clone().detach()
                        
                        obj_pred_softmax = F.softmax(obj_pred, dim=1)
                        
                        batch_size = min(128, edge_indices.shape[0])
                        if batch_size > 0:
                            sample_indices = torch.randperm(edge_indices.shape[0])[:batch_size]
                            sampled_edges = edge_indices[sample_indices]
                            
                            subject_indices = sampled_edges[:, 0]
                            object_indices = sampled_edges[:, 1]
                            subject_features = node_features[subject_indices]
                            object_features = node_features[object_indices]
                            relation_features = edge_feats[sample_indices]
                            
                            subject_cls_pred = obj_pred_softmax[subject_indices]
                            object_cls_pred = obj_pred_softmax[object_indices]
                            relation_cls_pred = rel_pred[sample_indices]
                            
                            subject_cls_idx = subject_cls_pred.argmax(dim=1)
                            object_cls_idx = object_cls_pred.argmax(dim=1)
                            
                            clip_rel_loss = 0
                            
                            for i in range(batch_size):
                                subject_name = self.obj_label_list[subject_cls_idx[i]]
                                object_name = self.obj_label_list[object_cls_idx[i]]
                                
                                if self.d_config.multi_rel:
                                    rel_idx = relation_cls_pred[i].argmax().item()
                                    relation_name = self.rel_label_list[rel_idx]
                                else:
                                    rel_idx = relation_cls_pred[i].argmax().item()
                                    relation_name = self.rel_label_list[rel_idx]
                                
                                triplet_text_emb = self._get_text_embedding(
                                    "a point cloud of a {subj} {pred} a {obj}", 
                                    {"subj": subject_name, "pred": relation_name, "obj": object_name}
                                )
                                
                                triplet_feature = self.triplet_projector(
                                    subject_features[i].unsqueeze(0), 
                                    object_features[i].unsqueeze(0), 
                                    relation_features[i].unsqueeze(0)
                                )
                                triplet_feature = F.normalize(triplet_feature, p=2, dim=1)
                                
                                clip_rel_loss += (1 - F.cosine_similarity(triplet_feature, triplet_text_emb)).mean()
                            
                            triplet_loss = clip_rel_loss / batch_size
                            self.meters['Train/Triplet_Loss'].update(triplet_loss.detach().item())
                    except Exception as e:
                        print(f"Error during calculate triplet loss: {str(e)}")
                        triplet_loss = torch.tensor(0.0, device=self.device)
                
                if hasattr(self, 'use_edge_loss') and self.use_edge_loss:
                    try:
                        rel_pred_softmax = rel_pred
                        
                        batch_size = min(128, edge_indices.shape[0])
                        if batch_size > 0:
                            sample_indices = torch.randperm(edge_indices.shape[0])[:batch_size]
                            relation_features = edge_feats[sample_indices]
                            relation_cls_pred = rel_pred_softmax[sample_indices]
                            
                            edge_loss = 0
                            
                            for i in range(batch_size):
                                if self.d_config.multi_rel:
                                    rel_idx = relation_cls_pred[i].argmax().item()
                                    relation_name = self.rel_label_list[rel_idx]
                                else:
                                    rel_idx = relation_cls_pred[i].argmax().item()
                                    relation_name = self.rel_label_list[rel_idx]
                                
                                relation_text_emb = self._get_text_embedding(
                                    "{pred}", 
                                    {"pred": relation_name}
                                )
                                
                                edge_feature = self.edge_projector(relation_features[i].unsqueeze(0))
                                edge_feature = F.normalize(edge_feature, p=2, dim=1)
                                
                                edge_loss += (1 - F.cosine_similarity(edge_feature, relation_text_emb)).mean()
                            
                            edge_text_loss = edge_loss / batch_size
                            self.meters['Train/Edge_Text_Loss'].update(edge_text_loss.detach().item())
                    except Exception as e:
                        print(f"Error during calculate edge text loss: {str(e)}")
                        edge_text_loss = torch.tensor(0.0, device=self.device)

                if hasattr(self, 'use_text_aux_loss') and self.use_text_aux_loss:
                    try:
                        obj_pred_softmax = F.softmax(obj_pred, dim=1)
                        rel_pred_softmax = rel_pred
                        
                        batch_size = min(128, edge_indices.shape[0])
                        if batch_size > 0:
                            sample_indices = torch.randperm(edge_indices.shape[0])[:batch_size]
                            relation_features = edge_feats[sample_indices]
                            
                            sampled_edges = edge_indices[sample_indices]
                            subject_indices = sampled_edges[:, 0]
                            object_indices = sampled_edges[:, 1]
                            
                            subject_cls_pred = obj_pred_softmax[subject_indices]
                            object_cls_pred = obj_pred_softmax[object_indices]
                            relation_cls_pred = rel_pred_softmax[sample_indices]
                            
                            subject_cls_idx = subject_cls_pred.argmax(dim=1)
                            object_cls_idx = object_cls_pred.argmax(dim=1)
                            
                            text_aux_loss = 0
                            
                            for i in range(batch_size):
                                subject_name = self.obj_label_list[subject_cls_idx[i]]
                                object_name = self.obj_label_list[object_cls_idx[i]]
                                
                                if self.d_config.multi_rel:
                                    rel_idx = relation_cls_pred[i].argmax().item()
                                    relation_name = self.rel_label_list[rel_idx]
                                else:
                                    rel_idx = relation_cls_pred[i].argmax().item()
                                    relation_name = self.rel_label_list[rel_idx]
                                
                                relation_text_emb = self._get_text_embedding(
                                    "a {subj} {pred} a {obj}", 
                                    {"subj": subject_name, "pred": relation_name, "obj": object_name}
                                )
                                
                                edge_feature = self.rel_encoder_projector(relation_features[i].unsqueeze(0))
                                edge_feature = F.normalize(edge_feature, p=2, dim=1)
                                
                                text_aux_loss += (1 - F.cosine_similarity(edge_feature, relation_text_emb)).mean()
                            
                            edge_text_aux_loss = text_aux_loss / batch_size
                            self.meters['Train/Edge_Text_Aux_Loss'].update(edge_text_aux_loss.detach().item())
                    except Exception as e:
                        print(f"Error during calculate edge_text_aux_loss: {str(e)}")
                        edge_text_aux_loss = torch.tensor(0.0, device=self.device)

                aug_loss = torch.tensor(0.0, device=self.device)
                if e >= self.aug_warmup_epochs:
                    aug_batch_size = min(int(edge_feats.size(0) * self.aug_batch_ratio), 8) # 조정
                    if aug_batch_size > 0:
                        sampled_features, sampled_classes = self._sample_from_replay_buffer(aug_batch_size)
                        
                        if sampled_features is not None and sampled_classes is not None:
                            augmented_features = self._apply_augmentation(sampled_features, sampled_classes)
                            
                            aug_rel_pred = self.model.rel_classifier(augmented_features)
                            
                            if self.d_config.multi_rel:
                                aug_gt_rel_label = torch.zeros_like(aug_rel_pred)
                                for i, cls in enumerate(sampled_classes):
                                    aug_gt_rel_label[i, cls] = 1.0
                            else:
                                aug_gt_rel_label = F.one_hot(sampled_classes, num_classes=self.num_rel_class).float()
                            
                            aug_rel_weight = self.__dynamic_rel_weight(aug_gt_rel_label)
                            aug_rel_loss = F.binary_cross_entropy(aug_rel_pred, aug_gt_rel_label, weight=aug_rel_weight)
                            
                            aug_weight = min(1.0, (e - self.aug_warmup_epochs + 1) / 10.0)
                            aug_loss = aug_rel_loss * aug_weight
                            
                            self.meters['Train/Augmented_Samples'].update(aug_batch_size)
                            self.meters['Train/Augmentation_Loss'].update(aug_loss.detach().item())

                self.meters['Train/Original_Samples'].update(edge_feats.size(0))

                # TODO: determine coefficient for each loss
                lambda_o = self.t_config.lambda_obj # 0.1
                lambda_r = self.t_config.lambda_rel
                lambda_g = self.t_config.lambda_geo
                lambda_v = self.t_config.lambda_view
                lambda_t = self.t_config.lambda_triplet
                lambda_e = self.t_config.lambda_edge
                lambda_t_a = self.t_config.lambda_text_aux
                # lambda_c = self.t_config.lambda_con # 0.1
                # + lambda_c * contrastive_loss \

                lambda_aug = self.lambda_aug if hasattr(self, 'lambda_aug') else 1.0
                    
                # Geo Aux: 0.3 or 1.0
                t_loss = lambda_o * c_obj_loss \
                    + lambda_r * c_rel_loss \
                    + lambda_g * geo_aux_loss \
                    + lambda_v * edge_clip_aux_loss \
                    + lambda_t * triplet_loss \
                    + lambda_e * edge_text_loss \
                    + lambda_t_a * edge_text_aux_loss \
                    + lambda_aug * aug_loss
                t_loss.backward()
                self.optimizer.step()
                self.meters['Train/Total_Loss'].update(t_loss.detach().item())
                self.meters['Train/Obj_Cls_Loss'].update(c_obj_loss.detach().item())
                self.meters['Train/Rel_Cls_Loss'].update(c_rel_loss.detach().item()) 
                # self.meters['Train/Contrastive_Loss'].update(contrastive_loss.detach().item()) 
                self.meters['Train/Geo_Aux_Loss'].update(geo_aux_loss.detach().item()) 
                self.meters['Train/Edge_CLIP_Aux_Loss'].update(edge_clip_aux_loss.detach().item()) 
                if hasattr(self, 'use_triplet_loss') and self.use_triplet_loss:
                    self.meters['Train/Triplet_Loss'].update(triplet_loss.detach().item())
                if hasattr(self, 'use_edge_loss') and self.use_edge_loss:
                    self.meters['Train/Edge_Text_Loss'].update(edge_text_loss.detach().item())
                if hasattr(self, 'use_text_aux_loss') and self.use_text_aux_loss:
                    self.meters['Train/Edge_Text_Aux_Loss'].update(edge_text_aux_loss.detach().item())
                
                self.meters['Train/Augmentation_Loss'].update(aug_loss.detach().item())
                t_log = [
                    ("train/rel_loss", c_rel_loss.detach().item()),
                    ("train/obj_loss", c_obj_loss.detach().item()),
                    # ("train/contrastive_loss", contrastive_loss.detach().item()),
                    ("train/total_loss", t_loss.detach().item()),
                    ("train/aug_loss", aug_loss.detach().item()),
                ]
                
                if hasattr(self, 'use_triplet_loss') and self.use_triplet_loss:
                    t_log.append(("train/triplet_loss", triplet_loss.detach().item()))

                if hasattr(self, 'use_edge_loss') and self.use_edge_loss:
                    t_log.append(("train/edge_text_loss", edge_text_loss.detach().item()))

                if hasattr(self, 'use_text_aux_loss') and self.use_text_aux_loss:
                    t_log.append(("train/edge_text_aux_loss", edge_text_aux_loss.detach().item()))
                
                t_log += [
                    ("Misc/epo", int(e)),
                    ("Misc/it", int(idx)),
                    ("lr", self.lr_scheduler.get_last_lr()[0]),
                    ("aug_ratio", self.aug_batch_ratio)
                ]
                
                if e % self.t_config.log_interval == 0:
                    logs = self.evaluate_train(obj_pred, gt_obj_label, rel_pred, gt_rel_label, edge_indices)
                    t_log += logs
                progbar.add(1, values=t_log)
            
            self.replay_buffer['update_class_difficulty'](self.class_accuracy)
            self.replay_buffer['update_class_frequency'](class_counts)
        
            self.lr_scheduler.step()
            if e % self.t_config.evaluation_interval == 0:
                mRecall_50 = self.evaluate_validation()
                if mRecall_50 >= val_metric:
                    self.save_checkpoint(self.exp_name, "best_model.pth")
                    val_metric = mRecall_50
                if e % self.t_config.save_interval == 0:
                    self.save_checkpoint(self.exp_name, 'ckpt_epoch_{epoch}.pth'.format(epoch=e))
            
            self.wandb_log["Train/learning_rate"] = self.lr_scheduler.get_last_lr()[0]
            self.wandb_log["Train/Augmentation_Loss"] = self.meters['Train/Augmentation_Loss'].avg
            self.wandb_log["Train/Original_Samples"] = self.meters['Train/Original_Samples'].avg
            self.wandb_log["Train/Augmented_Samples"] = self.meters['Train/Augmented_Samples'].avg
            self.wandb_log["Train/Augmentation_Ratio"] = self.aug_batch_ratio
            self.wandb_log["Train/Class_Accuracy_Mean"] = self.meters['Train/Class_Accuracy_Mean'].avg
            self.wandb_log["Train/Class_Accuracy_Min"] = self.meters['Train/Class_Accuracy_Min'].avg
            
            if hasattr(self, 'use_triplet_loss') and self.use_triplet_loss:
                self.wandb_log["Train/Triplet_Loss"] = self.meters['Train/Triplet_Loss'].avg
            if hasattr(self, 'use_edge_loss') and self.use_edge_loss:
                self.wandb_log["Train/Edge_Text_Loss"] = self.meters['Train/Edge_Text_Loss'].avg
            if hasattr(self, 'use_text_aux_loss') and self.use_text_aux_loss:
                self.wandb_log["Train/Edge_Text_Aux_Loss"] = self.meters['Train/Edge_Text_Aux_Loss'].avg
            
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
            # if hasattr(self.model, 'set_inference_mode'):
            #     self.model.set_inference_mode()
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
                
                _, obj_pred, rel_pred, _, _, _ = self.model(
                    obj_pts, rel_pts, edge_indices.t().contiguous(), descriptor, edge_feat_mask, batch_ids, None,  # attn_tfidf_weight
                    edge_2d_feats
                )
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
                gt_obj_list = np.concatenate((gt_obj_list, gt_obj_label.cpu().numpy()))
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
            # obj_mean_recall = get_obj_mean_recall(topk_obj_list, cls_matrix_list)
            
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
            obj_acc_mean_1, obj_acc_mean_5, obj_acc_mean_10 = self.compute_mean_object(gt_obj_list, topk_obj_list)
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
                # ("obj_mean_recall@1", obj_mean_recall[0]),
                # ("obj_mean_recall@5", obj_mean_recall[1]),
                # ("obj_mean_recall@10", obj_mean_recall[2]),
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
                
                ("Acc@1/obj_cls_acc_mean", obj_acc_mean_1),
                ("Acc@5/obj_cls_acc_mean", obj_acc_mean_5),
                ("Acc@10/obj_cls_acc_mean", obj_acc_mean_10),
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
            
            self.wandb_log["Validation/Acc@1/obj_cls_acc_mean"] =  obj_acc_mean_1
            self.wandb_log["Validation/Acc@5/obj_cls_acc_mean"] =  obj_acc_mean_5
            self.wandb_log["Validation/Acc@10/obj_cls_acc_mean"] =  obj_acc_mean_10

        # if hasattr(self.model, 'set_training_mode'):
        #     self.model.set_training_mode()

        return (obj_acc_1 + rel_acc_1 + rel_acc_mean_1 + mean_recall[0] + triplet_acc_50) / 5 
    

# print("Obj pts Shape:", obj_pts.shape)
# print("Rel pts Shape:", rel_pts.shape)
# print("Obj desc. Shape:", descriptor.shape)
# print("Rel label Shape:", gt_rel_label.shape)
# print("Obj label Shape:", gt_obj_label.shape)
# print("Edge index Shape:", edge_indices.shape)
# print("Batch idx Shape:", batch_ids.shape)