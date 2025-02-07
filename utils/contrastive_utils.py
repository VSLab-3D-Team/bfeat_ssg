from abc import ABC, abstractmethod
from config.define import *
from utils.os_utils import read_txt_to_list
from utils.data_utils import compute
from operator import itemgetter
import torch.nn.functional as F
import torch
import numpy as np
import clip
import random
import json

def compute_frequnecy_weight(obj_label_list, rel_label_list, data, selected_scans, multi_rel, device):
    _, _, o_obj_cls, o_rel_cls = compute(obj_label_list, rel_label_list, data, selected_scans, False)
        
    w_cls_obj = torch.from_numpy(np.array(o_obj_cls)).float().to(device)
    w_cls_rel = torch.from_numpy(np.array(o_rel_cls)).float().to(device)
    # for single relation output, we set 'None' relationship weight as 1e-3
    if not multi_rel:
        w_cls_rel[0] = w_cls_rel.max() * 10
    
    w_cls_obj = w_cls_obj.sum() / (w_cls_obj + 1) / w_cls_obj.sum()
    w_cls_rel = w_cls_rel.sum() / (w_cls_rel + 1) / w_cls_rel.sum()
    w_cls_obj = w_cls_obj / w_cls_obj.max()
    w_cls_rel = w_cls_rel / w_cls_rel.max()
    
    return w_cls_obj, w_cls_rel

class ContrastiveAbstractSampler(ABC):
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.d_config = config.dataset
        self.num_neg_samples = config.train.num_negative_sample
        self.obj_label_path = f"{SSG_DATA_PATH}/3DSSG_subset/classes.txt"
        self.rel_label_path = f"{SSG_DATA_PATH}/3DSSG_subset/relationships.txt"
        self.__read_cls()
        self.text_encoder, _ = clip.load("ViT-B/32", device=device)
        
        ## Get predicate category
        self.predicate_cat = PREDICATE_CATEGORY.keys()
        self.negative_index = { i: [] for i in range(26) }
        for p_i in range(26):
            for k, v in PREDICATE_CATEGORY.items():
                if k == "none":
                    continue
                if (p_i + 1 in v) and (not p_i == 23): 
                    self.negative_index[p_i].extend([ x - 1 for x in v if not x == (p_i + 1) ])
                elif (p_i == 23) and (not k == 'cover'):
                    self.negative_index[p_i].extend([ x - 1 for x in v ])
    
    def __read_cls(self):
        self.obj_label_list = read_txt_to_list(self.obj_label_path)
        self.rel_label_list = read_txt_to_list(self.rel_label_path)
        if self.d_config.multi_rel:
            self.rel_label_list.pop(0)

    @abstractmethod
    def sample(self):
        raise NotImplementedError
    

class ContrastiveSingleLabelSampler(ContrastiveAbstractSampler):
    def __init__(self, config, device):
        super().__init__(config, device)
    
    def __get_negative(self, gt_rel_index):
        return random.choice(self.negative_index[gt_rel_index])
    
    def sample(self, objs_target, rels_target, edges):
        target_pos_token, target_neg_token = [], []
        rel_index =  []
        for edge_index in range(len(edges)):
            idx_eo = edges[edge_index][0]
            idx_os = edges[edge_index][1]
            target_eo = self.obj_label_list[objs_target[idx_eo]]
            target_os = self.obj_label_list[objs_target[idx_os]]
            assert rels_target.ndim == 2
            if rels_target[edge_index].sum() == 0:
                pos_token = clip.tokenize(f"the {target_eo} and the {target_os} has no relation in the point cloud")
                target_pos_token.append(pos_token)
                
                neg_predicate = self.rel_label_list[self.__get_negative(0)]
                neg_token = clip.tokenize(f"a point cloud of a {target_eo} {neg_predicate} a {target_os}")
                target_neg_token.append(neg_token)
                
                rel_index.append(edge_index)
            else:
                # print(rels_target.shape[-1])
                for i in range(rels_target.shape[-1]):
                    if rels_target[edge_index][i] == 1:
                        target_rel = self.rel_label_list[i]
                        pos_token = clip.tokenize(f"a point cloud of a {target_eo} {target_rel} a {target_os}")
                        target_pos_token.append(pos_token)
                        
                        neg_predicate = self.rel_label_list[self.__get_negative(i)]
                        neg_token = clip.tokenize(f"a point cloud of a {target_eo} {neg_predicate} a {target_os}")
                        target_neg_token.append(neg_token)
                        
                        rel_index.append(edge_index)

        prompt_pos_features = torch.cat(target_pos_token).to(self.device)
        prompt_neg_features = torch.cat(target_neg_token).to(self.device)
        with torch.no_grad():
            triplet_pos_feats = self.text_encoder.encode_text(prompt_pos_features)
            triplet_neg_feats = self.text_encoder.encode_text(prompt_neg_features)
        
        target_pos_rel_feats, target_neg_rel_feats = [], []
        # deal with multi label
        for edge_index in range(len(edges)):
            multi_rel_idxs = torch.where(torch.tensor(rel_index) == edge_index)[0]
            target_pos_rel_feats.append(triplet_pos_feats[multi_rel_idxs].reshape(len(multi_rel_idxs), 512).mean(0))
            target_neg_rel_feats.append(triplet_neg_feats[multi_rel_idxs].reshape(len(multi_rel_idxs), 512).mean(0))
        
        assert len(target_pos_rel_feats) ==  len(edges)
        assert len(target_neg_rel_feats) ==  len(edges)
        p_target_rel_feats = torch.vstack(target_pos_rel_feats)
        p_target_rel_feats = p_target_rel_feats / p_target_rel_feats.norm(dim=-1, keepdim=True)
        n_target_rel_feats = torch.vstack(target_neg_rel_feats)
        n_target_rel_feats = n_target_rel_feats / n_target_rel_feats.norm(dim=-1, keepdim=True)
        
        return p_target_rel_feats.float(), n_target_rel_feats.float()


class ContrastiveFreqWeightedSampler(ContrastiveAbstractSampler):
    """
    Naive negative sampler w. frequency weighted multinomial distriubuton
    wo/ hard negative sampling (hard-negative sampling with same category)
    It can make unseen triplets as negative samples
    """
    def __init__(self, config, device):
        super().__init__(config, device)
        self.t_config = config.train
        data_path = f"{SSG_DATA_PATH}/3DSSG_subset"
        path_selection = f"{SSG_DATA_PATH}/3DSSG_subset"
        selected_scans = set()
        selected_scans = selected_scans.union(read_txt_to_list(os.path.join(path_selection,'train_scans.txt')))
        with open(os.path.join(data_path, 'relationships_train.json'), "r") as read_file:
            data = json.load(read_file)
        self.w_cls_obj, self.w_cls_rel = compute_frequnecy_weight(
            self.obj_label_list, 
            self.rel_label_list, 
            data, selected_scans, 
            self.d_config.multi_rel, device
        )
        self.__make_freq_prob_dist()
        
    def __make_freq_prob_dist(self):
        f_temperature = self.t_config.freq_temperature
        self.prob_obj_sample = F.softmax(self.w_cls_obj / f_temperature, dim=0)
        self.prob_rel_sample = F.softmax(self.w_cls_rel / f_temperature, dim=0)
    
    def __sample_negative_labels(self, anchor_idx):
        sample_dist = self.prob_rel_sample.clone()
        if not anchor_idx == -1: # If anchor is not none
            sample_dist[anchor_idx] = 0.
            sample_dist = sample_dist / sample_dist.sum()
        sample_indices = torch.multinomial(sample_dist, self.num_neg_samples, replacement=False).cpu().tolist()
        return itemgetter(*sample_indices)(self.rel_label_list)
    
    def sample(self, objs_target, rels_target, edges):
        """
        Inputs: 
            - objs_target: N X N_obj_cls
            - rels_target: N X N_rel_cls
            - edges: N X 2 
        Outputs:
        For multi-relaitonship, 
            - pos_target_rel_feats: M X N_feats
            - neg_target_rel_feats: M X N_neg X N_feats
            - rel_index, Relationship Index for G.T Labels: M X 1 \in [0, N-1]
        """
        # target_pos_token, target_neg_token = [], []
        # target_pos_feats: N X N_feats
        # target_neg_feats: N X N_neg X N_feats
        target_pos_feats, target_neg_feats = [], []
        rel_index = []
        for edge_index in range(len(edges)):
            idx_eo = edges[edge_index][0]
            idx_os = edges[edge_index][1]
            target_eo = self.obj_label_list[objs_target[idx_eo]]
            target_os = self.obj_label_list[objs_target[idx_os]]
            assert rels_target.ndim == 2
            with torch.no_grad():
                if rels_target[edge_index].sum() == 0:
                    # relationship = 'none'
                    pos_token = clip.tokenize(f"the {target_eo} and the {target_os} has no relation in the point cloud").to(self.device)
                    target_pos_feats.append(self.text_encoder.encode_text(pos_token))
                    
                    neg_samples = self.__sample_negative_labels(-1)
                    neg_tokens = clip.tokenize([ f"a point cloud of a {target_eo} {n_p} a {target_os}" for n_p in neg_samples ]).to(self.device)
                    target_neg_feats.append(self.text_encoder.encode_text(neg_tokens).unsqueeze(0))
                    rel_index.append(edge_index)
                else:
                    for i in range(rels_target.shape[-1]):
                        if rels_target[edge_index][i] == 1:
                            pos_rel = self.rel_label_list[i]
                            pos_token = clip.tokenize(f"a point cloud of a {target_eo} {pos_rel} a {target_os}").to(self.device)
                            target_pos_feats.append(self.text_encoder.encode_text(pos_token))
                            
                            neg_samples = self.__sample_negative_labels(i)
                            neg_tokens = clip.tokenize([ f"a point cloud of a {target_eo} {n_p} a {target_os}" for n_p in neg_samples ]).to(self.device)
                            target_neg_feats.append(self.text_encoder.encode_text(neg_tokens).unsqueeze(0)) # 1 X N_neg X N_feat
                            rel_index.append(edge_index)
        
        p_target_rel_feats = torch.vstack(target_pos_feats)
        n_target_rel_feats = torch.vstack(target_neg_feats)
        return p_target_rel_feats.float(), n_target_rel_feats.float(), torch.Tensor(rel_index).reshape(-1, 1).to(self.device).int()

class ContrastiveHybridTripletSampler(ContrastiveAbstractSampler):
    """
    Triplet based Hard negative sampler w. frequency weighted multinomial distriubuton
    Hard negative sampling in same category of anchor
    Since there are small numbers of labels in predicate categories, we decided to sample negative samples w. objects 
    """
    def __init__(self, config, device):
        super().__init__(config, device)
        data_path = f"{SSG_DATA_PATH}/3DSSG_subset"
        path_selection = f"{SSG_DATA_PATH}/3DSSG_subset"
        selected_scans = set()
        selected_scans = selected_scans.union(read_txt_to_list(os.path.join(path_selection,'train_scans.txt')))
        with open(os.path.join(data_path, 'relationships_train.json'), "r") as read_file:
            data = json.load(read_file)
        self.w_cls_obj, self.w_cls_rel = compute_frequnecy_weight(
            self.obj_label_list, 
            self.rel_label_list, 
            data, selected_scans, 
            self.d_config.multi_rel, device
        )
        self.__make_freq_prob_dist()
        
    def __make_freq_prob_dist(self):
        f_temperature = self.t_config.freq_temperature
        self.prob_obj_sample = F.softmax(self.w_cls_obj / f_temperature)
        self.prob_rel_sample = F.softmax(self.w_cls_rel / f_temperature)
    
    def __sample_neg_predicate(self, anchor_idx):
        sample_dist = self.prob_rel_sample.clone()
        if not anchor_idx == -1: # If anchor is not none
            sample_dist[anchor_idx] = 0.
            sample_dist = sample_dist / sample_dist.sum()
        sample_indices = torch.multinomial(sample_dist, self.num_neg_samples, replacement=False).cpu().tolist()
        return itemgetter(*sample_indices)(self.rel_label_list)
    
    def __sample_neg_object(self, anchor_idx):
        sample_dist = self.prob_obj_sample.clone()
        sample_dist[anchor_idx] = 0.
        sample_dist = sample_dist / sample_dist.sum()
        sample_indices = torch.multinomial(sample_dist, self.num_neg_samples, replacement=False).cpu().tolist()
        return itemgetter(*sample_indices)(self.obj_label_list)
    
    def sample(self, objs_target, rels_target, edges):
        """
        Inputs: 
            - objs_target: N X N_obj_cls
            - rels_target: N X N_rel_cls
            - edges: N X 2 
        Outputs:
        For multi-relaitonship, 
            - pos_target_rel_feats: M X N_feats
            - neg_target_rel_feats: M X N_neg X N_feats
            - rel_index, Relationship Index for G.T Labels: M X 1 \in [0, N-1]
        """
        # target_pos_token, target_neg_token = [], []
        # target_pos_feats: N X N_feats
        # target_neg_feats: N X N_neg X N_feats
        target_pos_feats, target_neg_feats = [], []
        rel_index = []
        for edge_index in range(len(edges)):
            idx_eo = edges[edge_index][0]
            idx_os = edges[edge_index][1]
            target_eo = self.obj_label_list[objs_target[idx_eo]]
            target_os = self.obj_label_list[objs_target[idx_os]]
            assert rels_target.ndim == 2
            with torch.no_grad():
                if rels_target[edge_index].sum() == 0:
                    # relationship = 'none'
                    pos_token = clip.tokenize(f"the {target_eo} and the {target_os} has no relation in the point cloud")
                    target_pos_feats.append(self.text_encoder.encode_text(pos_token))
                    
                    neg_samples = self.__sample_negative_labels(-1)
                    neg_tokens = clip.tokenize([ f"a point cloud of a {target_eo} {n_p} a {target_os}" for n_p in neg_samples ])
                    target_neg_feats.append(self.text_encoder.encode_text(neg_tokens).unsqueeze(0))
                    rel_index.append(edge_index)
                else:
                    for i in range(rels_target.shape[-1]):
                        if rels_target[edge_index][i] == 1:
                            pos_rel = self.rel_label_list[i]
                            pos_token = clip.tokenize(f"a point cloud of a {target_eo} {pos_rel} a {target_os}")
                            target_pos_feats.append(self.text_encoder(pos_token))
                            
                            neg_samples = self.__sample_negative_labels(i)
                            neg_tokens = clip.tokenize([ f"a point cloud of a {target_eo} {n_p} a {target_os}" for n_p in neg_samples ])
                            target_neg_feats.append(self.text_encoder.encode_text(neg_tokens).unsqueeze(0)) # 1 X N_neg X N_feat
                            rel_index.append(edge_index)
        
        p_target_rel_feats = torch.vstack(target_pos_feats)
        n_target_rel_feats = torch.vstack(target_neg_feats)
        return p_target_rel_feats.float(), n_target_rel_feats.float(), torch.Tensor(rel_index).reshape(-1, 1)
    