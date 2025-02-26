from abc import ABC, abstractmethod
from config.define import *
from utils.os_utils import read_txt_to_list
from utils.data_utils import compute
from utils.replay_buffer import Replay_Buffer
from operator import itemgetter
from einops import rearrange
import torch.nn.functional as F
import torch
import numpy as np
import clip
import random
import json
from tqdm import tqdm

def compute_frequnecy_weight(obj_label_list, rel_label_list, data, selected_scans, multi_rel, device):
    _, _, o_obj_cls, o_rel_cls = compute(obj_label_list, rel_label_list, data, selected_scans, False)
        
    w_cls_obj = torch.from_numpy(np.array(o_obj_cls)).float().to(device)
    w_cls_rel = torch.from_numpy(np.array(o_rel_cls)).float().to(device)
    # for single relation output, we set 'None' relationship weight as 1e-3
    if not multi_rel:
        w_cls_rel[0] = w_cls_rel.max() * 10
    
    w_cls_obj = (w_cls_obj / w_cls_obj.max()) * 10
    w_cls_rel = (w_cls_rel / w_cls_rel.max()) * 10    
    w_cls_obj = 1. / (w_cls_obj + 1)
    w_cls_rel = 1. / (w_cls_rel + 1)
    w_cls_rel[6] = -1 # There is no value for predicate "inside", so hard coded value will be inserted.
    
    return w_cls_obj, w_cls_rel

class ContrastiveAbstractSampler(ABC):
    def __init__(self, embedding_vector_loader, none_emb, config, device):
        self.config = config
        self.device = device
        self.d_config = config.dataset
        self.m_config = config.model
        self.num_neg_samples = config.train.num_negative_sample
        self.obj_label_path = f"{SSG_DATA_PATH}/3DSSG_subset/classes.txt"
        self.rel_label_path = f"{SSG_DATA_PATH}/3DSSG_subset/relationships.txt"
        self.__read_cls()
        self.text_encoder, _ = clip.load("ViT-B/32", device=device)
        self.embedding_vector_loader = embedding_vector_loader
        self.none_emb = none_emb
        
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
    def __init__(self, embedding_vector_loader, none_emb, config, device):
        super().__init__(embedding_vector_loader, none_emb, config, device)
    
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
    def __init__(self, embedding_vector_loader, none_emb, config, device):
        super().__init__(embedding_vector_loader, none_emb, config, device)
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
        print("Predicate Contrastive Sampler Distribution: ", self.prob_obj_sample)
        self.prob_rel_sample = F.softmax(self.w_cls_rel / f_temperature, dim=0)
    
    def __sample_negative_labels(self, anchor_idx):
        sample_dist = self.prob_rel_sample.clone()
        if not anchor_idx == -1: # If anchor is not none
            sample_dist[anchor_idx] = 0.
            sample_dist = sample_dist / sample_dist.sum()
        sample_indices = torch.multinomial(sample_dist, self.num_neg_samples, replacement=False)
        # s_list = np.array(self.rel_label_list)[sample_indices.cpu().numpy()]
        return sample_indices
    
    @torch.no_grad()
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
        # target_pos_feats: N X N_feats
        # target_neg_feats: N X N_neg X N_feats
        target_pos_token, target_neg_token = [], []
        
        rel_index = []
        for edge_index in range(len(edges)):
            idx_eo = edges[edge_index][0]
            idx_os = edges[edge_index][1]
            target_sub_idx = objs_target[idx_eo]
            target_obj_idx = objs_target[idx_os]
            assert rels_target.ndim == 2
            
            if rels_target[edge_index].sum() == 0:
                # relationship = 'none'
                pos_feat = self.none_emb[target_sub_idx, target_obj_idx, :]
                target_pos_token.append(pos_feat.unsqueeze(0)) # 1 X N_t
                # target_pos_feats.append(self.text_encoder.encode_text(pos_token))
                
                neg_samples_idx = self.__sample_negative_labels(-1)
                neg_feats = self.embedding_vector_loader[target_sub_idx, target_obj_idx, neg_samples_idx, :]
                target_neg_token.append(neg_feats.unsqueeze(0)) # 1 X N_neg X N_t
                # target_neg_feats.append(self.text_encoder.encode_text(neg_tokens).unsqueeze(0))
                rel_index.append(edge_index)
            else:
                for i in range(rels_target.shape[-1]):
                    if rels_target[edge_index][i] == 1:
                        pos_rel = self.rel_label_list[i]
                        pos_feat = self.embedding_vector_loader[target_sub_idx, target_obj_idx, i, :]
                        target_pos_token.append(pos_feat.unsqueeze(0)) # 1 X N_t
                        # target_pos_feats.append(self.text_encoder.encode_text(pos_token))
                        
                        neg_samples_idx = self.__sample_negative_labels(i)
                        neg_feats = self.embedding_vector_loader[target_sub_idx, target_obj_idx, neg_samples_idx, :]
                        target_neg_token.append(neg_feats.unsqueeze(0)) # 1 X N_neg X N_t
                        # target_neg_feats.append(self.text_encoder.encode_text(neg_tokens).unsqueeze(0)) # 1 X N_neg X N_feat
                        rel_index.append(edge_index)
    
        p_target_tokens = torch.vstack(target_pos_token).to(self.device) # M X N_t
        n_target_tokens = torch.vstack(target_neg_token).to(self.device) # M X N_neg X N_t
        return p_target_tokens.float(), n_target_tokens.float(), torch.Tensor(rel_index).reshape(-1, 1).to(self.device)

class ContrastiveTripletSampler(ContrastiveAbstractSampler):
    """
    Triplet based Hard negative sampler w. frequency weighted multinomial distriubuton
    Hard negative sampling in same category of anchor
    Since there are small numbers of labels in predicate categories, we decided to sample negative samples w. objects 
    """
    def __init__(self, embedding_vector_loader, none_emb, config, device):
        super().__init__(embedding_vector_loader, none_emb, config, device)
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
        assert self.num_neg_samples % 3 == 0, "# of Negative sample must be divided into 3"
        self.num_negs_per_type = self.num_neg_samples // 3
        
    def __make_freq_prob_dist(self):
        f_temperature = self.t_config.freq_temperature
        self.prob_obj_sample = F.softmax(self.w_cls_obj / f_temperature, dim=0)
        self.prob_rel_sample = F.softmax(self.w_cls_rel / f_temperature, dim=0)
    
    def __sample_neg_predicate(self, anchor_idx):
        if not anchor_idx == -1: # If anchor is not none
            sample_dist = self.prob_rel_sample.clone()
            sample_dist[anchor_idx] = 0.
            sample_dist = sample_dist / sample_dist.sum()
        else:
            sample_dist = self.prob_rel_sample.clone()
        sample_indices = torch.multinomial(sample_dist, self.num_negs_per_type, replacement=False)
        # s_list = np.array(self.rel_label_list)[sample_indices.cpu().numpy()]
        return sample_indices
    
    def __sample_neg_object(self, sub_anchor_idx, obj_anchor_idx):
        sample_dist = self.prob_obj_sample.clone()
        sample_dist[sub_anchor_idx] = 0.
        sample_dist[obj_anchor_idx] = 0.
        sample_dist = sample_dist / sample_dist.sum()
        sample_indices = torch.multinomial(sample_dist, 2 * self.num_negs_per_type, replacement=False)
        # s_list = np.array(self.obj_label_list)[sample_indices.cpu().numpy()]
        return sample_indices
    
    def __sample_neg_triplet(self, sub_anchor_idx, pred_anchor_idx, obj_anchor_idx):
        pred_neg_labels = self.__sample_neg_predicate(pred_anchor_idx)
        neg_labels = self.__sample_neg_object(sub_anchor_idx, obj_anchor_idx)
        obj_neg_labels = neg_labels[: self.num_negs_per_type]
        sub_neg_labels = neg_labels[self.num_negs_per_type: ]
        sub_neg_samples = self.embedding_vector_loader[sub_neg_labels, obj_anchor_idx, pred_anchor_idx, :].clone().detach() # N_neg X N_feat
        obj_neg_samples = self.embedding_vector_loader[sub_anchor_idx, obj_neg_labels, pred_anchor_idx, :].clone().detach() # N_neg X N_feat
        pred_neg_samples = self.embedding_vector_loader[sub_anchor_idx, obj_anchor_idx, pred_neg_labels, :].clone().detach() # N_neg X N_feat
        return torch.vstack([sub_neg_samples, obj_neg_samples, pred_neg_samples])
    
    @torch.no_grad()
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
        target_pos_token, target_neg_token = [], []
        rel_index = []
        for edge_index in range(len(edges)):
            idx_eo = edges[edge_index][0]
            idx_os = edges[edge_index][1]
            target_sub_idx = objs_target[idx_eo]
            target_obj_idx = objs_target[idx_os]
            assert rels_target.ndim == 2
            if rels_target[edge_index].sum() == 0:
                # relationship = 'none'
                pos_feat = self.none_emb[target_sub_idx, target_obj_idx, :]
                target_pos_token.append(pos_feat.unsqueeze(0))
                
                neg_feats = self.__sample_neg_triplet(target_sub_idx, -1, target_obj_idx)
                target_neg_token.append(neg_feats.unsqueeze(0))
                rel_index.append(edge_index)
            else:
                for i in range(rels_target.shape[-1]):
                    if rels_target[edge_index][i] == 1:
                        pos_feat = self.embedding_vector_loader[target_sub_idx, target_obj_idx, i, :]
                        target_pos_token.append(pos_feat.unsqueeze(0))
                        
                        neg_feats = self.__sample_neg_triplet(objs_target[idx_eo], i, objs_target[idx_os])
                        target_neg_token.append(neg_feats.unsqueeze(0)) # 1 X N_neg X N_feat
                        rel_index.append(edge_index)
        
        p_target_tokens = torch.vstack(target_pos_token).to(self.device)
        n_target_tokens = torch.vstack(target_neg_token).to(self.device)
        return p_target_tokens.float(), n_target_tokens.float(), torch.Tensor(rel_index).reshape(-1, 1).to(self.device)

class ContrastiveHybridTripletSampler(ContrastiveAbstractSampler):
    """
    Triplet based Hard negative sampler w. frequency weighted multinomial distriubuton
    Hard negative sampling in same category of anchor
    Since there are small numbers of labels in predicate categories, we decided to sample negative samples w. objects 
    """
    def __init__(self, embedding_vector_loader, none_emb, config, device):
        super().__init__(embedding_vector_loader, none_emb, config, device)
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
        assert self.num_neg_samples % 3 == 0, "# of Negative sample must be divided into 3"
        self.num_negs_per_type = self.num_neg_samples // 3
        
    def __make_freq_prob_dist(self):
        f_temperature = self.t_config.freq_temperature
        self.prob_obj_sample = F.softmax(self.w_cls_obj / f_temperature, dim=0)
        self.prob_rel_sample = F.softmax(self.w_cls_rel / f_temperature, dim=0)
    
    def __sample_neg_predicate(self, anchor_idx):
        if not anchor_idx == -1: # If anchor is not none
            f_temperature = self.t_config.freq_temperature
            pred_category_index = self.negative_index[anchor_idx]
            sample_dist = F.softmax(self.w_cls_rel[pred_category_index] / f_temperature, dim=0)
        else:
            sample_dist = self.prob_rel_sample.clone()
        sample_indices = torch.multinomial(sample_dist, self.num_negs_per_type, replacement=False)
        # s_list = np.array(self.rel_label_list)[sample_indices.cpu().numpy()]
        return sample_indices
    
    def __sample_neg_object(self, sub_anchor_idx, obj_anchor_idx):
        sample_dist = self.prob_obj_sample.clone()
        sample_dist[sub_anchor_idx] = 0.
        sample_dist[obj_anchor_idx] = 0.
        sample_dist = sample_dist / sample_dist.sum()
        sample_indices = torch.multinomial(sample_dist, 2 * self.num_negs_per_type, replacement=False)
        # s_list = np.array(self.obj_label_list)[sample_indices.cpu().numpy()]
        return sample_indices
    
    def __sample_neg_triplet(self, sub_anchor_idx, pred_anchor_idx, obj_anchor_idx):
        pred_neg_labels = self.__sample_neg_predicate(pred_anchor_idx)
        neg_labels = self.__sample_neg_object(sub_anchor_idx, obj_anchor_idx)
        obj_neg_labels = neg_labels[: self.num_negs_per_type]
        sub_neg_labels = neg_labels[self.num_negs_per_type: ]
        sub_neg_samples = self.embedding_vector_loader[sub_neg_labels, obj_anchor_idx, pred_anchor_idx, :].clone().detach() # N_neg X N_feat
        obj_neg_samples = self.embedding_vector_loader[sub_anchor_idx, obj_neg_labels, pred_anchor_idx, :].clone().detach() # N_neg X N_feat
        pred_neg_samples = self.embedding_vector_loader[sub_anchor_idx, obj_anchor_idx, pred_neg_labels, :].clone().detach() # N_neg X N_feat
        return torch.vstack([sub_neg_samples, obj_neg_samples, pred_neg_samples])
    
    @torch.no_grad()
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
        target_pos_token, target_neg_token = [], []
        rel_index = []
        for edge_index in range(len(edges)):
            idx_eo = edges[edge_index][0]
            idx_os = edges[edge_index][1]
            target_sub_idx = objs_target[idx_eo]
            target_obj_idx = objs_target[idx_os]
            assert rels_target.ndim == 2
            if rels_target[edge_index].sum() == 0:
                # relationship = 'none'
                pos_feat = self.none_emb[target_sub_idx, target_obj_idx, :]
                target_pos_token.append(pos_feat.unsqueeze(0))
                
                neg_feats = self.__sample_neg_triplet(target_sub_idx, -1, target_obj_idx)
                target_neg_token.append(neg_feats.unsqueeze(0))
                rel_index.append(edge_index)
            else:
                for i in range(rels_target.shape[-1]):
                    if rels_target[edge_index][i] == 1:
                        pos_feat = self.embedding_vector_loader[target_sub_idx, target_obj_idx, i, :]
                        target_pos_token.append(pos_feat.unsqueeze(0))
                        
                        neg_feats = self.__sample_neg_triplet(objs_target[idx_eo], i, objs_target[idx_os])
                        target_neg_token.append(neg_feats.unsqueeze(0)) # 1 X N_neg X N_feat
                        rel_index.append(edge_index)
        
        p_target_tokens = torch.vstack(target_pos_token).to(self.device)
        n_target_tokens = torch.vstack(target_neg_token).to(self.device)
        return p_target_tokens.float(), n_target_tokens.float(), torch.Tensor(rel_index).reshape(-1, 1).to(self.device)

class ContrastiveReplayBufferSampler(ContrastiveAbstractSampler):
    """
    Naive negative sampler w. frequency weighted multinomial distriubuton w replaybuffer
    wo/ hard negative sampling (hard-negative sampling with same category)
    It can make unseen triplets as negative samples
    """
    def __init__(self, embedding_vector_loader, none_emb, config, device):
        super().__init__(embedding_vector_loader, none_emb, config, device)
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
        
        self.replay_buffer=Replay_Buffer(1000000, "SET", True)
        
    def __make_freq_prob_dist(self):
        f_temperature = self.t_config.freq_temperature
        self.prob_obj_sample = F.softmax(self.w_cls_obj / f_temperature, dim=0)
        self.prob_rel_sample = F.softmax(self.w_cls_rel / f_temperature, dim=0)

    @torch.no_grad()
    def crazy_negative_embedding(self, token_vecs: torch.Tensor):
        """
        Embrace the bullshit.
        GPU is too expensive.
        FXXK YOU NVIDIA
        """
        target_feats = []
        num_seqs = token_vecs.shape[1]
        for n_i in range(num_seqs):
            t_tokens = token_vecs[:, n_i, :] # N_obj_cls X N_token
            target_feats.append(self.text_encoder.encode_text(t_tokens).float().unsqueeze(1))
        return torch.cat(target_feats, dim=1) # N_obj_cls X N_rel_cls X N_token        

    def __sample_negative_labels(self, sub_anchor_idx, pred_anchor_idx, obj_anchor_idx):
        neg_samples=[]
        buffer_neg_labels=[]
        num_buffer_sample=0
        
        sample_dist = self.prob_rel_sample.clone()
        if not pred_anchor_idx == -1: # If anchor is not none
            sample_dist[pred_anchor_idx] = 0.
            sample_dist = sample_dist / sample_dist.sum()
            
            num_buffer_sample=min(self.replay_buffer.Get_Sample_length((sub_anchor_idx,pred_anchor_idx,obj_anchor_idx)),self.num_neg_samples)
            buffer_neg_labels = [self.replay_buffer.Get_Sample((sub_anchor_idx,pred_anchor_idx,obj_anchor_idx)) for _ in range(num_buffer_sample)]
            
        sample_indices = torch.multinomial(sample_dist, self.num_neg_samples-num_buffer_sample, replacement=False)
        s_list = np.array(self.rel_label_list)[sample_indices.cpu().numpy()]
        s_list.tolist()
        s_list=[(sub_anchor_idx, pred_neg_idx, obj_anchor_idx) for pred_neg_idx in s_list]
        
        neg_samples=[
            *s_list,
            *buffer_neg_labels
        ]
        return neg_samples
    
    def __get_neg(self, objs_target, rels_target, edges, gt_edges, num_samples):
        #현재는 각 obj와 sub의 두개 모두 예측이 실패한 경우를 neg sample로 삼는다
        #예측 자체는 성공하더라도 정답 triplet을 제외하고 가장 가능성 높게나온 triplet 케이스를 샘플로 삼는다.
        #일단은 predicate는 맞고 sub, obj 두개 모두 틀린경우를 neg 샘플로 뽑는다.
        idx_list=[(i,j) for i in range(min(num_samples+1,len(objs_target)))  for j in range(min(num_samples+1,len(objs_target)))]
        neg_edges = []
        obj_conf = torch.argmax(objs_target, dim=-1, keepdim=True) # N_node X 1
        obj_sorted_idx = torch.argsort(obj_conf, dim=1, descending=True) # N_node X 1
        for edge_index in range(len(edges)):
            target_eo=[]
            target_os=[]
            target_rel=[]
            
            gt_sub=gt_edges[edge_index][0]
            gt_obj=gt_edges[edge_index][1]
            
            idx_list.sort(
                key = lambda x : obj_conf[obj_sorted_idx[x[0]]].item() + obj_conf[obj_sorted_idx[x[1]]].item(),
                reverse=True
            )
                        
            cnt=0
            for i,j in idx_list:
                target_sub_idx = obj_sorted_idx[i]
                target_obj_idx = obj_sorted_idx[j]
                if (target_sub_idx != gt_sub) and (target_obj_idx != gt_obj):
                    target_eo.append(target_sub_idx)
                    target_os.append(target_obj_idx)
                    cnt+=1
                if cnt>=num_samples:
                    break
            target_rel = rels_target[edge_index]
            
            neg_edges.append((target_eo, target_os, target_rel))
        return neg_edges
    
    def Add_sample_to_buffer(self, obj_pred, rel_pred, edges, gt_edges, num_samples):
        """
        Inputs: 
            - objs_target: N X N_obj_cls
            - obj_pred: N X N_obj_cls
            - rels_target: N X N_rel_cls
            - rel_pred: N X N_rel_cls
            - edges: N X 2 
        Outputs:
        None
        """
        neg_edges = self.__get_neg(obj_pred, rel_pred, edges, gt_edges, num_samples) # E X ([sub,...], [obj,..], [pred,])
        for edge_index in range(len(gt_edges)):
            sub_target_idx = gt_edges[edge_index][0] # 1
            pred_target_idx_list = gt_edges[edge_index][2] # N_gt_pred
            obj_traget_idx = gt_edges[edge_index][1] # 1
            
            sub_neg_idx_list = neg_edges[edge_index][0] # N_add_sample
            pred_neg_idx_list = neg_edges[edge_index][2] # N_gt_pred
            obj_neg_idx_list = neg_edges[edge_index][1] # N_add_sample
            assert len(sub_neg_idx_list)==len(obj_neg_idx_list)
            
            for pred_idx in range(len(pred_target_idx_list)):
                target_triplet=(sub_target_idx, pred_target_idx_list[pred_idx], obj_traget_idx)
                for obj_neg_idx in range(len(obj_neg_idx_list)):
                    neg_triplet=(sub_neg_idx_list[obj_neg_idx], pred_target_idx_list[pred_idx], obj_neg_idx_list[obj_neg_idx])
                    self.replay_buffer.Put_Sample(target_triplet,neg_triplet)

    @torch.no_grad()
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
        # target_pos_feats: N X N_feats
        # target_neg_feats: N X N_neg X N_feats
        target_pos_token, target_neg_token = [], []
        
        rel_index = []
        for edge_index in range(len(edges)):
            idx_eo = edges[edge_index][0]
            idx_os = edges[edge_index][1]
            target_eo = self.obj_label_list[objs_target[idx_eo]]
            target_os = self.obj_label_list[objs_target[idx_os]]
            assert rels_target.ndim == 2
            
            if rels_target[edge_index].sum() == 0:
                # relationship = 'none'
                pos_token = clip.tokenize(f"the {target_eo} and the {target_os} has no relation in the point cloud").to(self.device)
                target_pos_token.append(pos_token) # 1 X N_t
                # target_pos_feats.append(self.text_encoder.encode_text(pos_token))
                
                neg_samples = self.__sample_negative_labels(idx_eo, -1, idx_os)
                neg_tokens = clip.tokenize([ f"a point cloud of a {target_sub} {target_pred} a {target_obj}" for target_sub,target_pred,target_obj in neg_samples ]).to(self.device)
                target_neg_token.append(neg_tokens.unsqueeze(0)) # 1 X N_neg X N_t
                # target_neg_feats.append(self.text_encoder.encode_text(neg_tokens).unsqueeze(0))
                rel_index.append(edge_index)
            else:
                for i in range(rels_target.shape[-1]):
                    if rels_target[edge_index][i] == 1:
                        pos_rel = self.rel_label_list[i]
                        pos_token = clip.tokenize(f"a point cloud of a {target_eo} {pos_rel} a {target_os}").to(self.device)
                        target_pos_token.append(pos_token) # 1 X N_t
                        # target_pos_feats.append(self.text_encoder.encode_text(pos_token))
                        
                        neg_samples = self.__sample_negative_labels(idx_eo, i, idx_os)
                        neg_tokens = clip.tokenize([ f"a point cloud of a {target_sub} {target_pred} a {target_obj}" for target_sub,target_pred,target_obj in neg_samples ]).to(self.device)
                        target_neg_token.append(neg_tokens.unsqueeze(0)) # 1 X N_neg X N_t
                        # target_neg_feats.append(self.text_encoder.encode_text(neg_tokens).unsqueeze(0)) # 1 X N_neg X N_feat
                        rel_index.append(edge_index)
    
        p_target_tokens = torch.vstack(target_pos_token).to(self.device) # M X N_t
        n_target_tokens = torch.vstack(target_neg_token).to(self.device) # M X N_neg X N_t
        p_target_rel_feats = self.text_encoder.encode_text(p_target_tokens) # M X N_feats
        n_target_rel_feats = self.crazy_negative_embedding(n_target_tokens) # M X N_neg X N_feats
        return p_target_rel_feats.float(), n_target_rel_feats.float(), torch.Tensor(rel_index).reshape(-1, 1).to(self.device)