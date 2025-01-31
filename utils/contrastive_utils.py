from config.define import *
from utils.os_utils import read_txt_to_list
import torch
import numpy as np
import clip
import random

class ContrastiveSingleLabelSampler():
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.d_config = config.dataset
        self.obj_label_path = f"{SSG_DATA_PATH}/3DSSG_subset/classes.txt"
        self.rel_label_path = f"{SSG_DATA_PATH}/3DSSG_subset/relationships.txt"
        self.__read_cls()
        self.text_encoder, _ = clip.load("ViT-B/32", device=device)
        
        self.predicate_cat = PREDICATE_CATEGORY.keys()
        self.negative_index = { i: [] for i in range(27)}
        for p_i in range(27):
            for _, v in PREDICATE_CATEGORY.items():
                if (p_i in v) and ((not p_i == 0) or (not p_i == 24)): 
                    self.negative_index[p_i].extend([ x for x in v if not x == p_i ])
                elif (p_i == 0) or (p_i == 24):
                    self.negative_index[p_i].extend(v)
                else:
                    continue
        # print(self.negative_index)
    
    def __read_cls(self):        
        self.obj_label_list = read_txt_to_list(self.obj_label_path)
        self.rel_label_list = read_txt_to_list(self.rel_label_path)
    
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
