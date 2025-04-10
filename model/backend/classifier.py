import torch
import torch.nn as nn
import torch.nn.functional as F
from model.models.baseline import BaseNetwork
from einops import rearrange
from tqdm import tqdm
import clip

## TODO: Zero-Shot classifier
@torch.no_grad()
def consine_classification_obj(
    cls_matrix: torch.Tensor, # C X N_feat
    obj_feat: torch.Tensor   # B X N_feat
):
    _obj_feat = F.normalize(obj_feat, dim=-1)
    _cls_matrix = F.normalize(cls_matrix, dim=-1)
    sim_matrix = torch.mm(_obj_feat, _cls_matrix.T) / 0.1 # B X C
    obj_pred = F.softmax(sim_matrix, dim=1)
    return obj_pred

class RelCosineClassifier():
    def __init__(self, emb_vec, relation_cls, object_cls, device, d_feats):
        self.object_cls = object_cls
        self.relation_cls = relation_cls
        self.device = device
        self.encoder, _ = clip.load("ViT-B/32", device=device)
        self.embedding_vector_loader = emb_vec

    @torch.no_grad()
    def __call__(self, 
        edge_feat: torch.Tensor, # B_e X N_feat
        obj_pred: torch.Tensor, # B_o X N_feat
        edge_indices: torch.Tensor, # B_e X 2
    ):
        assert edge_feat.ndim == 2
        obj_pred_cls = torch.argmax(obj_pred, dim=1) # B_o X 1
        rel_text_feat = [] # B_e X N_rel_cls X N_feat
        for idx in range(len(edge_indices)):
            idx_eo = edge_indices[idx][0]
            idx_os = edge_indices[idx][1]
            sub_index = obj_pred_cls[idx_eo].int().item()
            obj_index = obj_pred_cls[idx_os].int().item()
            cls_mat = self.embedding_vector_loader[sub_index, obj_index, ...]
            rel_text_feat.append(cls_mat.unsqueeze(0)) # 1 X N_rel_cls X N_feat
        rel_feat_cls = torch.vstack(rel_text_feat) # B_e X N_rel_cls X N_feat
        
        edge_feat = F.normalize(edge_feat, dim=-1)
        rel_feat_cls = F.normalize(rel_feat_cls, dim=-1)
        edge_pred = torch.einsum('bn,bcn->bc', edge_feat, rel_feat_cls) / 0.1
        return torch.sigmoid(edge_pred) # B_e X N_rel_cls

class ObjectClsMulti(BaseNetwork):
    def __init__(self, k, in_size, batch_norm=True, drop_out=True, init_weights=True):
        super(ObjectClsMulti, self).__init__()
        self.name = 'pnetcls_obj'
        self.in_size=in_size
        self.use_bn = batch_norm
        self.use_drop_out = drop_out
        self.fc1 = nn.Linear(in_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)

        if self.use_drop_out:
            self.dropout = nn.Dropout(p=0.3)
        if self.use_bn:
            self.bn1 = nn.BatchNorm1d(512)
            self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

        if init_weights:
            self.init_weights('constant', 1, target_op = 'BatchNorm')
            self.init_weights('xavier_normal', 1)
            
    def forward(self, x):
        x = self.fc1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.use_drop_out:
            x = self.dropout(x)
        if self.use_bn:
            x = self.bn2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

class RelationClsMulti(BaseNetwork):

    def __init__(self, k=2, in_size=1024, batch_norm = True, drop_out = True,
                 init_weights=True):
        super(RelationClsMulti, self).__init__()
        self.name = 'pnetcls_rel'
        self.in_size=in_size
        self.use_bn = batch_norm
        self.use_drop_out = drop_out
        self.fc1 = nn.Linear(in_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)

        if self.use_drop_out:
            self.dropout = nn.Dropout(p=0.3)
        if self.use_bn:
            self.bn1 = nn.BatchNorm1d(512)
            self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

        if init_weights:
            self.init_weights('constant', 1, target_op = 'BatchNorm')
            self.init_weights('xavier_normal', 1)
    
    def forward(self, x):
        x = self.fc1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.use_drop_out:
            x = self.dropout(x)
        if self.use_bn:
            x = self.bn2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return torch.sigmoid(x)