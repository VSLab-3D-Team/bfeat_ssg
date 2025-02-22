from model.frontend.pointnet import PointNetEncoder
from model.backend.gat import BFeatVanillaGAT
from model.frontend.relextractor import *
from model.backend.classifier import RelationClsMulti, ObjectClsMulti
from model.backend.triplet import TripletContrastiveConvLayer, TripletContrastiveMLPLayer
from model.models.model_rel_only import BFeatRelObjConNet
from model.models.baseline import BaseNetwork
from utils.model_utils import Gen_Index
import torch

class BFeatTripletContrastvieGNN(BaseNetwork):
    def __init__(self, config, n_obj_cls, n_rel_cls, device):
        super(BFeatTripletContrastvieGNN, self).__init__()
        self.config = config
        self.t_config = config.train
        self.m_config = config.model
        self.dim_pts = 3
        if self.m_config.use_rgb:
            self.dim_pts += 3
        if self.m_config.use_normal:
            self.dim_pts += 3
        self.device = device
        
        self.feature_encoder = BFeatRelObjConNet(config, device)
        
        self.index_get = Gen_Index(flow=self.m_config.flow)
        self.gat = BFeatVanillaGAT(
            self.m_config.dim_obj_feats,
            self.m_config.dim_edge_feats,
            self.m_config.dim_attn,
            num_heads=self.m_config.num_heads,
            depth=self.m_config.num_graph_update,
            DROP_OUT_ATTEN=self.t_config.drop_out
        ).to(self.device)
        assert "triplet_feat_type" in self.m_config, "Triplet Contrastive Setting needs merging layer type: Concat or 1D Conv"
        if self.m_config.triplet_feat_type == "1dconv":
            self.triplet_encoder = TripletContrastiveConvLayer(
                self.m_config.dim_obj_feats, 
                self.m_config.num_layers
            )
        elif self.m_config.triplet_feat_type == "concat":
            self.triplet_encoder = TripletContrastiveMLPLayer(
                self.m_config.dim_obj_feats,
                self.m_config.num_layers
            )
        else:
            raise NotImplementedError
        
        self.obj_classifier = ObjectClsMulti(n_obj_cls, self.m_config.dim_obj_feats).to(self.device)
        self.rel_classifier = RelationClsMulti(n_rel_cls, self.m_config.dim_edge_feats).to(self.device)
    
    def forward(
        self, 
        obj_pts: torch.Tensor, 
        edge_indices: torch.Tensor, 
        descriptor: torch.Tensor, 
        batch_ids=None
    ):
        with torch.no_grad():
            _obj_feats, _edge_feats = self.feature_encoder(obj_pts, edge_indices.t().contiguous(), descriptor, is_train=False)
        obj_feats = _obj_feats.clone().detach()
        edge_feats = _edge_feats.clone().detach()
        
        obj_center = descriptor[:, :3].clone()
        obj_gnn_feats, edge_gnn_feats = self.gat(
            obj_feats, edge_feats, edge_indices, batch_ids, obj_center
        )
        sub_tri_feats, obj_tri_feats = self.index_get(obj_gnn_feats, edge_indices)
        tri_feats = self.triplet_encoder(sub_tri_feats, edge_gnn_feats, obj_tri_feats)
        
        obj_pred = self.obj_classifier(obj_gnn_feats)
        rel_pred = self.rel_classifier(edge_gnn_feats)
        
        return obj_pred, rel_pred, tri_feats