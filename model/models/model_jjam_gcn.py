from model.frontend.pointnet import PointNetEncoder
from model.backend.triplet_gcn import TripletGCN
from model.frontend.relextractor import *
from model.backend.classifier import RelationClsMulti, ObjectClsMulti
from model.models.baseline import BaseNetwork
from utils.model_utils import Gen_Index
import torch

class BFeatJJamTongTripletGCNNet(BaseNetwork):
    def __init__(self, config, n_obj_cls, n_rel_cls, device):
        super(BFeatJJamTongTripletGCNNet, self).__init__()
        self.config = config
        self.t_config = config.train
        self.m_config = config.model
        self.dim_pts = 3
        if self.m_config.use_rgb:
            self.dim_pts += 3
        if self.m_config.use_normal:
            self.dim_pts += 3
        self.device = device
        
        self.point_encoder = PointNetEncoder(device, channel=self.dim_pts)
        
        self.index_get = Gen_Index(flow=self.m_config.flow)
        assert "relation_type" in self.m_config, "Direct GNN needs Relation Encoder Type: ResNet or 1D Conv"
        if self.m_config.relation_type == "pointnet":
            self.relation_encoder = RelFeatPointExtractor(
                config, device
            )
        elif self.m_config.relation_type == "resnet":
            self.relation_encoder = RelFeatNaiveExtractor(
                self.m_config.dim_obj_feats,
                self.m_config.dim_geo_feats,
                self.m_config.dim_edge_feats,
                num_layers=self.m_config.num_layers
            ).to(self.device)
        elif self.m_config.relation_type == "1dconv":
            self.relation_encoder = RelFeatMergeExtractor(
                self.m_config.dim_obj_feats,
                self.m_config.dim_geo_feats,
                self.m_config.dim_edge_feats
            ).to(self.device)
        else:
            raise NotImplementedError
        
        self.gcn = TripletGCN(
            self.m_config.dim_obj_feats,
            self.m_config.dim_edge_feats,
            dim_hidden=512, 
            aggr='add', 
            use_bn=True
        ).to(self.device)
        
        self.obj_classifier = ObjectClsMulti(n_obj_cls, self.m_config.dim_obj_feats).to(self.device)
        self.rel_classifier = RelationClsMulti(n_rel_cls, self.m_config.dim_edge_feats).to(self.device)
        
    def forward(
        self, 
        obj_pts: torch.Tensor, 
        edge_pts, 
        edge_indices: torch.Tensor, 
        descriptor: torch.Tensor, 
        is_train = True
    ):
        if is_train:
            bsz = obj_pts.shape[0] // 2
            obj_feats_cat, _, _ = self.point_encoder(obj_pts)
            obj_t1_feats = obj_feats_cat[:bsz, ...]
            obj_t2_feats = obj_feats_cat[bsz:, ...]
            
            obj_feats = torch.stack([obj_t1_feats, obj_t2_feats]).mean(dim=0)
        else:
            obj_feats, _, _ = self.point_encoder(obj_pts)
        
        if not self.m_config.relation_type == "pointnet":
            x_i_feats, x_j_feats = self.index_get(obj_feats, edge_indices)
            geo_i_feats, geo_j_feats = self.index_get(descriptor, edge_indices)
            edge_feats = self.relation_encoder(x_i_feats, x_j_feats, geo_i_feats - geo_j_feats)
        else:
            edge_feats = self.relation_encoder(edge_pts)
        
        obj_gnn_feats, edge_gnn_feats = self.gcn(
            obj_feats, edge_feats, edge_indices
        )
        
        obj_pred = self.obj_classifier(obj_gnn_feats)
        rel_pred = self.rel_classifier(edge_gnn_feats)
        
        if is_train:
            return obj_gnn_feats, edge_gnn_feats, obj_pred, rel_pred, obj_t1_feats, obj_t2_feats
        else:
            return obj_pred, rel_pred
        
        
        
        
        