from model.frontend.pointnet import PointNetEncoder
from model.backend.gat import BFeatVanillaGAT
from model.frontend.relextractor import *
from model.backend.classifier import RelationClsMulti, ObjectClsMulti
from model.backend.triplet import TripletContrastiveConvLayer, TripletContrastiveMLPLayer
from model.models.baseline import BaseNetwork
from utils.model_utils import Gen_Index
import torch

class BFeatDirectGNNNet(BaseNetwork):
    def __init__(self, config, n_obj_cls, n_rel_cls, device):
        super(BFeatDirectGNNNet, self).__init__()
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
        self.point_encoder.load_state_dict(torch.load(self.t_config.ckp_path))
        self.point_encoder = self.point_encoder.to(self.device).eval()
        
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
        
        self.index_get = Gen_Index(flow=self.m_config.flow)
        self.gat = BFeatVanillaGAT(
            self.m_config.dim_obj_feats,
            self.m_config.dim_edge_feats,
            self.m_config.dim_attn,
            num_heads=self.m_config.num_heads,
            depth=self.m_config.num_graph_update,
            edge_attn=self.m_config.edge_attention,
            DROP_OUT_ATTEN=self.t_config.drop_out
        ).to(self.device)
        
        self.obj_classifier = ObjectClsMulti(n_obj_cls, self.m_config.dim_obj_feats).to(self.device)
        self.rel_classifier = RelationClsMulti(n_rel_cls, self.m_config.dim_edge_feats).to(self.device)
        
        
    def forward(
        self, 
        obj_pts: torch.Tensor, 
        edge_pts: torch.Tensor, 
        edge_indices: torch.Tensor, 
        descriptor: torch.Tensor, 
        batch_ids=None
    ):
        with torch.no_grad():
            _obj_feats, _, _ = self.point_encoder(obj_pts)
        obj_feats = _obj_feats.clone().detach()
        
        if not self.m_config.relation_type == "pointnet":
            x_i_feats, x_j_feats = self.index_get(obj_feats, edge_indices)
            geo_i_feats, geo_j_feats = self.index_get(descriptor, edge_indices)
            edge_feats = self.relation_encoder(x_i_feats, x_j_feats, geo_i_feats - geo_j_feats)
        else:
            edge_feats = self.relation_encoder(edge_pts)
        
        obj_center = descriptor[:, :3].clone()
        obj_gnn_feats, edge_gnn_feats = self.gat(
            obj_feats, edge_feats, edge_indices, batch_ids, obj_center
        )
        
        obj_pred = self.obj_classifier(obj_gnn_feats)
        rel_pred = self.rel_classifier(edge_gnn_feats)
        
        return edge_gnn_feats, obj_pred, rel_pred
        
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
        
        self.point_encoder = PointNetEncoder(device, channel=self.dim_pts)
        self.point_encoder.load_state_dict(torch.load(self.t_config.ckp_path))
        self.point_encoder = self.point_encoder.to(self.device).eval()
        
        assert "relation_type" in self.m_config, "Direct GNN needs Relation Encoder Type: ResNet or 1D Conv"
        if self.m_config.relation_type == "resnet":
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
        
        self.index_get = Gen_Index(flow=self.m_config.flow)
        self.gat = BFeatVanillaGAT(
            self.m_config.dim_obj_feats,
            self.m_config.dim_edge_feats,
            self.m_config.dim_attn,
            num_heads=self.m_config.num_heads,
            depth=self.m_config.num_graph_update,
            DROP_OUT_ATTEN=self.t_config.drop_out
        ).to(self.device)
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
        
        self.obj_classifier = ObjectClsMulti(n_obj_cls, self.m_config.dim_obj_feats).to(self.device)
        self.rel_classifier = RelationClsMulti(n_rel_cls, self.m_config.dim_edge_feats).to(self.device)
    
    def forward(
        self, 
        obj_pts: torch.Tensor, 
        edge_pts: torch.Tensor, # remaining for other processing domain
        edge_indices: torch.Tensor, 
        descriptor: torch.Tensor, 
        batch_ids=None
    ):
        with torch.no_grad():
            _obj_feats, _, _ = self.point_encoder(obj_pts)
        obj_feats = _obj_feats.clone().detach()
        
        if not self.m_config.relation_type == "pointnet":
            x_i_feats, x_j_feats = self.index_get(obj_feats, edge_indices)
            geo_i_feats, geo_j_feats = self.index_get(descriptor, edge_indices)
            edge_feats = self.relation_encoder(x_i_feats, x_j_feats, geo_i_feats - geo_j_feats)
        else:
            edge_feats = self.relation_encoder(edge_pts)
        
        obj_center = descriptor[:, :3].clone()
        obj_gnn_feats, edge_gnn_feats = self.gat(
            obj_feats, edge_feats, edge_indices, batch_ids, obj_center
        )
        # sub_tri_feats, obj_tri_feats = self.index_get(obj_gnn_feats, edge_indices)
        # tri_feats = self.triplet_encoder(sub_tri_feats, edge_gnn_feats, obj_tri_feats)
        
        obj_pred = self.obj_classifier(obj_gnn_feats)
        rel_pred = self.rel_classifier(edge_gnn_feats)
        
        return edge_gnn_feats, obj_pred, rel_pred # tri_feats,