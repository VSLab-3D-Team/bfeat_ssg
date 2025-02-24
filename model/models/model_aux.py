from model.frontend.pointnet import PointNetEncoder
from model.backend.gat import BFeatContrastiveAuxGAT
from model.frontend.relextractor import *
from model.backend.classifier import RelationClsMulti, ObjectClsMulti
from model.backend.triplet import TripletContrastiveConvLayer, TripletContrastiveMLPLayer
from model.models.baseline import BaseNetwork
from utils.model_utils import Gen_Index
import torch

class SGGFeatureEncoder(BaseNetwork):
    def __init__(self, config, device):
        super(SGGFeatureEncoder, self).__init__()
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
        self.relation_encoder = RelFeatNaiveExtractor(
            self.m_config.dim_obj_feats,
            self.m_config.dim_geo_feats,
            self.m_config.dim_edge_feats,
            self.m_config.num_layers
        ).to(self.device)
        
    def forward(
        self, 
        obj_pts: torch.Tensor, 
        edge_indices: torch.Tensor, 
        descriptor: torch.Tensor, 
    ):
        obj_feats, _, _ = self.point_encoder(obj_pts)
        
        x_i_feats, x_j_feats = self.index_get(obj_feats, edge_indices)
        geo_i_feats, geo_j_feats = self.index_get(descriptor, edge_indices)
        edge_feats = self.relation_encoder(x_i_feats, x_j_feats, geo_i_feats - geo_j_feats)
        
        return obj_feats, edge_feats

class BFeatAuxContrastvieGNN(BaseNetwork):
    def __init__(self, config, n_obj_cls, n_rel_cls, device):
        super(BFeatAuxContrastvieGNN, self).__init__()
        self.config = config
        self.t_config = config.train
        self.m_config = config.model
        self.dim_pts = 3
        if self.m_config.use_rgb:
            self.dim_pts += 3
        if self.m_config.use_normal:
            self.dim_pts += 3
        self.device = device
        
        self.con_encoder = SGGFeatureEncoder(config, device)
        self.sgg_encoder = SGGFeatureEncoder(config, device)
        
        self.index_get = Gen_Index(flow=self.m_config.flow)
        
        self.backend = BFeatContrastiveAuxGAT(
            self.m_config.dim_obj_feats,
            self.m_config.dim_edge_feats,
            self.m_config.dim_attn,
            num_heads=self.m_config.num_heads,
            depth=self.m_config.num_graph_update,
            DROP_OUT_ATTEN=self.t_config.drop_out
        ).to(self.device)
        # assert "triplet_feat_type" in self.m_config, "Triplet Contrastive Setting needs merging layer type: Concat or 1D Conv"
        # if self.m_config.triplet_feat_type == "1dconv":
        #     self.triplet_encoder = TripletContrastiveConvLayer(
        #         self.m_config.dim_obj_feats, 
        #         self.m_config.num_layers
        #     )
        # elif self.m_config.triplet_feat_type == "concat":
        #     self.triplet_encoder = TripletContrastiveMLPLayer(
        #         self.m_config.dim_obj_feats,
        #         self.m_config.num_layers
        #     )
        # else:
        #     raise NotImplementedError
        
        self.obj_classifier = ObjectClsMulti(n_obj_cls, self.m_config.dim_obj_feats).to(self.device)
        self.rel_classifier = RelationClsMulti(n_rel_cls, self.m_config.dim_edge_feats).to(self.device)
    
    def forward(
        self, 
        obj_pts: torch.Tensor, 
        edge_indices: torch.Tensor, 
        descriptor: torch.Tensor, 
        batch_ids=None,
        is_train=True
    ):
        con_obj_feats, con_edge_feats = self.con_encoder(obj_pts, edge_indices, descriptor)
        sgg_obj_feats, ssg_edge_feats = self.sgg_encoder(obj_pts, edge_indices, descriptor)
        
        ## Contrastive Auxiliary Task
        obj_center = descriptor[:, :3].clone()
        obj_feature_sgg, obj_feature_con, edge_feature_ssg, edge_feature_con = self.backend(
            sgg_obj_feats, 
            con_obj_feats, 
            ssg_edge_feats, 
            con_edge_feats, 
            edge_indices, 
            batch_ids, 
            obj_center=obj_center, 
            discriptor=descriptor, 
            istrain=is_train
        )
        # sub_tri_feats, obj_tri_feats = self.index_get(obj_feature_con, edge_indices)
        # tri_feats = self.triplet_encoder(sub_tri_feats, edge_feature_con, obj_tri_feats)
        
        obj_pred = self.obj_classifier(obj_feature_sgg)
        rel_pred = self.rel_classifier(edge_feature_ssg)
        
        return obj_feature_con, edge_feature_con, obj_pred, rel_pred # tri_feats, 