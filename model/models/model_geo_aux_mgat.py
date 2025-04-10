from model.frontend.pointnet import PointNetEncoder
from model.backend.gat import BFeatVanillaGAT
from model.backend.m_gat import BidirectionalEdgeGraphNetwork, BidirectionalEdgeGraphObjAttnNetwork
from model.frontend.relextractor import *
from model.backend.classifier import RelationClsMulti, ObjectClsMulti
from model.models.baseline import BaseNetwork
from utils.model_utils import Gen_Index, build_mlp
import torch
import torch.nn as nn
import torch.nn.functional as F

class BFeatGeoAuxMGATNet(BaseNetwork):
    def __init__(self, config, n_obj_cls, n_rel_cls, device):
        super(BFeatGeoAuxMGATNet, self).__init__()
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
        # self.obj_proj = build_mlp([
        #     self.m_config.dim_obj_feats, 
        #     self.m_config.dim_obj_feats // 2, 
        #     self.m_config.dim_obj_feats
        # ], do_bn=True, on_last=True)
        
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
        elif self.m_config.relation_type == "filmnet":
            self.relation_encoder = RelFeatMergeExtractorWithFiLM(
                self.m_config.dim_obj_feats,
                self.m_config.dim_geo_feats,
                self.m_config.dim_edge_feats,
                hidden_dim=256
            ).to(self.device)
        elif self.m_config.relation_type == "filmresnet":
            self.relation_encoder = RelFeatResNetWithFiLM(
                self.m_config.dim_obj_feats,
                self.m_config.dim_geo_feats,
                self.m_config.dim_edge_feats,
                num_layers=self.m_config.num_layers,
                hidden_dim=256
            ).to(self.device)
        else:
            raise NotImplementedError
        
        self.proj_clip_edge = build_mlp([
            self.m_config.dim_edge_feats, 
            self.m_config.dim_edge_feats // 2, 
            self.m_config.dim_edge_feats
        ], do_bn=True, on_last=True)
        self.proj_geo_desc = build_mlp([
            self.m_config.dim_edge_feats, 
            self.m_config.dim_edge_feats // 4, 
            11
        ], do_bn=True, on_last=True)
        
        if self.m_config.gat_type == "bidirectional":
            self.gat = BidirectionalEdgeGraphNetwork(
                dim_node=self.m_config.dim_obj_feats,
                dim_edge=self.m_config.dim_edge_feats,
                dim_atten=self.m_config.dim_attn,
                num_layers=self.m_config.num_graph_update,
                num_heads=self.m_config.num_heads,
                use_distance_mask=self.m_config.use_distance_mask,
                aggr='max',
                DROP_OUT_ATTEN=self.t_config.drop_out,
                use_bn=True
            )
        elif self.m_config.gat_type == "bidir_objattn":
            self.gat = BidirectionalEdgeGraphObjAttnNetwork(
                dim_node=self.m_config.dim_obj_feats,
                dim_edge=self.m_config.dim_edge_feats,
                dim_atten=self.m_config.dim_attn,
                num_layers=self.m_config.num_graph_update,
                num_heads=self.m_config.num_heads,
                use_distance_mask=self.m_config.use_distance_mask,
                aggr='max',
                DROP_OUT_ATTEN=self.t_config.drop_out,
                use_bn=True
            )
        else:
            self.gat = BFeatVanillaGAT(
                self.m_config.dim_obj_feats,
                self.m_config.dim_edge_feats,
                self.m_config.dim_attn,
                num_heads=self.m_config.num_heads,
                depth=self.m_config.num_graph_update,
                edge_attn=self.m_config.edge_attention,
                DROP_OUT_ATTEN=self.t_config.drop_out,
            ).to(self.device)
        
        self.obj_classifier = ObjectClsMulti(n_obj_cls, self.m_config.dim_obj_feats).to(self.device)
        self.rel_classifier = RelationClsMulti(n_rel_cls, self.m_config.dim_edge_feats).to(self.device)
        
        
    def forward(
        self, 
        obj_pts: torch.Tensor, 
        edge_pts: torch.Tensor, # remaining for other processing domain
        edge_indices: torch.Tensor, 
        descriptor: torch.Tensor, 
        edge_feat_mask: torch.Tensor,
        batch_ids=None,
        attn_weight=None,
    ):
        with torch.no_grad():
            _obj_feats, _, _ = self.point_encoder(obj_pts)
        obj_feats = _obj_feats.clone().detach() # B X N_feats
        # obj_feats = self.obj_proj(obj_feats)
        
        if not self.m_config.relation_type == "pointnet":
            x_i_feats, x_j_feats = self.index_get(obj_feats, edge_indices)
            geo_i_feats, geo_j_feats = self.index_get(descriptor, edge_indices)
            edge_feats = self.relation_encoder(x_i_feats, x_j_feats, geo_i_feats - geo_j_feats)
        else:
            edge_feats = self.relation_encoder(edge_pts)
        
        if self.m_config.gat_type == "bidirectional" and not self.m_config.use_distance_mask:
            obj_gnn_feats, edge_gnn_feats, _ = self.gat(
                obj_feats, edge_feats, edge_indices
            )
        elif self.m_config.gat_type == "bidirectional" and self.m_config.use_distance_mask:
            obj_gnn_feats, edge_gnn_feats, _ = self.gat(
                obj_feats, edge_feats, edge_indices, descriptor
            )
        elif self.m_config.gat_type == "bidir_objattn":
            obj_center = descriptor[:, :3].clone()
            obj_gnn_feats, edge_gnn_feats, _ = self.gat(
                obj_feats, edge_feats, edge_indices, descriptor, batch_ids, obj_center
            )
        else:
            obj_center = descriptor[:, :3].clone()
            obj_gnn_feats, edge_gnn_feats = self.gat(
                obj_feats, edge_feats, edge_indices, batch_ids, obj_center, attn_weight
            )
        
        obj_pred = self.obj_classifier(obj_gnn_feats)
        rel_pred = self.rel_classifier(edge_gnn_feats)
        
        pred_edge_clip, pred_geo_desc = \
            self.proj_clip_edge(edge_feats[edge_feat_mask, ...]), self.proj_geo_desc(edge_feats)
        
        return edge_feats, \
            obj_pred, \
            rel_pred, \
            pred_edge_clip, \
            pred_geo_desc, \
            geo_i_feats - geo_j_feats
        
        
        
        
        