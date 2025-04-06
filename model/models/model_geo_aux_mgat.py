from model.frontend.pointnet import PointNetEncoder
from model.backend.gat import BFeatVanillaGAT
from model.backend.m_gat import BidirectionalEdgeGraphNetwork
from model.frontend.relextractor import *
from model.backend.classifier import RelationClsMulti, ObjectClsMulti
from model.models.baseline import BaseNetwork
from utils.model_utils import Gen_Index, build_mlp
import torch
import torch.nn as nn
import torch.nn.functional as F

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

        self.n_obj_cls = n_obj_cls
        self.n_rel_cls = n_rel_cls
        
        try:
            if hasattr(config.dataset, 'obj_label_list'):
                with open(config.dataset.obj_label_list, 'r') as f:
                    self.obj_label_list = [line.strip() for line in f.readlines()]
            if hasattr(config.dataset, 'rel_label_list'):
                with open(config.dataset.rel_label_list, 'r') as f:
                    self.rel_label_list = [line.strip() for line in f.readlines()]
        except Exception as e:
            print(f"Warning: Could not load label lists: {e}")
            self.obj_label_list = [f"obj_{i}" for i in range(n_obj_cls)]
            self.rel_label_list = [f"rel_{i}" for i in range(n_rel_cls)]
        
        self.clip_text_encoder = CLIPTextEncoder(device=device).to(device)
        self._text_embeddings_cache = {}
        
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
        elif self.m_config.relation_type == "masking":
            self.relation_encoder = MaskingExtractor(
                self.m_config.dim_obj_feats,
                self.m_config.dim_geo_feats,
                self.m_config.dim_edge_feats,
                num_layers=self.m_config.num_layers,
                mask_ratio=0.3
            ).to(self.device)
        elif self.m_config.relation_type == "filmnet":
            self.relation_encoder = RelFeatMergeExtractorWithFiLM(
                self.m_config.dim_obj_feats,
                self.m_config.dim_geo_feats,
                self.m_config.dim_edge_feats,
                hidden_dim=256
            ).to(self.device)
        elif self.m_config.relation_type == "vlsat":
            self.relation_encoder = VLSAT3DEdgeEncoder(
                dim_geo_feats=self.m_config.dim_geo_feats,
                dim_edge_feats=self.m_config.dim_edge_feats
            ).to(self.device)
        elif self.m_config.relation_type == "image":
            self.relation_encoder = RelFeatImageAwareExtractor(
            self.m_config.dim_obj_feats,
            self.m_config.dim_geo_feats,
            512,
            self.m_config.dim_edge_feats,
            num_layers=self.m_config.num_layers
            ).to(self.device)
        elif self.m_config.relation_type == "mmodal":
            self.relation_encoder = RelFeatMultiModalExtractor(
            self.m_config.dim_obj_feats,
            self.m_config.dim_geo_feats,
            512,
            512,
            self.m_config.dim_edge_feats,
            num_layers=self.m_config.num_layers
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
        
    def _get_text_embedding(self, text_template, fill_values):
        text = text_template.format(**fill_values)
        
        if text in self._text_embeddings_cache:
            return self._text_embeddings_cache[text]
        
        with torch.no_grad():
            embedding = self.clip_text_encoder([text])
            embedding = F.normalize(embedding, p=2, dim=1)
            self._text_embeddings_cache[text] = embedding
            
        return embedding
        
    def forward(
        self, 
        obj_pts: torch.Tensor, 
        edge_pts: torch.Tensor, # remaining for other processing domain
        edge_indices: torch.Tensor, 
        descriptor: torch.Tensor, 
        edge_feat_mask: torch.Tensor,
        batch_ids=None,
        attn_weight=None,
        edge_2d_feats=None,
        edge_text_feats=None,
    ):
        with torch.no_grad():
            _obj_feats, _, _ = self.point_encoder(obj_pts)
        obj_feats = _obj_feats.clone().detach() # B X N_feats
        # obj_feats = self.obj_proj(obj_feats)
        
        if not self.m_config.relation_type == "pointnet":
            x_i_feats, x_j_feats = self.index_get(obj_feats, edge_indices)
            geo_i_feats, geo_j_feats = self.index_get(descriptor, edge_indices)
            
            if self.m_config.relation_type == "mmodal" and edge_text_feats is None and hasattr(self, 'clip_text_encoder'):
                with torch.no_grad():
                    obj_logits = self.obj_classifier(obj_feats)
                    obj_pred = F.softmax(obj_logits, dim=1)
                    
                    edge_text_feats_list = []
                    num_edges = edge_indices.shape[0]
                    
                    batch_size = min(128, num_edges)
                    
                    if batch_size > 0:
                        sample_indices = torch.arange(num_edges)[:batch_size]
                        
                        for i in sample_indices:
                            subj_idx = edge_indices[i, 0].item()
                            obj_idx = edge_indices[i, 1].item()
                            
                            subj_cls = obj_pred[subj_idx].argmax().item()
                            obj_cls = obj_pred[obj_idx].argmax().item()
                            
                            rel_cls = 0
                            
                            subject_name = self.obj_label_list[subj_cls]
                            object_name = self.obj_label_list[obj_cls]
                            relation_name = self.rel_label_list[rel_cls]
                            
                            text_emb = self._get_text_embedding(
                                "a {subj} {pred} a {obj}",
                                {"subj": subject_name, "pred": relation_name, "obj": object_name}
                            )
                            
                            edge_text_feats_list.append(text_emb)
                        
                        if edge_text_feats_list:
                            edge_text_feats = torch.cat(edge_text_feats_list, dim=0)
                            
                            if edge_text_feats.shape[0] < num_edges:
                                print(f"Padding edge_text_feats from shape {edge_text_feats.shape} to match {num_edges} edges")
                                
                                padded_feats = torch.zeros(num_edges, edge_text_feats.shape[1], 
                                                        device=edge_text_feats.device, 
                                                        dtype=edge_text_feats.dtype)
                                padded_feats[:edge_text_feats.shape[0]] = edge_text_feats
                                edge_text_feats = padded_feats
            
            if edge_2d_feats is not None:
                num_edges = x_i_feats.shape[0]
                
                print(f"Debug - x_i_feats shape: {x_i_feats.shape}, edge_2d_feats shape: {edge_2d_feats.shape}")
                
                if len(edge_2d_feats.shape) == 1:
                    edge_2d_feats = edge_2d_feats.unsqueeze(0).expand(num_edges, -1)
                
                elif edge_2d_feats.shape[0] != num_edges:
                    print(f"Adjusting edge_2d_feats from shape {edge_2d_feats.shape} to match {num_edges} edges")
                    
                    if edge_2d_feats.shape[0] < num_edges:
                        padded_feats = torch.zeros(num_edges, edge_2d_feats.shape[1], 
                                                device=edge_2d_feats.device, 
                                                dtype=edge_2d_feats.dtype)
                        padded_feats[:edge_2d_feats.shape[0]] = edge_2d_feats
                        edge_2d_feats = padded_feats
                    else:
                        edge_2d_feats = edge_2d_feats[:num_edges]
            
            if edge_text_feats is not None and self.m_config.relation_type == "mmodal":
                num_edges = x_i_feats.shape[0]
                
                print(f"Debug - x_i_feats shape: {x_i_feats.shape}, edge_text_feats shape: {edge_text_feats.shape}")
                
                if len(edge_text_feats.shape) == 1:
                    edge_text_feats = edge_text_feats.unsqueeze(0).expand(num_edges, -1)
                
                elif edge_text_feats.shape[0] != num_edges:
                    print(f"Adjusting edge_text_feats from shape {edge_text_feats.shape} to match {num_edges} edges")
                    
                    if edge_text_feats.shape[0] < num_edges:
                        padded_feats = torch.zeros(num_edges, edge_text_feats.shape[1], 
                                                device=edge_text_feats.device, 
                                                dtype=edge_text_feats.dtype)
                        padded_feats[:edge_text_feats.shape[0]] = edge_text_feats
                        edge_text_feats = padded_feats
                    else:
                        edge_text_feats = edge_text_feats[:num_edges]
            
            if self.m_config.relation_type == "mmodal":
                edge_feats = self.relation_encoder(
                    x_i_feats, x_j_feats, geo_i_feats - geo_j_feats, 
                    edge_2d_feats, edge_text_feats
                )
            elif edge_2d_feats is not None:
                if hasattr(self.relation_encoder, 'forward') and 'img_feats' in self.relation_encoder.forward.__code__.co_varnames:
                    edge_feats = self.relation_encoder(
                        x_i_feats, x_j_feats, geo_i_feats - geo_j_feats, edge_2d_feats
                    )
                else:
                    e_feats = self.relation_encoder(x_i_feats, x_j_feats, geo_i_feats - geo_j_feats)
                    
                    img_proj = getattr(self, 'img_proj', None)
                    if img_proj is None:
                        self.img_proj = nn.Linear(edge_2d_feats.shape[1], e_feats.shape[1]).to(e_feats.device)
                        img_proj = self.img_proj
                    
                    img_feats_proj = img_proj(edge_2d_feats)
                    
                    edge_feats = e_feats + img_feats_proj
            else:
                edge_feats = self.relation_encoder(
                    x_i_feats, x_j_feats, geo_i_feats - geo_j_feats
                )
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