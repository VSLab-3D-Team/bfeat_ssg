import torch
import torch.nn as nn
import torch.nn.functional as F
from model.models.baseline import ResidualBlock
from model.frontend.pointnet import PointNetEncoder

class RelFeatNaiveExtractor(nn.Module):
    def __init__(self, input_dim, geo_dim, out_dim, num_layers=6):
        super(RelFeatNaiveExtractor, self).__init__()
        self.obj_proj = nn.Linear(input_dim, 512)
        self.geo_proj = nn.Linear(geo_dim, 512)
        self.merge_layer = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=5, stride=1, padding="same")
        
        self.res_blocks = nn.Sequential(*[ResidualBlock(512) for _ in range(num_layers)])
        self.fc_out = nn.Linear(512, out_dim)  # 출력 레이어
        

    def forward(self, x_i: torch.Tensor, x_j: torch.Tensor, geo_feats: torch.Tensor):
        # All B X N_feat size
        p_i, p_j, g_ij = self.obj_proj(x_i), self.obj_proj(x_j), self.geo_proj(geo_feats)
        m_ij = torch.cat([
            p_i.unsqueeze(1), p_j.unsqueeze(1), g_ij.unsqueeze(1)
        ], dim=1)
        
        e_ij = self.merge_layer(m_ij).squeeze(1) # B X 512
        r_ij = self.res_blocks(e_ij)
        return self.fc_out(r_ij)

class RelFeatMergeExtractor(nn.Module):
    def __init__(self, dim_obj_feats, dim_geo_feats, dim_out_feats):
        super(RelFeatMergeExtractor, self).__init__()
        self.obj_proj = nn.Linear(dim_obj_feats, dim_out_feats)
        self.geo_proj = nn.Linear(dim_geo_feats, dim_out_feats)
        self.merge_layer = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=5, stride=1, padding="same")
    
    def forward(self, x_i, x_j, geo_feats): 
        # 다른 2개의 feature가 512 dimension인 것에 비하여, geometric descriptor는 11 차원으로 턱 없이 부족한 차원/정보양을 가짐
        # 이 둘을 적절하게 엮을 수 있는 network architecture를 고안할 필요가 있음.
        # 일단, visual feature는 생각하지 말고 여기에 집중하자.
        p_i, p_j, g_ij = self.obj_proj(x_i), self.obj_proj(x_j), self.geo_proj(geo_feats)
        m_ij = torch.cat([
            p_i.unsqueeze(1), p_j.unsqueeze(1), g_ij.unsqueeze(1)
        ], dim=1)
        
        edge_init_feats = self.merge_layer(m_ij).squeeze(1) # B X 512
        return edge_init_feats # think novel method
    
class RelFeatMergeExtractorWithFiLM(nn.Module):
    def __init__(self, dim_obj_feats, dim_geo_feats, dim_out_feats, hidden_dim=128):
        super(RelFeatMergeExtractorWithFiLM, self).__init__()
        
        self.obj_proj = nn.Linear(dim_obj_feats, dim_out_feats)
        
        self.geo_encoder = nn.Sequential(
            nn.Linear(dim_geo_feats, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.geo_proj = nn.Linear(dim_geo_feats, dim_out_feats)
        self.film_generator = nn.Linear(hidden_dim, dim_out_feats * 2)
        self.merge_layer = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=5, stride=1, padding="same")
    
    def forward(self, x_i, x_j, geo_feats):
        p_i = self.obj_proj(x_i)
        p_j = self.obj_proj(x_j)
        
        g_ij = self.geo_proj(geo_feats)
        
        geo_enc = self.geo_encoder(geo_feats)
        
        film_params = self.film_generator(geo_enc)
        gamma, beta = torch.chunk(film_params, 2, dim=1)
        
        p_i_mod = gamma * p_i + beta
        p_j_mod = gamma * p_j + beta
        
        m_ij = torch.cat([
            p_i_mod.unsqueeze(1), p_j_mod.unsqueeze(1), g_ij.unsqueeze(1)
        ], dim=1)
        
        edge_init_feats = self.merge_layer(m_ij).squeeze(1)
        
        return edge_init_feats

class RelFeatPointExtractor(nn.Module):
    def __init__(self, config, device, out_dims=512):
        super(RelFeatPointExtractor, self).__init__()
        self.config = config
        self.m_config = config.model
        self.dim_pts = 3
        if self.m_config.use_rgb:
            self.dim_pts += 3
        if self.m_config.use_normal:
            self.dim_pts += 3
        self.device = device
        self.point_encoder = PointNetEncoder(device, channel=self.dim_pts, out_dim=out_dims)
        
    def forward(self, rel_pts):
        rel_feats, _, _ = self.point_encoder(rel_pts)
        return rel_feats
    
class VLSAT3DEdgeEncoder(nn.Module):

    def __init__(self, dim_geo_feats, dim_edge_feats):
        super(VLSAT3DEdgeEncoder, self).__init__()
        
        self.geo_mlp = nn.Sequential(
            nn.Linear(dim_geo_feats, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, dim_edge_feats)
        )
    
    def forward(self, x_i=None, x_j=None, geo_feats=None):
        edge_feats = self.geo_mlp(geo_feats)
        
        return edge_feats
    
class MaskingExtractor(nn.Module):
    def __init__(self, input_dim, geo_dim, out_dim, num_layers=6, mask_ratio=0.3):
        super(MaskingExtractor, self).__init__()
        self.obj_proj = nn.Linear(input_dim, 512)
        self.geo_proj = nn.Linear(geo_dim, 512)
        self.merge_layer = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=5, stride=1, padding="same")
        
        self.res_blocks = nn.Sequential(*[ResidualBlock(512) for _ in range(num_layers)])
        self.fc_out = nn.Linear(512, out_dim)
        
        self.mask_ratio = mask_ratio
        self.training_mode = True
        
    def set_inference_mode(self):
        self.training_mode = False
        
    def set_training_mode(self):
        self.training_mode = True
        
    def apply_random_mask(self, x):
        if not self.training_mode:
            return x
            
        batch_size, feat_dim = x.shape
        mask = torch.ones_like(x)
        
        for i in range(batch_size):
            num_masked = int(feat_dim * self.mask_ratio)
            mask_indices = torch.randperm(feat_dim)[:num_masked]
            mask[i, mask_indices] = 0
            
        return x * mask

    def forward(self, x_i: torch.Tensor, x_j: torch.Tensor, geo_feats: torch.Tensor):
        x_i_masked = self.apply_random_mask(x_i)
        x_j_masked = self.apply_random_mask(x_j)
        
        p_i, p_j, g_ij = self.obj_proj(x_i_masked), self.obj_proj(x_j_masked), self.geo_proj(geo_feats)
        m_ij = torch.cat([
            p_i.unsqueeze(1), p_j.unsqueeze(1), g_ij.unsqueeze(1)
        ], dim=1)
        
        e_ij = self.merge_layer(m_ij).squeeze(1)
        r_ij = self.res_blocks(e_ij)
        return self.fc_out(r_ij)
    
class RelFeatImageAwareExtractor(nn.Module):
    def __init__(self, input_dim, geo_dim, img_dim, out_dim, num_layers=6):
        super(RelFeatImageAwareExtractor, self).__init__()
        self.obj_proj = nn.Linear(input_dim, 512)
        self.geo_proj = nn.Linear(geo_dim, 512)
        self.img_proj = nn.Linear(img_dim, 512)
        
        self.merge_layer = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=5, stride=1, padding="same")
        
        self.res_blocks = nn.Sequential(*[ResidualBlock(512) for _ in range(num_layers)])
        self.fc_out = nn.Linear(512, out_dim) 
        
    def forward(self, x_i: torch.Tensor, x_j: torch.Tensor, geo_feats: torch.Tensor, img_feats: torch.Tensor = None):
        p_i, p_j, g_ij = self.obj_proj(x_i), self.obj_proj(x_j), self.geo_proj(geo_feats)
        
        if img_feats is not None:
            i_ij = self.img_proj(img_feats)
            m_ij = torch.cat([
                p_i.unsqueeze(1), p_j.unsqueeze(1), g_ij.unsqueeze(1), i_ij.unsqueeze(1)
            ], dim=1)
        else:
            m_ij = torch.cat([
                p_i.unsqueeze(1), p_j.unsqueeze(1), g_ij.unsqueeze(1), 
                torch.zeros_like(p_i).unsqueeze(1)
            ], dim=1)
        
        e_ij = self.merge_layer(m_ij).squeeze(1)
        r_ij = self.res_blocks(e_ij)
        return self.fc_out(r_ij)
    
class RelFeatMultiModalExtractor(nn.Module):
    def __init__(self, input_dim, geo_dim, img_dim, text_dim, out_dim, num_layers=6):
        super(RelFeatMultiModalExtractor, self).__init__()
        self.obj_proj = nn.Linear(input_dim, 512)
        self.geo_proj = nn.Linear(geo_dim, 512)
        self.img_proj = nn.Linear(img_dim, 512)
        self.text_proj = nn.Linear(text_dim, 512)
        
        self.merge_layer = nn.Conv1d(in_channels=5, out_channels=1, kernel_size=5, stride=1, padding="same")
        
        self.res_blocks = nn.Sequential(*[ResidualBlock(512) for _ in range(num_layers)])
        self.fc_out = nn.Linear(512, out_dim)
        
    def forward(self, x_i: torch.Tensor, x_j: torch.Tensor, geo_feats: torch.Tensor, 
                img_feats: torch.Tensor = None, text_feats: torch.Tensor = None):
        p_i, p_j, g_ij = self.obj_proj(x_i), self.obj_proj(x_j), self.geo_proj(geo_feats)
        
        if img_feats is not None:
            i_ij = self.img_proj(img_feats)
        else:
            i_ij = torch.zeros_like(p_i)
        
        if text_feats is not None:
            t_ij = self.text_proj(text_feats)
        else:
            t_ij = torch.zeros_like(p_i)
        
        m_ij = torch.cat([
            p_i.unsqueeze(1), 
            p_j.unsqueeze(1), 
            g_ij.unsqueeze(1), 
            i_ij.unsqueeze(1),
            t_ij.unsqueeze(1)
        ], dim=1)
        
        e_ij = self.merge_layer(m_ij).squeeze(1)
        r_ij = self.res_blocks(e_ij)
        return self.fc_out(r_ij)