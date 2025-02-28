import torch
import torch.nn as nn
from model.models.baseline import ResidualBlock

class TripletContrastiveConvLayer(nn.Module):
    def __init__(self, d_feats, n_layers=6):
        super(TripletContrastiveConvLayer, self).__init__()
        self.merge_layer = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=5, stride=1, padding="same")
        self.res_blocks = nn.Sequential(*[ResidualBlock(d_feats) for _ in range(n_layers)])
        self.fc_out = nn.Linear(d_feats, d_feats)  # 출력 레이어
    
    def forward(
        self, 
        sub_feats: torch.Tensor, 
        pred_feats: torch.Tensor, 
        obj_feats: torch.Tensor
    ):
        m_ij = torch.cat([
            sub_feats.unsqueeze(1), 
            pred_feats.unsqueeze(1), 
            obj_feats.unsqueeze(1)
        ], dim=1)
        e_ij = self.merge_layer(m_ij).squeeze(1) # B X 512
        r_ij = self.res_blocks(e_ij)
        return self.fc_out(r_ij)

class TripletContrastiveMLPLayer(nn.Module):
    def __init__(self, d_feats, n_layers=6):
        super(TripletContrastiveMLPLayer, self).__init__()
        self.merge_layer = nn.Linear(3 * d_feats, d_feats)
        self.res_blocks = nn.Sequential(*[ResidualBlock(d_feats) for _ in range(n_layers)])
        self.fc_out = nn.Linear(d_feats, d_feats)  # 출력 레이어
        
    def forward(
        self, 
        sub_feats: torch.Tensor, 
        pred_feats: torch.Tensor, 
        obj_feats: torch.Tensor
    ):
        m_ij = torch.cat([ sub_feats, pred_feats, obj_feats ], dim=-1)
        e_ij = self.merge_layer(m_ij).squeeze(1) # B X 512
        r_ij = self.res_blocks(e_ij)
        return self.fc_out(r_ij)

class ProjectHead(nn.Module):
    def __init__(self, dims: list):
        super(ProjectHead, self).__init__()
        
        self.layers = nn.Sequential(
            *[ nn.Linear(dims[i - 1], dims[i]) for i in range(1, len(dims)) ]
        )
    
    def forward(self, x):
        return self.layers(x)