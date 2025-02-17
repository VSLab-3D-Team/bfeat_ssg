import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        residual = x  # Skip Connection
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out += residual  # Add Skip Connection
        out = self.activation(out)
        return out

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

## Discarded.
## 우선, High frequency information을 잘 capture할 수 있도록 고안된 모듈임
## 그걸 위해서 kernel trick을 활용하는 것일 뿐이다.
## 하지만, 우리는 high frequency feature가 필요한 것이 아니라, classification 문제를 풀어내는 것이 목표이다.
## 따라서, fourier feature 안은 폐기한다. (25/01/31)
## 혹시 모르니 함 해보자... (25/02/03)
class RelFeatFreqExtractor(nn.Module):
    def __init__(self, config):
        super(RelFeatFreqExtractor, self).__init__()
        self.config = config
    
    def forward(self, x_i, x_j):
        
        pass