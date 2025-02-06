import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualLinearNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResidualLinearNet, self).__init__()
    
    def forward(self, x_i, x_j):
        pass

class RelFeatNaiveExtractor(nn.Module):
    def __init__(self, input_dim, geo_dim, out_dim):
        super(RelFeatNaiveExtractor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim * 2 + geo_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, out_dim),
            nn.ReLU(),
        )

    def forward(self, x_i, x_j, geo_feats):
        x = torch.cat((x_i, x_j, geo_feats), dim=-1)
        return self.fc(x)

class RelFeatCrossExtractor(nn.Module):
    def __init__(self, dim_obj_feats, dim_geo_feats, dim_out_feats):
        super(RelFeatCrossExtractor, self).__init__()
        self.linear_obj_fc = nn.Linear(dim_obj_feats, dim_out_feats, bias=False)
        self.linear_geo_fc = nn.Linear(dim_geo_feats, dim_out_feats)
    
    def forward(self, x_i, x_j, geo_desc): 
        # 다른 2개의 feature가 512 dimension인 것에 비하여, geometric descriptor는 11 차원으로 턱 없이 부족한 차원/정보양을 가짐
        # 이 둘을 적절하게 엮을 수 있는 network architecture를 고안할 필요가 있음.
        # 일단, visual feature는 생각하지 말고 여기에 집중하자.
        R_i = self.linear_obj_fc(x_i)
        R_j = self.linear_obj_fc(x_j)
        R_ij = self.linear_geo_fc(geo_desc)
        
        
        return  # think novel method

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