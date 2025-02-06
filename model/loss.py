import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_logsumexp

class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Normalize embeddings
        anchor = F.normalize(anchor, dim=1)
        positive = F.normalize(positive, dim=1)
        negative = F.normalize(negative, dim=1)

        # Cosine similarities
        positive_similarity = (anchor * positive).sum(dim=1)
        negative_similarity = (anchor * negative).sum(dim=1)

        # Compute triplet loss
        loss = F.relu(self.margin + negative_similarity - positive_similarity)
        return loss.mean()

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, z1, z2, label):
        # Normalize embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Cosine similarity
        cosine_sim = (z1 * z2).sum(dim=1)

        # Contrastive loss calculation
        positive_loss = (1 - label) * (1 - cosine_sim) ** 2
        negative_loss = label * F.relu(cosine_sim - self.margin) ** 2
        loss = positive_loss + negative_loss
        return loss.mean()

class MultiLabelInfoNCELoss(nn.Module):
    def __init__(self, device, temperature=0.07):
        super(MultiLabelInfoNCELoss, self).__init__()
        self.device = device
        self.temperature = temperature
    
    def forward(
        self, 
        anchor: torch.Tensor, 
        positive: torch.Tensor, 
        negative: torch.Tensor, 
        rel_index: torch.Tensor
    ):
        # anchor: B X N_feat
        # positive: M X N_feat
        # negative: M X N_neg X N_feat
        # rel_index: M X 1 \in [0, N-1]
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negative = F.normalize(negative, dim=-1)
        
        ## Masking the Multi-labeled predicate mask
        B, M = anchor.shape[0], rel_index.shape[0]
        _mask = torch.zeros((B, M), dtype=torch.float32).to(self.device)
        _mask.scatter_(0, rel_index.reshape(-1).unsqueeze(0), 1.0)
        
        sim_ap = torch.matmul(anchor, positive.T) / self.temperature  # B x M
        sim_an = torch.einsum('nd,mkd->nmk', anchor, negative) / self.temperature  # B x M x 16
        
        sim_ap_exp = torch.exp(sim_ap) # B X M
        sim_an_exp = torch.sum(torch.exp(sim_an), dim=-1) # B X M
        
        info_nce_mat = torch.log(sim_ap_exp / sim_an_exp) * _mask # B X M
        loss = info_nce_mat[info_nce_mat != 0]
        return torch.mean(loss)