import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_logsumexp

import torch
import torch.nn as nn
import torch.nn.functional as F

def update_temperature_based_on_gradient(loss, temperature, alpha=0.001):
    grad = torch.autograd.grad(loss, temperature, retain_graph=True)[0]  # τ에 대한 Gradient
    temperature = temperature - alpha * grad  # Gradient Descent 방식 업데이트
    return torch.clamp(temperature, min=0.01, max=1.0)  # 범위 제한

class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        """
        클래스 불균형을 고려한 Focal Loss 구현

        class_weights: 클래스별 가중치 (Tensor 또는 None)
        alpha: Focal Loss의 기본 가중치
        gamma: Hard sample에 대한 가중치 조정
        reduction: 'mean', 'sum', 또는 'none'
        """
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self, 
        inputs: torch.Tensor, 
        targets: torch.Tensor, 
        class_weights = None
    ):
        """
        inputs: B X C
        targets: B X 1
        """
        # Cross Entropy Loss 계산
        ce_loss = F.cross_entropy(inputs, targets, weight=class_weights, reduction='none')

        # 예측 확률 p_t 계산
        probs = torch.softmax(inputs, dim=1)  # 확률값 변환
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # 정답 클래스 확률 선택

        # Focal Weight 계산
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = self.alpha * class_weights * focal_weight * ce_loss

        # 손실값 반환 방식 선택
        return focal_loss.mean()


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
        assert rel_index.max() < _mask.size(0), "Index out of bounds!"
        _mask.scatter_(0, rel_index.to(torch.int64).reshape(-1).unsqueeze(0), 1.0)
        
        sim_ap = torch.matmul(anchor, positive.T) / self.temperature  # B x M
        sim_an = torch.einsum('nd,mkd->nmk', anchor, negative) / self.temperature  # B x M x N_neg
        
        sim_ap_exp = torch.exp(sim_ap) # B X M
        sim_an_exp = torch.sum(torch.exp(sim_an), dim=-1) # B X M
        
        info_nce_mat = -torch.log(sim_ap_exp / (sim_an_exp)) * _mask # B X M
        
        # Calculate positive-term wise mean
        sum_non_zero = info_nce_mat.sum(dim=1, keepdim=True)
        count_non_zero = (info_nce_mat != 0).sum(dim=1, keepdim=True).clamp(1.0)
        loss = sum_non_zero / count_non_zero
        # loss = info_nce_mat[info_nce_mat != 0]
        return torch.mean(loss)

class ContrastiveSafeLoss(nn.Module):
    def __init__(self, device, temperature=0.07):
        super(ContrastiveSafeLoss, self).__init__()
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
        assert rel_index.max() < _mask.size(0), "Index out of bounds!"
        _mask.scatter_(0, rel_index.to(torch.int64).reshape(-1).unsqueeze(0), 1.0)
        
        sim_ap = torch.matmul(anchor, positive.T)  # B x M
        sim_an = torch.einsum('nd,mkd->nmk', anchor, negative)  # B x M x 16
        
        sim_an_sum = torch.sum((sim_an + 1) * 0.5, dim=-1) # B X M
        con_loss = ((1. - sim_ap) + sim_an_sum) * _mask
        
        # Calculate positive-term wise mean
        sum_non_zero = con_loss.sum(dim=1, keepdim=True)
        count_non_zero = (con_loss != 0).sum(dim=1, keepdim=True).clamp(1.0)
        loss = sum_non_zero / count_non_zero
        return torch.mean(loss)

class SupervisedCrossModalInfoNCE(nn.Module):
    def __init__(self, device, temperature=0.07):
        super(SupervisedCrossModalInfoNCE, self).__init__()
        self.device = device
        self.temperature = temperature
    
    def __nonzero_mean(self, x: torch.Tensor):
        """
        Crazy Indexing for accurate mean
        Holy Shit
        """
        non_zero_mask = x != 0
        sum_non_zero = x.sum(dim=2, keepdim=True).sum(dim=1, keepdim=True) # B X 1 X 1
        count_non_zero = non_zero_mask.sum(dim=2, keepdim=True).sum(dim=1, keepdim=True) # B X 1 X 1
        mean_non_zero = torch.where(count_non_zero > 0, sum_non_zero / count_non_zero, torch.zeros_like(sum_non_zero).to(self.device))
        return mean_non_zero
    
    def __text_nonzero_mean(self, x: torch.Tensor):
        non_zero_mask = x != 0
        sum_non_zero = x.sum(dim=1, keepdim=True) # B X 1
        count_non_zero = non_zero_mask.sum(dim=1, keepdim=True) # B X 1 
        mean_non_zero = torch.where(count_non_zero > 0, sum_non_zero / count_non_zero, torch.zeros_like(sum_non_zero).to(self.device))
        return mean_non_zero
    
    def forward(
        self, 
        z_p: torch.Tensor, 
        z_c: torch.Tensor,  
        labels: torch.Tensor,
        mask: torch.Tensor = None
    ):
        # z_p: B X N_feat
        # z_c: 
        #   - B X K X N_feat if Image 
        #   - B X N_feat     if Text
        # gt_label: B X 1
        # mask: B X K \in {0, 1}
        B = z_p.shape[0]
        z_p = F.normalize(z_p, dim=-1)
        z_c = F.normalize(z_c, dim=-1)
        labels = labels.unsqueeze(1)
        positive_mask = (labels == labels.T).float().to(self.device)
        negative_mask = (~(labels == labels.T)).float().to(self.device)
        
        if not mask == None: # if zero-mask is given, cross-modal loss w. RGB Image
            K = z_c.shape[1]
            valid_mask = mask.unsqueeze(1).repeat(1, B, 1)
            negative_mask = negative_mask.unsqueeze(2).repeat(1, 1, K)
            positive_mask = positive_mask.unsqueeze(2).repeat(1, 1, K)
            c_sim = torch.einsum('bn,mkn->bmk', z_p, z_c) / self.temperature
            exp_sim_mat = torch.exp(c_sim)  # B X B X K
            
            # Masking valid RGB Image which exists
            neg_sim = exp_sim_mat * valid_mask * negative_mask # B X B X K, non-zero w. negative sample
            l_neg_term = neg_sim.sum(1).sum(1) # B
            l_bp = exp_sim_mat / l_neg_term.unsqueeze(1).unsqueeze(1).repeat(1, B, K) # B X B X K
            l_bp = -torch.log(l_bp) * positive_mask * valid_mask
            loss_br = self.__nonzero_mean(l_bp)
            return torch.mean(loss_br)

        else: # otherwise, cross-modal loss w. Text
            c_sim = torch.mm(z_p, z_c.T) / self.temperature # B X B
            exp_sim_mat = torch.exp(c_sim)  # B X B X K
            
            neg_sim = exp_sim_mat * negative_mask
            l_neg_term = neg_sim.sum(1, keepdim=True) # B
            l_bp = exp_sim_mat / l_neg_term.repeat(1, B)
            l_bp = -torch.log(l_bp) * positive_mask
            loss_bt = self.__text_nonzero_mean(l_bp)
            return torch.mean(loss_bt)
        

class CrossModalInfoNCE(nn.Module):
    """
    Get cross-modality wo/ self-contrastive settings.
    Calculate only 3D-Text Cross-Modality
    """
    def __init__(self, device, temperature=0.07):
        super(CrossModalInfoNCE, self).__init__()
        self.device = device
        self.temperature = temperature
    
    def __nonzero_mean(self, x: torch.Tensor):
        """
        Crazy Indexing for accurate mean
        Holy Shit
        """
        non_zero_mask = x != 0
        sum_non_zero = x.sum(dim=2, keepdim=True).sum(dim=1, keepdim=True) # B X 1 X 1
        count_non_zero = non_zero_mask.sum(dim=2, keepdim=True).sum(dim=1, keepdim=True).clamp(1.0) # B X 1 X 1
        mean_non_zero = sum_non_zero / count_non_zero
        return mean_non_zero
    
    def __text_nonzero_mean(self, x: torch.Tensor):
        non_zero_mask = x != 0
        sum_non_zero = x.sum(dim=1, keepdim=True) # B X 1
        count_non_zero = non_zero_mask.sum(dim=1, keepdim=True).clamp(1.0) # B X 1 
        mean_non_zero = sum_non_zero / count_non_zero
        return mean_non_zero
    
    def forward(
        self, 
        z_p: torch.Tensor, 
        z_c: torch.Tensor,  
        mask: torch.Tensor = None
    ):
        # z_p: B X N_feat
        # z_c: 
        #   - B X K X N_feat if Image 
        #   - B X N_feat     if Text
        # mask: B X K \in {0, 1}
        B = z_p.shape[0]
        z_p = F.normalize(z_p, dim=-1)
        z_c = F.normalize(z_c, dim=-1)
        positive_mask = torch.eye(B).float().to(self.device)
        negative_mask = (~(positive_mask.bool())).float().to(self.device)
        
        if not mask == None: # if zero-mask is given, cross-modal loss w. RGB Image
            K = z_c.shape[1]
            negative_mask = negative_mask.unsqueeze(2).repeat(1, 1, K)
            positive_mask = positive_mask.unsqueeze(2).repeat(1, 1, K)
            valid_mask = mask.unsqueeze(1).repeat(1, B, 1)
            c_sim = torch.einsum('bn,mkn->bmk', z_p, z_c) / self.temperature
            exp_sim_mat = torch.exp(c_sim)  # B X B X K
            
            # Masking valid RGB Image which exists
            neg_sim = exp_sim_mat * valid_mask * negative_mask # B X B X K, non-zero w. negative sample
            l_neg_term = neg_sim.sum(1).sum(1) # B
            l_bp = exp_sim_mat / l_neg_term.unsqueeze(1).unsqueeze(1).repeat(1, B, K) # B X B X K
            l_bp = -torch.log(l_bp) * positive_mask * valid_mask
            loss_br = self.__nonzero_mean(l_bp)
            return torch.mean(loss_br)

        else: # otherwise, cross-modal loss w. Text
            c_sim = torch.mm(z_p, z_c.T) / self.temperature # B X B
            exp_sim_mat = torch.exp(c_sim)  # B X B X K
            
            neg_sim = exp_sim_mat * negative_mask
            l_neg_term = neg_sim.sum(1, keepdim=True) # B
            l_bp = exp_sim_mat / l_neg_term.repeat(1, B)
            l_bp = -torch.log(l_bp) * positive_mask
            loss_bt = self.__text_nonzero_mean(l_bp)
            return torch.mean(loss_bt)

class CrossModalNXTent(nn.Module):
    """
    Get cross-modality wo/ self-contrastive settings.
    Calculate only 3D-Text Cross-Modality
    """
    def __init__(self, device, temperature=0.07):
        super(CrossModalInfoNCE, self).__init__()
        self.device = device
        self.temperature = temperature
    
    def __nonzero_mean(self, x: torch.Tensor):
        """
        Crazy Indexing for accurate mean
        Holy Shit
        """
        non_zero_mask = x != 0
        sum_non_zero = x.sum(dim=2, keepdim=True).sum(dim=1, keepdim=True) # B X 1 X 1
        count_non_zero = non_zero_mask.sum(dim=2, keepdim=True).sum(dim=1, keepdim=True).clamp(1.0) # B X 1 X 1
        mean_non_zero = sum_non_zero / count_non_zero
        return mean_non_zero
    
    def __text_nonzero_mean(self, x: torch.Tensor):
        non_zero_mask = x != 0
        sum_non_zero = x.sum(dim=1, keepdim=True) # B X 1
        count_non_zero = non_zero_mask.sum(dim=1, keepdim=True).clamp(1.0) # B X 1 
        mean_non_zero = sum_non_zero / count_non_zero
        return mean_non_zero
    
    def forward(
        self, 
        z_p: torch.Tensor, 
        z_c: torch.Tensor,  
        mask: torch.Tensor = None
    ):
        # z_p: B X N_feat
        # z_c: 
        #   - B X K X N_feat if Image 
        #   - B X N_feat     if Text
        # mask: B X K \in {0, 1}
        B = z_p.shape[0]
        z_p = F.normalize(z_p, dim=-1)
        z_c = F.normalize(z_c, dim=-1)
        positive_mask = torch.eye(B).float().to(self.device)
        negative_mask = (~(positive_mask.bool())).float().to(self.device)
        
        # Intra-Modal negative similarity 
        pc_neg_sim = torch.mm('bn,mn->bm', z_p, z_p)
        exp_pc_neg_sim = torch.exp(pc_neg_sim) * negative_mask
        
        if not mask == None: # if zero-mask is given, cross-modal loss w. RGB Image
            K = z_c.shape[1]
            negative_mask = negative_mask.unsqueeze(2).repeat(1, 1, K)
            positive_mask = positive_mask.unsqueeze(2).repeat(1, 1, K)
            valid_mask = mask.unsqueeze(1).repeat(1, B, 1)
            c_sim = torch.einsum('bn,mkn->bmk', z_p, z_c) / self.temperature
            exp_sim_mat = torch.exp(c_sim)  # B X B X K
            
            # Masking valid RGB Image which exists
            exp_pc_neg_sim = exp_pc_neg_sim.unsqueeze(2).repeat(1, 1, K)
            neg_sim = exp_sim_mat * valid_mask + exp_pc_neg_sim * valid_mask # B X B X K, non-zero w. negative sample
            l_neg_term = neg_sim.sum(1).sum(1) # B
            l_bp = exp_sim_mat / l_neg_term.unsqueeze(1).unsqueeze(1).repeat(1, B, K) # B X B X K
            l_bp = -torch.log(l_bp) * positive_mask * valid_mask
            loss_br = self.__nonzero_mean(l_bp)
            return torch.mean(loss_br)

        else: # otherwise, cross-modal loss w. Text
            c_sim = torch.mm(z_p, z_c.T) / self.temperature # B X B
            exp_sim_mat = torch.exp(c_sim)  # B X B X K
            
            neg_sim = exp_sim_mat + exp_pc_neg_sim
            l_neg_term = neg_sim.sum(1, keepdim=True) # B
            l_bp = exp_sim_mat / l_neg_term.repeat(1, B)
            l_bp = -torch.log(l_bp) * positive_mask
            loss_bt = self.__text_nonzero_mean(l_bp)
            return torch.mean(loss_bt)

class IntraModalBarlowTwinLoss(nn.Module):
    def __init__(self, _lambda=5e-3):
        super(IntraModalBarlowTwinLoss, self).__init__()
        self._lambda = _lambda
        
    def forward(
        self, 
        z_a: torch.Tensor, 
        z_b: torch.Tensor
    ):
        z_a = F.normalize(z_a, dim=-1)
        z_b = F.normalize(z_b, dim=-1)
        
        z_a = (z_a - z_a.mean(dim=0)) / z_a.std(dim=0)
        z_b = (z_b - z_b.mean(dim=0)) / z_b.std(dim=0)
        # 2. 크로스 상관행렬 C 계산 (DxD 크기, D는 임베딩 차원)
        batch_size = z_a.size(0)
        c = torch.mm(z_a.T, z_b) / batch_size  # 배치 크기로 나누어 평균 상관값 계산
        # 3. 손실 함수 항목 계산
        identity = torch.eye(c.size(0)).to(c.device)  # 단위 행렬 (대각선이 1인 행렬)
        # - 대각선 요소를 1에 가깝게 만드는 불변성(invariance) 항
        invariance_term = (c.diag() - 1).pow(2).sum()
        # - 대각선 외 요소들을 0에 가깝게 만드는 중복 감소(redundancy reduction) 항
        redundancy_term = ((c - identity).pow(2).sum() - invariance_term)
        # 4. 최종 손실 계산
        loss = invariance_term + self._lambda * redundancy_term
        return loss