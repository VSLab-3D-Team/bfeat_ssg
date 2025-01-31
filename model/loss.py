import torch
import torch.nn as nn
import torch.nn.functional as F

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
