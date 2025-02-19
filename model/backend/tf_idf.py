import torch
import torch.nn as nn
import math
import numpy as np

class TfIdfLayer(nn.Module):
    def __init__(self, epsilon=0.0, gamma=0.0, bias=False):
        super(TfIdfLayer, self).__init__()
        self.epsilon = epsilon
        self.gamma = gamma
        self.bias = bias
        if self.bias:
            self.epsilon = nn.Parameter(torch.randn(1), requires_grad=True)
            self.gamma = nn.Parameter(torch.randn(1), requires_grad=True)
            
    def forward(self, x, labels):
        num_img = len(labels)
        t_id_list = []
        for label in labels:
            for ob_label in label.extra_fields['pred_labels']:
                tf_idf = self.tfidf(ob_label, label.extra_fields['pred_labels'], num_img, labels)
                t_id_list.append(tf_idf)
        weighted_x = torch.tensor(t_id_list).unsqueeze(-1).cuda() * x
        return weighted_x
    
    def tf(self, t, d):
        return torch.count_nonzero(torch.where(d == t, True, False)).item()
    
    def idf(self, t, N, docs):
        ni = 0
        for doc in docs:
            ni += t in doc.extra_fields['pred_labels']
        if self.bias:
            return math.log((N + self.epsilon) / (ni + 1 + self.gamma))
        else:
            return math.log(N / (ni + 1))
        
    def tfidf(self, t, d, N, docs):
        return self.tf(t, d) * self.idf(t, N, docs)