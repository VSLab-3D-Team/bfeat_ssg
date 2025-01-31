import numpy as np
import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h):
        super(ScaledDotProductAttention, self).__init__()
        self.proj_q = nn.Linear(d_model, h * d_k)
        self.proj_k = nn.Linear(d_model, h * d_k)
        self.proj_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        
        self.__init_weights()
        
    def __init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):  # nn.Linear 레이어만 선택
                nn.init.xavier_uniform_(m.weight)  # Xavier Uniform 방식으로 초기화
                if m.bias is not None:  # Bias가 존재하면 초기화
                    nn.init.constant_(m.bias, 0)  # Bias는 0으로 초기화
    
    def forward(
        self, 
        queries, keys, values, 
        attention_mask=None, attention_weights=None, 
        way='mul', use_knn=False
    ):
        bsz, n_q = queries.shape[:2]
        n_k = keys.shape[1]
        
        Q = self.proj_q(queries)
        Q = Q.view(bsz, n_q, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, n_q, d_k)
        K = self.proj_k(keys).view(bsz, n_k, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, n_k, d_k)
        V = self.proj_v(values).view(bsz, n_k, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)
        
        att = torch.matmul(Q, K) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        att_map = att.clone()
        if use_knn:
            att = attention_weights
        else:
            if attention_weights is not None:
                if way == 'mul':
                    att = att * attention_weights
                elif way == 'add':
                    #print(att.shape, attention_weights.shape, '<< att shape; add')
                    att = att + attention_weights
                else:
                    raise NotImplementedError(way)
        if attention_mask is not None:
            att = att.masked_fill(attention_mask==0, -np.inf)
        att = torch.softmax(att, -1)
        out = torch.matmul(att, V).permute(0, 2, 1, 3).contiguous().view(bsz, n_q, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out, att_map


class MultiHeadAttention(nn.Module):
    '''
    Multi-head attention layer with Dropout and Layer Normalization.
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.1, identity_map_reordering=False, can_be_stateful=False,
                 attention_module=None, attention_module_kwargs=None):
        super(MultiHeadAttention, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        if attention_module is not None:
            if attention_module_kwargs is not None:
                self.attention = attention_module(d_model=d_model, d_k=d_k, d_v=d_v, h=h, **attention_module_kwargs)
            else:
                self.attention = attention_module(d_model=d_model, d_k=d_k, d_v=d_v, h=h, m = 20)
        else:
            self.attention = ScaledDotProductAttention(d_model=d_model, d_k=d_k, d_v=d_v, h=h)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.can_be_stateful = can_be_stateful
        if self.can_be_stateful:
            self.register_state('running_keys', torch.zeros((0, d_model)))
            self.register_state('running_values', torch.zeros((0, d_model)))

    def forward(
        self, queries, keys, values, 
        attention_mask=None, attention_weights=None, way='mul', 
        use_knn=False, output_attn=False
    ):
        if self.can_be_stateful and self._is_stateful:
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            keys = self.running_keys

            self.running_values = torch.cat([self.running_values, values], 1)
            values = self.running_values

        if self.identity_map_reordering:
            q_norm = self.layer_norm(queries)
            k_norm = self.layer_norm(keys)
            v_norm = self.layer_norm(values)
            out, att = self.attention(q_norm, k_norm, v_norm, attention_mask, attention_weights, way)
            out = queries + self.dropout(torch.relu(out))
        else:
            out, att = self.attention(queries, keys, values, attention_mask, attention_weights, way, use_knn)
            out = self.dropout(out)
            out = self.layer_norm(queries + out)
        if output_attn:
            return out, att
        else:
            return out
