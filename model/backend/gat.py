import torch
import torch.nn as nn 
import torch.nn.functional as F
from model.backend.attention import MultiHeadAttention
from model.backend.fat import GraphEdgeAttenNetwork

class BFeatVanillaGAT(torch.nn.Module):
    def __init__(
        self, dim_node, dim_edge, dim_atten, num_heads=1, aggr='max', 
        use_bn=False, flow='target_to_source', attention='fat', 
        hidden_size=512, node_depth=2, edge_depth=1, use_edge:bool=True, 
        distance_threshold=None, k_neighbors=2, **kwargs,
    ):
        super(BFeatVanillaGAT, self).__init__()

        self.num_heads = num_heads
        self.node_depth = node_depth
        self.edge_depth = edge_depth
        self.k_neighbors = k_neighbors
        self.distance_threshold = distance_threshold
        self.edge_attention = kwargs["edge_attn"]

        self.self_attn = MultiHeadAttention(
            d_model=dim_node, d_k=dim_node // num_heads, 
            d_v=dim_node // num_heads, h=num_heads
        )

        if self.edge_attention:        
            self.self_attn_rel = MultiHeadAttention(
                d_model=dim_edge, d_k=dim_edge // num_heads, 
                d_v=dim_edge // num_heads, h=num_heads
            )
        
        self.gcn_3d = GraphEdgeAttenNetwork(
            num_heads,
            dim_node,
            dim_edge,
            dim_atten,
            aggr,
            use_bn=use_bn,
            flow=flow,
            attention=attention,
            use_edge=use_edge, 
            **kwargs
        )
           
        self.self_attn_fc = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, num_heads)
        )
        
        self.drop_out = torch.nn.Dropout(kwargs['DROP_OUT_ATTEN'])
    
    def forward(
        self, 
        obj_feature_3d,  
        edge_feature_3d, 
        edge_index, 
        batch_ids, 
        obj_center=None,
        attn_weight=None,
        istrain=False
    ):
        original_obj_feature = obj_feature_3d.clone()
        # original_edge_feature = edge_feature_3d.clone()
        
        if obj_center is not None:
            batch_size = batch_ids.max().item() + 1
            N_K = obj_feature_3d.shape[0]
            
            distances = torch.zeros(N_K, N_K).cuda()
            
            start_idx = 0
            for i in range(batch_size):
                idx_i = torch.where(batch_ids == i)[0]
                L = len(idx_i)
                
                center_A = obj_center[None, idx_i, :].clone().detach().repeat(L, 1, 1)
                center_B = obj_center[idx_i, None, :].clone().detach().repeat(1, L, 1)
                center_dist = (center_A - center_B)
                dist = center_dist.pow(2)
                dist = torch.sqrt(torch.sum(dist, dim=-1))  # L x L
                
                distances[start_idx:start_idx+L, start_idx:start_idx+L] = dist
                start_idx += L
            
            if self.distance_threshold is not None:
                distance_mask = (distances <= self.distance_threshold).float()
            else:
                distance_mask = torch.ones_like(distances)
            
            k_nearest_mask = torch.zeros_like(distances)
            for i in range(N_K):
                dist_i = distances[i].clone()
                dist_i[i] = 0.0
                
                dist_i_no_self = dist_i.clone()
                dist_i_no_self[i] = float('inf')
                
                if self.k_neighbors < N_K:
                    _, topk_indices = torch.topk(dist_i_no_self, min(self.k_neighbors, N_K-1), largest=False)
                    k_nearest_mask[i, topk_indices] = 1.0
                else:
                    k_nearest_mask[i, :] = 1.0
                
                k_nearest_mask[i, i] = 1.0
            
            final_mask = distance_mask * k_nearest_mask
            
            batch_mask = torch.zeros_like(distances)
            start_idx = 0
            for i in range(batch_size):
                idx_i = torch.where(batch_ids == i)[0]
                L = len(idx_i)
                batch_mask[start_idx:start_idx+L, start_idx:start_idx+L] = 1.0
                start_idx += L
            
            final_mask = final_mask * batch_mask
            
            obj_distance_weight = torch.zeros(1, self.num_heads, N_K, N_K).cuda()
            
            start_idx = 0
            for i in range(batch_size):
                idx_i = torch.where(batch_ids == i)[0]
                L = len(idx_i)
                
                center_A = obj_center[None, idx_i, :].clone().detach().repeat(L, 1, 1)
                center_B = obj_center[idx_i, None, :].clone().detach().repeat(1, L, 1)
                center_dist = (center_A - center_B)
                dist = center_dist.pow(2)
                dist = torch.sqrt(torch.sum(dist, dim=-1))[:, :, None]
                
                weights = torch.cat([center_dist, dist], dim=-1).unsqueeze(0)  # 1 L L 4
                dist_weights = self.self_attn_fc(weights).permute(0, 3, 1, 2)  # 1 num_heads L L
                
                obj_distance_weight[:, :, start_idx:start_idx+L, start_idx:start_idx+L] = dist_weights
                start_idx += L
            
            # attention_mask = final_mask.unsqueeze(0).unsqueeze(0)  # 1 1 N_K N_K
            
            hop_masks = []
            
            adj_matrix = torch.zeros(N_K, N_K).cuda()
            for e in range(edge_index.size(1)):
                src, dst = edge_index[0, e], edge_index[1, e]
                adj_matrix[src, dst] = 1.0
                adj_matrix[dst, src] = 1.0 # bidirectional
            
            hop1_mask = adj_matrix + torch.eye(N_K).cuda()
            hop_masks.append(hop1_mask)
            
            prev_hop_mask = hop1_mask
            for h in range(1, self.node_depth):
                next_hop_mask = torch.matmul(prev_hop_mask, adj_matrix)
                next_hop_mask = (next_hop_mask > 0).float()
                hop_masks.append(next_hop_mask)
                prev_hop_mask = next_hop_mask
        else:
            attention_mask = None
            obj_distance_weight = None
            hop_masks = [torch.ones(N_K, N_K).cuda() for _ in range(self.node_depth)]
        
        if self.edge_attention:
            edge_feature_3d = edge_feature_3d.unsqueeze(0)
            edge_feature_3d = self.self_attn_rel(edge_feature_3d, edge_feature_3d, edge_feature_3d)
            edge_feature_3d = edge_feature_3d.squeeze(0)
        
        updated_features = []
        current_features = original_obj_feature.clone()
        
        for hop in range(self.node_depth):
            if obj_center is not None:
                hop_attention_mask = (hop_masks[hop] * final_mask).unsqueeze(0).unsqueeze(0)
            else:
                hop_attention_mask = None
            
            current_features = current_features.unsqueeze(0)
            current_features = self.self_attn(
                current_features, current_features, current_features,
                attention_mask=hop_attention_mask,
                attention_weights=obj_distance_weight,
                way='add' if obj_distance_weight is not None else 'mul',
                use_knn=True
            )
            current_features = current_features.squeeze(0)
            
            if hop == 0:
                current_features, edge_feature_3d = self.gcn_3d(
                    current_features, edge_feature_3d, edge_index,
                    weight=attn_weight, istrain=istrain
                )
            else:
                current_features, _ = self.gcn_3d(
                    current_features, edge_feature_3d, edge_index,
                    weight=attn_weight, istrain=istrain
                )
            
            current_features = F.relu(current_features)
            current_features = self.drop_out(current_features)
            
            updated_features.append(current_features)
        
        final_obj_feature = updated_features[-1]
        
        return final_obj_feature, edge_feature_3d

class BFeatContrastiveAuxGAT(nn.Module):
    def __init__(
        self, dim_node, dim_edge, dim_atten, num_heads=1, aggr= 'max', 
        use_bn=False, flow='target_to_source', attention = 'fat', 
        hidden_size=512, depth=1, use_edge:bool=True, **kwargs,
    ):
        super(BFeatContrastiveAuxGAT, self).__init__()
        self.num_heads = num_heads
        self.depth = depth

        self.self_attn = nn.ModuleList(
            MultiHeadAttention(d_model=dim_node, d_k=dim_node // num_heads, d_v=dim_node // num_heads, h=num_heads) for i in range(depth))

        self.cross_attn = nn.ModuleList(
            MultiHeadAttention(d_model=dim_node, d_k=dim_node // num_heads, d_v=dim_node // num_heads, h=num_heads) for i in range(depth))

        self.cross_attn_rel = nn.ModuleList(
            MultiHeadAttention(d_model=dim_edge, d_k=dim_edge // num_heads, d_v=dim_edge // num_heads, h=num_heads) for i in range(depth))
        
        self.gcn_con = torch.nn.ModuleList()
        self.gcn_sgg = torch.nn.ModuleList()
        
        for _ in range(self.depth):

            self.gcn_con.append(GraphEdgeAttenNetwork(
                            num_heads,
                            dim_node,
                            dim_edge,
                            dim_atten,
                            aggr,
                            use_bn=use_bn,
                            flow=flow,
                            attention=attention,
                            use_edge=use_edge, 
                            **kwargs))
            
            self.gcn_sgg.append(GraphEdgeAttenNetwork(
                            num_heads,
                            dim_node,
                            dim_edge,
                            dim_atten,
                            aggr,
                            use_bn=use_bn,
                            flow=flow,
                            attention=attention,
                            use_edge=use_edge, 
                            **kwargs))
           
        self.self_attn_fc = nn.Sequential(  # 11 32 32 4(head)
            nn.Linear(4, 32),  # xyz, dist
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, num_heads)
        )
        
        self.drop_out = torch.nn.Dropout(kwargs['DROP_OUT_ATTEN'])
    
    def forward(
        self, 
        obj_feature_sgg: torch.Tensor, 
        obj_feature_con: torch.Tensor, 
        edge_feature_ssg: torch.Tensor, 
        edge_feature_con: torch.Tensor, 
        edge_index: torch.Tensor, 
        batch_ids: torch.Tensor, 
        obj_center=None, 
        discriptor=None, 
        istrain=False
    ):
        # compute weight for obj
        if obj_center is not None:
            # get attention weight for object
            batch_size = batch_ids.max().item() + 1
            N_K = obj_feature_sgg.shape[0]
            obj_mask = torch.zeros(1, 1, N_K, N_K).cuda()
            obj_distance_weight = torch.zeros(1, self.num_heads, N_K, N_K).cuda()
            count = 0

            for i in range(batch_size):

                idx_i = torch.where(batch_ids == i)[0]
                obj_mask[:, :, count:count + len(idx_i), count:count + len(idx_i)] = 1
            
                center_A = obj_center[None, idx_i, :].clone().detach().repeat(len(idx_i), 1, 1)
                center_B = obj_center[idx_i, None, :].clone().detach().repeat(1, len(idx_i), 1)
                center_dist = (center_A - center_B)
                dist = center_dist.pow(2)
                dist = torch.sqrt(torch.sum(dist, dim=-1))[:, :, None]
                weights = torch.cat([center_dist, dist], dim=-1).unsqueeze(0)  # 1 N N 4
                dist_weights = self.self_attn_fc(weights).permute(0,3,1,2)  # 1 num_heads N N
                
                attention_matrix_way = 'add'
                obj_distance_weight[:, :, count:count + len(idx_i), count:count + len(idx_i)] = dist_weights

                count += len(idx_i)
        else:
            obj_mask = None
            obj_distance = None
            attention_matrix_way = 'mul'


        for i in range(self.depth):

            obj_feature_sgg = obj_feature_sgg.unsqueeze(0)
            obj_feature_con = obj_feature_con.unsqueeze(0)
            
            obj_feature_sgg = self.self_attn[i](obj_feature_sgg, obj_feature_sgg, obj_feature_sgg, attention_weights=obj_distance_weight, way=attention_matrix_way, attention_mask=obj_mask, use_knn=False)
            obj_feature_con = self.cross_attn[i](obj_feature_con, obj_feature_sgg, obj_feature_sgg, attention_weights=obj_distance_weight, way=attention_matrix_way, attention_mask=obj_mask, use_knn=False)
            
            obj_feature_sgg = obj_feature_sgg.squeeze(0)
            obj_feature_con = obj_feature_con.squeeze(0)  


            obj_feature_sgg, edge_feature_ssg = self.gcn_sgg[i](obj_feature_sgg, edge_feature_ssg, edge_index, istrain=istrain)
            obj_feature_con, edge_feature_con = self.gcn_con[i](obj_feature_con, edge_feature_con, edge_index, istrain=istrain)

            
            edge_feature_con = edge_feature_con.unsqueeze(0)
            edge_feature_ssg = edge_feature_ssg.unsqueeze(0)
            
            edge_feature_con = self.cross_attn_rel[i](edge_feature_con, edge_feature_ssg, edge_feature_ssg, use_knn=False)
            
            edge_feature_con = edge_feature_con.squeeze(0)
            edge_feature_ssg = edge_feature_ssg.squeeze(0)

            if i < (self.depth-1) or self.depth==1:
                
                obj_feature_sgg = F.relu(obj_feature_sgg)
                obj_feature_sgg = self.drop_out(obj_feature_sgg)
                
                obj_feature_con = F.relu(obj_feature_con)
                obj_feature_con = self.drop_out(obj_feature_con)

                edge_feature_ssg = F.relu(edge_feature_ssg)
                edge_feature_ssg = self.drop_out(edge_feature_ssg)

                edge_feature_con = F.relu(edge_feature_con)
                edge_feature_con = self.drop_out(edge_feature_con)
        
        return obj_feature_sgg, obj_feature_con, edge_feature_ssg, edge_feature_con