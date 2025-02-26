import torch
import torch.nn as nn 
import torch.nn.functional as F
from model.backend.attention import MultiHeadAttention
from model.backend.fat import GraphEdgeAttenNetwork

class BFeatVanillaGAT(torch.nn.Module):

    def __init__(
        self, dim_node, dim_edge, dim_atten, num_heads=1, aggr= 'max', 
        use_bn=False, flow='target_to_source', attention = 'fat', 
        hidden_size=512, depth=1, use_edge:bool=True, **kwargs,
    ):
        super(BFeatVanillaGAT, self).__init__()

        self.num_heads = num_heads
        self.depth = depth

        self.self_attn = nn.ModuleList(
            MultiHeadAttention(d_model=dim_node, d_k=dim_node // num_heads, d_v=dim_node // num_heads, h=num_heads) 
            for _ in range(depth)
        )
        
        # self.self_attn_rel = nn.ModuleList(
        #     MultiHeadAttention(d_model=dim_edge, d_k=dim_edge // num_heads, d_v=dim_edge // num_heads, h=num_heads) 
        #     for i in range(depth)
        # )
        
        self.gcn_3ds = torch.nn.ModuleList()
        
        for _ in range(self.depth):
                
            self.gcn_3ds.append(
                GraphEdgeAttenNetwork(
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
            )
           
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
        obj_feature_3d,  
        edge_feature_3d, 
        edge_index, 
        batch_ids, 
        obj_center=None,
        attn_weight=None,
        istrain=False
    ):

        # compute weight for obj
        if obj_center is not None:
            # get attention weight for object
            batch_size = batch_ids.max().item() + 1
            N_K = obj_feature_3d.shape[0]
            N_R = edge_feature_3d.shape[0]
            obj_mask = torch.zeros(1, 1, N_K, N_K).cuda()
            rel_mask = torch.zeros(1, 1, N_R, N_R).cuda()
            obj_distance_weight = torch.zeros(1, self.num_heads, N_K, N_K).cuda()
            count = 0
            count_rel = 0

            for i in range(batch_size):

                idx_i = torch.where(batch_ids == i)[0]
                L = len(idx_i)
                obj_mask[:, :, count:count + L, count:count + L] = 1
                rel_mask[:, :, count_rel:count_rel + L * (L - 1), count_rel:count_rel + L * (L - 1)] = 1
                
                center_A = obj_center[None, idx_i, :].clone().detach().repeat(L, 1, 1)
                center_B = obj_center[idx_i, None, :].clone().detach().repeat(1, L, 1)
                center_dist = (center_A - center_B)
                dist = center_dist.pow(2)
                dist = torch.sqrt(torch.sum(dist, dim=-1))[:, :, None]
                weights = torch.cat([center_dist, dist], dim=-1).unsqueeze(0)  # 1 N N 4
                dist_weights = self.self_attn_fc(weights).permute(0,3,1,2)  # 1 num_heads N N
                
                attention_matrix_way = 'add'
                obj_distance_weight[:, :, count:count + L, count:count + L] = dist_weights

                count += L
                count_rel += L * (L - 1)
            # attn_weight = attn_weight.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            # obj_distance_weight *= attn_weight
        else:
            obj_mask = None
            obj_distance = None
            rel_mask = None
            attention_matrix_way = 'mul'


        for i in range(self.depth):

            obj_feature_3d = obj_feature_3d.unsqueeze(0)
            obj_feature_3d = self.self_attn[i](
                obj_feature_3d, obj_feature_3d, obj_feature_3d, 
                attention_weights=obj_distance_weight, way=attention_matrix_way, attention_mask=obj_mask, 
                use_knn=False
            )
            # edge_feature_3d = edge_feature_3d.unsqueeze(0)
            # edge_feature_3d = self.self_attn_rel[i](edge_feature_3d, edge_feature_3d, edge_feature_3d, attention_mask=rel_mask)
            
            obj_feature_3d = obj_feature_3d.squeeze(0)
            # edge_feature_3d = edge_feature_3d.squeeze(0)
            
            obj_feature_3d, edge_feature_3d = self.gcn_3ds[i](obj_feature_3d, edge_feature_3d, edge_index, weight=attn_weight, istrain=istrain)

            if i < (self.depth-1) or self.depth==1:
                # Final processing for Object Features 
                obj_feature_3d = F.relu(obj_feature_3d)
                obj_feature_3d = self.drop_out(obj_feature_3d)
                # Final processing for Edge Features
                edge_feature_3d = F.relu(edge_feature_3d)
                edge_feature_3d = self.drop_out(edge_feature_3d)
        
        return obj_feature_3d, edge_feature_3d

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