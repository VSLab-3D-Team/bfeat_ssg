from torch.utils.data import Dataset
from config.define import *
from utils.os_utils import read_3dssg_annotation
from utils.data_utils import gen_descriptor
import torch
import numpy as np
from tqdm import tqdm
from glob import glob
from itertools import product

## Dataloading strategy
### Load all data of each scan in constructor
### data: encoding vectors, metadata for graph relationship
class SSGFeatGeoAuxDataset(Dataset):
    def __init__(
        self, config, split, device, 
        scan_data, relationship_json, objs_json, scans
    ):
        super(SSGFeatGeoAuxDataset, self).__init__()
        
        self._scan_path = SSG_DATA_PATH
        self.config = config
        self.path_3rscan = f"{SSG_DATA_PATH}/3RScan/data/3RScan"
        self.path_selection = f"{SSG_DATA_PATH}/3DSSG_subset"
        self.for_train = True if split == "train_scans" else False
        self.use_rgb = True
        self.use_normal = True
        self.device = device
        self.dim_pts = 3
        if self.use_rgb:
            self.dim_pts += 3
        if self.use_normal:
            self.dim_pts += 3
        
        self.data_path = f"{SSG_DATA_PATH}/3DSSG_subset"
        self.classNames, self.relationNames, _, _ = \
            read_3dssg_annotation(self.data_path, self.path_selection, split)
        
        # for multi relation output, we just remove off 'None' relationship
        if self.config.multi_rel:
            self.relationNames.pop(0)
        
        self.relationship_json, self.objs_json, self.scans = relationship_json, objs_json, scans
        self.scan_data = scan_data
        
    def __len__(self):
        return len(self.scan_data)

    def norm_tensor(self, points):
        assert points.ndim == 2
        assert points.shape[1] == 3
        centroid = torch.mean(points, dim=0) # N, 3
        points -= centroid # n, 3, npts
        # furthest_distance = points.pow(2).sum(1).sqrt().max() # find maximum distance for each n -> [n]
        # points /= furthest_distance
        return points 
    
    def zero_mean(self, point):
        mean = torch.mean(point, dim=0)
        point -= mean.unsqueeze(0)
        ''' without norm to 1  '''
        # furthest_distance = point.pow(2).sum(1).sqrt().max() # find maximum distance for each n -> [n]
        # point /= furthest_distance
        return point  

    '''
    Cropping object point cloud from point cloud of entire scene 
    '''
    def __crop_obj_pts(self, s_point, obj_mask, instance_id, num_sample_pts, padding=0.2):
        obj_pointset = s_point[np.where(obj_mask == instance_id)[0]]
        min_box = np.min(obj_pointset[:,:3], 0) - padding
        max_box = np.max(obj_pointset[:,:3], 0) + padding
        obj_bbox = (min_box,max_box)  
        choice = np.random.choice(len(obj_pointset), num_sample_pts, replace=True)
        obj_pointset = obj_pointset[choice, :]
        descriptor = gen_descriptor(torch.from_numpy(obj_pointset)[:,:3])
        obj_pointset = torch.from_numpy(obj_pointset.astype(np.float32))
        obj_pointset[:,:3] = self.zero_mean(obj_pointset[:,:3])
        return obj_pointset, obj_bbox, descriptor
    
    '''
    Get training data from one scan
    - object point cloud
    - point cloud of two object w. relationship
    - edge indices
    - descriptors for geometric information
    '''
    def __get_data(
        self, scene_points, obj_masks, num_pts_normalized, relationships,
        instance_map, rel_json, rel_viewpoint_clip, padding=0.2, all_edge=True
    ):
        all_instance = list(np.unique(obj_masks))
        nodes_all = list(instance_map.keys())

        if 0 in all_instance: # remove background
            all_instance.remove(0)
        
        nodes = []
        for i, instance_id in enumerate(nodes_all):
            if instance_id in all_instance:
                nodes.append(instance_id)
        
        # get edge (instance pair) list, which is just index, nodes[index] = instance_id
        if all_edge:
            edge_indices = list(product(list(range(len(nodes))), list(range(len(nodes)))))
            # filter out (i,i)
            edge_indices = [i for i in edge_indices if i[0]!=i[1]]
        else:
            edge_indices = [(nodes.index(r[0]), nodes.index(r[1])) for r in rel_json if r[0] in nodes and r[1] in nodes]
        
        num_objects = len(nodes)
        dim_point = scene_points.shape[-1]
        
        instances_box, label_node = dict(), []
        obj_points = torch.zeros([num_objects, num_pts_normalized, dim_point])
        descriptor = torch.zeros([num_objects, 11])

        # obj_2d_feats = np.zeros([num_objects, 512])
        
        for i, instance_id in enumerate(nodes):
            assert instance_id in all_instance, "invalid instance id"
            # get node label name
            instance_name = instance_map[instance_id]
            label_node.append(self.classNames.index(instance_name))
            # get node point
            obj_pts, obj_bbox, desc = self.__crop_obj_pts(scene_points, obj_masks, instance_id, num_pts_normalized, padding=padding)
            instances_box[instance_id] = obj_bbox
            descriptor[i] = desc
            obj_points[i] = obj_pts

        # set gt label for relation
        len_object = len(nodes)
        if self.config.multi_rel:
            adj_matrix_onehot = np.zeros([len_object, len_object, len(relationships)])
        else:
            adj_matrix = np.zeros([len_object, len_object]) #set all to none label.
        
        for r in rel_json:
            if r[0] not in nodes or r[1] not in nodes: continue
            assert r[3] in relationships, "invalid relation name"
            r[2] = relationships.index(r[3]) # remap the index of relationships in case of custom relationNames

            if self.config.multi_rel:
                adj_matrix_onehot[nodes.index(r[0]), nodes.index(r[1]), r[2]] = 1
            else:
                adj_matrix[nodes.index(r[0]), nodes.index(r[1])] = r[2]
        
        # get relation union points
        if self.config.multi_rel:
            adj_matrix_onehot = torch.from_numpy(np.array(adj_matrix_onehot, dtype=np.float32))
            gt_rels = torch.zeros(len(edge_indices), len(relationships),dtype = torch.float)
        else:
            adj_matrix = torch.from_numpy(np.array(adj_matrix, dtype=np.int64))
            gt_rels = torch.zeros(len(edge_indices), dtype = torch.long)     
        
        rel_points = list()
        edge_feat_mask = torch.zeros((len(edge_indices),))
        edge_2d_feats = torch.zeros((len(edge_indices), 512))
        for e in range(len(edge_indices)):
            edge = edge_indices[e]
            index1 = edge[0]
            index2 = edge[1]
            instance1 = nodes[edge[0]]
            instance2 = nodes[edge[1]]

            if self.config.relpts == "obj_only":
                obj_pts_1, _, _ = self.__crop_obj_pts(scene_points, obj_masks, instance1, num_pts_normalized, padding) # dim: N_pts X self.dim_pts
                obj_pts_2, _, _ = self.__crop_obj_pts(scene_points, obj_masks, instance2, num_pts_normalized, padding) # dim: N_pts X self.dim_pts
                
                edge_pts = np.concatenate([obj_pts_1, obj_pts_2], axis=0) # dim: (2 * N_pts) X self.dim_pts
                rel_points.append(torch.from_numpy(edge_pts.astype(np.float32))) # 
            else:
                mask1 = (obj_masks == instance1).astype(np.int32) * 1
                mask2 = (obj_masks == instance2).astype(np.int32) * 2
                mask_ = np.expand_dims(mask1 + mask2, 1)
                bbox1 = instances_box[instance1]
                bbox2 = instances_box[instance2]
                min_box = np.minimum(bbox1[0], bbox2[0])
                max_box = np.maximum(bbox1[1], bbox2[1])
                filter_mask = (scene_points[:,0] > min_box[0]) * (scene_points[:,0] < max_box[0]) \
                            * (scene_points[:,1] > min_box[1]) * (scene_points[:,1] < max_box[1]) \
                            * (scene_points[:,2] > min_box[2]) * (scene_points[:,2] < max_box[2])
                
                # add with context, to distingush the different object's points
                points4d = np.concatenate([scene_points, mask_], 1)
                pointset = points4d[np.where(filter_mask > 0)[0], :]
                choice = np.random.choice(len(pointset), 2 * num_pts_normalized, replace=True)
                pointset = pointset[choice, :]
                pointset = torch.from_numpy(pointset.astype(np.float32))
                pointset[:,:3] = self.zero_mean(pointset[:,:3])
                rel_points.append(pointset)
            
            if self.config.multi_rel:
                gt_rels[e,:] = adj_matrix_onehot[index1,index2,:]
                if adj_matrix_onehot[index1, index2, :].sum() > 0.:
                    edge_feat_mask[e] = 1.
                    edge_feats = np.concatenate(rel_viewpoint_clip[index1][index2], axis=0)
                    edge_2d_feats[e] = torch.from_numpy(np.mean(edge_feats, axis=0)).float()
                else: 
                    edge_feat_mask[e] = 0.
            else:
                gt_rels[e] = adj_matrix[index1,index2]
                edge_feat_mask[e] = 1.

        if len(rel_points) > 0:
            rel_points = torch.stack(rel_points, 0)
        else:
            rel_points = torch.tensor([])
        
        label_node = torch.from_numpy(np.array(label_node, dtype=np.int64))
        edge_indices = torch.tensor(edge_indices,dtype=torch.long)
        edge_2d_feats = edge_2d_feats[edge_feat_mask.bool()]
        return obj_points, rel_points, descriptor, edge_2d_feats, gt_rels, label_node, edge_indices, edge_feat_mask.bool()
    
    ## Things to return
    ### Object features in the graph
    ### Edge Geometric descriptors
    ### Ground Truth Object label
    ### Ground Truth predicate label
    ### Edge Indices
    def __getitem__(self, index):
        scan_data = self.scan_data[index]
        obj_pts, rel_pts, descriptor, edge_2d_feats, gt_rels, gt_class, edge_indices, edge_feat_mask = \
            self.__get_data(
                scan_data["points"], scan_data["mask"], self.config.num_points_reg, 
                self.relationNames, scan_data["map_instid_name"], scan_data["rel_json"], scan_data["edge_view_rgb"],
                self.config.padding, self.config.all_edges
            )
        
        while(len(rel_pts) == 0 or gt_rels.sum()==0) and self.for_train:
            index = np.random.randint(self.__len__())
            obj_pts, rel_pts, descriptor, edge_2d_feats, gt_rels, gt_class, edge_indices, edge_feat_mask = \
                self.__getitem__(index)
        
        return obj_pts[..., :self.dim_pts], \
            rel_pts[..., :self.dim_pts], \
            descriptor, \
            edge_2d_feats, \
            gt_rels, gt_class, \
            edge_indices, edge_feat_mask

class SSGLWBFeat3DwMMGeo(Dataset):
    def __init__(
        self, config, split, device, d_feats,
        scan_data, relationship_json, objs_json, scans
    ):
        super(SSGLWBFeat3DwMMGeo, self).__init__()
        self._scan_path = SSG_DATA_PATH
        self.config = config
        self.path_3rscan = f"{SSG_DATA_PATH}/3RScan/data/3RScan"
        self.path_selection = f"{SSG_DATA_PATH}/3DSSG_subset"
        self.for_train = True if split == "train_scans" else False
        self.use_rgb = True
        self.use_normal = True
        self.device = device
        self.d_feats = d_feats
        self.dim_pts = 3
        if self.use_rgb:
            self.dim_pts += 3
        if self.use_normal:
            self.dim_pts += 3
        
        self.data_path = f"{SSG_DATA_PATH}/3DSSG_subset"
        self.classNames, self.relationNames, _, _ = \
            read_3dssg_annotation(self.data_path, self.path_selection, split)
        
        # for multi relation output, we just remove off 'None' relationship
        if self.config.multi_rel:
            self.relationNames.pop(0)
        
        self.relationship_json, self.objs_json, self.scans = relationship_json, objs_json, scans
        self.scan_data = scan_data
        
    def __len__(self):
        return len(self.scan_data)

    def norm_tensor(self, points):
        assert points.ndim == 2
        assert points.shape[1] == 3
        centroid = torch.mean(points, dim=0) # N, 3
        points -= centroid # n, 3, npts
        return points 
    
    def zero_mean(self, point):
        mean = torch.mean(point, dim=0)
        point -= mean.unsqueeze(0)
        ''' without norm to 1  '''
        return point  

    '''
    Cropping object point cloud from point cloud of entire scene 
    '''
    def __crop_obj_pts(self, s_point, obj_mask, instance_id, num_sample_pts, padding=0.2, is_obj=True):
        obj_pointset = s_point[np.where(obj_mask == instance_id)[0]]
        min_box = np.min(obj_pointset[:,:3], 0) - padding
        max_box = np.max(obj_pointset[:,:3], 0) + padding
        obj_bbox = (min_box,max_box)  
        choice = np.random.choice(len(obj_pointset), num_sample_pts, replace=True)
        obj_pointset = obj_pointset[choice, :]
        descriptor = gen_descriptor(torch.from_numpy(obj_pointset)[:,:3])
        obj_pointset = torch.from_numpy(obj_pointset.astype(np.float32))
        if is_obj:
            obj_pointset[:,:3] = self.zero_mean(obj_pointset[:,:3])
        return obj_pointset, obj_bbox, descriptor
    
    '''
    Get training data from one scan
    - object point cloud
    - point cloud of two object w. relationship
    - edge indices
    - descriptors for geometric information
    '''
    def __get_data(
        self, scene_points, obj_masks, mv_imgs_dict, num_pts_normalized, relationships,
        instance_map, rel_json, rel_viewpoint_clip, padding=0.2, all_edge=True
    ):
        all_instance = list(np.unique(obj_masks))
        nodes_all = list(instance_map.keys())

        if 0 in all_instance: # remove background
            all_instance.remove(0)
        
        nodes = []
        for i, instance_id in enumerate(nodes_all):
            if instance_id in all_instance:
                nodes.append(instance_id)
        
        # get edge (instance pair) list, which is just index, nodes[index] = instance_id
        if all_edge:
            edge_indices = list(product(list(range(len(nodes))), list(range(len(nodes)))))
            # filter out (i,i)
            edge_indices = [i for i in edge_indices if i[0]!=i[1]]
        else:
            edge_indices = [(nodes.index(r[0]), nodes.index(r[1])) for r in rel_json if r[0] in nodes and r[1] in nodes]
        
        num_objects = len(nodes)
        num_mv_img = self.config.num_mv_vecs
        dim_point = scene_points.shape[-1]
        
        instances_box, label_node = dict(), []
        obj_points = torch.zeros([num_objects, num_pts_normalized, dim_point])
        obj_mv_feats = torch.zeros([num_objects, num_mv_img, self.d_feats])
        
        obj_zero_mask = torch.ones([num_objects, num_mv_img])
        descriptor = torch.zeros([num_objects, 11])

        # obj_2d_feats = np.zeros([num_objects, 512])
        
        for i, instance_id in enumerate(nodes):
            assert instance_id in all_instance, "invalid instance id"
            # get node label name
            instance_name = instance_map[instance_id]
            label_node.append(self.classNames.index(instance_name))
            # get node point
            obj_pts, obj_bbox, desc = self.__crop_obj_pts(scene_points, obj_masks, instance_id, num_pts_normalized, padding=padding)
            instances_box[instance_id] = obj_bbox
            descriptor[i] = desc
            obj_points[i] = obj_pts
            
            ### Load Multi-View Image here
            ### Data loading process is too slow, 
            ### Object Point Cloud: B X N_pts X N_pc_feats
            ### Object Multi-View RGB CLIP Features: B X N_mv X N_feat
            obj_feats = mv_imgs_dict[instance_id]
            for n_i in range(num_mv_img):
                if n_i < len(obj_feats):
                    vec = obj_feats[n_i]
                    obj_mv_feats[i, n_i, :] = torch.from_numpy(vec[0])
                else:
                    obj_mv_feats[i, n_i, :] = torch.zeros((self.d_feats, ), dtype=torch.float32)
            obj_zero_mask[i, len(obj_feats):] = 0.

        # set gt label for relation
        len_object = len(nodes)
        if self.config.multi_rel:
            adj_matrix_onehot = np.zeros([len_object, len_object, len(relationships)])
        else:
            adj_matrix = np.zeros([len_object, len_object]) #set all to none label.
        
        for r in rel_json:
            if r[0] not in nodes or r[1] not in nodes: continue
            assert r[3] in relationships, "invalid relation name"
            r[2] = relationships.index(r[3]) # remap the index of relationships in case of custom relationNames

            if self.config.multi_rel:
                adj_matrix_onehot[nodes.index(r[0]), nodes.index(r[1]), r[2]] = 1
            else:
                adj_matrix[nodes.index(r[0]), nodes.index(r[1])] = r[2]
        
        # get relation union points
        if self.config.multi_rel:
            adj_matrix_onehot = torch.from_numpy(np.array(adj_matrix_onehot, dtype=np.float32))
            gt_rels = torch.zeros(len(edge_indices), len(relationships),dtype = torch.float)
        else:
            adj_matrix = torch.from_numpy(np.array(adj_matrix, dtype=np.int64))
            gt_rels = torch.zeros(len(edge_indices), dtype = torch.long)     
        
        rel_points = list()
        edge_feat_mask = torch.zeros((len(edge_indices),))
        edge_2d_feats = torch.zeros((len(edge_indices), 512))
        for e in range(len(edge_indices)):
            edge = edge_indices[e]
            index1 = edge[0]
            index2 = edge[1]
            instance1 = nodes[edge[0]]
            instance2 = nodes[edge[1]]

            if self.config.relpts == "obj_only":
                obj_pts_1, _, _ = self.__crop_obj_pts(scene_points, obj_masks, instance1, num_pts_normalized, padding, is_obj=False) # dim: N_pts X self.dim_pts
                obj_pts_2, _, _ = self.__crop_obj_pts(scene_points, obj_masks, instance2, num_pts_normalized, padding, is_obj=False) # dim: N_pts X self.dim_pts
                
                edge_pts = np.concatenate([obj_pts_1, obj_pts_2], axis=0) # dim: (2 * N_pts) X self.dim_pts
                rel_points.append(torch.from_numpy(edge_pts.astype(np.float32))) # 
            else:
                mask1 = (obj_masks == instance1).astype(np.int32) * 1
                mask2 = (obj_masks == instance2).astype(np.int32) * 2
                mask_ = np.expand_dims(mask1 + mask2, 1)
                bbox1 = instances_box[instance1]
                bbox2 = instances_box[instance2]
                min_box = np.minimum(bbox1[0], bbox2[0])
                max_box = np.maximum(bbox1[1], bbox2[1])
                filter_mask = (scene_points[:,0] > min_box[0]) * (scene_points[:,0] < max_box[0]) \
                            * (scene_points[:,1] > min_box[1]) * (scene_points[:,1] < max_box[1]) \
                            * (scene_points[:,2] > min_box[2]) * (scene_points[:,2] < max_box[2])
                
                # add with context, to distingush the different object's points
                points4d = np.concatenate([scene_points, mask_], 1)
                pointset = points4d[np.where(filter_mask > 0)[0], :]
                choice = np.random.choice(len(pointset), 2 * num_pts_normalized, replace=True)
                pointset = pointset[choice, :]
                pointset = torch.from_numpy(pointset.astype(np.float32))
                pointset[:,:3] = self.zero_mean(pointset[:,:3])
                rel_points.append(pointset)
            
            if self.config.multi_rel:
                gt_rels[e,:] = adj_matrix_onehot[index1,index2,:]
                if adj_matrix_onehot[index1, index2, :].sum() > 0.:
                    edge_feat_mask[e] = 1.
                    edge_feats = np.concatenate(rel_viewpoint_clip[index1][index2], axis=0)
                    edge_2d_feats[e] = torch.from_numpy(np.mean(edge_feats, axis=0)).float()
                else: 
                    edge_feat_mask[e] = 0.
            else:
                gt_rels[e] = adj_matrix[index1,index2]
                edge_feat_mask[e] = 1.
        
        if len(rel_points) > 0:
            rel_points = torch.stack(rel_points, 0)
        else:
            rel_points = torch.tensor([])
        
        label_node = torch.from_numpy(np.array(label_node, dtype=np.int64))
        edge_indices = torch.tensor(edge_indices,dtype=torch.long)
        edge_2d_feats = edge_2d_feats[edge_feat_mask.bool()]
        # obj_2d_feats = torch.from_numpy(obj_2d_feats.astype(np.float32))    
        
        return obj_points, \
            obj_mv_feats, \
            edge_2d_feats, \
            rel_points, \
            descriptor, \
            gt_rels, \
            label_node, \
            edge_indices, \
            obj_zero_mask, \
            edge_feat_mask.bool()
    
    ## Things to return
    ### Object features in the graph
    ### Edge Geometric descriptors
    ### Ground Truth Object label
    ### Ground Truth predicate label
    ### Edge Indices
    def __getitem__(self, index):
        scan_data = self.scan_data[index]
        obj_pts, obj_mv_feats, edge_2d_feats, rel_pts, descriptor, gt_rels, gt_class, edge_indices, obj_zero_mask, edge_feat_mask = \
            self.__get_data(
                scan_data["points"], scan_data["mask"], scan_data["mv_rgb"], self.config.num_points_reg, 
                self.relationNames, scan_data["map_instid_name"], scan_data["rel_json"], scan_data["edge_view_rgb"],
                self.config.padding, self.config.all_edges
            )
        
        while (len(rel_pts) == 0 or gt_rels.sum() == 0) and self.for_train:
            index = np.random.randint(self.__len__())
            obj_pts, obj_mv_feats, edge_2d_feats, rel_pts, descriptor, gt_rels, gt_class, edge_indices, obj_zero_mask, edge_feat_mask = \
                self.__getitem__(index)
        
        return obj_pts[..., :self.dim_pts], \
            obj_mv_feats, \
            edge_2d_feats, \
            rel_pts[..., :self.dim_pts], \
            descriptor, \
            gt_rels, \
            gt_class, \
            edge_indices, \
            obj_zero_mask, \
            edge_feat_mask