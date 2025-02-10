"""
Preprocessing is NOTHING!!!!!!!!!!!!!!!!!
This code is not used in anywhere for now
"""

from utils.os_utils import read_3dssg_annotation
from utils.data_utils import compute, read_labels
from config.define import *
from parse import parse
import PIL.Image as Image
import numpy as np
import trimesh
import torch
import clip
from glob import glob
import cv2

def load_mesh(path,label_file,use_rgb,use_normal):
    result=dict()
    if label_file == 'labels.instances.align.annotated.v2.ply' or label_file == 'labels.instances.align.annotated.ply':
        
        plydata = trimesh.load(os.path.join(path,label_file), process=False)
        points = np.array(plydata.vertices)
        instances = read_labels(plydata).flatten()
        
        if use_rgb:
            rgbs = np.array(plydata.visual.vertex_colors.tolist())[:,:3]
            points = np.concatenate((points, rgbs / 255.0), axis=1)
            
        if use_normal:
            normal = plydata.vertex_normals[:,:3]
            points = np.concatenate((points, normal), axis=1)
        
        result['points']=points
        result['instances']=instances
    else:
        raise NotImplementedError('')
    return result

## Save the encoder vectors from point cloud
class Preprocessor3RScan(object):
    def __init__(self, config, split, device):
        
        self.config = config.dataset
        self.path_3rscan = f"{SSG_DATA_PATH}/3RScan/data/3RScan"
        self.use_rgb = True
        self.use_normal = True
        self.device = device
        self.encoder, self.preprocessor = clip.load("ViT-B/32", device=device)
        
        self.data_path = f"{SSG_DATA_PATH}/3DSSG_subset"
        self.classNames, self.relationNames, data, selected_scans = \
            read_3dssg_annotation(self.data_path, self.data_path, split)
        
        wobjs, wrels, o_obj_cls, o_rel_cls = compute(self.classNames, self.relationNames, data,selected_scans, False)
        self.w_cls_obj = torch.from_numpy(np.array(o_obj_cls)).float().to(device)
        self.w_cls_rel = torch.from_numpy(np.array(o_rel_cls)).float().to(device)
        
        # for single relation output, we set 'None' relationship weight as 1e-3
        if not self.config.multi_rel:
            self.w_cls_rel[0] = self.w_cls_rel.max() * 10
        
        self.w_cls_obj = self.w_cls_obj.sum() / (self.w_cls_obj + 1) /self.w_cls_obj.sum()
        self.w_cls_rel = self.w_cls_rel.sum() / (self.w_cls_rel + 1) /self.w_cls_rel.sum()
        self.w_cls_obj /= self.w_cls_obj.max()
        self.w_cls_rel /= self.w_cls_rel.max()
     
        # print some info
        print('=== {} classes ==='.format(len(self.classNames)))
        for i in range(len(self.classNames)):
            print('|{0:>2d} {1:>20s}'.format(i,self.classNames[i]),end='')
            if self.w_cls_obj is not None:
                print(':{0:>1.3f}|'.format(self.w_cls_obj[i]),end='')
            if (i+1) % 2 ==0:
                print('')
        print('')
        print('=== {} relationships ==='.format(len(self.relationNames)))
        for i in range(len(self.relationNames)):
            print('|{0:>2d} {1:>20s}'.format(i,self.relationNames[i]),end=' ')
            if self.w_cls_rel is not None:
                print('{0:>1.3f}|'.format(self.w_cls_rel[i]),end='')
            if (i+1) % 2 ==0:
                print('')
        print('')    
        
        self.relationship_json, self.objs_json, self.scans = self.__read_rel_json(data, selected_scans)
        print('num of data:',len(self.scans))
        assert(len(self.scans)>0)
            
        self.dim_pts = 3
        if self.use_rgb:
            self.dim_pts += 3
        if self.use_normal:
            self.dim_pts += 3
    
    def __len__(self):
        return len(self.scans)
    
    def __read_rel_json(self, data, selected_scans:list):
        rel, objs, scans = dict(), dict(), []

        for scan_i in data['scans']:
            if scan_i["scan"] == 'fa79392f-7766-2d5c-869a-f5d6cfb62fc6':
                if LABEL_FILE_NAME == "labels.instances.align.annotated.v2.ply":
                    '''
                    In the 3RScanV2, the segments on the semseg file and its ply file mismatch. 
                    This causes error in loading data.
                    To verify this, run check_seg.py
                    '''
                    continue
            if scan_i['scan'] not in selected_scans:
                continue
                
            relationships_i = []
            for relationship in scan_i["relationships"]:
                relationships_i.append(relationship)
                
            objects_i = {}
            for id, name in scan_i["objects"].items():
                objects_i[int(id)] = name

            rel[scan_i["scan"] + "_" + str(scan_i["split"])] = relationships_i
            objs[scan_i["scan"]+"_"+str(scan_i['split'])] = objects_i
            scans.append(scan_i["scan"] + "_" + str(scan_i["split"]))

        return rel, objs, scans
    
    def process(self, index):
        scan_id = self.scans[index]
        scan_id_no_split = scan_id.rsplit('_',1)[0]
        map_instance2labelName = self.objs_json[scan_id]
        path = os.path.join(self.path_3rscan, scan_id_no_split)
        data = load_mesh(path, LABEL_FILE_NAME, self.use_rgb, self.use_normal)
        ### Get RGB Image
        all_instance = list(np.unique(data['instances']))
        nodes_all = list(map_instance2labelName.keys())

        if 0 in all_instance: # remove background
            all_instance.remove(0)
        
        nodes = []
        for i, instance_id in enumerate(nodes_all):
            if instance_id in all_instance:
                nodes.append(instance_id)
        # obj_2d_feats = np.zeros([num_objects, 512])
        
        label_node = []
        for i, instance_id in enumerate(nodes):
            assert instance_id in all_instance, "invalid instance id"
            # get node label name
            instance_name = map_instance2labelName[instance_id]
            label_node.append(self.classNames.index(instance_name))
            ### Load Multi-View Image here
            ### Data loading process is too slow, 
            image_path_list = glob(f"{path}/multi_view/instance_{instance_id}_class_{instance_name}_view*_*_*.jpg")
            for _p in image_path_list:
                pattern = "instance_{instance_id}_class_{instance_name}_view{v_id}_{iid_num}_{type_id}.jpg"
                result = parse(pattern, _p.split('/')[-1])
                v_id = result["v_id"]
                iid_num = result["iid_num"]
                type_id = result["type_id"]
                preprocess_clip_vec = self.preprocessor(
                    Image.fromarray(cv2.imread(_p)).transpose(Image.ROTATE_270)
                ).unsqueeze(0).to(self.device)
                np.save(
                    f"{path}/multi_view/clip_instance_{instance_id}_class_{instance_name}_view{v_id}_{iid_num}_{type_id}.npy", 
                    self.encoder.encode_image(preprocess_clip_vec).clone().detach().cpu().numpy()
                )
                
    
class PreprocessorScannet(object):
    def __init__(self):
        pass
    
    def process_one_scan(self, _path):
        pass
    
    def write_compressed(self, relationships):
        pass