from config.define import *
from utils.os_utils import read_3dssg_annotation
from tqdm.contrib.concurrent import process_map
import torch
import numpy as np
import os
import trimesh
import multiprocessing


lock = multiprocessing.Lock()

def compute_weights(labels, classes, count, verbose=False):
    if verbose: print("-------------")    
    sum_weights = 0
    for c in range(len(classes)):
        if classes[c] / count > 0:        
            sum_weights += count / classes[c]

    sum_weight_norm = 0
    weights=list()
    for c in range(len(classes)):
        if classes[c] / count > 0:
            weight = count / classes[c] / sum_weights
            sum_weight_norm += weight
            if verbose: print('{0:>20s} {1:>1.3f} {2:>6d}'.format(labels[c], weight, int(classes[c])))
            weights.append(weight)
        else:
            if verbose: print('{0:>20s} {1:>1.3f} {2:>6d}'.format(labels[c], 0.0, int(classes[c])))
            weights.append(0)
    if verbose: print("-------------")
    return weights

def compute(classNames,relationNames, relationship_data, selections:list = None, verbose=False):
    o_rel_cls = np.zeros((len(relationNames)))
    o_obj_cls = np.zeros((len(classNames)))
    classes_count = 0
    counter = 0
    
    exceed_ids = dict()
    scene_analysis = dict()
    cnn=0
    for scan in relationship_data['scans']:
        scan_id = scan["scan"]
        if selections is not None:
            if scan_id not in selections:
                continue
        instance2LabelName = {}
        
        for k, v in scan["objects"].items():
            instance2LabelName[int(k)] = v
            if v not in classNames:
                if verbose: print(v,'not in classNames')
            o_obj_cls[classNames.index(v)] += 1

        nnk=dict()
        for relationship in scan["relationships"]:
            if relationship[3] not in relationNames:
                if verbose: print(relationship[3],'not in relationNames')
                continue

            obj = relationship[0] # id object
            sub = relationship[1] # id subject
            rel = relationship[2] # id relationship
            
            if obj == 0 or sub == 0:
                raise RuntimeError('found obj or sub is 0')
            
            if not obj in instance2LabelName:
                RuntimeWarning('key not found:',obj)
                continue

            if not sub in instance2LabelName:
                RuntimeWarning('key not found:',sub)
                continue
            
            if relationNames.index(relationship[3]) >= len(relationNames): 
                if rel not in exceed_ids:
                    exceed_ids[relationNames.index(relationship[3])]=0
                else:
                    exceed_ids[relationNames.index(relationship[3])]+=1
                continue
            o_rel_cls[relationNames.index(relationship[3])] += 1
            classes_count += 1
            
            nn = str(obj)+'_'+str(sub)
            if nn not in nnk:
                nnk[nn] = 0
            nnk[str(obj)+'_'+str(sub)] +=1
        for v in nnk.values():
            if v > 1:
                cnn+=1
                
        counter += 1
        
        scene_analysis[scan_id] = dict()
        scene_analysis[scan_id]['num objects'] = len(scan["objects"].items())
        scene_analysis[scan_id]['num relationships'] = len(scan['relationships'])
    if verbose: print('num multi predicates:',cnn)
        
    if len(exceed_ids)>1:
        print('exceed map')
        for id, count in exceed_ids.items():
            print('\t',id,count)

    if verbose: print("objects:")
    wobjs = compute_weights(classNames, o_obj_cls, classes_count,verbose)
    if verbose: print("relationships:")
    wrels = compute_weights(relationNames, o_rel_cls, classes_count,verbose)
    return wobjs,wrels,o_obj_cls,o_rel_cls

def read_labels(plydata):
    data = plydata.metadata['ply_raw']['vertex']['data']
    try:
        labels = data['objectId']
    except:
        labels = data['label']
    return labels

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

def gen_descriptor(pts:torch.tensor):
    '''
    centroid_pts,std_pts,segment_dims,segment_volume,segment_lengths
    [3, 3, 3, 1, 1]
    '''
    assert pts.ndim==2
    assert pts.shape[-1]==3
    # centroid [n, 3]
    centroid_pts = pts.mean(0) 
    # # std [n, 3]
    std_pts = pts.std(0)
    # dimensions [n, 3]
    segment_dims = pts.max(dim=0)[0] - pts.min(dim=0)[0]
    # volume [n, 1]
    segment_volume = torch.log((segment_dims[0]*segment_dims[1]*segment_dims[2]).unsqueeze(0))
    # length [n, 1]
    segment_lengths = torch.log(segment_dims.max().unsqueeze(0))
    return torch.cat([centroid_pts,std_pts,segment_dims,segment_volume,segment_lengths],dim=0)

def __read_rel_json(data, selected_scans:list):
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

def __read_3dssg_scans(args):
    scan_id, instance = args
    scan_id_no_split = scan_id.rsplit('_',1)[0]
    map_instance2labelName = instance["objs_json"][scan_id]
    path = os.path.join(instance["path_3rscan"], scan_id_no_split)
    data = load_mesh(path, LABEL_FILE_NAME, instance["use_rgb"], instance["use_normal"])
    return {
        "map_instid_name": map_instance2labelName,
        "points": data['points'],
        "mask": data['instances'],
        "rel_json": instance["relationship_json"][scan_id]
    }

def read_scan_data(config, split, device):
    _scan_path = SSG_DATA_PATH
    config = config
    path_3rscan = f"{SSG_DATA_PATH}/3RScan/data/3RScan"
    path_selection = f"{SSG_DATA_PATH}/3DSSG_subset"
    use_rgb = True
    use_normal = True
    dim_pts = 3
    if use_rgb:
        dim_pts += 3
    if use_normal:
        dim_pts += 3
                            
    data_path = f"{SSG_DATA_PATH}/3DSSG_subset"
    classNames, relationNames, data, selected_scans = \
        read_3dssg_annotation(data_path, path_selection, split)
    
    wobjs, wrels, o_obj_cls, o_rel_cls = compute(classNames, relationNames, data, selected_scans, False)
    w_cls_obj = torch.from_numpy(np.array(o_obj_cls)).float().to(device)
    w_cls_rel = torch.from_numpy(np.array(o_rel_cls)).float().to(device)
    
    # for single relation output, we set 'None' relationship weight as 1e-3
    if not config.multi_rel:
        w_cls_rel[0] = w_cls_rel.max() * 10
    
    w_cls_obj = w_cls_obj.sum() / (w_cls_obj + 1) /w_cls_obj.sum()
    w_cls_rel = w_cls_rel.sum() / (w_cls_rel + 1) /w_cls_rel.sum()
    w_cls_obj /= w_cls_obj.max()
    w_cls_rel /= w_cls_rel.max()
    
    # print some info
    print('=== {} classes ==='.format(len(classNames)))
    for i in range(len(classNames)):
        print('|{0:>2d} {1:>20s}'.format(i,classNames[i]),end='')
        if w_cls_obj is not None:
            print(':{0:>1.3f}|'.format(w_cls_obj[i]),end='')
        if (i+1) % 2 ==0:
            print('')
    print('')
    print('=== {} relationships ==='.format(len(relationNames)))
    for i in range(len(relationNames)):
        print('|{0:>2d} {1:>20s}'.format(i,relationNames[i]),end=' ')
        if w_cls_rel is not None:
            print('{0:>1.3f}|'.format(w_cls_rel[i]),end='')
        if (i+1) % 2 ==0:
            print('')
    print('')    
    
    relationship_json, objs_json, scans = __read_rel_json(data, selected_scans)
    # Pre-load entire 3RScan/3DSSG dataset in main memory
    ## Main memory capacity of experiment environment: 128GB
    ## Required memory: about 85GB
    instance_data = {
        "path_3rscan": path_3rscan,
        "objs_json": objs_json,
        "use_rgb": use_rgb,
        "use_normal": use_normal,
        "relationship_json": relationship_json
    }
    scan_list = [ (s, instance_data) for s in scans ]
    scan_data = process_map(__read_3dssg_scans, scan_list, max_workers=12, chunksize=15, mininterval=0.1)
    print('num of data:',len(scans))
    assert(len(scans) > 0)
    
    return scan_data, relationship_json, objs_json, scans
    