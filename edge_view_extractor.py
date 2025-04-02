import torch.nn.functional as F
from asyncio import sleep
import json, os, trimesh, argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import PIL.Image as Image
import clip, torch, numpy as np

DATA_DIR = "/media/michael/ssd1/SceneGraph"
MAX_NUM = 5
switch_predicates = {
    "right": "on the rightside of",
    "left": "on the leftside of",
    "front": "in front of",
    "behind": "behind of", 
}

model, preprocess = clip.load("ViT-B/32", device='cuda')

def read_txt_to_list(file):
    output = [] 
    with open(file, 'r') as f: 
        for line in f: 
            entry = line.rstrip().lower() 
            output.append(entry) 
    return output

def process_imgs(imgs):
    # rotate images
    a = torch.stack([preprocess(Image.fromarray(img).transpose(Image.ROTATE_270)).cuda()for img in imgs], dim=0)
    return a

def read_pointcloud(scan_id):
    """
    Reads a pointcloud from a file and returns points with instance label.
    """
    plydata = trimesh.load(os.path.join(f'{DATA_DIR}/3DSSG/3RScan/data/3RScan', scan_id, 'labels.instances.annotated.v2.ply'), process=False)
    points = np.array(plydata.vertices)
    labels = np.array(plydata.metadata['ply_raw']['vertex']['data']['objectId'])

    return points, labels

def read_json(split):
    """
    Reads a json file and returns points with instance label.
    """
    selected_scans = set()
    if split == 'train' :
        selected_scans = selected_scans.union(read_txt_to_list(f'{DATA_DIR}/3DSSG/3DSSG_subset/train_scans.txt'))
        with open(f"{DATA_DIR}/3DSSG/3DSSG_subset/relationships_train.json", "r") as read_file:
            data = json.load(read_file)
    elif split == 'val':
        selected_scans = selected_scans.union(read_txt_to_list(f'{DATA_DIR}/3DSSG/3DSSG_subset/validation_scans.txt'))
        with open(f"{DATA_DIR}/3DSSG/3DSSG_subset/relationships_validation.json", "r") as read_file:
            data = json.load(read_file)
    else:
        raise RuntimeError('unknown split type:',split)

    # convert data to dict('scene_id': {'obj': [], 'rel': []})
    scene_data = dict()
    for i in data['scans']:
        if i['scan'] not in scene_data.keys():
            scene_data[i['scan']] = {'obj': dict(), 'rel': list()}
        scene_data[i['scan']]['obj'].update(i['objects'])
        scene_data[i['scan']]['rel'].extend(i['relationships'])

    return scene_data, selected_scans

def read_intrinsic(intrinsic_path, mode='rgb'):
    with open(intrinsic_path, "r") as f:
        data = f.readlines()
    
    m_versionNumber = data[0].strip().split(' ')[-1]
    m_sensorName = data[1].strip().split(' ')[-2]
    
    if mode == 'rgb':
        m_Width = int(data[2].strip().split(' ')[-1])
        m_Height = int(data[3].strip().split(' ')[-1])
        m_Shift = None
        m_intrinsic = np.array([float(x) for x in data[7].strip().split(' ')[2:]])
        m_intrinsic = m_intrinsic.reshape((4, 4))
    else:
        m_Width = int(data[4].strip().split(' ')[-1])
        m_Height = int(data[5].strip().split(' ')[-1])
        m_Shift = int(data[6].strip().split(' ')[-1])
        m_intrinsic = np.array([float(x) for x in data[9].strip().split(' ')[2:]])
        m_intrinsic = m_intrinsic.reshape((4, 4))
    
    m_frames_size = int(data[11].strip().split(' ')[-1])
    
    return dict(
        m_versionNumber=m_versionNumber,
        m_sensorName=m_sensorName,
        m_Width=m_Width,
        m_Height=m_Height,
        m_Shift=m_Shift,
        m_intrinsic=m_intrinsic,
        m_frames_size=m_frames_size
    )

def read_extrinsic(extrinsic_path):
    pose = []
    with open(extrinsic_path) as f:
        lines = f.readlines()
    for line in lines:
        pose.append([float(i) for i in line.strip().split(' ')])
    return pose

def read_scan_info(scan_id, mode='rgb'):
    scan_path = os.path.join(f"{DATA_DIR}/3DSSG/3RScan/data/3RScan", scan_id)
    sequence_path = os.path.join(scan_path, "sequence")
    intrinsic_path = os.path.join(sequence_path, "_info.txt")
    intrinsic_info = read_intrinsic(intrinsic_path, mode='rgb')
    mode_template = 'color.jpg' if mode == 'rgb' else 'depth.pgm'
    
    image_list, extrinsic_list = [], []
    
    for i in range(0, intrinsic_info['m_frames_size']):
        frame_path = os.path.join(sequence_path, "frame-%s." % str(i).zfill(6)+ mode_template)
        extrinsic_path = os.path.join(sequence_path, "frame-%s." % str(i).zfill(6)+ "pose.txt")
        assert os.path.exists(frame_path) and os.path.exists(extrinsic_path)
        image_list.append(np.array(plt.imread(frame_path)))
        # inverce the extrinsic matrix, from camera_2_world to world_2_camera
        extrinsic = np.matrix(read_extrinsic(extrinsic_path))
        extrinsic_list.append(extrinsic.I)
        sleep(1)
    
    return np.array(image_list), np.array(extrinsic_list), intrinsic_info

def crop_pc(points, instances, object_id, extrinsics, intrinsic, width, height):
    points_i = points[(instances == object_id).flatten()]
    if points_i.shape[0] == 0:
        return None, None
    points_i = np.concatenate((points_i, np.ones((points_i.shape[0],1))), axis=-1)
    # transform to camera coordinates
    w_2_c = (extrinsics @ points_i.T)   # n_frames x 4 x n_points
    # transform to image coordinates
    c_2_i = intrinsic[:3, :] @ w_2_c    # n_frames x 3 x n_points
    c_2_i = c_2_i.transpose(0, 2, 1)    # n_frames x n_points x 3
    c_2_i = c_2_i[...,:2] / c_2_i[..., 2:] # n_frames x n_points x 2
    # find the points in the image
    indexs = ((c_2_i[...,0]< width) & (c_2_i[...,0]>0) & (c_2_i[...,1]< height) & (c_2_i[...,1]>0))
    return c_2_i, indexs

"""
Maps a pointcloud to an image as relationships.
"""
def map_pc_relations(
    points, relationships, 
    instances, image_list, instance_names,  
    extrinsics, intrinsic, width, height,
    save_path, 
    topk=10
):
    rel_text_tokens = []
    objects = list(instance_names.keys())
    for rel in relationships:
        sub_id, obj_id, _, rel_name = rel
        if not (str(sub_id) in objects and str(obj_id) in objects):
            r_token = clip.tokenize(f"An image of nothing.") ## Exceptions in dataset of VL-SAT.
            rel_text_tokens.append(r_token)
        else:
            n_subject = instance_names[str(sub_id)]
            n_object = instance_names[str(obj_id)]
            rel_name = rel_name if not rel_name in list(switch_predicates.keys()) else switch_predicates[rel_name]
            r_token = clip.tokenize(f"An image of which {n_subject} is {rel_name} {n_object}.")
            rel_text_tokens.append(r_token)
    
    rel_text_tokens = torch.cat(rel_text_tokens, dim=0).to("cuda")
    with torch.no_grad():
        rel_text_feats = model.encode_text(rel_text_tokens)
    rel_text_feats = F.normalize(rel_text_feats, dim=-1)
    
    # get clip match rate to filter some transport noise
    image_input = process_imgs(image_list)
    with torch.no_grad():
        image_feature = model.encode_image(image_input)
    # image_feature /= image_feature.norm(dim=-1, keepdim=True)
    image_feature = F.normalize(image_feature, dim=-1)
    similarity = (image_feature @ rel_text_feats.T).softmax(dim=-1)

    # print(similarity.shape, similarity)
    top_k_imgs = []
    top_k_quality = []
    for i, rel in enumerate(relationships):
        # found the instance points, convert to homogeneous coordinates
        sub_id, obj_id, _, rel_name = rel
        if not (str(sub_id) in objects and str(obj_id) in objects):
            continue
        sub_pts, sub_indices = crop_pc(points, instances, sub_id, extrinsics, intrinsic, width, height)
        obj_pts, obj_indices = crop_pc(points, instances, obj_id, extrinsics, intrinsic, width, height)
        if (sub_pts is None) or (obj_pts is None):
            continue
        topk_index = (-similarity[:, i]).argsort()[:topk]

        top_k_rel_images = []
        top_k_quality_rel = []
        count = 0
        ## Quality A: Maximal CLIP similarity & Reprojected Point Cloud 
        for k in topk_index:
            sub_pts_k = sub_pts[k][sub_indices[k].reshape(-1)]
            obj_pts_k = obj_pts[k][obj_indices[k].reshape(-1)]
            if len(sub_pts_k) == 0 or len(obj_pts_k) == 0:
                continue
            top_k_rel_images.append(image_list[k])
            top_k_quality_rel.append("A")
            count += 1
            if count >= MAX_NUM:
                break

        ## Quality B: Maximal Reprojected Point Cloud, There is no image with two objects.
        if count == 0:
            sub_pc_ratio, obj_pc_ratio = sub_indices.mean(-1), obj_indices.mean(-1)
            pc_ratio = np.concatenate([sub_pc_ratio.reshape(-1, 1), obj_pc_ratio.reshape(-1, 1)], axis=-1).sum(axis=-1)
            f_top_k_indices = np.argsort(-pc_ratio)
            for k in f_top_k_indices:
                sub_pts_k = sub_pts[k][sub_indices[k].reshape(-1)]
                obj_pts_k = obj_pts[k][obj_indices[k].reshape(-1)]
                if len(sub_pts_k) == 0 and len(obj_pts_k) == 0:
                    continue
                top_k_rel_images.append(image_list[k])
                top_k_quality_rel.append("B")
                count += 1
                if count >= MAX_NUM:
                    break

        ## Quality C: Using only CLIP similarity. There is no image with two objects.
        if count == 0:
            assert sub_indices.mean() == 0 and obj_indices.mean() == 0
            c_top_k_indices = (-similarity[:, i]).argsort()[:topk]
            for k in c_top_k_indices:
                top_k_rel_images.append(image_list[k])
                top_k_quality_rel.append("C")
                count += 1
                if count >= MAX_NUM:
                    break
        
        top_k_imgs.append(top_k_rel_images) # N_relations X N_TOP_K
        top_k_quality.append(top_k_quality_rel)
        
        ## Get CLIP features for Edge images. (At least one)
        top_k_imgs_edge = np.array(top_k_rel_images)
        edge_image_token = process_imgs(top_k_imgs_edge)
        with torch.no_grad():
            edge_feats = model.encode_image(edge_image_token)
        edge_feats = edge_feats.mean(dim=0).cpu().numpy()
        np.save(
            os.path.join(save_path, f'edge_{i}_subject_{sub_id}_object_{obj_id}_rel_{rel_name}_mean.npy'), 
            edge_feats
        )
        fin = open(os.path.join(save_path, 'project_quality.txt'), 'a')
        qual_str = " ".join(top_k_quality_rel)
        fin.write(f'Edge:{i} sub:{instance_names[str(sub_id)]} obj: {instance_names[str(obj_id)]} Quality:{qual_str} \n')
        fin.close() 
    
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--mode', type=str, default='train', help='train or test')
    args = argparser.parse_args()
    print("========= Deal with {} ========".format(args.mode))
    scene_data, selected_scans = read_json(args.mode)
    for i in tqdm(selected_scans):
        pc_i, instances_i = read_pointcloud(i)
        instance_names = scene_data[i]['obj']
        relationships = scene_data[i]['rel']
        # print(f'======= read image and extrinsic for {i} =========')
        image_list, extrinsic_list, intrinsic_info = read_scan_info(i)
        save_path = f'{DATA_DIR}/3DSSG/3RScan/data/3RScan/{i}/edge_view'
        os.makedirs(save_path, exist_ok=True)
        # print(f'======= map pointcloud to image =========')
        map_pc_relations(
            pc_i, relationships, instances_i, image_list, instance_names,  
            extrinsic_list, intrinsic_info['m_intrinsic'], intrinsic_info['m_Width'], intrinsic_info['m_Height'], 
            save_path, topk=10
        )