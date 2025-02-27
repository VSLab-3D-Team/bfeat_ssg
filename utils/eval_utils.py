import numpy as np
import torch
import torch.nn.functional as F
import json

def get_gt(objs_target, rels_target, edges, multi_rel_outputs):
    gt_edges = []
    for edge_index in range(len(edges)):
        idx_eo = edges[edge_index][0]
        idx_os = edges[edge_index][1]
        target_eo = objs_target[idx_eo]
        target_os = objs_target[idx_os]
        target_rel = []
        if multi_rel_outputs:
            assert rels_target.ndim == 2
            for i in range(rels_target.shape[-1]):
                if rels_target[edge_index][i] == 1:
                    target_rel.append(i)
        else:
            assert rels_target.ndim == 1
            if rels_target[edge_index] > 0: # not None
                target_rel.append(rels_target[edge_index])
        gt_edges.append((target_eo, target_os, target_rel))
    return gt_edges


def evaluate_topk_object(objs_pred, objs_target, topk):
    res = []
    for obj in range(len(objs_pred)):
        obj_pred = objs_pred[obj]
        sorted_idx = torch.sort(obj_pred, descending=True)[1]
        gt = objs_target[obj]
        index = 1
        for idx in sorted_idx:
            if obj_pred[gt] >= obj_pred[idx] or index > topk:
                break
            index += 1
        res.append(index)
    return np.asarray(res)


def evaluate_topk_predicate(rels_preds, gt_edges, multi_rel_outputs, topk, confidence_threshold=0.5, epsilon=0.02):
    res = []
    for rel in range(len(rels_preds)):
        rel_pred = rels_preds[rel]
        # make the 'none' confidence the highest, if none of the rel classes are bigger than confidence_threshold
        # which means 'none' prediction in the multi binary cross entropy approach.
        # if multi_rel_outputs:
        #     if rel_pred.max() < confidence_threshold:
        #         rel_pred[0] = rel_pred.max() + epsilon
        
        sorted_conf_matrix, sorted_idx = torch.sort(rel_pred, descending=True)
        temp_topk = []
        rels_target = gt_edges[rel][2]
        
        if len(rels_target) == 0: # no gt relation
            indices = torch.where(sorted_conf_matrix < confidence_threshold)[0]
            if len(indices) == 0:
                index = topk + 1
            else:
                index = indices[0].item()+1
                #index = sorted(indices)[0].item()+1
            
            temp_topk.append(index)

        for gt in rels_target:
            index = 1
            for idx in sorted_idx:
                if rel_pred[gt] >= rel_pred[idx] or index > topk:
                    break
                index += 1
            temp_topk.append(index)
        
        temp_topk = sorted(temp_topk)
        counter = 0
        for tmp in temp_topk:
            res.append(tmp - counter)
            counter += 1
        #res += temp_topk
    return np.asarray(res)


def evaluate_topk(objs_pred, rels_pred, gt_rel, edges, multi_rel_outputs, topk, confidence_threshold=0.5, epsilon=0.02):
    res, cls = [], []
    # convert the score from log_softmax to softmax
    objs_pred = np.exp(objs_pred)
    if not multi_rel_outputs:
        rels_pred = np.exp(rels_pred)
    
    for edge in range(len(edges)):
        edge_from = edges[edge][0]
        edge_to = edges[edge][1]
        rel_predictions = rels_pred[edge]
        obj = objs_pred[edge_from]
        sub = objs_pred[edge_to]

        # make the 'none' confidence the highest, if none of the rel classes are bigger than confidence_threshold
        # which means 'none' prediction in the multi binary cross entropy approach.
        # if multi_rel_outputs:
        #     if rel_predictions.max() < confidence_threshold:
        #         rel_predictions[0] = rel_predictions.max() + epsilon

        size_o = len(obj)
        size_r = len(rel_predictions)

        node_score = np.matmul(obj.reshape(size_o, 1), sub.reshape(1, size_o))
        conf_matrix = np.matmul(node_score.reshape(size_o, size_o, 1), rel_predictions.reshape(1, size_r))
        conf_matrix_1d = conf_matrix.reshape(-1)
        sorted_args_1d = torch.sort(conf_matrix_1d, descending=True)[1]

        subject = gt_rel[edge][0]
        obj = gt_rel[edge][1]
        temp_topk, tmp_cls = [], []

        for predicate in gt_rel[edge][2]:
            index = 1
            for idx_1d in sorted_args_1d:
                idx = np.unravel_index(idx_1d, (size_o, size_o, size_r))
                gt_conf = conf_matrix[subject, obj, predicate]
                if gt_conf >= conf_matrix[idx] or index > topk:
                    break
                index += 1
            temp_topk.append(index)
            tmp_cls.append(predicate)
        
        temp_topk = sorted(temp_topk)
        counter = 0
        for tmp in temp_topk:
            assert (tmp - counter) > 0
            res.append(tmp - counter)
            counter += 1
        #res += temp_topk
        cls += tmp_cls
    
    return np.asarray(res), np.array(cls)


def evaluate_triplet_topk(objs_pred, rels_pred, gt_rel, edges, multi_rel_outputs, topk, confidence_threshold=0.5, epsilon=0.02, use_clip=False, obj_topk=None):
    res, triplet = [], []
    if not use_clip:
        # convert the score from log_softmax to softmax
        objs_pred = np.exp(objs_pred)
    else:
        # convert the score to softmax
        objs_pred = F.softmax(objs_pred, dim=-1)
    
    if not multi_rel_outputs:
        rels_pred = np.exp(rels_pred)

    sub_scores, obj_scores, rel_scores = [],  [],  []
    
    for edge in range(len(edges)):
        edge_from = edges[edge][0]
        edge_to = edges[edge][1]
        rel_predictions = rels_pred[edge]
        sub = objs_pred[edge_from]
        obj = objs_pred[edge_to]
        
        if obj_topk is not None:
            sub_pred = obj_topk[edge_from]
            obj_pred = obj_topk[edge_to]

        node_score = torch.einsum('n,m->nm',sub,obj)
        conf_matrix = torch.einsum('nl,m->nlm',node_score,rel_predictions)
        conf_matrix_1d = conf_matrix.reshape(-1)
        sorted_conf_matrix, sorted_args_1d = torch.sort(conf_matrix_1d, descending=True)
        
        # just take topk
        sorted_conf_matrix = sorted_conf_matrix[:topk]
        sorted_args_1d = sorted_args_1d[:topk]

        sub_gt= gt_rel[edge][0]
        obj_gt = gt_rel[edge][1]
        rel_gt = gt_rel[edge][2]
        temp_topk, tmp_triplet = [], []

        if len(rel_gt) == 0: # no gt relation
            indices = torch.where(sorted_conf_matrix < confidence_threshold)[0]
            if len(indices) == 0:
                index = topk + 1
            else:
                index = indices[0].item()+1
                #index = sorted(indices)[0].item()+1
            temp_topk.append(index)
            if obj_topk is not None:
                tmp_triplet.append([sub_gt.cpu(),sub_pred, obj_gt.cpu(), obj_pred, -1])
            else:
                tmp_triplet.append([sub_gt.cpu(),obj_gt.cpu(),-1])
        
        for predicate in rel_gt: # for multi class case
            gt_conf = conf_matrix[sub_gt, obj_gt, predicate]
            indices = torch.where(sorted_conf_matrix == gt_conf)[0]
            if len(indices) == 0:
                index = topk + 1
            else:
                index = indices[0].item()+1
                #index = sorted(indices)[0].item()+1
            temp_topk.append(index)
            if obj_topk is not None:
                tmp_triplet.append([sub_gt.cpu(),sub_pred, obj_gt.cpu(), obj_pred, predicate])
            else:
                tmp_triplet.append([sub_gt.cpu(), obj_gt.cpu(), predicate])
            
            sub_scores.append(sub.cpu())
            obj_scores.append(obj.cpu())
            rel_scores.append(rel_predictions.cpu())
            
   
        temp_topk = sorted(temp_topk)
        counter = 0
        for tmp in temp_topk:
            res.append(tmp - counter)
            counter += 1
        triplet += tmp_triplet
    
    return np.asarray(res), np.array(triplet), sub_scores, obj_scores, rel_scores


def evaluate_topk_recall(objs_pred, rels_pred, objs_target, rels_target, edges):
    top_k_obj = evaluate_topk_object(objs_pred, objs_target, topk=10)
    gt_edges = get_gt(objs_target, rels_target, edges, topk=10)
    top_k_predicate = evaluate_topk_predicate(rels_pred, gt_edges, multi_rel_outputs=True, topk=5)
    top_k = evaluate_triplet_topk(objs_pred, rels_pred, rels_target, edges, multi_rel_outputs=True, topk=100)
    return top_k, top_k_obj, top_k_predicate


def get_mean_recall(triplet_rank, cls_matrix, topk=[50, 100]):
    if len(cls_matrix) == 0:
        return np.array([0,0])

    mean_recall = [[] for _ in range(len(topk))]
    cls_num = int(cls_matrix.max())
    for i in range(cls_num):
        cls_rank = triplet_rank[cls_matrix[:,-1] == i]
        if len(cls_rank) == 0:
            continue
        for idx, top in enumerate(topk):
            mean_recall[idx].append((cls_rank <= top).sum() * 100 / len(cls_rank))
    mean_recall = np.array(mean_recall, dtype=np.float32)
    return mean_recall.mean(axis=1)


def read_txt_to_list(file):
    output = [] 
    with open(file, 'r') as f: 
        for line in f: 
            entry = line.rstrip().lower() 
            output.append(entry) 
    return output


def read_json(split):
    """
    Reads a json file and returns points with instance label.
    """
    selected_scans = set()
    if split == 'train' :
        selected_scans = selected_scans.union(read_txt_to_list('/data/wangziqin/project/3DSSG_Repo/data/3DSSG_subset/train_scans.txt'))
        with open("/data/wangziqin/project/3DSSG_Repo/data/3DSSG_subset/relationships_train.json", "r") as read_file:
            data = json.load(read_file)
    elif split == 'val':
        selected_scans = selected_scans.union(read_txt_to_list('/data/wangziqin/project/3DSSG_Repo/data/3DSSG_subset/validation_scans.txt'))
        with open("/data/wangziqin/project/3DSSG_Repo/data/3DSSG_subset/relationships_validation.json", "r") as read_file:
            data = json.load(read_file)
    else:
        raise RuntimeError('unknown split type:',split)

    return data

def get_zero_shot_recall(triplet_rank, cls_matrix, obj_names, rel_name):
   
    train_data = read_json('train')
    scene_data = dict()
    for i in train_data['scans']:
        objs = i['objects']
        for rel in i['relationships']:
            if str(rel[0]) not in objs.keys():
                #print(f'{rel[0]} not in objs in scene {i["scan"]} split {i["split"]}')
                continue
            if str(rel[1]) not in objs.keys():
                #print(f'{rel[1]} not in objs in scene {i["scan"]} split {i["split"]}')
                continue
            triplet_name = str(obj_names.index(objs[str(rel[0])])) + ' ' + str(obj_names.index(objs[str(rel[1])])) + ' ' + str(rel_name.index(rel[-1]))
            if triplet_name not in scene_data.keys():
                scene_data[triplet_name] = 1
            scene_data[triplet_name] += 1
    
    val_data = read_json('val')
    zero_shot_triplet = []
    count = 0
    for i in val_data['scans']:
        objs = i['objects']
        for rel in i['relationships']:
            count += 1
            triplet_name = str(obj_names.index(objs[str(rel[0])])) + ' ' + str(obj_names.index(objs[str(rel[1])])) + ' ' + str(rel_name.index(rel[-1]))
            if triplet_name not in scene_data.keys():
                zero_shot_triplet.append(triplet_name)
    
    # get valid triplet which not appears in train data
    valid_triplet = []
    non_zero_shot_triplet = []
    all_triplet = []

    for i in range(len(cls_matrix)):
        if cls_matrix[i, -1] == -1:
            continue
        if len(cls_matrix[i]) == 5:
            triplet_name = str(cls_matrix[i][0]) + ' ' + str(cls_matrix[i][2]) + ' ' + str(cls_matrix[i][-1])
        elif len(cls_matrix[i]) == 3:
            triplet_name = str(cls_matrix[i][0]) + ' ' + str(cls_matrix[i][1]) + ' ' + str(cls_matrix[i][-1])
        else:
            raise RuntimeError('unknown triplet length:', len(cls_matrix[i]))

        if triplet_name in zero_shot_triplet:
            valid_triplet.append(triplet_rank[i])
        else:
            non_zero_shot_triplet.append(triplet_rank[i])
        
        all_triplet.append(triplet_rank[i])
    
    valid_triplet = np.array(valid_triplet)
    non_zero_shot_triplet = np.array(non_zero_shot_triplet)
    all_triplet = np.array(all_triplet)

    zero_shot_50 = (valid_triplet <= 50).mean() * 100
    zero_shot_100 = (valid_triplet <= 100).mean() * 100

    non_zero_shot_50 = (non_zero_shot_triplet <= 50).mean() * 100
    non_zero_shot_100 = (non_zero_shot_triplet <= 100).mean() * 100

    all_50 = (all_triplet <= 50).mean() * 100
    all_100 = (all_triplet <= 100).mean() * 100

    return (zero_shot_50, zero_shot_100), (non_zero_shot_50, non_zero_shot_100), (all_50, all_100)

def evaluate_triplet_recallk(objs_pred, rels_pred, gt_rel, edges, multi_rel_outputs, topk, topk_each, use_clip=False, evaluate='triplet'):
    # objs_pred: N_o * 160
    # rels_pred: N_r * 26
    # gt_rel: N_r * 26, multiple
    # edges: N_r * 2, 0 - N_o-1 (obj index)
    # confidence_threshold: no
    res, triplet = [], []
    if not use_clip:
        # convert the score from log_softmax to softmax
        objs_pred = np.exp(objs_pred)
    else:
        # convert the score to softmax
        objs_pred = F.softmax(objs_pred, dim=-1)
    
    if not multi_rel_outputs:
        rels_pred = np.exp(rels_pred)

    all_topk_conf_matrix, all_topk_id = None, None
    topk_list = topk if isinstance(topk, list) else [topk]
    topk = np.max(topk_list)

    for edge in range(len(edges)):
        edge_from = edges[edge][0]
        edge_to = edges[edge][1]
        rel_predictions = rels_pred[edge]
        sub = objs_pred[edge_from]
        obj = objs_pred[edge_to]

        node_score = torch.einsum('n,m->nm',sub,obj)
        if evaluate == 'triplet':
            conf_matrix = torch.einsum('nl,m->nlm',node_score,rel_predictions)
            conf_matrix_1d = conf_matrix.reshape(-1)
        elif evaluate == 'rels':
            conf_matrix_1d = rel_predictions
        else:
            raise NotImplementedError('evaluate type', evaluate)

        curr_topk_conf_matrix, curr_topk_conf_id = conf_matrix_1d.topk(min(topk_each, conf_matrix_1d.shape[0]), largest=True)
        curr_edge_id = torch.zeros_like(curr_topk_conf_id) + edge
        # (edgeid, topk-conf-id)
        # edgeid represents (object and subject) id
        # topk-conf-id represents (sub type, obj type, rel type)
        curr_topk_id = torch.stack([curr_edge_id, curr_topk_conf_id], dim=-1)

        # select all --- matrix
        if all_topk_conf_matrix is None:
            all_topk_conf_matrix = curr_topk_conf_matrix
            all_topk_id = curr_topk_id
        else:
            # print(all_topk_conf_matrix.shape, all_topk_id.shape)
            all_topk_conf_matrix = torch.cat([all_topk_conf_matrix, curr_topk_conf_matrix], dim=0)
            all_topk_id = torch.cat([all_topk_id, curr_topk_id], dim=0)
        all_topk_conf_matrix, select_id = all_topk_conf_matrix.topk(min(topk, all_topk_conf_matrix.shape[0]), largest=True, sorted=True)
        all_topk_id = all_topk_id[select_id]

        # sorted_conf_matrix, sorted_args_1d = torch.sort(conf_matrix_1d, descending=True)
    # print(all_topk_id, all_topk_conf_matrix)
    # import ipdb; ipdb.set_trace()

    pred_triplets = []
    correct_number, all_number = [0 for i in topk_list], 0  # all_number: all correct number in the gt
    #all_number = sum([len(gt_edge[2]) for gt_edge in gt_rel])
    all_number = sum([min(1,len(gt_edge[2])) for gt_edge in gt_rel])
    # print(all_number, '<< gt edge number')
    # print(all_topk_conf_matrix, 'all conf')

    size_o, size_r = objs_pred.shape[1], rels_pred.shape[1]
    iscompute = [{} for _ in range(len(topk_list))]
    for idk, [edge, idx_1d] in enumerate(all_topk_id):  # calculate for each predicted edge
        conf_score = all_topk_conf_matrix[idk]

        # same edge id (same object and subject)
        edge_from = edges[edge][0]
        edge_to = edges[edge][1]
        rel_predictions = rels_pred[edge]

        sub_gt= gt_rel[edge][0]
        obj_gt = gt_rel[edge][1]
        rel_gt = gt_rel[edge][2]

        edge = int(edge)
        if evaluate == 'triplet':
            idx = np.unravel_index(idx_1d.item(), (size_o, size_o, size_r))
            if sub_gt == idx[0] and obj_gt == idx[1] and (idx[2] in rel_gt):
                for _, k in enumerate(topk_list):
                    if idk < k and edge not in iscompute[_].keys():
                        correct_number[_] += 1
                        iscompute[_][edge] = 1
                # print(conf_score, edge, 'idx', idx, 'edge from and to', edge_from, edge_to, 'gt type', sub_gt, obj_gt, rel_gt)
            pred_triplets.append(((edge_from, edge_to), idx, conf_score))  # edge, object&predicate cls type
        elif evaluate == 'rels':
            idx = idx_1d
            if idx in rel_gt:
                for _, k in enumerate(topk_list):
                    if idk < k and edge not in iscompute[_].keys():
                        correct_number[_] += 1
                        iscompute[_][edge] = 1
            pred_triplets.append(((edge_from, edge_to), (-1, -1, idx)))  # edge, object&predicate cls type
        else:
            raise NotImplementedError()

    # print(correct_number, all_number)
    correct_number = np.array(correct_number)
    #return pred_triplets, correct_number/all_number
    # print(correct_number, all_number, iscompute)
    return correct_number/all_number

def evaluate_triplet_mrecallk(objs_pred, rels_pred, gt_rel, edges, multi_rel_outputs, topk, topk_each, use_clip=False, evaluate='triplet'):
    # objs_pred: N_o * 160
    # rels_pred: N_r * 26
    # gt_rel: N_r * 26, multiple
    # edges: N_r * 2, 0 - N_o-1 (obj index)
    # confidence_threshold: no
    res, triplet = [], []
    if not use_clip:
        # convert the score from log_softmax to softmax
        objs_pred = np.exp(objs_pred)
    else:
        # convert the score to softmax
        objs_pred = F.softmax(objs_pred, dim=-1)
    
    if not multi_rel_outputs:
        rels_pred = np.exp(rels_pred)

    all_topk_conf_matrix, all_topk_id = None, None
    topk_list = topk if isinstance(topk, list) else [topk]
    topk = np.max(topk_list)

    for edge in range(len(edges)):
        edge_from = edges[edge][0]
        edge_to = edges[edge][1]
        rel_predictions = rels_pred[edge]
        sub = objs_pred[edge_from]
        obj = objs_pred[edge_to]

        node_score = torch.einsum('n,m->nm',sub,obj)
        if evaluate == 'triplet':
            conf_matrix = torch.einsum('nl,m->nlm',node_score,rel_predictions)
            conf_matrix_1d = conf_matrix.reshape(-1)
        elif evaluate == 'rels':
            conf_matrix_1d = rel_predictions
        else:
            raise NotImplementedError('evaluate type', evaluate)

        curr_topk_conf_matrix, curr_topk_conf_id = conf_matrix_1d.topk(min(topk_each, conf_matrix_1d.shape[0]), largest=True)
        curr_edge_id = torch.zeros_like(curr_topk_conf_id) + edge
        # (edgeid, topk-conf-id)
        # edgeid represents (object and subject) id
        # topk-conf-id represents (sub type, obj type, rel type)
        curr_topk_id = torch.stack([curr_edge_id, curr_topk_conf_id], dim=-1)

        # select all --- matrix
        if all_topk_conf_matrix is None:
            all_topk_conf_matrix = curr_topk_conf_matrix
            all_topk_id = curr_topk_id
        else:
            # print(all_topk_conf_matrix.shape, all_topk_id.shape)
            all_topk_conf_matrix = torch.cat([all_topk_conf_matrix, curr_topk_conf_matrix], dim=0)
            all_topk_id = torch.cat([all_topk_id, curr_topk_id], dim=0)
        all_topk_conf_matrix, select_id = all_topk_conf_matrix.topk(min(topk, all_topk_conf_matrix.shape[0]), largest=True, sorted=True)
        all_topk_id = all_topk_id[select_id]

        # sorted_conf_matrix, sorted_args_1d = torch.sort(conf_matrix_1d, descending=True)
    # print(all_topk_id, all_topk_conf_matrix)
    # import ipdb; ipdb.set_trace()

    pred_triplets = []
    correct_number = [[0 for _ in topk_list] for _ in range(26)] 
    #all_number = sum([len(gt_edge[2]) for gt_edge in gt_rel])
    all_number_perclass = [ sum([1 if i in gt_edge[2] else 0 for gt_edge in gt_rel]) for i in range(26)]
    # print(all_number, '<< gt edge number')
    # print(all_topk_conf_matrix, 'all conf')

    size_o, size_r = objs_pred.shape[1], rels_pred.shape[1]
    iscompute = [{} for _ in range(len(topk_list))]
    
    for idk, [edge, idx_1d] in enumerate(all_topk_id):  # calculate for each predicted edge
        conf_score = all_topk_conf_matrix[idk]

        # same edge id (same object and subject)
        edge_from = edges[edge][0]
        edge_to = edges[edge][1]
        rel_predictions = rels_pred[edge]

        sub_gt= gt_rel[edge][0]
        obj_gt = gt_rel[edge][1]
        rel_gt = gt_rel[edge][2]

        edge = int(edge)
        if evaluate == 'triplet':
            idx = np.unravel_index(idx_1d.item(), (size_o, size_o, size_r))
            if sub_gt == idx[0] and obj_gt == idx[1] and (idx[2] in rel_gt):
                for _, k in enumerate(topk_list):
                    #for cls in range(26):
                    if idk < k and edge not in iscompute[_].keys():
                        for r in rel_gt:
                            correct_number[r][_] += 1
                        iscompute[_][edge] = 1
                # print(conf_score, edge, 'idx', idx, 'edge from and to', edge_from, edge_to, 'gt type', sub_gt, obj_gt, rel_gt)
            pred_triplets.append(((edge_from, edge_to), idx, conf_score))  # edge, object&predicate cls type
        elif evaluate == 'rels':
            idx = idx_1d
            if idx in rel_gt:
                for _, k in enumerate(topk_list):
                    if idk < k and edge not in iscompute[_].keys():
                        for r in rel_gt:
                            correct_number[r][_] += 1
                        iscompute[_][edge] = 1
                    # for cls in range(26):
                    #     if idk < k and cls == idx:
                    #         correct_number[cls][_] += 1

            #pred_triplets.append(((edge_from, edge_to), (-1, -1, idx)))  # edge, object&predicate cls type
        else:
            raise NotImplementedError()

    # print(correct_number, all_number)
    correct_number = np.array(correct_number)

    #return pred_triplets, correct_number/all_number
    # print(correct_number, all_number, iscompute)
    return [[correct_number[j][i] / all_number_perclass[j] if all_number_perclass[j]!=0 else -1 for i in range(3)] for j in range(26)]

def get_rel_mean_recall(topk_pred_list, cls_matrix, topk=[3, 5]):
    if len(cls_matrix) == 0:
        return np.array([0,0])

    mean_recall = [[] for _ in range(len(topk))]
    cls_num = int(cls_matrix.max())
    for i in range(cls_num):
        cls_rank = topk_pred_list[cls_matrix[:,-1] == i]
        if len(cls_rank) == 0:
            continue
        for idx, top in enumerate(topk):
            mean_recall[idx].append((cls_rank <= top).sum() * 100 / len(cls_rank))
    mean_recall = np.array(mean_recall, dtype=np.float32)
    return mean_recall.mean(axis=1)

def handle_mean_recall(recall_input):
    '''
    recall_list : N * 26 * 3:List
    '''
    recall_input = np.array(recall_input)
    num_list = [0 for i in range(recall_input.shape[1])]
    recall_list = [[0.0, 0.0, 0.0] for i in range(recall_input.shape[1])]
    for one_batch in recall_input:
        for idx, recall in enumerate(one_batch):
            if -1 in recall:
                continue
            num_list[idx] += 1
            recall_list[idx][0] += recall[0]
            recall_list[idx][1] += recall[1]
            recall_list[idx][2] += recall[2]
    
    for idx in range(len(recall_list)):
        if num_list[idx] == 0:
            continue
        recall_list[idx][0] /= num_list[idx]
        recall_list[idx][1] /= num_list[idx]
        recall_list[idx][2] /= num_list[idx]
    
    num = sum(1 for value in num_list if value != 0)
    result = np.array(recall_list).sum(0) / num
    return result