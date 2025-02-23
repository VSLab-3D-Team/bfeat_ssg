from experiments.base_runner import BaseExperimentRunner
from utils.eval_utils import *
from utils.logger import Progbar
import numpy as np
import torch.nn as nn

class ClassificationExperiment(BaseExperimentRunner):
    def __init__(
        self,
        model: nn.Module, 
        config, 
        device
    ):
        super().__init__(config, device)
        self.model = model
        self.model.load_state_dict(torch.load(self.t_config.ckp_path))
        
    @torch.no_grad()
    def validate(self):
        n_iters = len(self.v_dataloader)
        progbar = Progbar(n_iters, width=40, stateful_metrics=['Misc/it'])
        loader = iter(self.v_dataloader)
        
        topk_obj_list, topk_rel_list, topk_triplet_list, cls_matrix_list = np.array([]), np.array([]), np.array([]), []
        sub_scores_list, obj_scores_list, rel_scores_list = [], [], []
        sgcls_recall_list, predcls_recall_list  = [],[]
        logs = []
        
        self.model = self.model.eval()
        for i, (
            obj_pts, 
            rel_pts, 
            descriptor,
            gt_rel_label,
            gt_obj_label,
            edge_indices,
            batch_ids
        ) in enumerate(loader):
            (
                obj_pts, 
                rel_pts, 
                descriptor,
                gt_rel_label,
                gt_obj_label,
                edge_indices,
                batch_ids
            ) = self.to_device(obj_pts, rel_pts, descriptor, gt_rel_label, gt_obj_label, edge_indices, batch_ids)
            
            obj_pts = obj_pts.transpose(2, 1).contiguous()
            rel_pts = rel_pts.transpose(2, 1).contiguous()
            tfidf_class = self.tfidf.get_mask(gt_obj_label, batch_ids)
            attn_tfidf_weight = tfidf_class[gt_obj_label.long()] # N_obj X 1 
            
            _, obj_pred, rel_pred = self.model(obj_pts, rel_pts, edge_indices.t().contiguous(), descriptor, batch_ids, attn_tfidf_weight)
            top_k_obj = evaluate_topk_object(obj_pred.detach(), gt_obj_label, topk=11)
            gt_edges = get_gt(gt_obj_label, gt_rel_label, edge_indices, self.d_config.multi_rel)
            top_k_rel = evaluate_topk_predicate(rel_pred.detach(), gt_edges, self.d_config.multi_rel, topk=6)
            top_k_triplet, cls_matrix, sub_scores, obj_scores, rel_scores = \
                evaluate_triplet_topk(
                    obj_pred.detach(), rel_pred.detach(), 
                    gt_edges, edge_indices, self.d_config.multi_rel, 
                    topk=101, use_clip=True, obj_topk=top_k_obj
                )
            
            sgcls_recall = evaluate_triplet_recallk(obj_pred.detach(), rel_pred.detach(), gt_edges, edge_indices, self.d_config.multi_rel, [20,50,100], 100, use_clip=True, evaluate='triplet')
            predcls_recall = evaluate_triplet_recallk(obj_pred.detach(), rel_pred.detach(), gt_edges, edge_indices, self.d_config.multi_rel, [20,50,100], 100, use_clip=True, evaluate='rels')
            
            sgcls_recall_list.append(sgcls_recall)
            predcls_recall_list.append(predcls_recall)
            
            topk_obj_list = np.concatenate((topk_obj_list, top_k_obj))
            topk_rel_list = np.concatenate((topk_rel_list, top_k_rel))
            topk_triplet_list = np.concatenate((topk_triplet_list, top_k_triplet))
            if cls_matrix is not None:
                cls_matrix_list.extend(cls_matrix)
                sub_scores_list.extend(sub_scores)
                obj_scores_list.extend(obj_scores)
                rel_scores_list.extend(rel_scores)
            
            logs += [
                ("Acc@1/obj_cls_acc", (topk_obj_list <= 1).sum() * 100 / len(topk_obj_list)),
                ("Acc@5/obj_cls_acc", (topk_obj_list <= 5).sum() * 100 / len(topk_obj_list)),
                ("Acc@10/obj_cls_acc", (topk_obj_list <= 10).sum() * 100 / len(topk_obj_list)),
                ("Acc@1/rel_cls_acc", (topk_rel_list <= 1).sum() * 100 / len(topk_rel_list)),
                ("Acc@3/rel_cls_acc", (topk_rel_list <= 3).sum() * 100 / len(topk_rel_list)),
                ("Acc@5/rel_cls_acc", (topk_rel_list <= 5).sum() * 100 / len(topk_rel_list)),
                ("Acc@50/triplet_acc", (topk_triplet_list <= 50).sum() * 100 / len(topk_triplet_list)),
                ("Acc@100/triplet_acc", (topk_triplet_list <= 100).sum() * 100 / len(topk_triplet_list))
            ]

            progbar.add(1, values=logs)
        
        cls_matrix_list = np.stack(cls_matrix_list)
        sub_scores_list = np.stack(sub_scores_list)
        obj_scores_list = np.stack(obj_scores_list)
        rel_scores_list = np.stack(rel_scores_list)
        mean_recall = get_mean_recall(topk_triplet_list, cls_matrix_list)
        
        obj_acc_1 = (topk_obj_list <= 1).sum() * 100 / len(topk_obj_list)
        obj_acc_5 = (topk_obj_list <= 5).sum() * 100 / len(topk_obj_list)
        obj_acc_10 = (topk_obj_list <= 10).sum() * 100 / len(topk_obj_list)
        rel_acc_1 = (topk_rel_list <= 1).sum() * 100 / len(topk_rel_list)
        rel_acc_3 = (topk_rel_list <= 3).sum() * 100 / len(topk_rel_list)
        rel_acc_5 = (topk_rel_list <= 5).sum() * 100 / len(topk_rel_list)
        triplet_acc_50 = (topk_triplet_list <= 50).sum() * 100 / len(topk_triplet_list)
        triplet_acc_100 = (topk_triplet_list <= 100).sum() * 100 / len(topk_triplet_list)
        
        sgcls_recall_list=np.array(sgcls_recall_list) # N_graph X [correct@20,correct@50,correct@100]
        predcls_recall_list=np.array(predcls_recall_list) # N_graph X [correct@20,correct@50,correct@100]
        
        sgcls_recall=np.mean(sgcls_recall_list,axis=0)
        predcls_recall=np.mean(predcls_recall_list,axis=0)
        
        rel_acc_mean_1, rel_acc_mean_3, rel_acc_mean_5 = self.compute_mean_predicate(cls_matrix_list, topk_rel_list)
        self.compute_predicate_acc_per_class(cls_matrix_list, topk_rel_list)
        logs += [
            ("Acc@1/obj_cls_acc", obj_acc_1),
            ("Acc@5/obj_cls_acc", obj_acc_5),
            ("Acc@10/obj_cls_acc", obj_acc_10),
            ("Acc@1/rel_cls_acc", rel_acc_1),
            ("Acc@1/rel_cls_acc_mean", rel_acc_mean_1),
            ("Acc@3/rel_cls_acc", rel_acc_3),
            ("Acc@3/rel_cls_acc_mean", rel_acc_mean_3),
            ("Acc@5/rel_cls_acc", rel_acc_5),
            ("Acc@5/rel_cls_acc_mean", rel_acc_mean_5),
            ("Acc@50/triplet_acc", triplet_acc_50),
            ("Acc@100/triplet_acc", triplet_acc_100),
            ("mean_recall@50", mean_recall[0]),
            ("mean_recall@100", mean_recall[1]),
            
            ("SGcls@20", sgcls_recall[0]),
            ("SGcls@50", sgcls_recall[1]),
            ("SGcls@100", sgcls_recall[2]),
            ("Predcls@20", predcls_recall[0]),
            ("Predcls@50", predcls_recall[1]),
            ("Predcls@100", predcls_recall[2]),
        ] 
        return (obj_acc_1 + rel_acc_1 + rel_acc_mean_1 + mean_recall[0] + triplet_acc_50) / 5 