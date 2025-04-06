from model.models.model_vanilla import BFeatVanillaNet
from model.models.model_direct_gnn import BFeatDirectGNNNet
from experiments.base_runner import BaseExperimentRunner
from utils.eval_utils import *
from utils.logger import Progbar
import numpy as np

class EntireExperimentRunners(BaseExperimentRunner):
    def __init__(
        self,
        model_name: str,
        ckp_path: str, 
        config, 
        device
    ):
        super().__init__(config, device)
        
        if model_name == "vanilla":
            self.model = BFeatVanillaNet(
                config,
                self.num_obj_class,
                self.num_rel_class,
                device
            )
        elif model_name == "direct_gnn":
            self.model = BFeatDirectGNNNet(
                config,
                self.num_obj_class,
                self.num_rel_class,
                device
            )
        else:
            raise NotImplementedError
        self.model.load_state_dict(torch.load(ckp_path))
        ckp_name = ckp_path.split("/")[-1].split(".")[0]
        self.model_name = model_name
        self.ckp_name = ckp_name
        
    @torch.no_grad()
    def validate(self):
        n_iters = len(self.v_dataloader)
        progbar = Progbar(n_iters, width=40, stateful_metrics=['Misc/it'])
        loader = iter(self.v_dataloader)
        
        topk_obj_list, topk_rel_list, topk_triplet_list, cls_matrix_list = np.array([]), np.array([]), np.array([]), []
        gt_rel_cls_list = np.array([])
        sub_scores_list, obj_scores_list, rel_scores_list = [], [], []
        sgcls_recall_list_wo, predcls_recall_list_wo, sgcls_recall_list_w, predcls_recall_list_w, sgcls_mean_recall_list_w, predcls_mean_recall_list_w  = [],[],[],[],[],[]
        logs = []
        
        # self.model.set_inference_mode()
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
            
            _, obj_pred, rel_pred = self.model(obj_pts, rel_pts, edge_indices.t().contiguous(), descriptor, batch_ids)
            top_k_obj = evaluate_topk_object(obj_pred.detach(), gt_obj_label, topk=11)
            gt_edges = get_gt(gt_obj_label, gt_rel_label, edge_indices, self.d_config.multi_rel)
            top_k_rel = evaluate_topk_predicate(rel_pred.detach(), gt_edges, self.d_config.multi_rel, topk=6, confidence_threshold=0.5)
            top_k_triplet, cls_matrix, sub_scores, obj_scores, rel_scores = \
                evaluate_triplet_topk(
                    obj_pred.detach(), rel_pred.detach(), 
                    gt_edges, edge_indices, self.d_config.multi_rel, 
                    topk=101, use_clip=True, obj_topk=top_k_obj, confidence_threshold=0.5
                )
            
            sgcls_recall_wo = evaluate_triplet_recallk(obj_pred.detach(), rel_pred.detach(), gt_edges, edge_indices, self.d_config.multi_rel, [20,50,100], 100, use_clip=True, evaluate='triplet')
            predcls_recall_wo = evaluate_triplet_recallk(obj_pred.detach(), rel_pred.detach(), gt_edges, edge_indices, self.d_config.multi_rel, [20,50,100], 100, use_clip=True, evaluate='rels')
            sgcls_recall_w = evaluate_triplet_recallk(obj_pred.detach(), rel_pred.detach(), gt_edges, edge_indices, self.d_config.multi_rel, [20,50,100], 1, use_clip=True, evaluate='triplet')
            predcls_recall_w = evaluate_triplet_recallk(obj_pred.detach(), rel_pred.detach(), gt_edges, edge_indices, self.d_config.multi_rel, [20,50,100], 1, use_clip=True, evaluate='rels')
            sgcls_mean_recall_w = evaluate_triplet_recallk(obj_pred.detach(), rel_pred.detach(), gt_edges, edge_indices, self.d_config.multi_rel, [20,50,100], 1, use_clip=True, evaluate='triplet')
            predcls_mean_recall_w = evaluate_triplet_recallk(obj_pred.detach(), rel_pred.detach(), gt_edges, edge_indices, self.d_config.multi_rel, [20,50,100], 1, use_clip=True, evaluate='rels')
            
            sgcls_recall_list_wo.append(sgcls_recall_wo)
            predcls_recall_list_wo.append(predcls_recall_wo)
            sgcls_recall_list_w.append(sgcls_recall_w)
            predcls_recall_list_w.append(predcls_recall_w)
            sgcls_mean_recall_list_w.append(sgcls_mean_recall_w)
            predcls_mean_recall_list_w.append(predcls_mean_recall_w)
            
            topk_obj_list = np.concatenate((topk_obj_list, top_k_obj))
            topk_rel_list = np.concatenate((topk_rel_list, top_k_rel))
            topk_triplet_list = np.concatenate((topk_triplet_list, top_k_triplet))
            
            gt_rel_cls_ = np.concatenate((torch.where(gt_rel_label.sum(dim=1)==0)[0].cpu().numpy(), torch.nonzero(gt_rel_label, as_tuple=True)[1].cpu().numpy()))
            gt_rel_cls_list = np.concatenate((gt_rel_cls_list, gt_rel_cls_))
            
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
                ("Acc@100/triplet_acc", (topk_triplet_list <= 100).sum() * 100 / len(topk_triplet_list)),
                ("R@3/rel_cls_recall", (topk_rel_list <= 3).sum() * 100 / len(gt_rel_cls_list)),
                ("R@5/rel_cls_recall", (topk_rel_list <= 5).sum() * 100 / len(gt_rel_cls_list)),
                ("R@50/triplet_recall", (topk_triplet_list <= 50).sum() * 100 / len(gt_rel_cls_list)),
                ("R@100/triplet_recall", (topk_triplet_list <= 100).sum() * 100 / len(gt_rel_cls_list))
            ]

            progbar.add(1, values=logs)
        
        cls_matrix_list = np.stack(cls_matrix_list)
        sub_scores_list = np.stack(sub_scores_list)
        obj_scores_list = np.stack(obj_scores_list)
        rel_scores_list = np.stack(rel_scores_list)
        mean_recall = get_mean_recall(topk_triplet_list, cls_matrix_list)
        rel_mean_recall = get_rel_mean_recall(topk_rel_list, cls_matrix_list)
        obj_mean_recall = get_obj_mean_recall(topk_obj_list, cls_matrix_list)
        
        obj_acc_1 = (topk_obj_list <= 1).sum() * 100 / len(topk_obj_list)
        obj_acc_5 = (topk_obj_list <= 5).sum() * 100 / len(topk_obj_list)
        obj_acc_10 = (topk_obj_list <= 10).sum() * 100 / len(topk_obj_list)
        rel_acc_1 = (topk_rel_list <= 1).sum() * 100 / len(topk_rel_list)
        rel_acc_3 = (topk_rel_list <= 3).sum() * 100 / len(topk_rel_list)
        rel_acc_5 = (topk_rel_list <= 5).sum() * 100 / len(topk_rel_list)
        triplet_acc_50 = (topk_triplet_list <= 50).sum() * 100 / len(topk_triplet_list)
        triplet_acc_100 = (topk_triplet_list <= 100).sum() * 100 / len(topk_triplet_list)
        rel_recall_3 = (topk_rel_list <= 3).sum() * 100 / len(gt_rel_cls_list)
        rel_recall_5 = (topk_rel_list <= 5).sum() * 100 / len(gt_rel_cls_list)
        triplet_recall_50 = (topk_triplet_list <= 50).sum() * 100 / len(gt_rel_cls_list)
        triplet_recall_100 = (topk_triplet_list <= 100).sum() * 100 / len(gt_rel_cls_list)
        
        sgcls_recall_list_wo=np.array(sgcls_recall_list_wo) # N_graph X [correct@20,correct@50,correct@100]
        predcls_recall_list_wo=np.array(predcls_recall_list_wo) # N_graph X [correct@20,correct@50,correct@100]
        sgcls_recall_list_w=np.array(sgcls_recall_list_w)
        predcls_recall_list_w=np.array(predcls_recall_list_w) 
        sgcls_mean_recall_list_w=np.array(sgcls_mean_recall_list_w) 
        predcls_mean_recall_list_w=np.array(predcls_mean_recall_list_w) 
        
        sgcls_recall_wo=np.mean(sgcls_recall_list_wo,axis=0)
        predcls_recall_wo=np.mean(predcls_recall_list_wo,axis=0)
        # sgcls_recall_w=np.mean(sgcls_recall_list_w, axis=0)
        # predcls_recall_w=np.mean(predcls_recall_list_w, axis=0) 
        # sgcls_mean_recall_w = handle_mean_recall(sgcls_mean_recall_list_w)
        # predcls_mean_recall_w = handle_mean_recall(predcls_mean_recall_list_w)
        
        rel_acc_mean_1, rel_acc_mean_3, rel_acc_mean_5 = self.compute_mean_predicate(cls_matrix_list, topk_rel_list)
        self.compute_predicate_acc_per_class(cls_matrix_list, topk_rel_list)
        
        
        results = [
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
            
            ("R@3/rel_cls_recall", rel_recall_3),
            ("R@5/rel_cls_recall", rel_recall_5),
            ("R@50/triplet_recall", triplet_recall_50),
            ("R@100/triplet_recall", triplet_recall_100),
            ("rel_mean_recall@3", rel_mean_recall[0]),
            ("rel_mean_recall@5", rel_mean_recall[1]),
            ("obj_mean_recall@1", obj_mean_recall[0]),
            ("obj_mean_recall@5", obj_mean_recall[1]),
            ("obj_mean_recall@10", obj_mean_recall[2]),
            ("mean_recall@50", mean_recall[0]),
            ("mean_recall@100", mean_recall[1]),
            
            ("SGcls_wo@20", sgcls_recall_wo[0]),
            ("SGcls_wo@50", sgcls_recall_wo[1]),
            ("SGcls_wo@100", sgcls_recall_wo[2]),
            ("Predcls_wo@20", predcls_recall_wo[0]),
            ("Predcls_wo@50", predcls_recall_wo[1]),
            ("Predcls_wo@100", predcls_recall_wo[2]),
            # ("SGcls_w@20", sgcls_recall_w[0]),
            # ("SGcls_w@50", sgcls_recall_w[1]),
            # ("SGcls_w@100", sgcls_recall_w[2]),
            # ("Predcls_w@20", predcls_recall_w[0]),
            # ("Predcls_w@50", predcls_recall_w[1]),
            # ("Predcls_w@100", predcls_recall_w[2]),
            # ("SGcls_mean_w@20", sgcls_mean_recall_w[0]),
            # ("SGcls_mean_w@50", sgcls_mean_recall_w[1]),
            # ("SGcls_mean_w@100", sgcls_mean_recall_w[2]),
            # ("Predcls_mean_w@20", predcls_mean_recall_w[0]),
            # ("Predcls_mean_w@50", predcls_mean_recall_w[1]),
            # ("Predcls_mean_w@100", predcls_mean_recall_w[2]),
        ] 
        progbar.add(1, values=logs)
        with open(f"./outputs/results_{self.model_name}_{self.ckp_name}.txt", 'w') as f:
            f.write("Evaluation results:\n")
            f.write("-----------------------------------------------------\n")
            for name, value in results:
                f.write(f"{name}: {value}\n")
            f.write("-----------------------------------------------------")
        