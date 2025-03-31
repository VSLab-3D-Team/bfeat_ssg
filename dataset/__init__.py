from dataset.dataset_3dssg import SSGLWBFeat3D, SSGLWBFeat3DwMultiModal
from dataset.dataset_geo import SSGFeatGeoAuxDataset
from utils.data_utils import read_scan_data, read_scan_data_with_rgb, read_scan_data_with_edge_view
from dataset.dataloader import CustomDataLoader, \
    collate_fn_bfeat, \
    collate_fn_bfeat_mv, \
    collate_fn_geo_aux, \
    SSGImbalanceSampler

def build_dataset(config, split, device):
    scan_data, relationship_json, objs_json, scans = read_scan_data(config, split, device)
    dataset = SSGLWBFeat3D(
        config, split, device,
        scan_data, relationship_json, objs_json, scans
    )
    return dataset

def build_dataset_multi_view(config, split, device, d_feats):
    scan_data, relationship_json, objs_json, scans = read_scan_data_with_rgb(config, split, device)
    dataset = SSGLWBFeat3DwMultiModal(
        config, split, device, d_feats,
        scan_data, relationship_json, objs_json, scans
    )
    return dataset

def build_dataset_edge_view(config, split, device):
    scan_data, relationship_json, objs_json, scans = read_scan_data_with_edge_view(config, split, device)
    dataset = SSGFeatGeoAuxDataset(
        config, split, device,
        scan_data, relationship_json, objs_json, scans
    )
    return dataset

def build_dataset_and_loader(data_type, config, device, batch_size, num_workers, oversampling=False, dfeats=None):
    if data_type == "vanilla":
        t_dataset = build_dataset(config, split="train_scans", device=device)
        v_dataset = build_dataset(config, split="validation_scans", device=device)
        w_sampler = SSGImbalanceSampler(t_dataset) if oversampling else None
        is_shuffle = True if not oversampling else False
        t_dataloader = CustomDataLoader(
            config, 
            t_dataset, 
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=w_sampler,
            shuffle=is_shuffle,
            drop_last=True,
            collate_fn=collate_fn_bfeat
        )
        v_dataloader = CustomDataLoader(
            config, 
            v_dataset, 
            batch_size=1,
            num_workers=num_workers,
            shuffle=False,
            drop_last=True,
            collate_fn=collate_fn_bfeat
        )
    elif data_type == "multi-view":
        t_dataset = build_dataset_multi_view(config, split="train_scans", device=device, d_feats=dfeats)
        v_dataset = build_dataset_multi_view(config, split="validation_scans", device=device, d_feats=dfeats)
        w_sampler = SSGImbalanceSampler(t_dataset) if oversampling else None
        is_shuffle = True if not oversampling else False
        t_dataloader = CustomDataLoader(
            config, 
            t_dataset, 
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=w_sampler,
            shuffle=is_shuffle,
            drop_last=True,
            collate_fn=collate_fn_bfeat_mv
        )
        v_dataloader = CustomDataLoader(
            config, 
            v_dataset, 
            batch_size=1,
            num_workers=num_workers,
            shuffle=False,
            drop_last=True,
            collate_fn=collate_fn_bfeat_mv
        )
    elif data_type == "edge_view_aux":
        t_dataset = build_dataset_edge_view(config, split="train_scans", device=device)
        v_dataset = build_dataset_edge_view(config, split="validation_scans", device=device)
        w_sampler = SSGImbalanceSampler(t_dataset) if oversampling else None
        is_shuffle = True if not oversampling else False
        t_dataloader = CustomDataLoader(
            config, 
            t_dataset, 
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=w_sampler,
            shuffle=is_shuffle,
            drop_last=True,
            collate_fn=collate_fn_geo_aux
        )
        v_dataloader = CustomDataLoader(
            config, 
            v_dataset, 
            batch_size=1,
            num_workers=num_workers,
            shuffle=False,
            drop_last=True,
            collate_fn=collate_fn_geo_aux
        )
    else:
        raise NotImplementedError
    return t_dataset, v_dataset, t_dataloader, v_dataloader