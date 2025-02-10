from dataset.dataset_3dssg import SSGLWBFeat3D, SSGLWBFeat3DwMultiModal
from utils.data_utils import read_scan_data, read_scan_data_with_rgb

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