from dataset.dataset_3dssg import SSGLWBFeat3D
from utils.data_utils import read_scan_data

def build_dataset(config, split, device):
    scan_data, relationship_json, objs_json, scans = read_scan_data(config, split, device)
    dataset = SSGLWBFeat3D(
        config, split, device,
        scan_data, relationship_json, objs_json, scans
    )
    return dataset