import os 
import json

def check_file_exists(path):
    if not os.path.exists(path):
        raise RuntimeError(f"File does not exist: {path}")
    
def read_txt_to_list(file):
    output = [] 
    with open(file, 'r') as f: 
        for line in f: 
            entry = line.rstrip().lower() 
            output.append(entry) 
    return output

def read_3dssg_annotation(
    root:str, 
    pth_selection:str,
    split:str,
):  
    # read object class
    pth_catfile = os.path.join(pth_selection, 'classes.txt')
    classNames = read_txt_to_list(pth_catfile)
    # read relationship class
    pth_relationship = os.path.join(pth_selection, 'relationships.txt')
    check_file_exists(pth_relationship)
    relationNames = read_txt_to_list(pth_relationship)
    # read relationship json
    selected_scans=set()
    if split == 'train_scans':
        selected_scans = selected_scans.union(read_txt_to_list(os.path.join(pth_selection,'train_scans.txt')))
        with open(os.path.join(root, 'relationships_train.json'), "r") as read_file:
            data = json.load(read_file)
    elif split == 'validation_scans':
        selected_scans = selected_scans.union(read_txt_to_list(os.path.join(pth_selection,'validation_scans.txt')))
        with open(os.path.join(root, 'relationships_validation.json'), "r") as read_file:
            data = json.load(read_file)
    else:
        raise RuntimeError('unknown split type:',split)
    return  classNames, relationNames, data, selected_scans