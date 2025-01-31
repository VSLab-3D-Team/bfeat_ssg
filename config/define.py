import os

SSG_DATA_PATH = "/media/michael/ssd1/SceneGraph/3DSSG"
SSG_PREPROCESS_PATH = "/media/michael/ssd1/SceneGraph/LWBFeat3DSG"
OBJ_LABEL_PATH = f"{SSG_DATA_PATH}/3DSSG_subset/classes.txt"
REL_LABEL_PATH = f"{SSG_DATA_PATH}/3DSSG_subset/relations.txt"
PROJECT_PATH = os.path.abspath(".")

# 3RScan file names
LABEL_FILE_NAME_RAW = 'labels.instances.annotated.v2.ply'
LABEL_FILE_NAME = 'labels.instances.align.annotated.v2.ply'
SEMSEG_FILE_NAME = 'semseg.v2.json'
MTL_NAME = 'mesh.refined.mtl'
OBJ_NAME = 'mesh.refined.v2.obj'
TEXTURE_NAME = 'mesh.refined_0.png'

PREDICATE_CATEGORY = {
    "spatial": [2, 3, 4, 5, 6, 10, 11],
    "size_comparison": [8, 9, 12, 13],
    "support_contact": [1, 14, 15, 16, 17, 18, 19, 26], 
    "containment_affiliation": [7, 20, 21, 22, 23, 25],
    "cover": [24],
    "none": [0]
}