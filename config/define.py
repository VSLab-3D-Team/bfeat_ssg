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

SYNONYM_LIST = {
    1: ["carried by", "anchored by"],
    2: ["left-hand", "leftward"], 
    3: ["right-hand"],
    4: ["in front of", "forward", "frontal"],
    5: ["backward"], 
    6: ["nearby", "adjacent to", "next to"],
    7: ["within"],
    8: ["larger than"], 
    9: ["tinier than", "more compact than"],
    10: ["above"],
    11: ["below"], 
    12: ["same shape as"],
    13: ["not different with"],
    14: [""], 
    15: [],
    16: [],
    17: [], 
    18: [],
    19: [],
    20: [], 
    21: [],
    22: [],
    23: [], 
    24: [],
    25: [],
    26: [],
}


# 1. supported by
# 2. left
# 3. right
# 4. front
# 5. behind
# 6. close by
# 7. inside
# 8. bigger than
# 9. smaller than
# 10. higher than
# 11. lower than
# 12. same symmetry as
# 13. same as
# 14. attached to
# 15. standing on
# 16. lying on
# 17. hanging on
# 18. connected to
# 19. leaning against
# 20. part of
# 21. belonging to
# 22. build in
# 23. standing in
# 24. cover
# 25. lying in
# 26. hanging in