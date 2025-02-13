import torch
import torch.nn as nn
import torch.nn.functional as F
from model.models.baseline import BaseNetwork

## TODO: Zero-Shot classifier
@torch.no_grad()
def consine_classification(
    cls_matrix: torch.Tensor, # C X N_feat
    obj_feat: torch.Tensor   # B X N_feat
):
    # cls_matrix = F.normalize(cls_matrix, dim=-1)
    # obj_feat = F.normalize(obj_feat, dim=-1)
    sim_matrix = torch.mm(obj_feat, cls_matrix.T) # B X C
    # obj_pred = (sim_matrix + 1) * 0.5
    obj_pred = F.softmax(sim_matrix, dim=1)
    return obj_pred

class ObjectClsMulti(BaseNetwork):
    def __init__(self, k, in_size, batch_norm=True, drop_out=True, init_weights=True):
        super(ObjectClsMulti, self).__init__()
        self.name = 'pnetcls_obj'
        self.in_size=in_size
        self.use_bn = batch_norm
        self.use_drop_out = drop_out
        self.fc1 = nn.Linear(in_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)

        if self.use_drop_out:
            self.dropout = nn.Dropout(p=0.3)
        if self.use_bn:
            self.bn1 = nn.BatchNorm1d(512)
            self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

        if init_weights:
            self.init_weights('constant', 1, target_op = 'BatchNorm')
            self.init_weights('xavier_normal', 1)
            
    def forward(self, x):
        x = self.fc1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.use_drop_out:
            x = self.dropout(x)
        if self.use_bn:
            x = self.bn2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

class RelationClsMulti(BaseNetwork):

    def __init__(self, k=2, in_size=1024, batch_norm = True, drop_out = True,
                 init_weights=True):
        super(RelationClsMulti, self).__init__()
        self.name = 'pnetcls_rel'
        self.in_size=in_size
        self.use_bn = batch_norm
        self.use_drop_out = drop_out
        self.fc1 = nn.Linear(in_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)

        if self.use_drop_out:
            self.dropout = nn.Dropout(p=0.3)
        if self.use_bn:
            self.bn1 = nn.BatchNorm1d(512)
            self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

        if init_weights:
            self.init_weights('constant', 1, target_op = 'BatchNorm')
            self.init_weights('xavier_normal', 1)
    
    def forward(self, x):
        x = self.fc1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.use_drop_out:
            x = self.dropout(x)
        if self.use_bn:
            x = self.bn2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return torch.sigmoid(x)