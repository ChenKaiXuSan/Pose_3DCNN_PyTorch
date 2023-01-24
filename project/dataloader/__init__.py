import sys 

try:
    from data_loader import *
except:
    sys.path.append('/workspace/Pose_3DCNN_PyTorch/project/dataloader')
    from data_loader import *