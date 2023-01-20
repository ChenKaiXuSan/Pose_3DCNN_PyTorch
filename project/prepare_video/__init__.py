import sys 

try:
    from split_part import *
except:
    sys.path.append('/workspace/Pose_3DCNN_PyTorch/project/helper')
    from split_part import *