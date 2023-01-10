import sys 

try:
    from make_model import *
    from pytorchvideo_models import *
except:
    sys.path.append('/workspace/Pose_3DCNN_PyTorch/project/models')
    from make_model import *
    from pytorchvideo_models import *