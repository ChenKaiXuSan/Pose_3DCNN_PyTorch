import sys 

try:
    from data_loader import *
except:
    sys.path.append('/workspace/Walk_Video_PyTorch/project/dataloader')
    from data_loader import *