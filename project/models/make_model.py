# %%
from pytorchvideo.models import resnet, csn, r2plus1d, x3d, slowfast

import torch
import torch.nn as nn
import copy


# %%

class MakeVideoModule(nn.Module):
    '''
    the module zoo from the PytorchVideo lib.

    Args:
        nn (_type_): 
    '''

    def __init__(self, hparams) -> None:

        super().__init__()

        self.model_class_num = hparams.model_class_num
        self.model_depth = hparams.model_depth

        self.transfor_learning = hparams.transfor_learning

    def set_parameter_requires_grad(self, model: torch.nn.Module, flag:bool = True):

        for param in model.parameters():
            param.requires_grad = flag

    def make_walk_csn(self):

        if self.transfor_learning:
            CSN = torch.hub.load("facebookresearch/pytorchvideo:main", model='csn_r101', pretrained=True)
            CSN.blocks[-1].proj = nn.Linear(2048, self.model_class_num)
        
        else:
            CSN = csn.create_csn(
            input_channel=3,
            model_depth=self.model_depth,
            model_num_class=self.model_class_num,
            norm=nn.BatchNorm3d,
            activation=nn.ReLU,
            )

        return CSN

    def make_walk_r2plus1d(self) -> nn.Module:

        if self.transfor_learning:
            model = torch.hub.load("facebookresearch/pytorchvideo:main", model='r2plus1d_r50', pretrained=True)

            # change the head layer.
            model.blocks[-1].proj = nn.Linear(2048, self.model_class_num)
            model.blocks[-1].activation = None
        
        else:
            model = r2plus1d.create_r2plus1d(
                input_channel=3,
                model_depth=self.model_depth,
                model_num_class=self.model_class_num,
                norm=nn.BatchNorm3d,
                activation=nn.ReLU,
                head_activation=None,
            )

        return model

    def make_walk_c2d(self) -> nn.Module:

        if self.transfor_learning:
            model = torch.hub.load("facebookresearch/pytorchvideo:main", model='c2d_r50', pretrained=True)
            model.blocks[-1].proj = nn.Linear(2048, self.model_class_num, bias=True)

            return model
        else:
            print('no orignal model supported!')

    def make_walk_i3d(self) -> nn.Module:
        
        if self.transfor_learning:
            model = torch.hub.load("facebookresearch/pytorchvideo:main", model='i3d_r50', pretrained=True)
            model.blocks[-1].proj = nn.Linear(2048, self.model_class_num)

            return model
        else:
            print('no orignal model supported!')

    def make_walk_x3d(self) -> nn.Module:

        if self.transfor_learning:
            model = torch.hub.load("facebookresearch/pytorchvideo:main", model='x3d_m', pretrained=True)
            model.blocks[-1].proj = nn.Linear(2048, self.model_class_num)
            model.blocks[-1].activation = None

        else:
            model = x3d.create_x3d(
                input_channel=3, 
                input_clip_length=16,
                input_crop_size=224,
                model_num_class=1,
                norm=nn.BatchNorm3d,
                activation=nn.ReLU,
                head_activation=None,
            )

        return model

    def make_walk_slow_fast(self) -> nn.Module:

        if self.transfor_learning:
            model = torch.hub.load("facebookresearch/pytorchvideo:main", model='slowfast_r50', pretrained=True)
            model.blocks[-1].proj = nn.Linear(2048, self.model_class_num)

        else:

            model = slowfast.create_slowfast(
                input_channels=3,
                model_depth=self.model_depth,
                model_num_class=self.model_class_num,
                norm=nn.BatchNorm3d,
                activation=nn.ReLU,
            )

        return model


    def make_walk_resnet(self):
        
        # make model
        if self.transfor_learning:
            slow = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
            
            # change the knetics-400 output 400 to model class num
            slow.blocks[-1].proj = nn.Linear(2048, self.model_class_num)

        else:
            slow = resnet.create_resnet(
                input_channel=3,
                model_depth=self.model_depth,
                model_num_class=self.model_class_num,
                norm=nn.BatchNorm3d,
                activation=nn.ReLU,
            )

        return slow

# # %%
# class opt: 

#     model_class_num = 1
#     model_depth = 50
#     transfor_learning = True
#     fix_layer = 'stage_head'

# make_video_module = MakeVideoModule(opt)

# model = make_video_module.make_walk_i3d()

# from torchinfo import summary

# summary(model, input_size=(4, 3, 16, 224, 224))
# # %%

# single_frame_model = single_frame(opt)

# batch_video = torch.randn(size=[4, 3, 16, 224, 224])

# output = single_frame_model(batch_video)

# %%
# list the model in repo.
torch.hub.list('facebookresearch/pytorchvideo', force_reload=True)
# # %%