# %%
from pytorchvideo.models import resnet, csn, r2plus1d, x3d, slowfast
from pytorchvideo.models.head import create_res_basic_head
from pytorchvideo.models.resnet import create_bottleneck_block, create_res_block
from pytorchvideo.models.x3d import create_x3d_head
from pytorchvideo.models.net import Net

import torch
import torch.nn as nn
from torchvision.ops.misc import MLP

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

        self.part = hparams.part
        self.transfor_learning = hparams.transfor_learning

    def set_parameter_requires_grad(self, model: torch.nn.Module, flag: bool = True):

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

    def make_walk_x3d_l(self) -> nn.Module:

        if self.transfor_learning:
            # x3d l model, param 6.15 with 16 frames. more smaller maybe more faster.
            # top1 acc is 77.44
            model = torch.hub.load("facebookresearch/pytorchvideo:main", model='x3d_l', pretrained=True)
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

# start multi part fusion 

class MakeMultipartVideoModule(MakeVideoModule):

    def __init__(self, hparams) -> None:

        super().__init__(hparams)

        self.fuse_flag = hparams.fuse_flag

        # init the different network stem with stage.
        self.make_multipart_stage()

        self.multipart_head = self.fuse_head()

    def make_multipart_stage(self):
        '''
        init the different network stem and stage.
        '''

        self.head_net = Net(blocks = self.make_walk_x3d_l().blocks[:-1])
        self.upper_net = Net(blocks = self.make_walk_x3d_l().blocks[:-1])
        self.lower_net = Net(blocks = self.make_walk_x3d_l().blocks[:-1])
        self.body_net = Net(blocks = self.make_walk_x3d_l().blocks[:-1])

    def fuse_head(self):
        # todo 比较一下不同的fusion方法？

        head_list = []

        if self.fuse_flag == 'concat':

            head = MLP(
            in_channels=4*192,
            hidden_channels=(384, 192, 96, self.model_class_num),
            norm_layer=nn.BatchNorm2d,
            dropout=0.5
            )

        elif self.fuse_flag in ['sum', 'max']:
            
            head = create_x3d_head(
                dim_in= 192,
                dim_inner= 432,
                dim_out=864,
                # pool_kernel_size=(16, 10, 10),
                activation=None,
                num_classes= self.model_class_num
            )

        elif self.fuse_flag == 'conv':
            
            head = create_x3d_head(
                dim_in= 4 * 192,
                dim_inner= 432,
                dim_out=864,
                # pool_kernel_size=(16, 10, 10),
                activation=None,
                num_classes= self.model_class_num

            )
            head_list.append(head)

        else:

            head = create_res_basic_head(
                in_features=4 * 2048,
                out_features=self.model_class_num,
            )

        # FIXME have err because the nn.modulelist 
        return Net(blocks=nn.ModuleList(head_list))

    def forward(self, part_Dict: dict):

        head = part_Dict['head']
        upper = part_Dict['upper']
        lower = part_Dict['lower']
        body = part_Dict['body']

        head_output = self.head_net(head)
        upper_output = self.upper_net(upper)
        lower_output = self.lower_net(lower)
        body_output = self.body_net(body)

        # todo remake the code
        if self.fuse_flag in ['concat', 'conv']:
            feat = torch.cat((head_output, upper_output, lower_output, body_output), dim=1)
        elif self.fuse_flag == 'max':
            feat1 = torch.maximum(head_output, upper_output)
            feat2 = torch.maximum(lower_output, body_output)
            feat = torch.maximum(feat1, feat2)
        elif self.fuse_flag == 'sum':
            feat = head_output + upper_output + lower_output + body_output

        output = self.multipart_head(feat)

        return output

# %%
#todo 这个方法到底有没有必要把他抽象成一个单独的类
#todo 另一个问题就是，用了mlp之后结果会变好吗
# class FuseHead(MakeVideoModule, nn.Module):
#     def __init__(self, hparams) -> None:
#         super().__init__(hparams)

#         self.fuse_flag = hparams.fuse_flag

#         # init the fuse head 
#         self.head = self.fuse_head()

#     def fuse_head(self):
#         head_list = []
#         slow = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)

#         if self.fuse_flag == 'concat':
#             slow.blocks[-1].proj = nn.Linear(2048 * 4, self.model_class_num)
#             head = slow.blocks[-1]
#             head_list.append(head)

#         elif self.fuse_flag in ['sum', 'max']:
#             # todo MLP?
#             # slow.blocks[-1].proj = nn.Linear(2048, self.model_class_num)
#             head = slow.blocks[-1]

#             head_list.append(head.pool)

#             head_list.append(
#                 MLP(
#                     in_channels=2048,
#                     hidden_channels=(1024*3, 512, 256, self.model_class_num),
#                     norm_layer=nn.BatchNorm2d,
#                     dropout=0.5
#                 ))

#             head_list.append(
#                 nn.AdaptiveAvgPool3d(output_size=self.model_class_num)
#             )

#         elif self.fuse_flag == 'conv':

#             # conv part
#             stage = create_bottleneck_block(
#                 dim_in=4 * 2048,
#                 dim_inner=2 * 2048,
#                 dim_out=2048,
#             )

#             head_list.append(stage)

#             # change the knetics-400 output 400 to model class num
#             slow.blocks[-1].proj = nn.Linear(2048, self.model_class_num)
#             head = slow.blocks[-1]
#             head.pool = torch.nn.AvgPool3d(kernel_size=(5, 4, 4), stride=(1, 1, 1), padding=(0, 0, 0))
#             head_list.append(head)

#             return Net(blocks=nn.ModuleList(head_list))

#         else:

#             head = create_res_basic_head(
#                 in_features=4 * 2048,
#                 out_features=self.model_class_num,
#             )

#             return head

#     def forward(self, x_list: torch.Tensor) -> torch.Tensor:
        
#         head = x_list[0]
#         upper = x_list[1]
#         lower = x_list[2]
#         body = x_list[3]

#         if self.fuse_flag in ['concat', 'conv']:
#             feat = torch.cat((head, upper, lower, body), dim=1)
#         elif self.fuse_flag == 'max':
#             feat1 = torch.maximum(head, upper)
#             feat2 = torch.maximum(lower, body)
#             feat = torch.maximum(feat1, feat2)
#         elif self.fuse_flag == 'sum':
#             feat = head + upper + lower + body

#         return self.head(feat)


# %%
# list the model in repo.
# torch.hub.list('facebookresearch/pytorchvideo', force_reload=True)
# # %%
