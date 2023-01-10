# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from pytorch_lightning import LightningModule

from make_model import MakeVideoModule
from helper.split_part import BatchSplit
from utils.metrics import *

# for metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt 
import seaborn as sns

# test 
from torchvision.io.video import write_video

# %%

class WalkVideoClassificationLightningModule(LightningModule):
    
    def __init__(self, hparams):
        super().__init__()

        # return model type name
        self.model_type=hparams.model
        self.img_size = hparams.img_size

        self.lr=hparams.lr
        self.num_class = hparams.model_class_num
        self.uniform_temporal_subsample_num = hparams.uniform_temporal_subsample_num

        self.model = MakeVideoModule(hparams)

        # select the network structure 
        if self.model_type == 'resnet':
            self.model=self.model.make_walk_resnet()

        elif self.model_type == 'r2plus1d':
            self.model = self.model.make_walk_r2plus1d()

        elif self.model_type == 'csn':
            self.model=self.model.make_walk_csn()

        elif self.model_type == 'x3d':
            self.model = self.model.make_walk_x3d()

        elif self.model_type == 'slowfast':
            self.model = self.model.make_walk_slow_fast()

        elif self.model_type == 'i3d':
            self.model = self.model.make_walk_i3d()

        elif self.model_type == 'c2d':
            self.model = self.model.make_walk_c2d()

        self.transfor_learning = hparams.transfor_learning

        # save the hyperparameters to the file and ckpt
        self.save_hyperparameters()

        # select the metrics
        self._accuracy = get_Accuracy(self.num_class)
        self._precision = get_Precision(self.num_class)
        self._confusion_matrix = get_Confusion_Matrix()

        # self.dice = get_Dice()
        # self.average_precision = get_Average_precision(self.num_class)
        # self.AUC = get_AUC()
        # self.f1_score = get_F1Score()
        # self.precision_recall = get_Precision_Recall()
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        '''
        train steop when trainer.fit called

        Args:
            batch (3D tensor): b, c, t, h, w
            batch_idx (_type_): _description_

        Returns:
            loss: the calc loss
        '''
        
        # input and label
        video = batch['video'].detach()

        label = batch['label'].detach() # b, class_num

        # split different part 
        body_keypoints = BatchSplit(self.img_size, self.uniform_temporal_subsample_num)
        head, upper, lower, raw = body_keypoints.handle_batch_video(batch) # b, c, t, h, w

        # for test, save one batch
        write_video('%s_head_pard.mp4' % batch_idx, video_array=head[1].cpu().permute(1, 2, 3, 0), fps=30)
        write_video('%s_upper_pard.mp4' % batch_idx, video_array=upper[1].cpu().permute(1, 2, 3, 0), fps=30)
        write_video('%s_lower_pard.mp4' % batch_idx, video_array=lower[1].cpu().permute(1, 2, 3, 0), fps=30)
        # write_video('raw_pard.mp4', video_array=raw[1].cpu().permute(1, 2, 3, 0), fps=30)

        # classification task
        y_hat = self.model(video)

        # when torch.size([1]), not squeeze.
        if y_hat.size()[0] != 1 or len(y_hat.size()) != 1 :
            y_hat = y_hat.squeeze(dim=-1)
            
            y_hat_sigmoid = torch.sigmoid(y_hat)
        
        else:
            y_hat_sigmoid = torch.sigmoid(y_hat)

        loss = F.binary_cross_entropy_with_logits(y_hat, label.float())
        # soft_margin_loss = F.soft_margin_loss(y_hat_sigmoid, label.float())

        accuracy = self._accuracy(y_hat_sigmoid, label)
        precision = self._precision(y_hat_sigmoid, label)

        self.log_dict({'train_loss': loss, 'train_acc': accuracy, 'train_precision': precision})

        return loss

    # def training_epoch_end(self, outputs) -> None:
    #     '''
    #     after validattion_step end.

    #     Args:
    #         outputs (list): a list of the train_step return value.
    #     '''
        
    #     # log epoch metric
    #     # self.log('train_acc_epoch', self.accuracy)
    #     pass

    def validation_step(self, batch, batch_idx):
        '''
        val step when trainer.fit called.

        Args:
            batch (3D tensor): b, c, t, h, w
            batch_idx (_type_): _description_

        Returns:
            loss: the calc loss 
            accuract: selected accuracy result.
        '''

        # input and label
        video = batch['video'].detach() # b, c, t, h, w

        label = batch['label'].detach() # b, class_num

        # split different part 
        body_keypoints = BatchSplit(self.img_size)
        head, upper, lower, raw = body_keypoints.handle_batch_video(batch) # b, c, t, h, w

        # for test, save one batch
        # write_video('head_pard.mp4', video_array=head[1].cpu().permute(1, 2, 3, 0), fps=30)
        # write_video('upper_pard.mp4', video_array=upper[1].cpu().permute(1, 2, 3, 0), fps=30)
        # write_video('lower_pard.mp4', video_array=lower[1].cpu().permute(1, 2, 3, 0), fps=30)
        # write_video('raw_pard.mp4', video_array=raw[1].cpu().permute(1, 2, 3, 0), fps=30)

        # self.model.eval()

        # pred the video frames
        with torch.no_grad():
            head_preds = self.model(head)
            preds = self.model(video)

        # when torch.size([1]), not squeeze.
        if preds.size()[0] != 1 or len(preds.size()) != 1 :
            preds = preds.squeeze(dim=-1)
            preds_sigmoid = torch.sigmoid(preds)
        else:
            preds_sigmoid = torch.sigmoid(preds)

        # squeeze(dim=-1) to keep the torch.Size([1]), not null.
        val_loss = F.binary_cross_entropy_with_logits(preds, label.float())

        # calc the metric, function from torchmetrics
        accuracy = self._accuracy(preds_sigmoid, label)

        precision = self._precision(preds_sigmoid, label)

        confusion_matrix = self._confusion_matrix(preds_sigmoid, label)

        # log the val loss and val acc, in step and in epoch.
        self.log_dict({'val_loss': val_loss, 'val_acc': accuracy, 'val_precision': precision}, on_step=False, on_epoch=True)
        
        return accuracy

    def validation_epoch_end(self, outputs):
        pass
        
        # val_metric = torch.stack(outputs, dim=0)

        # final_acc = (torch.sum(val_metric) / len(val_metric)).item()

        # print('Epoch: %s, avgAcc: %s' % (self.current_epoch, final_acc))

        # self.ACC[self.current_epoch] = final_acc

    def on_validation_end(self) -> None:
        pass
            
    def test_step(self, batch, batch_idx):
        '''
        test step when trainer.test called

        Args:
            batch (3D tensor): b, c, t, h, w
            batch_idx (_type_): _description_
        '''

        # input and label
        video = batch['video'].detach() # b, c, t, h, w

        if self.fusion_method == 'single_frame': 
            label = batch['label'].detach()

            # when batch > 1, for multi label, to repeat label in (bxt)
            label = label.repeat_interleave(self.uniform_temporal_subsample_num).squeeze()

        else:
            label = batch['label'].detach() # b, class_num

        self.model.eval()

        # pred the video frames
        with torch.no_grad():
            preds = self.model(video)

        # when torch.size([1]), not squeeze.
        if preds.size()[0] != 1 or len(preds.size()) != 1 :
            preds = preds.squeeze(dim=-1)
            preds_sigmoid = torch.sigmoid(preds)
        else:
            preds_sigmoid = torch.sigmoid(preds)

        # squeeze(dim=-1) to keep the torch.Size([1]), not null.
        val_loss = F.binary_cross_entropy_with_logits(preds, label.float())

        # calc the metric, function from torchmetrics
        accuracy = self._accuracy(preds_sigmoid, label)

        precision = self._precision(preds_sigmoid, label)

        confusion_matrix = self._confusion_matrix(preds_sigmoid, label)

        # log the val loss and val acc, in step and in epoch.
        self.log_dict({'test_loss': val_loss, 'test_acc': accuracy, 'test_precision': precision}, on_step=False, on_epoch=True)

        return {
            'pred': preds_sigmoid.tolist(),
            'label': label.tolist()
        }
        
    def test_epoch_end(self, outputs):
        #todo try to store the pred or confusion matrix
        pred_list = []
        label_list = []

        for i in outputs:
            for number in i['pred']:
                if number > 0.5:
                    pred_list.append(1)
                else:
                    pred_list.append(0)
            for number in i['label']:
                label_list.append(number)

        pred = torch.tensor(pred_list)
        label = torch.tensor(label_list)

        cm = confusion_matrix(label, pred)
        ax = sns.heatmap(cm, annot=True, fmt="3d")

        ax.set_title('confusion matrix')
        ax.set(xlabel="pred class", ylabel="ground truth")
        ax.xaxis.tick_top()
        plt.show()
        plt.savefig('test.png')

        return cm 

    def configure_optimizers(self):
        '''
        configure the optimizer and lr scheduler

        Returns:
            optimizer: the used optimizer.
            lr_scheduler: the selected lr scheduler.
        '''

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
                "monitor": "val_loss",
            },
        }
        # return torch.optim.SGD(self.parameters(), lr=self.lr)

    def _get_name(self):
        return self.model_type