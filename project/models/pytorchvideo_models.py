# %%
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_lightning import LightningModule

from make_model import MakeVideoModule, MakeMultipartVideoModule

from torchmetrics.functional.classification import \
    binary_f1_score, \
    binary_accuracy, \
    binary_cohen_kappa, \
    binary_auroc, \
    binary_confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# test
from torchvision.io.video import write_video

# %%


class WalkVideoClassificationLightningModule(LightningModule):

    def __init__(self, hparams):
        super().__init__()

        # return model type name
        self.model_type = hparams.model
        self.img_size = hparams.img_size

        self.lr = hparams.lr
        self.num_class = hparams.model_class_num

        # frame rate
        self.uniform_temporal_subsample_num = hparams.uniform_temporal_subsample_num

        # body part
        self.part = hparams.part

        if self.part == 'all':
            self.multipart_model = MakeMultipartVideoModule(hparams)

        elif self.part == 'all_loss':

            self.model = MakeVideoModule(hparams)

            # init the different network part
            self.head_net = self.model.make_walk_resnet()
            self.upper_net = self.model.make_walk_resnet()
            self.lower_net = self.model.make_walk_resnet()
            self.body_net = self.model.make_walk_resnet()

        else:

            self.model = MakeVideoModule(hparams)

            # select the network structure
            if self.model_type == 'resnet':
                self.model = self.model.make_walk_resnet()

            elif self.model_type == 'x3d_l':
                self.model = self.model.make_walk_x3d_l()

        # save the hyperparameters to the file and ckpt
        self.save_hyperparameters()

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

        batch_part_video_Dict = {}
        batch_part_label_Dict = {}

        # input and label
        for part in batch.keys():
            batch_part_video_Dict[part] = batch[part]['video'].detach()
            batch_part_label_Dict[part] = batch[part]['label'].detach()

        if self.part == 'all':

            y_hat = self.multipart_model(batch_part_video_Dict)
            label = batch_part_label_Dict['body']

        elif self.part == 'all_loss':

            # todo how to train the network.
            head_pred = self.head_net(batch_part_video_Dict['head'])
            upper_pred = self.upper_net(batch_part_video_Dict['upper'])
            lower_pred = self.lower_net(batch_part_video_Dict['lower'])
            body_pred = self.body_net(batch_part_video_Dict['body'])

        else:
            y_hat = self.model(batch_part_video_Dict[self.part])
            label = batch_part_label_Dict[self.part]

        # when torch.size([1]), not squeeze.
        if y_hat.size()[0] != 1 or len(y_hat.size()) != 1:
            y_hat = y_hat.squeeze(dim=-1)

            y_hat_sigmoid = torch.sigmoid(y_hat)

        else:
            y_hat_sigmoid = torch.sigmoid(y_hat)

        loss = F.binary_cross_entropy_with_logits(y_hat, label.float())

        # metrics
        accuracy = binary_accuracy(y_hat_sigmoid, label)
        f1_score = binary_f1_score(y_hat_sigmoid, label)
        auroc = binary_auroc(y_hat_sigmoid, label)
        cohen_kappa = binary_cohen_kappa(y_hat_sigmoid, label)
        cm = binary_confusion_matrix(y_hat_sigmoid, label)

        # log to tensorboard
        self.log_dict({'train_loss': loss,
                       'train_acc': accuracy,
                       'train_f1_score': f1_score, 
                       'train_auroc': auroc, 
                       'train_cohen_kappa': cohen_kappa, 
                       })
        
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
            accuracy: selected accuracy result.
        '''
        batch_part_video_Dict = {}
        batch_part_label_Dict = {}

        for part in batch.keys():
            batch_part_video_Dict[part] = batch[part]['video'].detach()
            batch_part_label_Dict[part] = batch[part]['label'].detach()

        if self.part == 'all':
            with torch.no_grad():
                preds = self.multipart_model(batch_part_video_Dict)

            label = batch_part_label_Dict['body']

        else:
            # pred the video frames
            with torch.no_grad():
                preds = self.model(batch_part_video_Dict[self.part])

            label = batch_part_label_Dict[self.part]

        # when torch.size([1]), not squeeze.
        if preds.size()[0] != 1 or len(preds.size()) != 1:
            preds = preds.squeeze(dim=-1)
            preds_sigmoid = torch.sigmoid(preds)
        else:
            preds_sigmoid = torch.sigmoid(preds)

        # squeeze(dim=-1) to keep the torch.Size([1]), not null.
        val_loss = F.binary_cross_entropy_with_logits(preds, label.float())

        # calc the metric, function from torchmetrics
        
        # metrics
        accuracy = binary_accuracy(preds_sigmoid, label)
        f1_score = binary_f1_score(preds_sigmoid, label)
        auroc = binary_auroc(preds_sigmoid, label)
        cohen_kappa = binary_cohen_kappa(preds_sigmoid, label)
        cm = binary_confusion_matrix(preds_sigmoid, label)

        # log to tensorboard
        self.log_dict({'val_loss': val_loss,
                       'val_acc': accuracy,
                       'val_f1_score': f1_score, 
                       'val_auroc': auroc, 
                       'val_cohen_kappa': cohen_kappa, 
                       })
        
        print(cm)

        return val_loss

    def validation_epoch_end(self, outputs):
        pass

        # val_metric = torch.stack(outputs, dim=0)

        # final_acc = (torch.sum(val_metric) / len(val_metric)).item()

        # print('Epoch: %s, avgAcc: %s' % (self.current_epoch, final_acc))

        # self.ACC[self.current_epoch] = final_acc

    def on_validation_end(self) -> None:
        pass

    # def test_step(self, batch, batch_idx):
    #     '''
    #     test step when trainer.test called

    #     Args:
    #         batch (3D tensor): b, c, t, h, w
    #         batch_idx (_type_): _description_
    #     '''

    #     # input and label
    #     video = batch['video'].detach()  # b, c, t, h, w

    #     if self.fusion_method == 'single_frame':
    #         label = batch['label'].detach()

    #         # when batch > 1, for multi label, to repeat label in (bxt)
    #         label = label.repeat_interleave(self.uniform_temporal_subsample_num).squeeze()

    #     else:
    #         label = batch['label'].detach()  # b, class_num

    #     self.model.eval()

    #     # pred the video frames
    #     with torch.no_grad():
    #         preds = self.model(video)

    #     # when torch.size([1]), not squeeze.
    #     if preds.size()[0] != 1 or len(preds.size()) != 1:
    #         preds = preds.squeeze(dim=-1)
    #         preds_sigmoid = torch.sigmoid(preds)
    #     else:
    #         preds_sigmoid = torch.sigmoid(preds)

    #     # squeeze(dim=-1) to keep the torch.Size([1]), not null.
    #     val_loss = F.binary_cross_entropy_with_logits(preds, label.float())

    #     # calc the metric, function from torchmetrics
    #     accuracy = self._accuracy(preds_sigmoid, label)

    #     precision = self._precision(preds_sigmoid, label)

    #     confusion_matrix = self._confusion_matrix(preds_sigmoid, label)

    #     # log the val loss and val acc, in step and in epoch.
    #     self.log_dict({'test_loss': val_loss, 'test_acc': accuracy,
    #                   'test_precision': precision}, on_step=False, on_epoch=True)

    #     return {
    #         'pred': preds_sigmoid.tolist(),
    #         'label': label.tolist()
    #     }

    # def test_epoch_end(self, outputs):
    #     # todo try to store the pred or confusion matrix
    #     pred_list = []
    #     label_list = []

    #     for i in outputs:
    #         for number in i['pred']:
    #             if number > 0.5:
    #                 pred_list.append(1)
    #             else:
    #                 pred_list.append(0)
    #         for number in i['label']:
    #             label_list.append(number)

    #     pred = torch.tensor(pred_list)
    #     label = torch.tensor(label_list)

    #     cm = confusion_matrix(label, pred)
    #     ax = sns.heatmap(cm, annot=True, fmt="3d")

    #     ax.set_title('confusion matrix')
    #     ax.set(xlabel="pred class", ylabel="ground truth")
    #     ax.xaxis.tick_top()
    #     plt.show()

    #     return cm

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
