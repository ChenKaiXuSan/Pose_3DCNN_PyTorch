'''
this project were based the pytorch, pytorch lightning and pytorch video library, 
for rapid development.
'''

# %%
import os
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers
# callbacks
from pytorch_lightning.callbacks import TQDMProgressBar, RichModelSummary, RichProgressBar, ModelCheckpoint, EarlyStopping
from pl_bolts.callbacks import PrintTableMetricsCallback, TrainingDataMonitor
from utils.utils import get_ckpt_path

from dataloader.data_loader import WalkDataModule
from models.pytorchvideo_models import WalkVideoClassificationLightningModule
from argparse import ArgumentParser

import pytorch_lightning
# %%

def get_parameters():
    '''
    The parameters for the model training, can be called out via the --h menu
    '''
    parser = ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'csn', 'r2plus1d', 'x3d', 'slowfast', 'c2d', 'i3d'])
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--version', type=str, default='test', help='the version of logger, such data')
    parser.add_argument('--model_class_num', type=int, default=1, help='the class num of model')
    parser.add_argument('--model_depth', type=int, default=50, choices=[50, 101, 152], help='the depth of used model')

    # Training setting
    parser.add_argument('--max_epochs', type=int, default=50, help='numer of epochs of training')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size for the dataloader')
    parser.add_argument('--num_workers', type=int, default=2, help='dataloader for load video')
    parser.add_argument('--clip_duration', type=int, default=1, help='clip duration for the video')
    parser.add_argument('--uniform_temporal_subsample_num', type=int,
                        default=10, help='num frame from the clip duration')
    parser.add_argument('--gpu_num', type=int, default=0, choices=[0, 1], help='the gpu number whicht to train')
    parser.add_argument('--part', type=str, default='all', choices=['all', 'body', 'head', 'upper', 'lower'], help='which part to used.')
    parser.add_argument('--fuse_flag', type=str, default='conv', choices=['sum', 'max', 'concat', 'conv'], help='how to fuse the different feature when part is all.')

    # ablation experment 
    # pre process flag
    parser.add_argument('--pre_process_flag', action='store_true', help='if use the pre process video, which detection.')
    # Transfor_learning
    parser.add_argument('--transfor_learning', action='store_true', help='if use the transformer learning')

    # TTUR
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate for optimizer')
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)

    # Path
    parser.add_argument('--data_path', type=str, default="/workspace/data/dataset/", help='meta dataset path')
    parser.add_argument('--pose_data_path', type=str,
                        default="/workspace/data/new_Pose_dataset_512", help="pose based dataset, split person to four different part. [body, head, upper, lower]")
    parser.add_argument('--split_pad_data_path', type=str, default="/workspace/data/split_pad_dataset_512/",
                        help="split and pad dataset with detection method.")

    parser.add_argument('--log_path', type=str, default='./logs', help='the lightning logs saved path')

    # add the parser to ther Trainer
    # parser = Trainer.add_argparse_args(parser)

    return parser.parse_known_args()

# %%

def train(hparams):

    # fixme will occure bug, with deterministic = true
    # seed_everything(42, workers=True)

    classification_module = WalkVideoClassificationLightningModule(hparams)

    # instance the data module
    data_module = WalkDataModule(hparams)

    # for the tensorboard
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.join(hparams.log_path, hparams.model), name=hparams.log_version, version=hparams.fold)

    # some callbacks
    progress_bar = TQDMProgressBar(refresh_rate=100)
    rich_model_summary = RichModelSummary(max_depth=2)
    rich_progress_bar = RichProgressBar(refresh_rate=hparams.batch_size)

    # define the checkpoint becavier.
    model_check_point = ModelCheckpoint(
        filename='{epoch}-{val_loss:.2f}-{val_acc:.4f}',
        auto_insert_metric_name=True,
        monitor="val_acc",
        mode="max",
        save_last=True,
        save_top_k=5,

    )

    # define the early stopping.
    # stop when val acc dont increase in 5 epochs.
    early_stopping = EarlyStopping(
        monitor='val_acc',
        patience=5,
        mode='max',
    )

    # bolts callbacks
    table_metrics_callback = PrintTableMetricsCallback()
    monitor = TrainingDataMonitor(log_every_n_steps=50)

    trainer = Trainer(
                      devices=[hparams.gpu_num,],
                      accelerator="gpu",
                      max_epochs=hparams.max_epochs,
                      logger=tb_logger,
                      #   log_every_n_steps=100,
                      check_val_every_n_epoch=1,
                      callbacks=[progress_bar, rich_model_summary, table_metrics_callback, monitor, model_check_point, early_stopping],
                      #   deterministic=True
                      multiple_trainloader_mode='min_size'
                      )

    # from the params
    # trainer = Trainer.from_argparse_args(hparams)

    # training and val
    trainer.fit(classification_module, data_module)

    Acc_list = trainer.validate(classification_module, data_module, ckpt_path='best')

    # return the best acc score.
    return model_check_point.best_model_score.item()
 
# %%
if __name__ == '__main__':

    # for test in jupyter
    config, unkonwn = get_parameters()

    #############
    # K Fold CV
    #############

    if config.pre_process_flag:
        DATA_PATH = config.pose_data_path
    else:
        DATA_PATH = config.data_path

    # get the fold number
    fold_num = os.listdir(DATA_PATH)
    fold_num.sort()
    fold_num.remove('raw')

    store_Acc_Dict = {}
    sum_list = []

    for fold in fold_num:
        #################
        # start k Fold CV
        #################

        print('#' * 50)
        print('Strat %s' % fold)
        print('#' * 50)

        config.train_path = os.path.join(DATA_PATH, fold)
        config.fold = fold

        # connect the version + model + depth, for tensorboard logger.
        config.log_version = config.version + '_' + config.model + '_depth' + str(config.model_depth)

        Acc_score = train(config)

        store_Acc_Dict[fold] = Acc_score
        sum_list.append(Acc_score)

    print('#' * 50)
    print('different fold Acc:')
    print(store_Acc_Dict)
    print('Final avg Acc is: %s' % (sum(sum_list) / len(sum_list)))
    