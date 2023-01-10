from argparse import ArgumentParser

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
    parser.add_argument('--batch_size', type=int, default=8, help='batch size for the dataloader')
    parser.add_argument('--num_workers', type=int, default=8, help='dataloader for load video')
    parser.add_argument('--clip_duration', type=int, default=1, help='clip duration for the video')
    parser.add_argument('--uniform_temporal_subsample_num', type=int,
                        default=8, help='num frame from the clip duration')
    parser.add_argument('--gpu_num', type=int, default=0, choices=[0, 1], help='the gpu number whicht to train')

    # ablation experment 
    # different fusion method 
    # parser.add_argument('--fusion_method', type=str, default='slow_fusion', choices=['single_frame', 'early_fusion', 'late_fusion', 'slow_fusion'], help="select the different fusion method from ['single_frame', 'early_fusion', 'late_fusion']")
    # pre process flag
    parser.add_argument('--pre_process_flag', action='store_true', help='if use the pre process video, which detection.')
    # Transfor_learning
    parser.add_argument('--transfor_learning', action='store_true', help='if use the transformer learning')
    # parser.add_argument('--fix_layer', type=str, default='all', choices=['all', 'head', 'stem_head', 'stage_head'], help="select the ablation study within the choices ['all', 'head', 'stem_head', 'stage_head'].")

    # TTUR
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate for optimizer')
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)

    # Path
    parser.add_argument('--data_path', type=str, default="/workspace/data/dataset/", help='meta dataset path')
    parser.add_argument('--split_data_path', type=str,
                        default="/workspace/data/splt_dataset_512", help="split dataset path")
    parser.add_argument('--split_pad_data_path', type=str, default="/workspace/data/split_pad_dataset_512/",
                        help="split and pad dataset with detection method.")

    parser.add_argument('--log_path', type=str, default='./logs', help='the lightning logs saved path')

    # using pretrained
    parser.add_argument('--pretrained_model', type=bool, default=False,
                        help='if use the pretrained model for training.')

    # add the parser to ther Trainer
    # parser = Trainer.add_argparse_args(parser)

    return parser.parse_known_args()
