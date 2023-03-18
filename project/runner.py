"""
help function for experiment.

"""

import time
import subprocess
from argparse import ArgumentParser

VIDEO_LENGTH = ['1', '2']
VIDEO_FRAME = ['16']

MAIN_FILE_PATH = '/workspace/Pose_3DCNN_PyTorch/project/main.py'

def get_parameters():
    '''
    The parameters for the model training, can be called out via the --h menu
    '''
    parser = ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--model', type=str, default='x3d_l', choices=['resnet', 'x3d_l'])
    parser.add_argument('--model_depth', type=int, default=50, choices=[50, 101, 152], help='the depth of used model')

    # Training setting
    parser.add_argument('--gpu_num', type=int, default=0, choices=[0, 1], help='the gpu number whicht to train')
    parser.add_argument('--part', type=str, default='all', choices=['all', 'body', 'head', 'upper', 'lower'], help='which part to used.')
    parser.add_argument('--fuse_flag', type=str, default='conv', choices=['sum', 'max', 'concat', 'conv'], help='how to fuse the different feature when part is all.')

    # Transfor_learning
    parser.add_argument('--transfor_learning', action='store_true', help='if use the transformer learning')

    # pre process flag
    parser.add_argument('--pre_process_flag', action='store_true', help='if use the pre process video, which detection.')

    # Path
    parser.add_argument('--log_path', type=str, default='./logs', help='the lightning logs saved path')

    return parser.parse_known_args()


if __name__ == '__main__':

    config, unkonwn = get_parameters()

    transfor_learning = config.transfor_learning
    pre_process_flag = config.pre_process_flag
    model = config.model
    part = config.part

    if part != 'all':
        img_size = 312
    else:
        img_size = 224

    symbol = '_'

    for length in VIDEO_LENGTH:
        for frames in VIDEO_FRAME:

            data = str(time.localtime().tm_mon) + str(time.localtime().tm_mday)
            
            if transfor_learning:
                
                if not pre_process_flag:

                    version = symbol.join([data, length, frames, part, 'not_pre_process'])
                    log_path = '/workspace/Pose_3DCNN_PyTorch/logs/' + symbol.join([version, model]) + '.log'

                    with open(log_path, 'w') as f:

                        # start one train.
                        subprocess.run(['python', MAIN_FILE_PATH,
                                        '--version', version,
                                        '--model', model,
                                        '--clip_duration', length,
                                        '--fuse_flag', str(config.fuse_flag),
                                        '--uniform_temporal_subsample_num', frames,
                                        '--part', part,
                                        '--gpu_num', str(config.gpu_num),
                                        # '--pre_process_flag',
                                        '--transfor_learning',
                                        '--img_size', str(img_size),
                                        ], stdout=f, stderr=f)

                else:

                    version = symbol.join([data, length, frames, part])
                    log_path = '/workspace/Pose_3DCNN_PyTorch/logs/' + symbol.join([version, model]) + '.log'

                    with open(log_path, 'w') as f:

                        # start one train.
                        subprocess.run(['python', MAIN_FILE_PATH,
                                        '--version', version,
                                        '--model', model,
                                        '--clip_duration', length,
                                        '--uniform_temporal_subsample_num', frames,
                                        '--part', part,
                                        '--gpu_num', str(config.gpu_num),
                                        '--fuse_flag', str(config.fuse_flag),
                                        '--pre_process_flag',
                                        '--transfor_learning',
                                        '--img_size', str(img_size),
                                        ], stdout=f, stderr=f)

            else:

                version = symbol.join([data, length, frames, part, 'not_transfor_learning'])
                log_path = '/workspace/Pose_3DCNN_PyTorch/logs/' + symbol.join([version, model]) + '.log'

                with open(log_path, 'w') as f:

                    # start one train.
                    subprocess.run(['python', MAIN_FILE_PATH,
                                    '--version', version,
                                    '--model', model,
                                    '--clip_duration', length,
                                    '--uniform_temporal_subsample_num', frames,
                                    '--fuse_flag', str(config.fuse_flag),
                                    '--part', part,
                                    '--gpu_num', str(config.gpu_num),
                                    '--pre_process_flag',
                                    '--max_epochs', '100'
                                    # '--transfor_learning',
                                    '--img_size', str(img_size),
                                    ], stdout=f, stderr=f)

            print('finish %s' % log_path)
