'''
main entrance of split video.
split one person to different part,
head part
upper part
lower part 
masked raw img
'''

# %%
import os
import sys

# there should exchange the path with yourself path.
sys.path.append('/workspace/Pose_3DCNN_PyTorch/project')

import warnings
warnings.filterwarnings("ignore")

from utils.utils import make_folder, count_File_Number
from argparse import ArgumentParser
from torchvision.io import read_video, read_video_timestamps, write_video
import torch
from split_part import BatchSplit

FOLD = [str(i) for i in range(5)]
PART = ['body', 'head', 'upper', 'lower']

# %%

def make_different_part_folder(path: str):
    '''
    make the split pad video folder, for /train/ASD; /train/ASD_not;, /val/ASD; /val/ASD_not

    Args:
        split_pad_data_path (str): '/workspace/data/split_pad_dataset'
    '''
    for f in FOLD:
        fold_num = 'fold' + f

        for part in PART:
            
            for flag in ('train', 'val'):

                data_path_flag = os.path.join(path, fold_num, part, flag)

                for diease_flag in ('ASD', 'ASD_not'):

                    final_split_pad_data_path = os.path.join(data_path_flag, diease_flag)

                    make_folder(final_split_pad_data_path)

# %%


def get_Path_List(data_path: str):
    '''
    get the prefix data path list, like "/workspace/data/dataset/train/ASD", len = 4

    Args:
        data_path (str): meta data path

    Returns:
        list: list of the prefix data path list, len=4
    '''

    diease_path_list = []

    for flag in ('train', 'val'):

        data_path_flag = os.path.join(data_path, flag)

        for diease_flag in (os.listdir(data_path_flag)):  # ASD, ASD_not

            data_path_diease_flag = os.path.join(data_path_flag, diease_flag)

            diease_path_list.append(data_path_diease_flag)

    return diease_path_list

# %%


def get_final_video_path_Dict(prefix_path_list: list):
    '''
    get the all final video full path, store in dict. 
    the keys mean the unique data with the disease.
    the values mean in on data with disease, how many file they are, store in List.

    Args:
        prefix_path_list (list): the prefix path list, like /train/ASD; /train/ASD_not; /val/ASD; /val/ASD_not.

    Returns:
        Dict: the all final video full path, store in dict.
    '''

    final_video_path_Dict = {}

    for data_path_flag_diease in prefix_path_list:

        compare_video_name_list = []

        # all video path
        video_path_list = os.listdir(data_path_flag_diease)

        # start compare video path list
        for compare_video_name in video_path_list:

            compare_video_name_list.append(compare_video_name[:15])

        compare_video_file_name = list(set(compare_video_name_list))

        for name in compare_video_file_name:

            video_same_path_list = []

            for video_path_name in video_path_list:

                now_video_path_name = video_path_name[:15]

                if now_video_path_name in name:

                    # store the full path of unique data with diease in a list.
                    video_same_path_list.append(os.path.join(data_path_flag_diease, video_path_name))

            video_same_path_list.sort()

            final_video_path_Dict[os.path.join(data_path_flag_diease, name[:15])] = video_same_path_list

    return final_video_path_Dict

# %%

def read_and_write_video_from_List(path_list: list, video_save_path: str, splitter):

    
    for path in path_list:

        train_flag, disease_flag, video_name = path.split('/')[-3:]
        different_part_list = []

        for part in PART:
            different_part_list.append(os.path.join(video_save_path, part, train_flag, disease_flag, video_name))

        video_frame, audio_frame, video_info = read_video(path)  # (t, h, w, c)

        print("currect file:", path)

        # 之前写的代码，这里带不上lable，怎么办
        head, upper, lower, body = splitter.handle_batch_video(video_frame)  # c, t, h, w

        print(video_info['video_fps'], path, video_frame.shape)

        if head is not None:
            # for test, save one batch
            for p, v in zip(different_part_list, [body, head, upper, lower]):
                
                write_video(p, v.permute(1, 2, 3, 0), fps=30, video_codec='h264')

            print('clip %i frames!, %i frams lost!' %
                (head.size()[1], video_frame.size()[0] - head.size()[1]))
            print('----------------------------------------------------------------------')
        else:
            print('video have empty list, drop it!')
            print('----------------------------------------------------------------------')

    return video_frame, path, video_info['video_fps']


# %%

def get_parameters():

    parser = ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--img_size', type=int, default=512)

    # Training setting
    parser.add_argument('--num_workers', type=int, default=2, help='dataloader for load video')

    # Path
    parser.add_argument('--data_path', type=str, default="/workspace/data/dataset/", help='meta dataset path')
    parser.add_argument('--split_pad_data_path', type=str, default="/workspace/data/split_pad_dataset_512",
                        help="split and pad dataset with detection method.")
    parser.add_argument('--split_data_path', type=str, default="/workspace/data/split_dataset",
                        help="split dataset with detection method.")

    return parser.parse_known_args()


# %%
if __name__ == '__main__':

    # for test in jupyter
    parames, unkonwn = get_parameters()

    DATA_PATH = parames.data_path
    SPLIT_PAD_DATA_PATH = parames.split_pad_data_path

    IMG_SIZE = parames.img_size

    # make folder with img size
    MAKED_DATA_PATH = "/workspace/data/Pose_dataset_" + str(IMG_SIZE)
    make_different_part_folder(MAKED_DATA_PATH)

    # instance pose class.
    split = BatchSplit(img_size=IMG_SIZE)

    # get different fold path
    for f in FOLD:
        
        fold_num = 'fold' + f
        VIDEO_SAVE_PATH = os.path.join(MAKED_DATA_PATH, fold_num) 

        prefix_path_list = get_Path_List(os.path.join(SPLIT_PAD_DATA_PATH, fold_num))  # four folder, train/ASD, train/ASD_not, val/ASD, val/ASD_not

        final_video_path_Dict = get_final_video_path_Dict(prefix_path_list)

        for key in final_video_path_Dict.keys():

            now_video_path_list = final_video_path_Dict[key]
            now_video_path_list.sort()

            video_frame, now_video_path_list, video_info = read_and_write_video_from_List(
                now_video_path_list, video_save_path=VIDEO_SAVE_PATH, splitter=split)

    print('Finish split and pad video!')

    count_File_Number(SPLIT_PAD_DATA_PATH)
