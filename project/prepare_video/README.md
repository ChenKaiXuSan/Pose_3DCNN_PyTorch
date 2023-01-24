
<div align="center">

# How to split different part from whole body video

</div>

## Description  

Here we show how to split whole body part into different part, head, upper, lower.
We use MediaPipe to estimate the joint keypoint, and then use them to split different part.

Techniques used include object detection, pose estimation.

## Folder structure  

``` markdown
Walk_Video_PyTorch
-- project
    |-- prepare_video:
    |   |-- move_video.ipynb
    |   |   move splitted part video into 5 fold Cross Validation.
    |   |-- split_part.py
    |   |   batch split method, defined how to split one batch video into different part.
    |   |-- split_video.py
    |   |   main entrance of split video. 
    |   |-- README.md
            this document.
```

## How to run

1. install dependencies

[MediaPipe in Python](https://google.github.io/mediapipe/getting_started/python) is needed for this method.

You can install from pip source:

``` bash 
$ pip install mediapipe

```

2. from whole body to split into different part

```bash
# module folder
cd /workspace/Pose_3DCNN_PyTorch/project/prepare_video

# run script 
python split_video.py > ./xxx.log 
```

3. take splitted video into 5 fold Cross Validation
``` batch 
move_video.ipynb
``` 

## Implementation 

The implementation of the split_video.py file,  

``` python  
usage: split_video.py [-h] [--img_size IMG_SIZE] [--num_workers NUM_WORKERS]
                      [--data_path DATA_PATH]
                      [--split_pad_data_path SPLIT_PAD_DATA_PATH]
                      [--split_data_path SPLIT_DATA_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --img_size IMG_SIZE
  --num_workers NUM_WORKERS
                        dataloader for load video
  --data_path DATA_PATH
                        meta dataset path
  --split_pad_data_path SPLIT_PAD_DATA_PATH
                        split and pad dataset with detection method.
  --split_data_path SPLIT_DATA_PATH
                        split dataset with detection method.

```

for example,  

``` bash  
cd /workspace/Pose_3DCNN_PyTorch/project/prepare_video

python split_video.py --img_size 512 --data_path [meta dataset path] --split_pad_data_path [split and pad dataset path] --split_data_path [split dataset path] > ./split_log.log &

```

⚠️ You need to replace the content in [ ] with your own actual path.

## About the lib  

stop building wheels 😄

### MediaPipe

[MediaPipe](https://mediapipe.dev/) offers open source cross-platform, customizable ML solutions for live and streaming media.

> There also have some problem with MediaPipe.
I found that I can't use GPU to do inference. Maybe compile from source can resolve this problem, but a little complex for me. So I decide first split the whole data to different part, and store them. Then load them when training.