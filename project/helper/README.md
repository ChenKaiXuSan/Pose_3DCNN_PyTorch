
<div align="center">

# How to prepare a Spinal Disease Dataset

</div>

## Description  

Here we show how to pre-process the spinal disease dataset.
Techniques used include object detection, extrapolation of figure centre coordinates and the extent of cropping.

## Folder structure  

``` markdown
Walk_Video_PyTorch
-- project
    |-- prepare_video
    |   |-- batch_dection.py
            use detectron2 library to detecated the person centered video.
    |   |-- prepare_video.py
            main entrance for prepare video method.
    |   |-- README.md
            this document.
    |   one stage prepare video code location
```

## How to run

1. install dependencies
[Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) is needed for this method.

You can build Detectron2 from source:
``` bash 
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
# (add --user if you don't have permission)

# Or, to install it from a local clone:
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
```

2. navigate to any file and run it.

```bash
# module folder
cd Walk_Video_PyTorch/project/prepare_video

# run module 
python prepare_video > ./xxx.log 
```

### stage one 

In this stage, the interface provided by [detector2](https://detectron2.readthedocs.io/en/latest/index.html) is used for video pre-processing, in order to extract the area centered on the person and save a series of frames as video with a uniform FPS = 30.

The implementation of the prepare_video.py file,  

``` python  
usage: prepare_video.py [-h] [--img_size IMG_SIZE] [--num_workers NUM_WORKERS] [--data_path DATA_PATH] [--split_pad_data_path SPLIT_PAD_DATA_PATH] [--split_data_path SPLIT_DATA_PATH] [--pad_flag PAD_FLAG]

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
  --pad_flag PAD_FLAG   flag that pad or not

```

for example,  

``` python  
cd Walk_Video_PyTorch/project/prepare_video/

python prepare_video.py --img_size 512 --data_path [meta dataset path] --split_pad_data_path [split and pad dataset path] --split_data_path [split dataset path] > ./split_log.log &

```

⚠️ You need to replace the content in [ ] with your own actual path.

# Experimental setup

## Dataset

Due to the limitation of data, we divide 80% of the data into the training set and set the remaining 20% as the validation and test data sets.

The detail number of the split dataset are given in next table.

| the number of video | ASD | ASD_not |
| ------------------- | --- | ------- |
| train               | 923 | 815     |
| val                 | 123 | 96      |

The detail number of different disease and the number of every disease object are given in next talbe.

| disease | number of video | object for train | object for val |
| ------- | --------------- | ---------------- | -------------- |
| ASD     | 1046            | 48               | 6              |
| DHS     | 587             | 14               | 2              |
| HipOA   | 64              | 3                | null           |
| LCS     | 260             | 7                | 2              |

## About the lib  

stop building wheels 😄

### detectron2

[Detectron2](https://detectron2.readthedocs.io/en/latest/index.html) is Facebook AI Research's next generation library that provides state-of-the-art detection and segmentation algorithms. It is the successor of Detectron and maskrcnn-benchmark. It supports a number of computer vision research projects and production applications in Facebook.