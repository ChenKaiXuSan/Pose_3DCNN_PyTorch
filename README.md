<div align="center">    
 
# Pose Based Gait Posture Classification

</div>
 
## Description   
📓 This project made with the Pytorch, Pytorch Lightning, Pytorch Video library.

This project implements the task of classifying different human gait posture, based on four different pose.

classification of ASD and non-ASD.

The whole procedure is divided into these steps:
1. use meidapipe to detect the pose, get the joint keypoint.
2. use the estimated joint keypoint location, to split the whole body into three different part, such as head, upper, lower.
3. feed different part video into 3D CNN network, get the final predict results.
4. try to fusion different part features, to get a better accuracy.

## Folder structure

``` bash 
Pose based PyTorch
|-- logs
|   output logs and saved model .ckpt file location.
|-- project
|   |-- attribution
|   | data visualization and draw curve for predict results.
|   |-- dataloader
|   |   pytorch lightning data module based dataloader, to prepare the train/val/test dataloader for different stage.
|   |-- models
|   |   pytorch lightning modeul based model, where used for train and val.
|   |-- prepare_video
|   |   use joint keypoint to split the raw video to different part.
|   `-- utils
|       some tools for the code.
`-- tests
    test logs file and code.

```

# How to run

First, install dependencies   
```bash
# clone project   
git clone https://github.com/YourGithubName/deep-learning-project-template

# install project   
cd deep-learning-project-template 
pip install -e .   
pip install -r requirements.txt
 ```   
 Next, navigate to any file and run it.   
 ```bash
# module folder
cd project

# run module (example: mnist as your main contribution)   
python lit_classifier_main.py    
```

## Experimental setup

## Docker  

We recommend using docker to build the training environment.

1. pull the official docker image, where release in the [pytorchlightning/pytorch_lightning](https://hub.docker.com/r/pytorchlightning/pytorch_lightning)

``` bash  
docker pull pytorchlightning/pytorch_lightning
```

2. create container.

``` bach  
docker run -itd -v $(pwd)/path:/path --gpus all --name container_name --shm-size 32g --ipc="host" <images:latest> bash 

```

3. enter the container and run the code.

``` bash  
docker exec -it container_name bash
```


## About the libary
stop building wheels. 😄

### PyTorch Lightning  

[PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/) is the deep learning framework for professional AI researchers and machine learning engineers who need maximal flexibility without sacrificing performance at scale. Lightning evolves with you as your projects go from idea to paper/production.

### PyTorch Video  

[link](https://pytorchvideo.org/)
A deep learning library for video understanding research.

### MediaPipe

[MediaPipe](https://mediapipe.dev/) offers open source cross-platform, customizable ML solutions for live and streaming media.

### Torch Metrics

[TorchMetrics](https://torchmetrics.readthedocs.io/en/latest/) is a collection of 80+ PyTorch metrics implementations and an easy-to-use API to create custom metrics.