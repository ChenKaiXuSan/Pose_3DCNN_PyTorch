<div align="center">    
 
# Pose Based Gait Posture Classification

</div>
 
## Description   
This project implements the task of classifying different human gait posture disease.

The current phase performs a dichotomous classification task for four different disorders. classification of ASD and non-ASD.

## Folder structure

``` bash
Walk_Video_PyTorch
|-- imgs
|   imgs for markdown.
|-- logs
|   output logs and saved model .ckpt file location.
`-- project
    |-- dataloader
    |   pytorch lightning data module based dataloader, to prepare the train/val/test dataloader for different stage.
    |-- models
    |   pytorch lightning modeul based model, where used for train and val.
    |-- prepare_video
    |   one stage prepare video code location
    |-- tests
    |   `-- logs
    |   test logs location and code
    `-- utils
        tools for the code.
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

## Imports
This project is setup as a package which means you can now easily import any file into any other file like so:
```python
from project.datasets.mnist import mnist
from project.lit_classifier_main import LitClassifier
from pytorch_lightning import Trainer

# model
model = LitClassifier()

# data
train, val, test = mnist()

# train
trainer = Trainer()
trainer.fit(model, train, val)

# test using the best model!
trainer.test(test_dataloaders=test)
```

### Citation   
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```   
