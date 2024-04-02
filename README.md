# Image Generation (Deep Convolutional GAN) project

 This is a repository for Image Generation project based on one homework of DL-2 course (HSE). The task is to generate images, in this case a dataset containing pictures of cat faces was used.
## Repository structure

`src` - directory included all project files.
* `dataset` - functions and class for parsing protocols from ASVSpoof2019 (Dataset), audio preprocessing (we need 4 sec during both train and evaluation part).
* `model` - DCGAN model architecture (from [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434v2)).
* `trainer` - train loop, logging in W&B.
* `utils` - crucial functions (collator, wandb writer, utils).

`train_config` - file with training configuration.
## Installation guide

As usual, clone repository, change directory and install requirements:

```shell
!git clone https://github.com/KemmerEdition/base_folder.git
!cd /content/base_folder
!pip install -r requirements.txt
```
## Train
Train model with command below (you should use Kaggle for training because there is a dataset needed in this task. 

You need to add the [dataset](https://www.kaggle.com/datasets/spandan2/cats-faces-64x64-for-generative-models) to your workspace, only then run the script). If you want to log something in W&B - set `-w True` else `-w False`.
   ```shell
   !python train.py -w True/False
   ``` 
Enjoy the results!

![cats_gen2.gif](https://github.com/KemmerEdition/DcganCat/blob/master/src/cats_gen2.gif)
