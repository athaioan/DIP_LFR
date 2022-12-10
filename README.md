# Module 1 Assignment Learning Feature Representation WASP course 

## Unsupervised pre-training
Completing pre-training will save the network's weight in the "stored" folder (which will be created if does not already exist)

### DIM
Unsupervised pre-training on the Tiny ImageNet using the DIM framework (https://arxiv.org/abs/1808.06670)

```python pre_train_DIM.py```

### dAE
Unsupervised pre-training on the Tiny ImageNet using the denoising-AE

```python pre_train_dAE.py```



### Fine Tuning
Supervised training on the CIFAR10 

```python fine_tune_cifar10.py```

### Dataset format

The data structure should look like this. The cifar10 folder will create itself upon running any of the training scripts. We need to create the imagenet_tiny structure ourselves based the .bin file we were given.

        ├── data
        │   ├──imagenet_tiny
        │   │   ├── image_tensor.bin        
        │   ├──cifar10
        └── ...
