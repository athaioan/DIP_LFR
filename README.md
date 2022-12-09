# Module 1 Assignment Learning Feature Representation WASP course 

### Unsupervised pre-training

## DIM

```python pre_train_DIM.py```

### Dataset format

The data structure should look like this. The cifar10 folder will create itself upon running any of the training scripts. We need to create the imagenet_tiny structure ourselves based the .bin file we were given.

        ├── data
        │   ├──imagenet_tiny
        │   │   ├── image_tensor.bin        
        │   ├──cifar10
        └── ...
