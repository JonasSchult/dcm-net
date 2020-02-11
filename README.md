# Dual Convolution Mesh Network (DCM Net)
![prediction example](doc/teaser.png)

## Coming soon...
Please *stay tuned*; we are currently working hard to get the code out quickly.


## Requirements
All of our dependencies can be installed with conda or pip.
* Python 3.7
* Open3D
* PyTorch 1.1 Cuda 10.0
* TensorboardX (Tensorflow and Tensorboard are unfortunately also needed to install this)
* Our fork of PyTorch Geometric (with its accompanying libraries as torch_scatter, torch_cluster, torch_sparse)
* tqdm

Since we adapted PyTorch Geometric to enable graph level support, you need to install our fork as follows:
    
    cd pytorch_geometric
    python setup.py install

## Preprocessing
Please refer to https://github.com/ScanNet/ScanNet to get access to the ScanNet dataset. Our method relies on the .ply as well as the .labels.ply files.

### Start a new training:

    python train_wrapper.py \
    -c PATH_TO_EXPERIMENTS_FILE.json
    
### Resume a training:

    python train_wrapper.py \
    -c PATH_TO_EXPERIMENTS_FILE.json \
    -r PATH_TO_CHECKPOINT.pth

### Reproduce the scores of our paper:

    python run.py \
    -c experiments/EXPERIMENT_NAME.json \
    -r paper_checkpoints/EXPERIMENT_NAME.pth \
    -e