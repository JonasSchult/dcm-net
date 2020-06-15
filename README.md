# DualConvMesh-Net: Joint Geodesic and Euclidean Convolutions on 3D Meshes
Created by [Jonas Schult*](https://www.vision.rwth-aachen.de/person/schult), [Francis Engelmann*](https://www.vision.rwth-aachen.de/person/14/), [Theodora Kontogianni](https://www.vision.rwth-aachen.de/person/15/) and [Bastian Leibe](https://www.vision.rwth-aachen.de/person/1/) from RWTH Aachen University.

![prediction example](doc/teaser.png)

## Introduction
This work is based on our paper 
[DualConvMesh-Net: Joint Geodesic and Euclidean Convolutions on 3D Meshes](https://arxiv.org/abs/2004.01002),
which appeared at the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2020. 

You can also check our [project page](https://visualcomputinginstitute.github.io/dcm-net/) for further details.

We propose DualConvMesh-Nets (DCM-Net) a family of deep hierarchical convolutional networks over 3D geometric data that combines two types of convolutions. The first type, *geodesic convolutions*, defines the kernel weights over mesh surfaces or graphs. That is, the convolutional kernel weights are mapped to the local surface of a given mesh. The second type, *Euclidean convolutions*, is independent of any underlying mesh structure. The convolutional kernel is applied on a neighborhood obtained from a local affinity representation based on the Euclidean distance between 3D points. Intuitively, geodesic convolutions can easily separate objects that are spatially close but have disconnected surfaces, while Euclidean convolutions can represent interactions between nearby objects better, as they are oblivious to object surfaces. To realize a multi-resolution architecture, we borrow well-established mesh simplification methods from the geometry processing domain and adapt them to define mesh-preserving pooling and unpooling operations. We experimentally show that combining both types of convolutions in our architecture leads to significant performance gains for 3D semantic segmentation, and we report competitive results on three scene segmentation benchmarks.

*In this repository, we release code for training and testing DualConvMesh-Nets on arbitrary datasets.*

## Citation
If you find our work useful in your research, please consider citing us:

    @inproceedings{Schult20CVPR,
      author    = {Jonas Schult* and
                   Francis Engelmann* and
                   Theodora Kontogianni and
                   Bastian Leibe},
      title     = {{DualConvMesh-Net: Joint Geodesic and Euclidean Convolutions on 3D Meshes}},
      booktitle = {{IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}},
      year      = {2020}
    }


## Installation
Our code requires **CUDA 10.0** for running correctly. Please make sure that your `$PATH`, `$CPATH` and `$LD_LIBRARBY_PATH` environment variables point to the right CUDA version.

    conda deactivate
    conda create -y -n dualmesh python=3.7
    conda activate dualmesh

    conda install -y -c open3d-admin open3d=0.6.0.0
    conda install -y pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch
    conda install -y -c conda-forge tensorboardx=1.7
    conda install -y -c conda-forge tqdm=4.31.1
    conda install -y -c omnia termcolor=1.1.0

    # Execute pip installs one after each other
    pip install --no-cache-dir torch-scatter==1.3.1
    pip install --no-cache-dir torch_cluster==1.4.3
    pip install --no-cache-dir torch_sparse==0.4.0
    pip install "pillow<7" # torchvision bug

### Batching of hierarchical meshes (PyTorch Geometric Fork)
We created a fork of [PyTorch Geometric](https://github.com/JonasSchult/pytorch_geometric_fork) in order to support hierarchical mesh structures interlinked with pooling trace maps.

    git clone https://github.com/JonasSchult/pytorch_geometric_fork.git
    cd pytorch-geometric-fork
    pip install --no-cache-dir . 

### Mesh Simplification Preprocessing (VCGlib)
We adapted [VCGlib](https://github.com/JonasSchult/vcglib) to generate pooling trace maps for vertex clustering and quadric error metrics.

    git clone https://github.com/JonasSchult/vcglib.git

    # QUADRIC ERROR METRICS
    cd vcglib/apps/tridecimator/
    qmake
    make

    # VERTEX CLUSTERING
    cd ../sample/trimesh_clustering
    qmake
    make

Add `vcglib/apps/tridecimator` and `vcglib/apps/sample/trimesh_clustering` to your environment path variable!

## Preparation

### Prepare the dataset
Please refer to https://github.com/ScanNet/ScanNet and https://github.com/niessner/Matterport to get access to the ScanNet and Matterport dataset. Our method relies on the .ply as well as the .labels.ply files.
We train on crops and we evaluate on full rooms.
After inserting the paths to the dataset and deciding on the parameters, execute the scripts in `utils/preprocess/scripts/{scannet, matterport}/rooms` and *subsequently* in `utils/preprocess/scripts/scannet, matterport}/crops` to generate mesh hierarchies on rooms and crop areas for training.
Please note that the scripts are developed for a SLURM batch system. If your lab does not use SLURM, please consider adapting the scripts for your purposes.
More information about the parameters are provided in the corresponding scripts in `utils/preprocess`.

### Symbolic Links pointing to the dataset
Create symlinks to the dataset such that our framework can find it.
For example:

    ln -s /path/to/scannet/rooms/ data/scannet/scannet_qem_rooms

Alternatively, you can also directly set the paths in the corresponding experiment files.

### Model Checkpoints
We provide the [model checkpoints](https://omnomnom.vision.rwth-aachen.de/data/dcm_net_checkpoints/dcm_net_checkpoints.zip) on our server.

## Training
An example training script is given in `example_scripts/train_scannet.sh` 

## Inference
An example inference script is given in `example_scripts/inference_scannet.sh` 

## Visualization
An example visualization script is given in `example_scripts/visualize_scannet.sh`.
We show qualitative results on the ScanNet validation set.
Please note that a symlink to the ScanNet mesh folder has to be in placed in `data/scannet/scans`.
The visualization tool is based on [open3D](http://www.open3d.org/) and handles the following key events:
* h = RGB
* j = prediction
* k = ground truth
* f = color-coded positive/negative predictions
* l = local lighting on/off
* s = smoothing mesh on/off
* b = back-face culling on/off
* d = save current meshes as .ply in `visualizations/` folder (useful, if you plan to make some decent rendering with Blender, later on :) )
* q = quit and show next room

Use your mouse to navigate in the mesh.

## ToDo's
- Preprocessing code for S3DIS data set

## Acknowledgements
This project is based on the [PyTorch-Template](https://github.com/victoresque/pytorch-template) by [@victoresque](https://github.com/victoresque).