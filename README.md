# [Learning Guided Convolutional Network for Depth Completion](https://arxiv.org/pdf/1908.01238).


## Introduction

This is the pytorch implementation of our paper.

## Dependency
```
PyTorch 1.4
PyTorch-Encoding v1.4.0
```

## Setup
Compile the C++ and CUDA code:
```
cd exts
python setup.py install
```

## Dataset
Please download KITTI [depth completion](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion)
dataset.
The structure of data directory:
```
└── datas
    └── kitti
        ├── data_depth_annotated
        │   ├── train
        │   └── val
        ├── data_depth_velodyne
        │   ├── train
        │   └── val
        ├── raw
        │   ├── 2011_09_26
        │   ├── 2011_09_28
        │   ├── 2011_09_29
        │   ├── 2011_09_30
        │   └── 2011_10_03
        ├── test_depth_completion_anonymous
        │   ├── image
        │   ├── intrinsics
        │   └── velodyne_raw
        └── val_selection_cropped
            ├── groundtruth_depth
            ├── image
            ├── intrinsics
            └── velodyne_raw
```

## Configs
The config of different settings:
- GN.yaml
- GNS.yaml

*Compared to **GN**, **GNS** uses fewer parameters to generate the guided kernels, 
but achieves slightly better results.*


## Trained Models
You can directly download the trained model and put it in *checkpoints*:
- [GN](https://drive.google.com/file/d/1-sa2pnMMjSv2dV2bRwuyLxPr1onmVykj/view?usp=sharing)
- [GNS](https://drive.google.com/file/d/16tVrZQEDBucgjZmTjZl4iFkklkjfeDcs/view?usp=sharing)

## Train 
You can also train by yourself:
```
python train.py
```
*Pay attention to the settings in the config file (e.g. gpu id).*

## Test
With the trained model, 
you can test and save depth images.
```
python test.py
```

## Citation
If you find this work useful in your research, please consider citing:
```
@article{guidenet,
  title={Learning guided convolutional network for depth completion},
  author={Tang, Jie and Tian, Fei-Peng and Feng, Wei and Li, Jian and Tan, Ping},
  journal={IEEE Transactions on Image Processing},
  volume={30},
  pages={1116--1129},
  year={2020},
  publisher={IEEE}
}
```