# DeepSFM

This is a PyTorch implementation of the ECCV2020 (Oral) paper 
"DeepSFM: Structure From Motion Via Deep Bundle Adjustment".

In this work, we design a physical driven architecture, namely DeepSFM, 
inspired by traditional Bundle Adjustment (BA), 
which consists of two cost volume based architectures for depth and
 pose estimation respectively, iteratively running to improve both. 
 The explicit constraints on both depth (structure) and pose (motion), when combined with the learning components, 
 bring the merit from both traditional BA and emerging deep 
 learning technology.
Our framework receives frames of a scene from different viewpoints, and produces
depth maps and camera poses for all frames. 

Please check the [paper](https://arxiv.org/abs/1912.09697) 
and the [project webpage](https://weixk2015.github.io/DeepSFM/) for more details.

If you have any question, please contact Xingkui Wei <xkwei19@fudan.edu.cn>.

#### Citation

If you use this code for any purpose, please consider citing:

```
@inProceedings{wei2020deepsfm,
  title={DeepSFM: Structure From Motion Via Deep Bundle Adjustment},
  author={Xingkui Wei and Yinda Zhang and Zhuwen Li and Yanwei Fu and Xiangyang Xue},
  booktitle={ECCV},
  year={2020}
}
```

## Requirements

Building and using requires the following libraries and programs

    Pytorch 0.4.0
    CUDA 9.0
    python 3.6.4
    scipy
    argparse
    tensorboardX
    progressbar2
    path.py
    transforms3d
    minieigen
    
The versions match the configuration we have tested on an ubuntu 16.04 system.

## Data Preparation 
Training data preparation requires the following libraries and programs

    opencv
    imageio
    joblib
    h5py
    lz4
    
1. Download DeMoN data (https://github.com/lmb-freiburg/demon)
2. Convert data

[Training data]
    
```
bash download_traindata.sh
python ./dataset/preparation/preparedata_train.py
```

[Test data]
    
```
bash download_testdata.sh
python ./dataset/preparation/preparedata_test.py
```

[DeMoN Initialization]

The network assume initial depth maps and camera poses are given. 
The initialization is not necessary to be accurate.
In this implementation the initialization is obtained
from "DeMoN: Depth and Motion Network".

Please refer DeMoN(https://github.com/lmb-freiburg/demon) 
for the details of initial pose and depth map generation.

The final input file structure is shown as follows:
```
dataset/train
│   train.txt
│   val.txt    
│
└───scene0
│   │   0000.jpg
│   │   0001.jpg
│   │    ...
│   │   0000.npy
│   │   0001.npy  
│   │    ...
│   │   0000_demon.npy
│   │   0001_demon.npy  
│   │    ...
│   │   cam.txt
│   │   poses.txt
│   │   demon_poses.txt
└───scene1
    ...

dataset/test
│   test.txt
│
└───test_scene0
│   │   0000.jpg
│   │   0001.jpg
│   │    ...
│   │   0000.npy
│   │   0001.npy  
│   │    ...
│   │   cam.txt
│   │   poses.txt
│   │   demon_poses.txt
└───test_scene1
    ...
```
    
## Train
The released code implements the depth map prediction subnet and 
the camera pose prediction subnet independently.

#### Depth 

The training process and the implementation
 of the depth subnet is similar to [DPSNet](https://github.com/sunghoonim/DPSNet). 
 The only difference is that the local geometric consistency constraints is 
 introduced by a additional initial depth maps warping.


```
python train.py ./dataset/train/ --mindepth 0.5 --nlabel 64 --pose_init demon --depth_init demon 
```
- pose_init: the prefix of pose txt file. The network will use **"%s_poses.txt" % args.pose_init** 
as the initial pose file. If not set, **poses.txt** will be adopted.

- depth_init: the prefix of depth npy file. The network will use **"0000_%s.npy" % args.depth_init** 
as the initial depth file of image **0000.jpg**. If not set, **0000.npy** will be adopted.


#### Pose
The architecture and implementation
 of the pose subnet is similar to depth subnet. 

```
python pose_train.py ./dataset/train/ --std_tr 0.27 --std_rot 0.12 --nlabel 10 --pose_init demon --depth_init demon 
``` 
- pose_init: the prefix of pose txt file. The network will use **"%s_poses.txt" % args.pose_init** 
as the initial pose file. If not set, **poses.txt** will be adopted.

- depth_init: the prefix of depth npy file. The network will use **"0000_%s.npy" % args.depth_init** 
as the initial depth file of image **0000.jpg**. If not set, **0000.npy** will be adopted.


## Test
#### Depth 
```
python test.py ./dataset/test/ --sequence-length 2  --pretrained-dps depth.pth.tar --pose_init demon --depth_init demon --save I0
```
- pose_init: the prefix of pose txt file. The network will use **"%s_poses.txt" % args.pose_init** 
as the initial pose file. If not set, **poses.txt** will be adopted.

- depth_init: the prefix of depth npy file. The network will use **"0000_%s.npy" % args.depth_init** 
as the initial depth file of image **0000.jpg**. If not set, **0000.npy** will be adopted.

- save: the prefix of saved depth npy file. The network will use **"0000_%s.npy" % args.save** 
as the save path of predicted depth file of image **0000.jpg**. If not set or the file already exists, 
the predicted depth will not be saved.

#### Pose
```
python pose_test.py ./dataset/test/ --sequence-length 2  --pretrained-dps pose.pth.tar --pose_init demon --depth_init demon --save I0
```
- pose_init: the prefix of pose txt file. The network will use **"%s_poses.txt" % args.pose_init** 
as the initial pose file. If not set, **poses.txt** will be adopted.

- depth_init: the prefix of depth npy file. The network will use **"0000_%s.npy" % args.depth_init** 
as the initial depth file of image **0000.jpg**. If not set, **0000.npy** will be adopted.

- save: the prefix of saved pose txt file. The network will use **"%s_poses.txt" % args.save** 
as the save path of predicted pose file. If not set or the file already exists, 
the predicted pose will not be saved.

To run iteratively, change the value of pose_init, depth_init and save.
```
python test.py ./dataset/train/ --mindepth 0.5 --nlabel 64 --pose_init demon --depth_init demon --save I0
python pose_test.py ./dataset/train/ --std_tr 0.27 --std_rot 0.12 --nlabel 10 --pose_init demon --depth_init demon --save I0
python test.py ./dataset/train/ --mindepth 0.5 --nlabel 64 --pose_init I0 --depth_init I0 --save I1
python pose_test.py ./dataset/train/ --std_tr 0.27 --std_rot 0.12 --nlabel 10 --pose_init I0 --depth_init I0 --save I1
python test.py ./dataset/train/ --mindepth 0.5 --nlabel 64 --pose_init I1 --depth_init I1 --save I2
python pose_test.py ./dataset/train/ --std_tr 0.27 --std_rot 0.12 --nlabel 10 --pose_init I1 --depth_init I1 --save I2
...
``` 
The pretrained models can be downloaded at [Google drive](https://drive.google.com/drive/folders/1GGzFKaNV39M9Z8XlqMIPuoCsvpA-wgVY?usp=sharing).
Due to the stochastic nature during training. The performance of the pre-trained model may be slightly different from it in the paper.

## Acknowledgments

the implementation codes borrows heavily from [DPSNet](https://github.com/sunghoonim/DPSNet). Thanks for the sharing.
