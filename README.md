# ThicknessVAE

This project is for paper [ThicknessVAE: Learning a Side Prior for Clothed Human Body Reconstruction](https://openreview.net/pdf?id=FyZGjyeghg).

![Overview of ThicknessVAE](images/ThicknessVAE_overview.png "Overview of ThicknessVAE")

## Abstract

Sandwich-like structure has demonstrated excellent performance in reconstructing clothed humans. However, these approaches often generate unrealistic side geometries due to inadequate handling of the side region. In this paper, we address this issue by incorporating the side geometry of the clothed human as a prior. To this end, we propose ThicknessVAE, our two-stage method. Our work makes two key contributions: (1) We leverage a VQVAE architecture to learn a prototype for the side part of clothed human to extract common and accurate geometry features. (2) We use the this prototype as prior to transform the geometric features into a thickness map associated with clothed human images, enabling a refined normal integration result for sandwich-like reconstruction methods. By seamlessly integrating our model into the sandwich-like reconstruction pipeline, we are able to achieve highly realistic side views. Qualitative and quantitative experiments demonstrate that our approach outperforms SOTA methods in terms of side view realism for loose clothing and in-the-wild images.

## Install

You can follow this command to create a conda virtual environment:

```bash
conda env create -f env.yml
```

Other packages can be installed by:

```bash
pip install -r requirements.txt
```

This code has been tested on Linux Ubuntu 18.08, cuda 10.2 (also works on cuda 10.1), python 3.7.9, pytorch==1.7.0, torchvision==0.8.0.

## Dataset

You are supposed to have THuman 2.0 dataset for both training and testing.

To train ThicknessVAE, the dataset files should be placed with the following structure:

```
data
├──THuman2.0_Release/
│	├─0000/
│	│   ├─0000_pt_side.ply
│	│   ├─0000.obj
│	│   ├─depth_F.png
│	│   ├─depth_B_.png
│	│   ├─material0.jpeg
│	│   ├─material0.mtl
│	│   └─thickness_2.npy
│	├─0001/
│	├─0002/
│	├─...
│	├─0525/
│	└─items.txt
├─...
```

To illustrate the meaning of each document:

| file             | statement                                    |
| ---------------- | -------------------------------------------- |
| 0000_pt_side.ply | The side part point cloud of each .obj file. |
| 0000.obj         | The original .obj file in THuman 2.0.        |
| depth_F.png      | Depth map from front view.                   |
| depth_B_.png     | Depth map from back view.                    |
| material0.jpeg   | The original material file in THuman 2.0.   |
| material0.mtl    | The original material file in THuman 2.0.   |
| thickness_2.npy  | Ground truth of thickness map for each item. |

## VQ module - Stage (a)

The best pretrained checkpoint of stage (a) is [best_l1_cd.pth](log/thuman_pu_1/checkpoints/best_l1_cd.pth)

### Training

In order to train the VQ module, please use the script:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --use_env train_vq.py --exp_name thuman --lr 0.0001 --epochs 1000 --batch_size 8
```

The training process will cost about 10 hours.

You can use TensorBoard to visualize the training process:

```bash
tensorboard --logdir=log/thuman
```

### Testing

Use this script for testing VQ module:

```bash
python test.py --exp_name thuman --ckpt_path log/thuman/checkpoints/best_l1_cd.pth --batch_size 32 --num_workers 8
```

Testing will cost about 4 hours.

## ThicknessVAE - Stage (b)

The best pretrained checkpoint of stage (b) is [thickness_1218.pth](log/thickness_1218/checkpoints/thickness_1218.pth)

### Training

If you have the VQ checkpoint, Use the script below to train ThicknessVAE:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --use_env train_thickness.py --exp_name [expext_name] --lr 0.0001 --epochs 1200 --batch_size 8
```

Training process for stage (b) will cost about 24 hours.

After getting a ThicknessVAE checkpoint, you can assert this module into ECON.

## Citation

* [PCN: Point Completion Network](https://arxiv.org/pdf/1808.00671.pdf)
* [PCN&#39;s official Tensorflow implementation](https://github.com/wentaoyuan/pcn)
