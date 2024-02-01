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