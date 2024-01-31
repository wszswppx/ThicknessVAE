import open3d as o3d
import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def o3d_visualize_pc(pc):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pc)
    o3d.visualization.draw_geometries([point_cloud])


def plot_pcd_one_view(filename, pcds, titles, suptitle='', sizes=None, cmap='Reds', zdir='y',
                         xlim=(-0.5, 0.5), ylim=(-0.5, 0.5), zlim=(-0.5, 0.5)):
    if sizes is None:
        sizes = [0.5 for i in range(len(pcds))]
    fig = plt.figure(figsize=(len(pcds) * 3 * 1.4, 3 * 1.4))
    elev = 30  # 水平倾斜
    azim = -45  # 旋转
    for j, (pcd, size) in enumerate(zip(pcds, sizes)):
        color = pcd[:, 0]
        ax = fig.add_subplot(1, len(pcds), j + 1, projection='3d')
        ax.view_init(elev, azim)
        ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir=zdir, c=color, s=size, cmap=cmap, vmin=-1.0, vmax=0.5)
        ax.set_title(titles[j])
        ax.set_axis_off()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)
    plt.suptitle(suptitle)
    fig.savefig(filename)
    plt.close(fig)

def view_depth(input1, input2, pred, GT, filename):
    
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(input1)
    plt.title('Input 1')

    plt.subplot(2, 2, 2)
    plt.imshow(input2)
    plt.title('Input 2')
    
    plt.subplot(2, 2, 3)
    plt.imshow(pred)
    plt.title('Prediction')
    
    plt.subplot(2, 2, 4)
    plt.imshow(GT)
    plt.title('Ground Truth')

    # 调整子图之间的间隔
    plt.tight_layout()

    # 保存图片
    plt.savefig(filename)
    