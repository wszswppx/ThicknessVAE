import open3d as o3d

def mesh_to_point_cloud(mesh_file, num_points):
    # 加载PLY格式的网格文件
    mesh = o3d.io.read_triangle_mesh(mesh_file)

    # 将网格转换为点云
    point_cloud = mesh.sample_points_poisson_disk(num_points)

    return point_cloud

def save_point_cloud(point_cloud, output_file):
    # 保存点云为PLY格式
    o3d.io.write_point_cloud(output_file, point_cloud)

item_list = range(0,526)
for i in item_list:
    # 将i填充为4位，不足部分用0填充:
    file_name = str(i).zfill(4)
    # 输入文件路径
    input_file = "./data/THuman2.0_Release/%s/%s_side.ply" % (file_name, file_name)
    # 输出文件路径
    output_file = "./data/THuman2.0_Release/%s/%s_pt_side.ply" % (file_name, file_name)
    # 欲得到的点的数量
    num_points = 16384

    # 将网格转换为点云
    point_cloud = mesh_to_point_cloud(input_file, num_points)

    # 保存点云为PLY格式
    save_point_cloud(point_cloud, output_file)

    print("Point cloud saved as", output_file)