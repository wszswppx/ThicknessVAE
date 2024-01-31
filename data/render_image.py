import pyvista as pv
import os
from tqdm import tqdm
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def render_image(data_path, mesh, texture):
    p = pv.Plotter(off_screen=True)
    p.add_mesh(mesh,texture=texture)
    p.camera.position=(0,0,2)
    p.camera.focal_point=(0,0,0)
    p.camera.up=(0,1,0)
    p.window_size=(512,512)
    p.enable_parallel_projection()
    name = "rendered_image.png"
    rendered_file = os.path.join(data_path, name)                  # ./THuman2.0_Release/0000/rendered_image.png
    p.screenshot(filename=rendered_file, transparent_background=False, return_img=False)
    
items = 526
current = 277

progress_bar = tqdm(total=items-current, desc='Rendering')
dataset_path = os.path.dirname('./THuman2.0_Release/')

for i in range(current,items):
    # 将i填充为4位，不足部分用0填充:
    file_name = str(i).zfill(4)                                 # 0000
    data_path = os.path.join(dataset_path, file_name)           # ./THuman2.0_Release/0000
    mesh_path = os.path.join(data_path, '%s.obj'%(file_name))   # ./THuman2.0_Release/0000/0000.obj
    material = os.path.join(data_path, 'material0.jpeg')

    mesh = pv.read(mesh_path)
    texture = pv.read_texture(material)
    
    render_image(data_path, mesh, texture,)
    
    del mesh
    
    progress_bar.update(1)

progress_bar.close()