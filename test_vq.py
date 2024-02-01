'''
Description: Test code for ThicknessVAE Stage 1: VQ module
Author: XiaotaoWu
Date: 2023-08-31 16:09:10
LastEditTime: 2024-02-01 14:12:24
LastEditors: XiaotaoWu
'''

import os
import argparse

import numpy as np
import open3d as o3d
import torch
import torch.utils.data as Data

from models import PCN
from dataset import thuman
from visualization import plot_pcd_one_view
from metrics.metric import l1_cd, l2_cd, emd, f_score


def make_dir(dir_path):
    """
    Create a directory if it does not exist.

    Args:
        dir_path (str): The path of the directory to be created.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def export_ply(filename, points):
    """
    Export a point cloud to a PLY file.

    Args:
        filename (str): The path to the output PLY file.
        points (list): List of 3D points.

    Returns:
        None
    """
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pc, write_ascii=True)


def test_single_category(model, params, save=True):
    """
    Test the model on a single category.

    Args:
        model (torch.nn.Module): The model to be tested.
        params (Namespace): The parameters for testing.
        save (bool, optional): Whether to save the results. Defaults to True.

    Returns:
        float: The average L1 Chamfer distance.
        float: The average L2 Chamfer distance.
        float: The average F-score.
    """
    if save:
        cat_dir = os.path.join(params.result_dir, params.exp_name)
        image_dir = os.path.join(cat_dir, 'image')
        output_dir = os.path.join(cat_dir, 'output')
        make_dir(cat_dir)
        make_dir(image_dir)
        make_dir(output_dir)

    test_dataset = thuman('./data/THuman2.0_Release', 'test_novel' if params.novel else 'test')
    test_dataloader = Data.DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False)

    index = 1
    total_l1_cd, total_l2_cd, total_f_score = 0.0, 0.0, 0.0
    with torch.no_grad():
        for p,c in test_dataloader:
            p = p.to(params.device)
            c = c.to(params.device)
            vq, _, c_, perplexity = model(p)
            
            #_, c_ = model(p)
            total_l1_cd += l1_cd(c_, c).item()
            total_l2_cd += l2_cd(c_, c).item()
            
            for i in range(len(p)):
                input_pc = p[i].detach().cpu().numpy()
                output_pc = c_[i].detach().cpu().numpy()
                gt_pc = p[i].detach().cpu().numpy()
                total_f_score += f_score(output_pc, gt_pc)
                if save:
                    plot_pcd_one_view(os.path.join(image_dir, '{:03d}.png'.format(index)), [input_pc, output_pc, gt_pc], ['Input', 'Output', 'GT'], xlim=(-0.35, 0.35), ylim=(-0.35, 0.35), zlim=(-0.35, 0.35))
                    export_ply(os.path.join(output_dir, '{:03d}.ply'.format(index)), output_pc)
                index += 1
    
    avg_l1_cd = total_l1_cd / len(test_dataset)
    avg_l2_cd = total_l2_cd / len(test_dataset)
    avg_f_score = total_f_score / len(test_dataset)

    return avg_l1_cd, avg_l2_cd, avg_f_score


def test(params, save=False):
    """
    Function to test the PCN model.

    Args:
        params (object): Parameters object containing various settings.
        save (bool, optional): Flag indicating whether to save the results. Defaults to False.
    """
    if save:
        make_dir(os.path.join(params.result_dir, params.exp_name))

    print(params.exp_name)

    # load pretrained model
    #model = PCN(16384, 1024, 4).to(params.device)
    model = PCN(16384, 1024, 4, 0.99).to(params.device)
    #model.load_state_dict(torch.load(params.ckpt_path))
    checkpoint = torch.load(params.ckpt_path)
    model.load_state_dict({k.replace('module.',''):v for k,v in checkpoint.items()})
    model.eval()

    print('\033[33m{:20s}{:20s}{:20s}\033[0m'.format('L1_CD(1e-3)', 'L2_CD(1e-4)', 'FScore-0.01(%)'))
    print('\033[33m{:20s}{:20s}{:20s}\033[0m'.format('-----------', '-----------', '--------------'))
    
    avg_l1_cd, avg_l2_cd, avg_f_score = test_single_category(model, params, save)
    print('{:<20.4f}{:<20.4f}{:<20.4f}'.format(1e3 * avg_l1_cd, 1e4 * avg_l2_cd, 1e2 * avg_f_score))


def test_single_category_emd(model, params):
    """
    Calculate the average Earth Mover's Distance (EMD) for a single category in the test dataset.

    Args:
        model (torch.nn.Module): The model used for inference.
        params (Namespace): The parameters for the test.

    Returns:
        float: The average EMD for the single category.
    """
    test_dataset = thuman('data/THuman2.0_Release', 'test_novel' if params.novel else 'test')
    test_dataloader = Data.DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False)

    total_emd = 0.0
    with torch.no_grad():
        for p in test_dataloader:
            p = p.to(params.device)
            vq, coarse, fine, perplexity = model(p) #vq_loss, coarse, fine, perplexity
            total_emd += emd(p, fine).item()
        
    avg_emd = total_emd / len(test_dataset) / fine.shape[1]
    return avg_emd


def test_emd(params):
    """
    Test the Earth Mover's Distance (EMD) for a given model and parameters.

    Args:
        params (object): The parameters object containing the necessary information.

    Returns:
        None
    """
    print(params.exp_name)

    # load pretrained model
    model = PCN(16384, 1024, 4).to(params.device)
    #model.load_state_dict(torch.load(params.ckpt_path))
    checkpoint = torch.load(params.ckpt_path)
    model.load_state_dict({k.replace('module.',''):v for k,v in checkpoint.items()})
    model.eval()

    print('\033[33m{:20s}\033[0m'.format('EMD(1e-3)'))
    print('\033[33m{:20s}\033[0m'.format('---------'))

    
    avg_emd = test_single_category_emd(model, params)
    print('{:<20.4f}'.format(1e3 * avg_emd))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Point Cloud Completion Testing')
    parser.add_argument('--exp_name', type=str, help='Tag of experiment')
    parser.add_argument('--result_dir', type=str, default='results', help='Results directory')
    parser.add_argument('--ckpt_path', type=str, help='The path of pretrained model.')
    #parser.add_argument('--category', type=str, default='all', help='Category of point clouds')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for data loader')
    parser.add_argument('--num_workers', type=int, default=6, help='Num workers for data loader')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for testing')
    parser.add_argument('--save', type=bool, default=False, help='Saving test result')
    parser.add_argument('--novel', type=bool, default=False, help='unseen categories for testing')
    parser.add_argument('--emd', type=bool, default=False, help='Whether evaluate emd')
    params = parser.parse_args()

    if not params.emd:
        test(params, params.save)
    else:
        test_emd(params)
