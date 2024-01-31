import argparse
import os
import datetime
import random

import torch
import torch.nn as nn
import torch.optim as Optim
import torch.nn.functional as F
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter

from dataset.thickness_data import thickness_loader
from models.thickness_net import thickness_Autoencoder
from metrics.metric import l1_cd
from metrics.loss import cd_loss_L1, emd_loss, Loss
from visualization import view_depth

import pdb

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    else:
        pass


def log(fd,  message, time=True):
    if time:
        message = ' ==> '.join([datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), message])
    fd.write(message + '\n')
    fd.flush()
    print(message)


def prepare_logger(params):
    # prepare logger directory
    make_dir(params.log_dir)
    make_dir(os.path.join(params.log_dir, params.exp_name))

    logger_path = os.path.join(params.log_dir, params.exp_name, )
    ckpt_dir = os.path.join(params.log_dir, params.exp_name, 'checkpoints')
    epochs_dir = os.path.join(params.log_dir, params.exp_name, 'epochs')

    make_dir(logger_path)
    make_dir(ckpt_dir)
    make_dir(epochs_dir)

    logger_file = os.path.join(params.log_dir, params.exp_name, 'logger.log')
    log_fd = open(logger_file, 'a')

    log(log_fd, "Experiment: {}".format(params.exp_name), False)
    log(log_fd, "Logger directory: {}".format(logger_path), False)
    log(log_fd, str(params), False)

    train_writer = SummaryWriter(os.path.join(logger_path, 'train'))
    val_writer = SummaryWriter(os.path.join(logger_path, 'val'))

    return ckpt_dir, epochs_dir, log_fd, train_writer, val_writer


def train(params, device, writer):
    torch.backends.cudnn.benchmark = True

    #ckpt_dir, epochs_dir, log_fd, train_writer, val_writer = prepare_logger(params)
    ckpt_dir, epochs_dir, log_fd, train_writer, val_writer = writer[0], writer[1], writer[2], writer[3], writer[4]

    log(log_fd, 'Loading Data...')

    train_dataset = thickness_loader(params.train_set_path, 'train', )
    val_dataset = thickness_loader(params.train_set_path, 'valid', )

    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=False,sampler=DistributedSampler(train_dataset))
    val_dataloader = DataLoader(val_dataset, batch_size=params.batch_size, shuffle=False,sampler=DistributedSampler(val_dataset))

    log(log_fd, "Dataset loaded!")


    pdb.set_trace()
    model = thickness_Autoencoder(in_dim=2).to(device)
    # load vq model
    checkpoint = torch.load(params.vq_path)
    vq_dict = {k.replace('module._vq_vae.',''): v for k, v in checkpoint.items() if '_vq_vae' in k}
    model.vq.load_state_dict(vq_dict, strict=False)
    for name, param in model.named_parameters():
        if "vq" in name:
            param.requires_grad = False
        
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                        device_ids=[local_rank],
                                                        output_device=local_rank,
                                                        find_unused_parameters=True)
    
    # optimizer
    optimizer = Optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=params.lr, betas=(0.9, 0.999), eps=1e-2)
    lr_schedual = Optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.7)
    Loss_fn = Loss(emd_weight=1,
                 uniform_weight=1,
                 repulsion_weight=1,
                 TVLoss_weight=0.1,
                 l1_weight=5,
                 ssim_weight=1,
                 msssim_weight=2,)

    #step = len(train_dataloader) // params.log_frequency    # 27 // 10 = 2
    step = 5

    # load pretrained model and optimizer
    if params.ckpt_path is not None:
        model.load_state_dict(torch.load(params.ckpt_path))

    # training
    best_val_loss = 1e8
    best_epoch_val = -1
    train_step, val_step = 0, 0
    for epoch in range(1, params.epochs + 1):
        

        # training
        model.train()
        for i, (depth_F,depth_B, thickness, mask) in enumerate(train_dataloader):
            depth_F = depth_F.to(device)
            depth_B = depth_B.to(device)
            # rendered = rendered.to(device)
            # input_data = torch.cat((depth_F, depth_B, rendered), dim=1)
            input_data = torch.cat((depth_F, depth_B), dim=1)
            thickness = thickness.to(device).unsqueeze(1)
            mask = mask.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            
            pred = model(input_data)
            
            
            l1_loss = Loss_fn.get_matrix_l1_loss(pred[mask], thickness[mask])
            msssim_loss = Loss_fn.get_msssim_loss(pred, thickness)
            tv_loss = Loss_fn.get_tv_loss(pred)
            
            loss = l1_loss + msssim_loss + tv_loss
            
            loss.backward()
            optimizer.step()
            
            if (i + 1) % step == 0:
                log(log_fd, "Training Epoch [{:03d}/{:03d}] - Training Step [{:05d}]: l1_loss = {:.3f}, ssim_loss = {:.3f}, tv_loss = {:.3f}, total = {:.3f}"
                    .format(epoch, params.epochs, train_step, l1_loss.item() * 1e3, msssim_loss.item() * 1e3, tv_loss.item() * 1e3, loss.item() * 1e3))
            
            train_writer.add_scalar('L1', l1_loss.item(), train_step)
            train_writer.add_scalar('MS-SSIM', msssim_loss.item(), train_step)
            train_writer.add_scalar('TV', tv_loss.item(), train_step)
            train_writer.add_scalar('Total', loss.item(), train_step)
            train_step += 1
            
        lr_schedual.step()

        # evaluation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            rand_iter = random.randint(0, len(val_dataloader) - 1)  # for visualization
            
            for i, (depth_F,depth_B, thickness, mask) in enumerate(train_dataloader):
                depth_F = depth_F.to(device)
                depth_B = depth_B.to(device)
                # rendered = rendered.to(device)
                # input_data = torch.cat((depth_F, depth_B, rendered), dim=1)
                input_data = torch.cat((depth_F, depth_B), dim=1)
                
                thickness = thickness.to(device).unsqueeze(1)
                mask = mask.to(device).unsqueeze(1)
                
                pred = model(input_data)
                
                l1_loss = Loss_fn.get_matrix_l1_loss(pred[mask], thickness[mask])
                msssim_loss = Loss_fn.get_msssim_loss(pred, thickness)
                total_val_loss += l1_loss.item() + msssim_loss.item()
                
                if rand_iter == i and epoch%10==0 :
                    view_depth(depth_F[0][0].detach().cpu().numpy(),
                               depth_B[0][0].detach().cpu().numpy(),
                               pred[0][0].detach().cpu().numpy(), 
                               thickness[0][0].detach().cpu().numpy(), 
                               os.path.join(epochs_dir, 'epoch_{:03d}.png'.format(epoch)))
                    
            
            total_val_loss /= len(val_dataset)
            val_writer.add_scalar('Validate Loss', total_val_loss, val_step)
            val_step += 1

            log(log_fd, "Validate Epoch [{:03d}/{:03d}]: Validate Loss = {:.6f}".format(epoch, params.epochs, total_val_loss * 1e3))
        
        if total_val_loss < best_val_loss:
            best_epoch_val = epoch
            best_val_loss = total_val_loss
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f'{params.exp_name}.pth'))
            
    log(log_fd, 'Best val model in epoch {}, the minimum val loss is {}'.format(best_epoch_val, best_val_loss * 1e3))
    log_fd.close()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('thickness')
    parser.add_argument('--exp_name', type=str, help='Tag of experiment')
    parser.add_argument('--log_dir', type=str, default='log', help='Logger directory')
    parser.add_argument('--ckpt_path', type=str, default=None, help='The path of pretrained model')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    #parser.add_argument('--category', type=str, default='all', help='Category of point clouds')
    parser.add_argument('--epochs', type=int, default=400, help='Epochs of training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for data loader')
    #parser.add_argument('--coarse_loss', type=str, default='cd', help='loss function for coarse point cloud')
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers for data loader')
    #parser.add_argument('--device', type=str, default='cuda:0', help='device for training')
    parser.add_argument('--log_frequency', type=int, default=10, help='Logger frequency in every epoch')
    parser.add_argument('--save_frequency', type=int, default=10, help='Model saving frequency')
    parser.add_argument('--train_set_path', type=str, default='data/THuman2.0_Release', help='Path of training set (THuman2.0_Release)')
    parser.add_argument('--vq_path', type=str, default='./log/thuman_pu_1/checkpoints/best_l1_cd.pth', help='Path of pretrained vq model')
    params = parser.parse_args()
    
    writer = prepare_logger(params)
    
    torch.distributed.init_process_group(backend="nccl")
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    #pdb.set_trace()
    train(params, device, writer)
