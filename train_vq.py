'''
Description: ThicknessVAE Stage 1: train the codebook
Author: XiaotaoWu
Date: 2023-09-19 21:48:59
LastEditTime: 2024-02-01 14:01:24
LastEditors: XiaotaoWu
'''
import argparse
import os
import datetime
import random

import torch
import torch.nn as nn
import torch.optim as Optim

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter

from dataset import thuman
from models import PCN
from metrics.metric import l1_cd
from metrics.loss import cd_loss_L1, emd_loss, Loss
from visualization import plot_pcd_one_view
import pdb

def make_dir(dir_path):
    """
    The function `make_dir` creates a directory at the specified path if it does not already exist.
    
    :param dir_path: The `dir_path` parameter is a string that represents the path of the directory that
    needs to be created
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def log(fd,  message, time=True):
    """
    Write a log message to a file descriptor and print it to the console.

    Parameters:
    - fd: The file descriptor to write the log message to.
    - message: The log message to write.
    - time: Whether to include the current timestamp in the log message. Default is True.
    """
    if time:
        message = ' ==> '.join([datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), message])
    fd.write(message + '\n')
    fd.flush()
    print(message)


def prepare_logger(params):
    """
    Prepare the logger for the experiment.

    Args:
        params (object): The parameters for the experiment.

    Returns:
        tuple: A tuple containing the checkpoint directory, epochs directory, log file descriptor,
               train writer, and validation writer.
    """
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
    """
    Train the VQ-VAE model.

    Args:
        params (object): Parameters for training.
        device (torch.device): Device to perform training on.
        writer (list): List of writer objects for logging.
    """
    torch.backends.cudnn.benchmark = True

    ckpt_dir, epochs_dir, log_fd, train_writer, val_writer = writer[0], writer[1], writer[2], writer[3], writer[4]

    log(log_fd, 'Loading Data...')

    # load dataset
    train_dataset = thuman(params.train_set_path, 'train', )
    val_dataset = thuman(params.train_set_path, 'valid', )
    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=False,sampler=DistributedSampler(train_dataset))
    val_dataloader = DataLoader(val_dataset, batch_size=params.batch_size, shuffle=False,sampler=DistributedSampler(val_dataset))
    log(log_fd, "Dataset loaded!")

    # model
    model = PCN(num_dense=16384, latent_dim=1024, grid_size=4, decay=0.99).to(device)
    #model = PCN(num_dense=16384, latent_dim=1024, grid_size=4).to(device)
    Loss_fn=Loss()
    
    # parallel training
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                        device_ids=[local_rank],
                                                        output_device=local_rank,
                                                        find_unused_parameters=True)
    
    # optimizer
    optimizer = Optim.Adam(model.parameters(), lr=params.lr, betas=(0.9, 0.999))
    lr_schedual = Optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.7)

    step = len(train_dataloader) // params.log_frequency

    # load pretrained model and optimizer
    if params.ckpt_path is not None:
        model.load_state_dict(torch.load(params.ckpt_path))

    # training
    best_cd_l1 = 1e8
    best_epoch_l1 = -1
    train_step, val_step = 0, 0
    for epoch in range(1, params.epochs + 1):
        # hyperparameter alpha
        if train_step < 2000:#10000:
            alpha = 0.01
        elif train_step < 3000:#20000:
            alpha = 0.1
        elif train_step < 4000:#50000:
            alpha = 0.5
        else:
            alpha = 1.0

        # training
        model.train()
        for i, (p,c) in enumerate(train_dataloader):
            p = p.to(device)    # partial point cloud (coarse)
            c = c.to(device)    # complete point cloud (dense)

            optimizer.zero_grad()

            # forward propagation
            vq_loss, coarse_pred, dense_pred, perplexity = model(p)
            #coarse_pred, dense_pred = model(p)
            
            # loss function
            coarse_c = c[:,:1024,:]
            loss1 = emd_loss(coarse_pred, coarse_c)
            loss2 = cd_loss_L1(dense_pred, c)
            repulsion_loss = 5 * Loss_fn.get_repulsion_loss(dense_pred)
            uniform_loss = 10 * Loss_fn.get_uniform_loss(dense_pred)
            loss = loss1 + alpha * loss2 + vq_loss + repulsion_loss + uniform_loss

            # back propagation
            loss.backward()
            optimizer.step()

            # log
            if (i + 1) % step == 0:
                log(log_fd, "Training Epoch [{:03d}/{:03d}] - Training Step [{:05d}]: coarse = {:.6f}, dense = {:.6f}, repulsion = {:.6f}, uniform = {:.6f}, total loss = {:.6f}"
                    .format(epoch, params.epochs, train_step, loss1.item() * 1e3, loss2.item() * 1e3, repulsion_loss.item()* 1e3, uniform_loss.item()* 1e3, loss.item() * 1e3))
            if train_step%200==0:
                log(log_fd, "alpha = %f"%(alpha))
            
            # tensorboard
            train_writer.add_scalar('coarse', loss1.item(), train_step)
            train_writer.add_scalar('dense', loss2.item(), train_step)
            train_writer.add_scalar('vq', vq_loss.item(), train_step)
            train_writer.add_scalar('repulsion', repulsion_loss.item(), train_step)
            train_writer.add_scalar('uniform', uniform_loss.item(), train_step)
            train_writer.add_scalar('total', loss.item(), train_step)
            train_step += 1
        
        lr_schedual.step()

        # evaluation
        model.eval()
        total_cd_l1 = 0.0
        with torch.no_grad():
            rand_iter = random.randint(0, len(val_dataloader) - 1)  # for visualization

            for i, (p,c) in enumerate(val_dataloader):
                p = p.to(device)
                c = c.to(device)
                
                vq_loss, coarse_pred, dense_pred, perplexity = model(p)
                #coarse_pred, dense_pred = model(p)
                
                total_cd_l1 += l1_cd(dense_pred, c).item()

                # save eval into image
                if rand_iter == i and epoch%10==0 :
                    index = random.randint(0, dense_pred.shape[0] - 1)
                    plot_pcd_one_view(os.path.join(epochs_dir, 'epoch_{:03d}.png'.format(epoch)),
                                      [p[index].detach().cpu().numpy(), coarse_pred[index].detach().cpu().numpy(), 
                                       dense_pred[index].detach().cpu().numpy(), c[index].detach().cpu().numpy()],
                                      ['Input', 'Coarse', 
                                       'Dense', 'Ground Truth'], 
                                      xlim=(-0.35, 0.35), ylim=(-0.35, 0.35), zlim=(-0.35, 0.35))
            
            # tensorboard
            total_cd_l1 /= len(val_dataset)
            val_writer.add_scalar('l1_cd', total_cd_l1, val_step)
            val_step += 1

            # log
            log(log_fd, "Validate Epoch [{:03d}/{:03d}]: L1 Chamfer Distance = {:.6f}".format(epoch, params.epochs, total_cd_l1 * 1e3))
        
        if total_cd_l1 < best_cd_l1:
            best_epoch_l1 = epoch
            best_cd_l1 = total_cd_l1
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best_l1_cd.pth'))
            
    log(log_fd, 'Best l1 cd model in epoch {}, the minimum l1 cd is {}'.format(best_epoch_l1, best_cd_l1 * 1e3))
    log_fd.close()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('thuman')
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
    params = parser.parse_args()
    
    # prepare logger
    writer = prepare_logger(params)
    
    # distributed training
    torch.distributed.init_process_group(backend="nccl")
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    #device = torch.device("cuda")
    
    train(params, device, writer)
