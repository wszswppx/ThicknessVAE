import torch
import torch.nn as nn
import torch.nn.functional as F

import os,sys
sys.path.append('../')
from auction_match import auction_match
import pointnet2.pointnet2_utils as pn2_utils
import math
from knn_cuda import KNN
import pytorch_msssim

from extensions.chamfer_distance.chamfer_distance import ChamferDistance
from extensions.earth_movers_distance.emd import EarthMoverDistance


CD = ChamferDistance()
EMD = EarthMoverDistance()


def cd_loss_L1(pcs1, pcs2):
    """
    L1 Chamfer Distance.

    Args:
        pcs1 (torch.tensor): (B, N, 3)
        pcs2 (torch.tensor): (B, M, 3)
    """
    dist1, dist2 = CD(pcs1, pcs2)
    dist1 = torch.sqrt(dist1)
    dist2 = torch.sqrt(dist2)
    return (torch.mean(dist1) + torch.mean(dist2)) / 2.0


def cd_loss_L2(pcs1, pcs2):
    """
    L2 Chamfer Distance.

    Args:
        pcs1 (torch.tensor): (B, N, 3)
        pcs2 (torch.tensor): (B, M, 3)
    """
    dist1, dist2 = CD(pcs1, pcs2)
    return torch.mean(dist1) + torch.mean(dist2)


def emd_loss(pcs1, pcs2):
    """
    EMD Loss.

    Args:
        xyz1 (torch.Tensor): (b, N, 3)
        xyz2 (torch.Tensor): (b, N, 3)
    """
    dists = EMD(pcs1, pcs2)
    return torch.mean(dists)


class Loss(nn.Module):
    def __init__(self,radius=1.0,
                 emd_weight=1,
                 uniform_weight=1,
                 repulsion_weight=1,
                 TVLoss_weight=1,
                 l1_weight=1,
                 ssim_weight=1,
                 msssim_weight=1,
                 ):
        super(Loss,self).__init__()
        self.radius=radius
        self.knn_uniform=KNN(k=2,transpose_mode=True)
        self.knn_repulsion=KNN(k=20,transpose_mode=True)
        self.TVLoss_weight = TVLoss_weight
        self.emd_weight = emd_weight
        self.uniform_weight = uniform_weight
        self.repulsion_weight = repulsion_weight
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.msssim_weight = msssim_weight
        
    def get_emd_loss(self,pred,gt,radius=1.0):
        '''
        pred and gt is B N 3
        '''
        idx, _ = auction_match(pred.contiguous(), gt.contiguous())
        #gather operation has to be B 3 N
        #print(gt.transpose(1,2).shape)
        matched_out = pn2_utils.gather_operation(gt.transpose(1, 2).contiguous(), idx)
        matched_out = matched_out.transpose(1, 2).contiguous()
        dist2 = (pred - matched_out) ** 2
        dist2 = dist2.contiguous().view(dist2.shape[0], -1)  # <-- ???
        dist2 = torch.mean(dist2, dim=1, keepdims=True)  # B,
        dist2 /= radius
        return self.emd_weight * torch.mean(dist2)
    
    def get_uniform_loss(self,pcd,percentage=[0.004,0.006,0.008,0.010,0.012],radius=1.0):
        B,N,C=pcd.shape[0],pcd.shape[1],pcd.shape[2]
        npoint=int(N*0.05)
        loss=0
        further_point_idx = pn2_utils.furthest_point_sample(pcd.contiguous(), npoint)
        new_xyz = pn2_utils.gather_operation(pcd.permute(0, 2, 1).contiguous(), further_point_idx)  # B,C,N
        for p in percentage:
            nsample=int(N*p)
            r=math.sqrt(p*radius)
            disk_area=math.pi*(radius**2)/N

            idx=pn2_utils.ball_query(r,nsample,pcd.contiguous(),new_xyz.permute(0,2,1).contiguous()) #b N nsample

            expect_len=math.sqrt(disk_area)

            grouped_pcd=pn2_utils.grouping_operation(pcd.permute(0,2,1).contiguous(),idx)#B C N nsample
            grouped_pcd=grouped_pcd.permute(0,2,3,1) #B N nsample C

            grouped_pcd=torch.cat(torch.unbind(grouped_pcd,dim=1),dim=0)#B*N nsample C

            dist,_=self.knn_uniform(grouped_pcd,grouped_pcd)
            #print(dist.shape)
            uniform_dist=dist[:,:,1:] #B*N nsample 1
            uniform_dist=torch.abs(uniform_dist+1e-8)
            uniform_dist=torch.mean(uniform_dist,dim=1)
            uniform_dist=(uniform_dist-expect_len)**2/(expect_len+1e-8)
            mean_loss=torch.mean(uniform_dist)
            mean_loss=mean_loss*math.pow(p*100,2)
            loss+=mean_loss
        return self.uniform_weight * loss/len(percentage)
    
    def get_repulsion_loss(self,pcd,h=0.0005):
        dist,idx=self.knn_repulsion(pcd,pcd)#B N k

        dist=dist[:,:,1:5]**2 #top 4 cloest neighbors

        loss=torch.clamp(-dist+h,min=0)
        loss=torch.mean(loss)
        #print(loss)
        return self.repulsion_weight * loss
    
    def get_tv_loss(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]
    
    def get_matrix_l1_loss(self, pred, GT):
        return self.l1_weight * torch.mean(torch.abs(pred-GT))
    
    def get_ssim_loss(self, pred, GT):
        return self.ssim_weight * (1.0 - pytorch_msssim.ssim(pred, GT))
    
    def get_msssim_loss(self, pred, GT):
        return self.msssim_weight * (1.0 - pytorch_msssim.msssim(pred, GT, normalize="relu"))