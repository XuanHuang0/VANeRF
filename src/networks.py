import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import math
import smplx
import numpy as np
from src.utils import *
from pytorch3d.ops import knn_points, knn_gather
from torch import autograd

from spconv.pytorch.conv import (SparseConv2d, SparseConv3d,
                                 SparseConvTranspose2d,
                                 SparseConvTranspose3d, SparseInverseConv2d,
                                 SparseInverseConv3d, SubMConv2d, SubMConv3d)
from spconv.pytorch.core import SparseConvTensor
from spconv.pytorch.identity import Identity
from spconv.pytorch.modules import SparseModule, SparseSequential
from spconv.pytorch.ops import ConvAlgo
from spconv.pytorch.pool import SparseMaxPool2d, SparseMaxPool3d
from spconv.pytorch.tables import AddTable, ConcatTable

num_v=int(1558/2)

def KNN_vis(query, vert, vert_feat, vert_vis, k):
    _,mink_idxs,_=knn_points(query,vert,K=k)
    vert_feat_knn=vert_feat[:,mink_idxs[0,:,:]]*vert_vis[:,mink_idxs[0,:,:]]
    vert_feat_toh=torch.cat([vert_feat[:,num_v:],vert_feat[:,:num_v]],dim=1)
    vert_vis_toh=torch.cat([vert_vis[:,num_v:],vert_vis[:,:num_v]],dim=1)
    vert_feat_knn_toh=vert_feat_toh[:,mink_idxs[0,:,:]]*vert_vis_toh[:,mink_idxs[0,:,:]]
    return vert_feat_knn.view(*vert_feat_knn.shape[:2],-1), vert_feat_knn_toh.view(*vert_feat_knn_toh.shape[:2],-1), vert_vis[:,mink_idxs[0,:,:]].squeeze(-1), vert_vis_toh[:,mink_idxs[0,:,:]].squeeze(-1)

def KNN(query, vert, vert_feat, vert_vis, k):
    _,mink_idxs,_=knn_points(query,vert,K=k)
    vert_feat_knn=vert_feat[:,mink_idxs[0,:,:]]
    vert_feat_toh=torch.cat([vert_feat[:,num_v:],vert_feat[:,:num_v]],dim=1)
    vert_vis_toh=torch.cat([vert_vis[:,num_v:],vert_vis[:,:num_v]],dim=1)
    vert_feat_knn_toh=vert_feat_toh[:,mink_idxs[0,:,:]]
    return vert_feat_knn.view(*vert_feat_knn.shape[:2],-1), vert_feat_knn_toh.view(*vert_feat_knn_toh.shape[:2],-1), vert_vis[:,mink_idxs[0,:,:]].squeeze(-1), vert_vis_toh[:,mink_idxs[0,:,:]].squeeze(-1)

class GeoVisFusion(nn.Module):
    def __init__(self, n=200704, fgt_ch=42, q_feat_in=486, q_feat_out=390, ka_feat_in=72):
        super(GeoVisFusion, self).__init__()

        self.fconv_at= nn.Sequential(
            nn.Conv1d(196, 10, 1, padding=0, bias=False),
            nn.ReLU(True),
            nn.Conv1d(10, 3, 1, padding=0, bias=False),
            nn.Sigmoid(),
        )

        self.fconv_ated= nn.Sequential(
            nn.Conv1d(196, 64, 1, padding=0, bias=False),
            nn.ReLU(True),
            nn.Conv1d(64, 64, 1, padding=0, bias=False)
        )

        self.fconv_at1= nn.Sequential(
            nn.Conv1d(28, 10, 1, padding=0, bias=False),
            nn.ReLU(True),
            nn.Conv1d(10, 3, 1, padding=0, bias=False),
            nn.Sigmoid(),
        )

        self.fconv_ated1= nn.Sequential(
            nn.Conv1d(28, 8, 1, padding=0, bias=False),
            nn.ReLU(True),
            nn.Conv1d(8, 8, 1, padding=0, bias=False)
        )

        self.sample_func = feat_sample

    def forward(self, vert_xy, fg, feat_sampled, vert, v, vert_vis, query_vis, closest_face, query_sdf):
        # fg1 ([1, 64, 32, 32])
        # fg2 ([1, 8, 128, 128])
        # ft1 ([1, 8, 64, 64]) 
        # y 518
        B = vert_xy.shape[0]

        feat_sampled_fused = []
        vert_feat = self.sample_func(fg[0], vert_xy) #3*64
        vert_feat_knn, vert_feat_knn_toh, vert_vis_th, vert_vis_toh=KNN_vis(v, vert, vert_feat, vert_vis, 1)
 
        fused_feat=torch.cat([feat_sampled[0].squeeze(1),vert_feat_knn,vert_feat_knn_toh],dim=2)
        fused_feat=torch.cat([fused_feat,query_sdf,query_vis,vert_vis_th,vert_vis_toh],dim=2) 
        fused_feat=fused_feat.float()
        fused_feat_at=self.fconv_at(fused_feat.permute(0,2,1)).permute(0,2,1)

        fused_feat_ated=torch.cat([feat_sampled[0].squeeze(1)*fused_feat_at[:,:,0:1],vert_feat_knn*fused_feat_at[:,:,1:2],vert_feat_knn_toh*fused_feat_at[:,:,2:3]],dim=2)
        fused_feat_ated=torch.cat([fused_feat_ated,query_sdf,query_vis,vert_vis_th,vert_vis_toh],dim=2)   
        fused_feat_ated=self.fconv_ated(fused_feat_ated.permute(0,2,1)).permute(0,2,1)
        feat_sampled_fused.append(fused_feat_ated.view(B, 1, *fused_feat_ated.shape[-2:]))

        vert_feat_f = self.sample_func(fg[1], vert_xy) #3*8
        vert_feat_knn, vert_feat_knn_toh, vert_vis_th, vert_vis_toh=KNN_vis(v, vert, vert_feat_f, vert_vis, 1)
        fused_feat=torch.cat([feat_sampled[1].squeeze(1),vert_feat_knn,vert_feat_knn_toh],dim=2)
        fused_feat=torch.cat([fused_feat,query_sdf,query_vis,vert_vis_th,vert_vis_toh],dim=2) 
        fused_feat_at=self.fconv_at1(fused_feat.permute(0,2,1)).permute(0,2,1)
        fused_feat_ated=torch.cat([feat_sampled[1].squeeze(1)*fused_feat_at[:,:,0:1],vert_feat_knn*fused_feat_at[:,:,1:2],vert_feat_knn_toh*fused_feat_at[:,:,2:3]],dim=2)
        fused_feat_ated=torch.cat([fused_feat_ated,query_sdf,query_vis,vert_vis_th,vert_vis_toh],dim=2)   
        fused_feat_ated=self.fconv_ated1(fused_feat_ated.permute(0,2,1)).permute(0,2,1)
        feat_sampled_fused.append(fused_feat_ated.view(B, 1, *fused_feat_ated.shape[-2:]))

        return feat_sampled_fused

class GeoVisFusion_spconv(nn.Module):
    def __init__(self, n=200704, fgt_ch=42, q_feat_in=486, q_feat_out=390, ka_feat_in=72):
        super(GeoVisFusion_spconv, self).__init__()

        self.linear_at = nn.Sequential(
            nn.Linear(193, 10),
            nn.LayerNorm(10, 1e-6),
            nn.ReLU(True),
            nn.Linear(10, 5),
            nn.Sigmoid(),
        )

        self.linear_vis_at = nn.Sequential(
            nn.Linear(3, 10),
            nn.LayerNorm(10, 1e-6),
            nn.ReLU(True),
            nn.Linear(10, 5),
            nn.Sigmoid(),
        )

        self.linear_ated = nn.Sequential(
            nn.Linear(196, 64),
            nn.LayerNorm(64, 1e-6),
            nn.ReLU(True),
            nn.Linear(64, 64),
        )

        self.linear_at1 = nn.Sequential(
            nn.Linear(73, 10),
            nn.LayerNorm(10, 1e-6),
            nn.ReLU(True),
            nn.Linear(10, 5),
            nn.Sigmoid(),
        )

        self.linear_vis_at1 = nn.Sequential(
            nn.Linear(3, 10),
            nn.LayerNorm(10, 1e-6),
            nn.ReLU(True),
            nn.Linear(10, 5),
            nn.Sigmoid(),
        )

        self.linear_ated1 = nn.Sequential(
            nn.Linear(76, 8),
            nn.LayerNorm(8, 1e-6),
            nn.ReLU(True),
            nn.Linear(8, 8),
        )

        self.linear = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32, 1e-6),
            nn.ReLU(True),
            nn.Linear(32, 16),
        )

        self.sample_func = feat_sample
        self.xyzc_net = SparseConvNet(f_in=16,f_up=32)
        self.xyzc_net_f = SparseConvNet(f_in=8,f_up=16)

    def forward(self, vert_xy, fg, feat_sampled, vert, v, vert_vis, query_vis, closest_face, query_sdf, coord, out_sh, bounds):
        # fg1 ([1, 64, 32, 32])
        # fg2 ([1, 8, 128, 128])
        # ft1 ([1, 8, 64, 64]) 
        # y 518
        B = vert_xy.shape[0]
        feat_sampled_fused = []
        vert_feat = self.sample_func(fg[0], vert_xy) #3*64
        vert_feat=self.linear(vert_feat)
        xyzc = SparseConvTensor(vert_feat[0], coord, out_sh, B)
        grid_coords = get_grid_coords(v, bounds, out_sh)
        grid_coords = grid_coords.unsqueeze(1)
        xyzc_feature = self.xyzc_net(xyzc, grid_coords.float())
        xyzc_feature =xyzc_feature.permute(0,2,1)
        vert_feat_knn, vert_feat_knn_toh, vert_vis_th, vert_vis_toh=KNN(v, vert, vert_feat, vert_vis, 1)
        
        fused_feat=torch.cat([feat_sampled[0].squeeze(1),vert_feat_knn,vert_feat_knn_toh,xyzc_feature,query_sdf],dim=2)
        vis_feat=torch.cat([query_vis,vert_vis_th,vert_vis_toh],dim=2) 
        fused_feat_at=self.linear_at(fused_feat)
        vis_feat_at=self.linear_vis_at(vis_feat.float())
        fused_feat_at=fused_feat_at*vis_feat_at

        fused_feat_ated=torch.cat([feat_sampled[0].squeeze(1)*fused_feat_at[:,:,0:1],vert_feat_knn*fused_feat_at[:,:,1:2],vert_feat_knn_toh*fused_feat_at[:,:,2:3],xyzc_feature*fused_feat_at[:,:,3:4],query_sdf*fused_feat_at[:,:,4:5]],dim=2)
        
        fused_feat_ated=torch.cat([fused_feat_ated,query_vis,vert_vis_th,vert_vis_toh],dim=2)   
        fused_feat_ated=self.linear_ated(fused_feat_ated)

        feat_sampled_fused.append(fused_feat_ated.view(B, 1, *fused_feat_ated.shape[-2:]))

        vert_feat_f = self.sample_func(fg[1], vert_xy) #3*8
        vert_feat_knn, vert_feat_knn_toh, vert_vis_th, vert_vis_toh=KNN(v, vert, vert_feat_f, vert_vis, 1)
        
        xyzc = SparseConvTensor(vert_feat_f[0], coord, out_sh, B)
        grid_coords = get_grid_coords(v, bounds, out_sh)
        grid_coords = grid_coords.unsqueeze(1)
        xyzc_feature = self.xyzc_net_f(xyzc, grid_coords.float())
        xyzc_feature =xyzc_feature.permute(0,2,1)

        fused_feat=torch.cat([feat_sampled[1].squeeze(1),vert_feat_knn,vert_feat_knn_toh,xyzc_feature,query_sdf],dim=2)
        vis_feat=torch.cat([query_vis,vert_vis_th,vert_vis_toh],dim=2) 
        fused_feat_at=self.linear_at1(fused_feat)
        vis_feat_at=self.linear_vis_at1(vis_feat.float())
        fused_feat_at=fused_feat_at*vis_feat_at 
        fused_feat_ated=torch.cat([feat_sampled[1].squeeze(1)*fused_feat_at[:,:,0:1],vert_feat_knn*fused_feat_at[:,:,1:2],vert_feat_knn_toh*fused_feat_at[:,:,2:3],xyzc_feature*fused_feat_at[:,:,3:4],query_sdf*fused_feat_at[:,:,4:5]],dim=2)
        fused_feat_ated=torch.cat([fused_feat_ated,query_vis,vert_vis_th,vert_vis_toh],dim=2)   
        fused_feat_ated=self.linear_ated1(fused_feat_ated)
        feat_sampled_fused.append(fused_feat_ated.view(B, 1, *fused_feat_ated.shape[-2:]))

        return feat_sampled_fused

class TexVisFusion(nn.Module):
    def __init__(self, n=200704, fgt_ch=42, q_feat_in=72+24, q_feat_out=16+24):
        super(TexVisFusion, self).__init__()
        if_ch3=8

        self.fconv= nn.Sequential(
            nn.Conv1d(q_feat_in, q_feat_in, 1, padding=0, bias=False),
            nn.ReLU(True),
            nn.Conv1d(q_feat_in, q_feat_out, 1, padding=0, bias=False)
        )

        self.fconv_at= nn.Sequential(
            nn.Conv1d(q_feat_in, q_feat_in, 1, padding=0, bias=False),
            nn.ReLU(True),
            nn.Conv1d(q_feat_in, 6, 1, padding=0, bias=False),
            nn.Sigmoid()
        )

        self.fconv_gt= nn.Sequential(
            nn.Conv1d(42, num_v, 3, padding=1, bias=False),
            nn.LayerNorm(18, 1e-6),
            nn.ReLU(True),
            nn.Conv1d(num_v, num_v*2, 3, padding=1, bias=False),
            nn.LayerNorm(18, 1e-6),
            nn.ReLU(True)
        )

        self.fconv3 = nn.Sequential(
            nn.Conv2d(if_ch3, 21, 3, padding=1, bias=False),
            nn.LayerNorm([64,64], 1e-6),
            nn.ReLU(True),
            nn.Conv2d(21, 42, 3, padding=1, bias=False),
            nn.LayerNorm([64,64], 1e-6),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(3)
        )

        self.fconv4 = nn.Sequential(
            nn.Conv2d(3, 21, 3, padding=1, bias=False),
            nn.LayerNorm([256,256], 1e-6),
            nn.ReLU(True),
            nn.Conv2d(21, 42, 3, padding=1, bias=False),
            nn.LayerNorm([256,256], 1e-6),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(3)
        )

        self.sample_func = feat_sample

    def forward(self, vert_xy, ft1, ft_xy, vert, v, vert_vis, query_vis, img_xy, img_fmap, latent_fused):
        B = vert_xy.shape[0]
        vert_feat = self.sample_func(ft1, vert_xy) #1,779*2,8
        vert_img_feat = self.sample_func(img_fmap, vert_xy) #1,779*2,3
        vert_feat=torch.cat([vert_img_feat,vert_feat],dim=2)
        gf = self.fconv3(ft1)
        gf = gf.reshape(*gf.shape[:2],-1)
        gf_img= self.fconv4(img_fmap)
        gf_img = gf_img.reshape(*gf_img.shape[:2],-1)
        gf=torch.cat([gf_img,gf],dim=-1)
        gf_vert_feat=self.fconv_gt(gf)
        vert_feat=torch.cat([vert_feat,gf_vert_feat],dim=2)

        vert_feat_knn, vert_feat_knn_toh, vert_vis_th, vert_vis_toh=KNN_vis(v, vert, vert_feat, vert_vis, 1)
        vert_feat_knn_gf, vert_feat_knn_toh_gf=vert_feat_knn[:,:,11:], vert_feat_knn_toh[:,:,11:]
        vert_feat_knn, vert_feat_knn_toh=vert_feat_knn[:,:,:11], vert_feat_knn_toh[:,:,:11]
        query_feat=torch.cat([img_xy,ft_xy],dim=2)  #12
        y_feat=torch.cat([query_feat,vert_feat_knn,vert_feat_knn_toh, vert_feat_knn_gf, vert_feat_knn_toh_gf],dim=2)
        y_feat=torch.cat([y_feat, latent_fused, query_vis, vert_vis_th, vert_vis_toh],dim=2) 
        y_feat=y_feat.float()
        y_feat_at=self.fconv_at(y_feat.permute(0,2,1)).permute(0,2,1)
        y_feat_ated=torch.cat([query_feat*y_feat_at[:,:,0:1],vert_feat_knn*y_feat_at[:,:,1:2],vert_feat_knn_toh*y_feat_at[:,:,2:3], vert_feat_knn_gf*y_feat_at[:,:,3:4], vert_feat_knn_toh_gf*y_feat_at[:,:,4:5], latent_fused*y_feat_at[:,:,5:6]],dim=2)
        y_feat_ated=torch.cat([y_feat_ated,query_vis,vert_vis_th,vert_vis_toh],dim=2) 
        y_feat_ated=self.fconv(y_feat_ated.permute(0,2,1)).permute(0,2,1)

        return y_feat_ated

class TexVisFusion_spconv(nn.Module):
    def __init__(self, n=200704, fgt_ch=42, q_feat_in=72+24, q_feat_out=16+24):
        super(TexVisFusion_spconv, self).__init__()
        if_ch3=8

        self.linear = nn.Sequential(
            nn.Linear(215+3, q_feat_in),
            nn.LayerNorm(q_feat_in, 1e-6),
            nn.ReLU(True),
            nn.Linear(q_feat_in, q_feat_out),
        )

        self.linear_at= nn.Sequential(
            nn.Linear(215, q_feat_in),
            nn.LayerNorm(q_feat_in, 1e-6),
            nn.ReLU(True),
            nn.Linear(q_feat_in, 7),
            nn.Sigmoid()
        )

        self.linear_vis_at= nn.Sequential(
            nn.Linear(3, 10),
            nn.LayerNorm(10, 1e-6),
            nn.ReLU(True),
            nn.Linear(10, 7),
            nn.Sigmoid()
        )


        self.fconv_gt= nn.Sequential(
            nn.Conv1d(42, num_v, 3, padding=1),
            nn.LayerNorm(18, 1e-6),
            nn.ReLU(True),
            nn.Conv1d(num_v, num_v*2, 3, padding=1),
            nn.LayerNorm(18, 1e-6),
            nn.ReLU(True)
        )

        self.fconv3 = nn.Sequential(
            nn.Conv2d(if_ch3, 21, 3, padding=1),
            nn.LayerNorm([64,64], 1e-6),
            nn.ReLU(True),
            nn.Conv2d(21, 42, 3, padding=1),
            nn.LayerNorm([64,64], 1e-6),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(3)
        )

        self.fconv4 = nn.Sequential(
            nn.Conv2d(3, 21, 3, padding=1),
            nn.LayerNorm([256,256], 1e-6),
            nn.ReLU(True),
            nn.Conv2d(21, 42, 3, padding=1),
            nn.LayerNorm([256,256], 1e-6),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(3)
        )

        self.softmax=nn.Softmax(dim=-1)
        self.sample_func = feat_sample
        self.xyzc_net = SparseConvNet(f_in=29,f_up=32)

    def forward(self, vert_xy, ft1, ft_xy, vert, v, vert_vis, query_vis, img_xy, img_fmap, latent_fused, coord, out_sh, bounds):
        # ft1 ([1, 8, 64, 64]) 
        # img_fmap 1,3,256,256
        # img_xy 3 ft_xy 8 vert_xy 24 vert_xy_toh 24 gf 8 latent_fused 24
        B = vert_xy.shape[0]
        vert_feat = self.sample_func(ft1, vert_xy) #1,779*2,8
        vert_img_feat = self.sample_func(img_fmap, vert_xy) #1,779*2,3
        vert_feat=torch.cat([vert_img_feat,vert_feat],dim=2) #12
        gf = self.fconv3(ft1)
        gf = gf.reshape(*gf.shape[:2],-1)
        gf_img= self.fconv4(img_fmap)
        gf_img = gf_img.reshape(*gf_img.shape[:2],-1)
        gf=torch.cat([gf_img,gf],dim=-1)
        gf_vert_feat=self.fconv_gt(gf) #18
        vert_feat=torch.cat([vert_feat,gf_vert_feat],dim=2) #29
   
        xyzc = SparseConvTensor(vert_feat[0], coord, out_sh, B)
        grid_coords = get_grid_coords(v, bounds, out_sh)
        grid_coords = grid_coords.unsqueeze(1)
    
        xyzc_feature = self.xyzc_net(xyzc, grid_coords.float())
        xyzc_feature =xyzc_feature.permute(0,2,1)
        vert_feat_knn, vert_feat_knn_toh, vert_vis_th, vert_vis_toh=KNN(v, vert, vert_feat, vert_vis, 1)
        vert_feat_knn_gf, vert_feat_knn_toh_gf=vert_feat_knn[:,:,11:], vert_feat_knn_toh[:,:,11:]
        vert_feat_knn, vert_feat_knn_toh=vert_feat_knn[:,:,:11], vert_feat_knn_toh[:,:,:11]
        query_feat=torch.cat([img_xy,ft_xy],dim=2)  #12

        y_feat=torch.cat([query_feat,vert_feat_knn,vert_feat_knn_toh, vert_feat_knn_gf, vert_feat_knn_toh_gf, xyzc_feature, latent_fused],dim=2)
        vis_feat=torch.cat([query_vis, vert_vis_th, vert_vis_toh],dim=2) 
    
        y_feat_at=self.linear_at(y_feat) #72+24-3
        vis_feat_at=self.linear_vis_at(vis_feat.float())
        y_feat_at=y_feat_at*vis_feat_at
        y_feat_ated=torch.cat([query_feat*y_feat_at[:,:,0:1],vert_feat_knn*y_feat_at[:,:,1:2],vert_feat_knn_toh*y_feat_at[:,:,2:3], vert_feat_knn_gf*y_feat_at[:,:,3:4], vert_feat_knn_toh_gf*y_feat_at[:,:,4:5],xyzc_feature*y_feat_at[:,:,5:6], latent_fused*y_feat_at[:,:,6:7]],dim=2)
        y_feat_ated=torch.cat([y_feat_ated,query_vis,vert_vis_th,vert_vis_toh],dim=2) 
        y_feat_ated=self.linear(y_feat_ated)

        return y_feat_ated

def get_grid_coords(pts, bounds, out_sh):
    # convert xyz to the voxel coordinate dhw
    dhw = pts[..., [2, 1, 0]]
    min_dhw = bounds[:, 0, [2, 1, 0]]
    dhw = dhw - min_dhw[:, None, None]
    dhw = dhw / torch.tensor([0.005, 0.005, 0.005]).to(dhw)
    # convert the voxel coordinate to [-1, 1]
    out_sh = torch.tensor(out_sh).to(dhw)
    dhw = dhw / out_sh * 2 - 1
    # convert dhw to whd, since the occupancy is indexed by dhw
    grid_coords = dhw[..., [2, 1, 0]]
    return grid_coords

def single_conv(in_channels, out_channels, indice_key=None):
    return SparseSequential(
        SubMConv3d(in_channels,
                   out_channels,
                   1,
                   bias=False,
                   indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def double_conv(in_channels, out_channels, indice_key=None):
    return SparseSequential(
        SubMConv3d(in_channels,
                   out_channels,
                   3,
                   bias=False,
                   indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        SubMConv3d(out_channels,
                   out_channels,
                   3,
                   bias=False,
                   indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def triple_conv(in_channels, out_channels, indice_key=None):
    return SparseSequential(
        SubMConv3d(in_channels,
                   out_channels,
                   3,
                   bias=False,
                   indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        SubMConv3d(out_channels,
                   out_channels,
                   3,
                   bias=False,
                   indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        SubMConv3d(out_channels,
                   out_channels,
                   3,
                   bias=False,
                   indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def stride_conv(in_channels, out_channels, indice_key=None):
    return SparseSequential(
        SparseConv3d(in_channels,
                     out_channels,
                     3,
                     2,
                     padding=1,
                     bias=False,
                     indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01), nn.ReLU())


class SparseConvNet(nn.Module):
    def __init__(self,f_in=29,f_up=32):
        super(SparseConvNet, self).__init__()

        self.conv0 = double_conv(f_in, f_in, 'subm0')
        self.down0 = stride_conv(f_in, f_in, 'down0')

        self.conv1 = double_conv(f_in, f_in, 'subm1')
        self.down1 = stride_conv(f_in, f_in, 'down1')

        self.conv2 = triple_conv(f_in, f_in, 'subm2')
        self.down2 = stride_conv(f_in, f_up, 'down2')

        self.conv3 = triple_conv(f_up, f_up, 'subm3')
        self.down3 = stride_conv(f_up, f_up, 'down3')

        self.conv4 = triple_conv(f_up, f_up, 'subm4')

    def forward(self, x, grid_coords):
        net = self.conv0(x)
        net = self.down0(net)

        net = self.conv1(net)
        net1 = net.dense()
        feature_1 = F.grid_sample(net1,
                                  grid_coords,
                                  padding_mode='zeros',
                                  align_corners=True)

        net = self.down1(net)
        net = self.conv2(net)
        net2 = net.dense()
        feature_2 = F.grid_sample(net2,
                                  grid_coords,
                                  padding_mode='zeros',
                                  align_corners=True)
        net = self.down2(net)
        net = self.conv3(net)
        net3 = net.dense()
        feature_3 = F.grid_sample(net3,
                                  grid_coords,
                                  padding_mode='zeros',
                                  align_corners=True)
        net = self.down3(net)
        net = self.conv4(net)
        net4 = net.dense()
        feature_4 = F.grid_sample(net4,
                                  grid_coords,
                                  padding_mode='zeros',
                                  align_corners=True)

        features = torch.cat((feature_1, feature_2, feature_3, feature_4),
                             dim=1)
        features = features.view(features.size(0), -1, features.size(4))

        return features

class Discriminator_vis(nn.Module):
    def __init__(self):
        super(Discriminator_vis, self).__init__()

        self.fconv3 = nn.Sequential(
            nn.Conv2d(12, 10, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(10, 10,3, padding=1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fconv4 = nn.Sequential(
            nn.Conv2d(12, 20, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(20, 20, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(20, 12, 3, padding=1),
        )

        self.fconv2 = nn.Sequential(
            nn.Conv2d(24, 30, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(30, 20, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(20, 1, 3, padding=1),
            nn.Sigmoid(),
        )

        self.linear = nn.Sequential(
            nn.Linear(10, 3),
            nn.ReLU(True),
            nn.Linear(3, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_img, input_densepose, tar_densepose, pred):
        # ft1 ([1, 8, 64, 64]) 
        # img_fmap 1,3,56,56
        # img_xy 3 ft_xy 8 vert_xy 24 vert_xy_toh 24 gf 8
        B = pred.shape[0]
        img=torch.cat([input_img, input_densepose, tar_densepose, pred],dim=1) #12
        gf_img= self.fconv3(img).squeeze(3).squeeze(2) #1,10
        img_vis= self.fconv4(img) #1,10
        img_cat=torch.cat([img,img_vis],dim=1)
        img_vis= self.fconv2(img_cat)
        gan_pred=self.linear(gf_img)
        return gan_pred,img_vis

def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(outputs=real_pred.sum(),
                               inputs=real_img,
                               create_graph=True)

    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty

def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss
