
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import sys
import numpy as np
from torch.utils.checkpoint import checkpoint
from tqdm.notebook import tqdm,trange

import sklearn.metrics as metrics
import matplotlib.pyplot as plt


def evaluate_img(path, enc, classifier, save_to, no_mild = False):
    print('loading data')
    verse_all = np.load(path, allow_pickle=True)['arr_0'].item(0)

    vertebrae_val = verse_all['vertebrae_all']
    print(vertebrae_val.shape)
    #vertebrae_val = verse_all['vertebrae_val']
    #vertebrae_val = verse_all
    print('for images I should add .cuda().unsqueeze(1).flip(3)')
    vertebrae3d = torch.from_numpy(vertebrae_val).cuda().flip(3)
    
    val_fx_g = torch.from_numpy(verse_all['all_fx_g'])
    #val_fx_g = torch.from_numpy(verse_all['val_fx_g'])
    val_fx_g_ = val_fx_g[val_fx_g.isnan()==False].long().cuda()

    ##remove class 4
    vertebrae3d = vertebrae3d[val_fx_g_!=4]
    val_fx_g_ = val_fx_g_[val_fx_g_!=4]
    vertebrae3d = vertebrae3d[val_fx_g_!=5]
    val_fx_g_ = val_fx_g_[val_fx_g_!=5]

    len_data = vertebrae3d.shape[0]

    print('data loading done, number of vertebrae: ', )

    #gt = torch.empty(0)
    

    if no_mild == False:
        target = val_fx_g_.long().cpu()
        target[target == 1] = 0.5
        target[target == 2] = .75
        target[target == 3] = 1.
        target[target > 3] = 0.5
        gt = (val_fx_g_>0).long().cpu()
        gt_no1 = (val_fx_g_>1).long().cpu()
    else:
        vertebrae3d = vertebrae3d[val_fx_g_!=1]
        val_fx_g_ = val_fx_g_[val_fx_g_!=1]
        len_data = vertebrae3d.shape[0]

        target = val_fx_g_.long().cpu()
        target[target == 1] = 0.
        target[target == 2] = 1
        target[target == 3] = 1.
        target[target > 3] = 0.
        gt = (val_fx_g_>0).long().cpu()
        gt_no1 = (val_fx_g_>1).long().cpu()

        print(val_fx_g_.shape, vertebrae3d.shape, gt.shape)

    
    est = torch.empty(0)
    output_val = torch.empty(0,2)

    enc.eval()
    classifier.eval()
    
    B=3
    for ix in trange(len_data//B):
        output = torch.zeros(vertebrae_val[ix].shape[0],7).cuda()
        with torch.inference_mode():
            idx = torch.arange(ix*B,(ix+1)*B)
            vertebrae_in0 = vertebrae3d[idx]#.unsqueeze(1)
   
            q = enc(vertebrae_in0.float())
            output = classifier(q.view(B,-1,1)).squeeze(2)
            est = torch.cat((est,output.argmax(1).float().cpu()))
            output_val = torch.cat((output_val,torch.softmax(output,1).cpu()),0)

            #gt = torch.cat((gt,coords_val[ix][:,3].float()-5))
    
    print(output_val.shape, gt.shape)

    print(est[:10])
    print(gt[:10])
    print(gt_no1[:10])
    print(target[:10])

    print('AUC_gt_no1: ', metrics.roc_auc_score(gt_no1, output_val[:,1]))
    print('AUC_gt: ', metrics.roc_auc_score(gt, output_val[:,1]))
    print('AUC_target: ', metrics.roc_auc_score(target, output_val[:,1]))

    #dict_test = {'data_test': vertebrae3d, 'gt_test': val_fx_g_}
    #torch.save(dict_test, 'verse19_test_nomild.pth')

    torch.save(est, save_to + 'est.pth')
    torch.save(output_val, save_to + 'output_val.pth')
    #torch.save(gt, 'cls_eval/cls_results/gt_img.pth')

    return None


def evaluate_pointenc(path, enc, classifier, num_p, save_to, no_mild = False):
    print('loading data')
    verse_all = np.load(path, allow_pickle=True)['arr_0'].item(0)
    vertebrae_val = verse_all['vertebrae_all']
    #vertebrae_val = verse_all['vertebrae_val']
    vertebrae3d = torch.from_numpy(vertebrae_val).cuda().flip(3)
    
    val_fx_g = torch.from_numpy(verse_all['all_fx_g'])
    #val_fx_g = torch.from_numpy(verse_all['val_fx_g'])
    val_fx_g_ = val_fx_g[val_fx_g.isnan()==False].long().cuda()

    ##remove class 4
    vertebrae3d = vertebrae3d[val_fx_g_!=4]
    val_fx_g_ = val_fx_g_[val_fx_g_!=4]
    vertebrae3d = vertebrae3d[val_fx_g_!=5]
    val_fx_g_ = val_fx_g_[val_fx_g_!=5]
    #len_data = len(val_fx_g_)

    print(len(val_fx_g_), vertebrae3d.shape, num_p)
    #num_p = 1920*2

    in_vert3d_unsmoothed = torch.sigmoid(vertebrae3d.cuda())*2-1
    points_all, too_small = point_sampling(in_vert3d_unsmoothed.squeeze(1), num_p, fps=True)
    # points_all = torch.load('points_all_fps_3840_verse19test.pth')
    #data_all = np.delete(data_all, too_small, axis=0)
    print(too_small)
    #val_fx_g_ = torch.cat((val_fx_g_[:263], val_fx_g_[265:]),0)

    len_data = len(points_all)
    print('remaining vertebrae: ',len_data, val_fx_g_.shape)
    #points_all = torch.stack(points_all,0).cuda()

    print('data loading done, number of vertebrae: ', points_all.shape[0])

    if no_mild == False:
        target = val_fx_g_.long().cpu()
        target[target == 1] = 0.5
        target[target == 2] = .75
        target[target == 3] = 1.
        target[target > 3] = 0.5
        gt = (val_fx_g_>0).long().cpu()
        gt_no1 = (val_fx_g_>1).long().cpu()
    else:
        points_all = points_all[val_fx_g_!=1]
        val_fx_g_ = val_fx_g_[val_fx_g_!=1]
        len_data = points_all.shape[0]

        target = val_fx_g_.long().cpu()
        target[target == 1] = 0.
        target[target == 2] = 1
        target[target == 3] = 1.
        target[target > 3] = 0.
        gt = (val_fx_g_>0).long().cpu()
        gt_no1 = (val_fx_g_>1).long().cpu()

        print(val_fx_g_.shape, vertebrae3d.shape, gt.shape)

    #gt = torch.empty(0)
    est = torch.empty(0)
    output_val = torch.empty(0,2)
    output_val5 = torch.empty(0,5)

    enc.eval()
    classifier.eval()

    B=3
    for ix in trange(len_data//B):
        
        output = torch.zeros(vertebrae_val[ix].shape[0],7).cuda()
        #with torch.inference_mode():
        with torch.no_grad():
            idx = torch.arange(ix*B,(ix+1)*B)
            points_in = points_all[idx].cuda().squeeze(1).permute(0,2,1)#.view(2,3,-1)

            q = enc(points_in)

            output = classifier(q.view(B,-1,1)).squeeze(2)
            est = torch.cat((est,output.argmax(1).float().cpu()))
            output_val = torch.cat((output_val,torch.softmax(output,1).cpu()),0)


    print(est[:10])
    print(gt[:10])
    print(gt_no1[:10])
    print(target[:10])
    print(est.shape)

    print('AUC_gt_no1: ', metrics.roc_auc_score(gt_no1, output_val[:,1]))
    print('AUC_gt: ', metrics.roc_auc_score(gt, output_val[:,1]))
    print('AUC_target: ', metrics.roc_auc_score(target, output_val[:,1]))

    torch.save(est, save_to + 'est.pth')
    torch.save(output_val, save_to + 'output_val.pth')
    #torch.save(gt, 'cls_eval/cls_results/gt_point.pth')
    torch.save(gt, 'cls_eval/cls_results/gt_test.pth')


    return None


def evaluate_dgcnn(path, enc, classifier,k, num_p, save_to, no_mild = False):

    print('loading data')
    verse_all = np.load(path, allow_pickle=True)['arr_0'].item(0)#['a']#['arr_0'].item(0)
    vertebrae_val = verse_all['vertebrae_all']
    #vertebrae_val = verse_all['vertebrae_val']
    vertebrae3d = torch.from_numpy(vertebrae_val).cuda().flip(3)
    
    val_fx_g = torch.from_numpy(verse_all['all_fx_g'])
    #val_fx_g = torch.from_numpy(verse_all['val_fx_g'])
    val_fx_g_ = val_fx_g[val_fx_g.isnan()==False].long().cuda()

    ##remove class 4
    vertebrae3d = vertebrae3d[val_fx_g_!=4]
    val_fx_g_ = val_fx_g_[val_fx_g_!=4]
    vertebrae3d = vertebrae3d[val_fx_g_!=5]
    val_fx_g_ = val_fx_g_[val_fx_g_!=5]
    #len_data = len(val_fx_g_)
    #len_data = len(val_fx_g_)

    print(len(val_fx_g_), vertebrae3d.shape, num_p)
    #num_p = 1920*2

    in_vert3d_unsmoothed = torch.sigmoid(vertebrae3d.cuda())*2-1

    points_all, too_small = point_sampling(in_vert3d_unsmoothed.squeeze(1), num_p, fps=True)
    # points_all = torch.load('points_all_fps_3840_verse19test.pth')
    #data_all = np.delete(data_all, too_small, axis=0)
    print(points_all.shape, too_small)

    len_data = len(points_all)
    print('remaining vertebrae: ',len_data, val_fx_g_.shape)
    #points_all = torch.stack(points_all,0).cuda()

    print('data loading done, number of vertebrae: ', points_all.shape[0])

    if no_mild == False:
        target = val_fx_g_.long().cpu()
        target[target == 1] = 0.5
        target[target == 2] = .75
        target[target == 3] = 1.
        target[target > 3] = 0.5
        gt = (val_fx_g_>0).long().cpu()
        gt_no1 = (val_fx_g_>1).long().cpu()
    else:
        points_all = points_all[val_fx_g_!=1]
        val_fx_g_ = val_fx_g_[val_fx_g_!=1]
        len_data = points_all.shape[0]

        target = val_fx_g_.long().cpu()
        target[target == 1] = 0.
        target[target == 2] = 1
        target[target == 3] = 1.
        target[target > 3] = 0.
        gt = (val_fx_g_>0).long().cpu()
        gt_no1 = (val_fx_g_>1).long().cpu()

        print(val_fx_g_.shape, vertebrae3d.shape, gt.shape)

    #gt = torch.empty(0)
    est = torch.empty(0)
    output_val = torch.empty(0,2)
    output_val5 = torch.empty(0,5)

    enc.eval()
    classifier.eval()

    
    B=1
    for ix in trange(len_data//B):
        
        output = torch.zeros(vertebrae_val[ix].shape[0],7).cuda()
        #with torch.inference_mode():
        with torch.no_grad():
            idx = torch.arange(ix*B,(ix+1)*B)
            
            points_in = points_all[idx].cuda().squeeze(1)

            q = enc(points_in,k)

            output = classifier(q.view(B,-1,1)).squeeze(2)
            est = torch.cat((est,output.argmax(1).float().cpu()))
            output_val = torch.cat((output_val,torch.softmax(output,1).cpu()),0)

            #gt = torch.cat((gt,coords_val[ix][:,3].float()-5))

    print(est[:10])
    print(gt[:10])
    print(gt_no1[:10])
    print(target[:10])

    print('AUC_gt_no1: ', metrics.roc_auc_score(gt_no1, output_val[:,1]))
    print('AUC_gt: ', metrics.roc_auc_score(gt, output_val[:,1]))
    print('AUC_target: ', metrics.roc_auc_score(target, output_val[:,1]))

    torch.save(est, save_to + 'est.pth')
    torch.save(output_val, save_to + 'output_val.pth')
    torch.save(gt, 'cls_eval/cls_results/gt_gtseg.pth')
    print("saved gt")


    return None




def point_sampling(data_all, num_points, fps = True):

    print('generating pointset and removing smaller vertebrae below number of points threshold')
    points_all = []
    too_small  = []
    b,d,h,w = data_all.shape
    print('starting with num_vertebrae: ', b)
    for j in range(len(data_all)):
        mesh = F.affine_grid(torch.eye(3,4).unsqueeze(0).cuda(),(1,1,d,h,w), align_corners=False).reshape(-1,3)
        pts = mesh[data_all[j].reshape(-1)>.1,:]
        if fps:
            if pts.shape[0] > num_points:
                kpts = farthest_point_sampling(pts.unsqueeze(0), num_points)[0]
                points_all.append(kpts)

            else:
                too_small.append(j)
        else:
            if pts.shape[0] > num_points:
                ids = torch.cat((torch.randperm(pts.shape[0]),torch.randperm(pts.shape[0]),torch.randperm(pts.shape[0])),0)[:num_points]
                points_all.append(pts[ids])
            else: 
                too_small.append(j)

    print('remaining vertebrae: ',len(points_all))
    points_all = torch.stack(points_all,0).cuda()

    #torch.save(points_all, 'points_all_fps_3840_verse19test.pth')

    return points_all, too_small


def farthest_point_sampling(kpts, num_points):
    _, N, _ = kpts.size()
    ind = torch.zeros(num_points).long()
    ind[0] = torch.randint(N, (1,))
    dist = torch.sum((kpts - kpts[:, ind[0], :]) ** 2, dim=2)
    for i in range(1, num_points):
        ind[i] = torch.argmax(dist)
        dist = torch.min(dist, torch.sum((kpts - kpts[:, ind[i], :]) ** 2, dim=2))

    while N < num_points:
        add_points = min(N, num_points - N)
        ind = torch.cat([ind[:N], ind[:add_points]])
        N += add_points

    return kpts[:, ind, :], ind

