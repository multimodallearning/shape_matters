
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import sys
import numpy as np
from torch.utils.checkpoint import checkpoint
from tqdm.notebook import tqdm,trange


from torch.utils.tensorboard import SummaryWriter

from .training_mlp_datasplit import generate_data_split


def point_sampling(data_all, num_points, fps = True):
    points_all = []
    too_small  = []
    b,d,h,w = data_all.shape
    for j in range(len(data_all)):
        mesh = F.affine_grid(torch.eye(3,4).unsqueeze(0).cuda(),(1,1,d,h,w), align_corners=False).reshape(-1,3)
        pts = mesh[data_all[j].reshape(-1)>.1,:]
        if pts.shape[0] >= num_points-1:
            if fps:
                kpts = farthest_point_sampling(pts.unsqueeze(0), num_points)[0]
                points_all.append(kpts)
            else:
                ids = torch.cat((torch.randperm(pts.shape[0]),torch.randperm(pts.shape[0]),torch.randperm(pts.shape[0])),0)[:num_points]
                points_all.append(pts[ids])
        else:
                too_small.append(j)
        
    points_all = torch.stack(points_all,0).cuda()
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


def train_CNN_mlp_image2fx(path, enc, mlp, config, save_to, data_ratio=1, no_mild = False , ratio=False):
    num_iterations = 6000
    seed = torch.random.seed()
    print(seed)

    ##load data
    print('loading data')
    verse_all = np.load(path, allow_pickle=True)['arr_0'].item(0)
    vertebrae_all = verse_all['vertebrae_all']
    vertebrae3d = torch.from_numpy(vertebrae_all).cuda().flip(3)
    writer = SummaryWriter('runs_mlp/img_e2e_image')
    
    all_fx_g = torch.from_numpy(verse_all['all_fx_g']).float().cuda()
    all_fx_g_ = all_fx_g[all_fx_g.isnan()==False]
    all_fx_g_ = all_fx_g_.long().cuda()

    ##remove class 4
    vertebrae3d = vertebrae3d[all_fx_g_!=4]
    all_fx_g_ = all_fx_g_[all_fx_g_!=4]
    if no_mild:
        ##remove class 1
        vertebrae3d = vertebrae3d[all_fx_g_!=1]
        all_fx_g_ = all_fx_g_[all_fx_g_!=1].cuda()

     ##data_ratio
    if ratio:
        vertebrae3d, all_fx_g_ = generate_data_split(path, data_ratio, no_mild)
        writer_n = 'runs_mlp/data_split_img2end_r' + str(data_ratio)
        writer = SummaryWriter(writer_n)
        num_iterations = 6000
    else:
        writer = SummaryWriter('runs_mlp/img2end')

    len_data = all_fx_g_.shape[0]
    print(len_data)
    num_p = 1920*2

    points_all, too_small = point_sampling(vertebrae3d.squeeze(1), num_p, fps=True)
    data_all = np.delete(vertebrae3d.cpu().numpy(), too_small, axis=0)

    len_data = points_all.shape[0]
    
    enc.train()
    enc.cuda()
        
    optimizer = torch.optim.Adam(mlp.parameters(),lr=0.0001)
    run_loss = torch.zeros(num_iterations)
    t0 = time.time()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,num_iterations//3,2)
    scaler = torch.cuda.amp.GradScaler()


    B = 16


    print('starting training', num_iterations)
    with tqdm(total=num_iterations, file=sys.stdout) as pbar:
        t_run = time.time()

        for i in range(num_iterations):
            idx = torch.randperm(len_data)[:B]
            input0 = torch.from_numpy(data_all[idx]).cuda()

            target = all_fx_g_[idx].long()
            if no_mild: 
                target[target == 1] = 0
            else:
                target[target == 1] = 1
            target[target == 2] = 1
            target[target == 3] = 1
            target[target > 3] = 0
            #optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                    
                affine_matrix = (0.07*torch.randn(B,3,4)+torch.eye(3,4).unsqueeze(0)).cuda()
                augment_grid = F.affine_grid(affine_matrix,(B,1,96,64,80),align_corners=False)
                vertebrae_in0 = F.grid_sample(input0,augment_grid,align_corners=False)

                q = enc(vertebrae_in0)

                output = mlp(q.view(B,-1,1))

            loss = nn.CrossEntropyLoss()(output, target.view(B,1).long())

            run_loss[i] = loss.item()

 
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            
            writer.add_scalar('loss/no_mild/train/', run_loss[i], i)


            if i % 1000 == 0 and i != 0:
                print("loss: ", run_loss[i-15:i].mean())


        #torch.save(run_loss, 'run_loss_' + save_to +'_mlp.pth')
        torch.save([enc, mlp], 'run_checkpoint' + save_to + str(seed) +'_encmlp.pth')
        t_run2 = time.time()
        
    return


def train_DGCNN_mlp_graph2fx(path, enc, k, mlp, config, save_to, data_ratio=1, no_mild = False, ratio= False):
    num_iterations = 6000
    seed = torch.random.seed()
    print(seed)

    ##load data
    print('loading data')
    verse_all = np.load(path, allow_pickle=True)['arr_0'].item(0)
    vertebrae_all = verse_all['vertebrae_all']
    vertebrae3d = torch.from_numpy(vertebrae_all).flip(3).cuda()
    print('hello this is important, flip(3)? yes, move forward', vertebrae3d.shape)
    
    all_fx_g = torch.from_numpy(verse_all['all_fx_g']).float().cuda()
    all_fx_g_ = all_fx_g[all_fx_g.isnan()==False]
    all_fx_g_ = all_fx_g_.long().cuda()

    ##remove class 4
    vertebrae3d = vertebrae3d[all_fx_g_!=4]
    all_fx_g_ = all_fx_g_[all_fx_g_!=4]
    if no_mild:
        ##remove class 1
        vertebrae3d = vertebrae3d[all_fx_g_!=1]
        all_fx_g_ = all_fx_g_[all_fx_g_!=1].cuda()

     ##data_ratio
    if ratio:
        vertebrae3d, all_fx_g_ = generate_data_split(path, data_ratio, no_mild)
        writer_n = 'runs_mlp/data_split_dgcnn2end_r' + str(data_ratio)
        writer = SummaryWriter(writer_n)
        num_iterations = 6000
    else:
        writer = SummaryWriter('runs_mlp/dgcnn2end')

    len_data = all_fx_g_.shape[0]
    print(len_data)
    num_p = 1920*2

    writer = SummaryWriter('runs_mlp/e2e_graph')

    points_all, too_small = point_sampling(vertebrae3d.squeeze(1), num_p, fps=True)
    data_all = np.delete(vertebrae3d.cpu(), too_small, axis=0)

    len_data = points_all.shape[0]

    print('alarm new len_data incoming: ', len_data)
    
    enc.eval()
    enc.cuda()
        
    optimizer = torch.optim.Adam(mlp.parameters(),lr=0.0001)
    B=16
    #B_acc = 4
    run_loss = torch.zeros(num_iterations)#*B_acc
    t0 = time.time()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,num_iterations//3,2)
    scaler = torch.cuda.amp.GradScaler()
    smooth1 = nn.Sequential(nn.AvgPool3d(3,stride=1,padding=1)).cuda()
    _,_,D,H,W = data_all.shape


    print('starting training', num_iterations)
    with tqdm(total=num_iterations, file=sys.stdout) as pbar:
        t_run = time.time()

        for i in range(num_iterations):
            idx = torch.randperm(len_data)[:B]
            #input0 = data_all[idx].cuda()

            target = all_fx_g_[idx].long()
            if no_mild: 
                target[target == 1] = 0
            else:
                target[target == 1] = 1
            target[target == 2] = 1
            target[target == 3] = 1
            target[target > 3] = 0
            #optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                in_vert3d = torch.sigmoid(smooth1(data_all[idx].cuda())*5)*2-1
                in_vert3d_unsmoothed = torch.sigmoid(data_all[idx].cuda())*2-1
                A = (torch.eye(3,4).unsqueeze(0)+.025*torch.randn(B,3,4)).cuda()
                #target = F.grid_sample(in_vert3d,F.affine_grid(A,(B,1,D,H,W)), align_corners=False)
                in_vert3d_unsmoothed = F.grid_sample(in_vert3d_unsmoothed,F.affine_grid(A,(B,1,D,H,W)), align_corners=False)
                points_in, too_small = point_sampling(in_vert3d_unsmoothed.squeeze(1), num_p, fps=False)#[0].permute(0,2,1)
                
                points_in = points_in.squeeze(1).cuda()

                q = enc(points_in,k)
                
                output = mlp(q.view(B,-1,1))

            loss = nn.CrossEntropyLoss()(output, target.view(B,1).long())

            run_loss[i] = loss.item()

 
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            
            writer.add_scalar('loss/no_mild/train/', run_loss[i], i)


            if i % 1000 == 0 and i != 0:
                print("loss: ", run_loss[i-15:i].mean())


        #torch.save(run_loss, 'run_loss_' + save_to +'_mlp.pth')
        torch.save([enc, mlp], 'run_checkpoint' + save_to + str(seed) +'_encmlp.pth')
        t_run2 = time.time()
        
    return




def train_Pointnet_mlp_point2fx(path, enc, mlp, config, save_to, data_ratio=1, no_mild = False, ratio = False):
    num_iterations = 6000
    seed = torch.random.seed()
    print(seed)

    ##load data
    print('loading data')
    verse_all = np.load(path, allow_pickle=True)['arr_0'].item(0)
    vertebrae_all = verse_all['vertebrae_all']
    vertebrae3d = torch.from_numpy(vertebrae_all).flip(3).cuda()
    print('hello this is important, flip(3)? yes, move forward', vertebrae3d.shape)
    
    all_fx_g = torch.from_numpy(verse_all['all_fx_g']).float().cuda()
    all_fx_g_ = all_fx_g[all_fx_g.isnan()==False]
    all_fx_g_ = all_fx_g_.long().cuda()

    ##remove class 4
    vertebrae3d = vertebrae3d[all_fx_g_!=4]
    all_fx_g_ = all_fx_g_[all_fx_g_!=4]
    if no_mild:
        ##remove class 1
        vertebrae3d = vertebrae3d[all_fx_g_!=1]
        all_fx_g_ = all_fx_g_[all_fx_g_!=1].cuda()

    ##data_ratio
    if ratio:
        vertebrae3d, all_fx_g_ = generate_data_split(path, data_ratio, no_mild)
        writer_n = 'runs_mlp/data_split_point2end_r' + str(data_ratio)
        writer = SummaryWriter(writer_n)
        num_iterations = 6000
    else:
        writer = SummaryWriter('runs_mlp/point2end')

    len_data = all_fx_g_.shape[0]
    print(len_data)
    num_p = 1920*2

    writer = SummaryWriter('runs_mlp/e2e_point')

    points_all, too_small = point_sampling(vertebrae3d.squeeze(1), num_p, fps=True)
    data_all = np.delete(vertebrae3d.cpu(), too_small, axis=0)

    len_data = points_all.shape[0]

    print('alarm new len_data incoming: ', len_data)
    
    enc.eval()
    enc.cuda()
        
    optimizer = torch.optim.Adam(mlp.parameters(),lr=0.0001)
    B=16
    #B_acc = 4
    run_loss = torch.zeros(num_iterations)#*B_acc
    t0 = time.time()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,num_iterations//3,2)
    scaler = torch.cuda.amp.GradScaler()
    smooth1 = nn.Sequential(nn.AvgPool3d(3,stride=1,padding=1)).cuda()
    _,_,D,H,W = data_all.shape


    print('starting training', num_iterations)
    with tqdm(total=num_iterations, file=sys.stdout) as pbar:
        t_run = time.time()

        for i in range(num_iterations):
            idx = torch.randperm(len_data)[:B]
            #input0 = data_all[idx].cuda()

            target = all_fx_g_[idx].long()
            if no_mild: 
                target[target == 1] = 0
            else:
                target[target == 1] = 1
            target[target == 2] = 1
            target[target == 3] = 1
            target[target > 3] = 0
            #optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                in_vert3d = torch.sigmoid(smooth1(data_all[idx].cuda())*5)*2-1
                in_vert3d_unsmoothed = torch.sigmoid(data_all[idx].cuda())*2-1
                A = (torch.eye(3,4).unsqueeze(0)+.025*torch.randn(B,3,4)).cuda()
                #target = F.grid_sample(in_vert3d,F.affine_grid(A,(B,1,D,H,W)), align_corners=False)
                in_vert3d_unsmoothed = F.grid_sample(in_vert3d_unsmoothed,F.affine_grid(A,(B,1,D,H,W)), align_corners=False)
                points_in, too_small = point_sampling(in_vert3d_unsmoothed.squeeze(1), num_p, fps=False)#[0].permute(0,2,1)
                
                points_in = points_in.squeeze(1).cuda().permute(0,2,1)

                q = enc(points_in)
                
                output = mlp(q.view(B,-1,1))

            loss = nn.CrossEntropyLoss()(output, target.view(B,1).long())

            run_loss[i] = loss.item()

 
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            
            writer.add_scalar('loss/no_mild/train/', run_loss[i], i)


            if i % 1000 == 0 and i != 0:
                print("loss: ", run_loss[i-15:i].mean())


        #torch.save(run_loss, 'run_loss_' + save_to +'_mlp.pth')
        torch.save([enc, mlp], 'run_checkpoint' + save_to + str(seed) +'_encmlp.pth')
        t_run2 = time.time()
        
    return


