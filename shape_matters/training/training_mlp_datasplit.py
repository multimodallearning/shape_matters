import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import sys
import numpy as np
from torch.utils.checkpoint import checkpoint
from tqdm.notebook import tqdm,trange


def generate_data_split(path, ratio = 1., no_mild = True):
    print('loading data --- using training and test data')
    verse_all = np.load(path, allow_pickle=True)['arr_0'].item(0)
    vertebrae_all = np.concatenate([verse_all['vertebrae_all'],verse_all['vertebrae_val']],0)
    vertebrae3d = torch.from_numpy(vertebrae_all)
    
    all_fx_g = torch.cat([torch.from_numpy(verse_all['all_fx_g']).float(), torch.from_numpy(verse_all['val_fx_g']).float()],0)
    all_fx_g_ = all_fx_g[all_fx_g.isnan()==False]
    all_fx_g_ = all_fx_g_.long()

    ##remove class 4
    vertebrae3d = vertebrae3d[all_fx_g_!=4]
    all_fx_g_ = all_fx_g_[all_fx_g_!=4]

    ##remove class 1
    if no_mild:
        vertebrae3d = vertebrae3d[all_fx_g_!=1]
        all_fx_g_ = all_fx_g_[all_fx_g_!=1]

    len_all = len(vertebrae3d)

    vertebrae3d_healthy = vertebrae3d[all_fx_g_==0]
    vertebrae3d_fx  = vertebrae3d[all_fx_g_>0]

    fx_g_0 = all_fx_g_[all_fx_g_==0]
    fx_g_1 = all_fx_g_[all_fx_g_>0]

    #len_subset = int(len_all * ratio)
    len_subset_healthy = int(len(vertebrae3d_healthy)*ratio)
    len_subset_fx = int(len(vertebrae3d_fx)*ratio)

    idx_h = torch.randperm(len(vertebrae3d_healthy))[:len_subset_healthy]
    idx_fx = torch.randperm(len(vertebrae3d_fx))[:len_subset_fx]

    print("created new subset: healthy - ", len_subset_healthy, " and fractured - ", len_subset_fx, "with ratio - ", ratio)

    vertebrae_subset = torch.cat((vertebrae3d_healthy[idx_h],vertebrae3d_fx[idx_fx]),0)
    fx_g_subset = torch.cat((fx_g_0[idx_h],fx_g_1[idx_fx]),0)
    print(len(vertebrae_subset))

    return vertebrae_subset.cuda(), fx_g_subset.cuda()

''' #deprecated
def train_mlp_bin_fx_ratio(path, enc, mlp, ratio, config):
    num_iterations = int(12000 * ratio)

    ##load data and apply ratio
    vertebrae3d, all_fx_g_ = generate_data_split(path, ratio)

    len_data = vertebrae3d.shape[0]
    
        
    optimizer = torch.optim.Adam(mlp.parameters(),lr=0.0001)
    run_loss = torch.zeros(num_iterations)
    t0 = time.time()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,num_iterations//3,2)
    scaler = torch.cuda.amp.GradScaler()

    B=64
    print('starting training')
    with tqdm(total=num_iterations, file=sys.stdout) as pbar:
        t_run = time.time()

        for i in range(num_iterations):

            idx = torch.randperm(len_data)[:B]
            input0 = vertebrae3d[idx]
            target = all_fx_g_[idx].long()
            target[target == 1] = 1
            target[target == 2] = 1.#0.75
            target[target == 3] = 1.
            target[target > 3] = 0.5
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    
                    affine_matrix = (0.07*torch.randn(B,3,4)+torch.eye(3,4).unsqueeze(0)).cuda()
                    augment_grid = F.affine_grid(affine_matrix,(B,1,96,64,80),align_corners=False)
                    vertebrae_in0 = F.grid_sample(input0,augment_grid,align_corners=False)

                    q = enc(vertebrae_in0)
                    

                output = mlp(q.view(B,-1,1))
                #print(output.shape)
                #loss_f = nn.CrossEntropyLoss()(output[:,:5],coeff[target])#target2) 
                #loss_l = nn.CrossEntropyLoss()(output[:,5:7],(target>0).long())

                loss = nn.CrossEntropyLoss()(output, target.view(B,1).long())
                

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            run_loss[i] = loss.item()

        torch.save(run_loss, 'run_loss_'+f'{i}' + '_mlp_2.pth')
        torch.save(mlp, 'run_checkpoint'+ f'{i}' + '_mlp_2.pth')
        t_run2 = time.time()
        print('runtime for run:', t_run2-t_run)    

    return

def train_mlp_bin_fx_varenc(path, enc, mlp, config):
    num_iterations = 12000

    ##load data
    print('loading data')
    verse_all = np.load(path, allow_pickle=True)['arr_0'].item(0)
    vertebrae_all = verse_all['vertebrae_all']
    vertebrae3d = torch.from_numpy(vertebrae_all).cuda()
    
    all_fx_g = torch.from_numpy(verse_all['all_fx_g']).float().cuda()
    all_fx_g_ = all_fx_g[all_fx_g.isnan()==False]
    all_fx_g_ = all_fx_g_.long().cuda()
    
    ##remove class 4
    vertebrae3d = vertebrae3d[all_fx_g_!=4]
    all_fx_g_ = all_fx_g_[all_fx_g_!=4]
        
    optimizer = torch.optim.Adam(mlp.parameters(),lr=0.001)
    run_loss = torch.zeros(num_iterations)
    t0 = time.time()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,num_iterations//3,2)
    scaler = torch.cuda.amp.GradScaler()

    B=64
    print('starting training')
    with tqdm(total=num_iterations, file=sys.stdout) as pbar:
        t_run = time.time()

        for i in range(num_iterations):

            idx = torch.randperm(833)[:B]
            input0 = vertebrae3d[idx]
            target = all_fx_g_[idx].long()
            target[target == 1] = 1
            target[target == 2] = 1
            target[target == 3] = 1
            target[target > 3] = 0.5
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    
                    affine_matrix = (0.07*torch.randn(B,3,4)+torch.eye(3,4).unsqueeze(0)).cuda()
                    augment_grid = F.affine_grid(affine_matrix,(B,1,96,64,80),align_corners=False)
                    vertebrae_in0 = F.grid_sample(input0,augment_grid,align_corners=False)

                    q, kl = enc(vertebrae_in0)
                    

                output = mlp(q.view(B,-1,1))
                #print(output.shape)
                #loss_f = nn.CrossEntropyLoss()(output[:,:5],coeff[target])#target2) 
                #loss_l = nn.CrossEntropyLoss()(output[:,5:7],(target>0).long())

                loss = nn.CrossEntropyLoss()(output, target.view(B,1).long())
                #print(kl)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            run_loss[i] = loss.item()

        torch.save(run_loss, 'run_loss_'+f'{i}' + '_mlp.pth')
        torch.save(mlp, 'run_checkpoint'+ f'{i}' + '_mlp.pth')
        t_run2 = time.time()
        print('runtime for run:', t_run2-t_run)    

    return



def train_mlp_bin_fx_pbin(path, enc, mlp, config):
    num_iterations = 12000

    ##load data
    print('loading data')
    verse_all = np.load(path, allow_pickle=True)['arr_0'].item(0)
    vertebrae_all = verse_all['vertebrae_all']
    vertebrae3d = torch.from_numpy(vertebrae_all).cuda()
    
    all_fx_g = torch.from_numpy(verse_all['all_fx_g']).float().cuda()
    all_fx_g_ = all_fx_g[all_fx_g.isnan()==False]
    all_fx_g_ = all_fx_g_.long().cuda()

    ##remove class 4
    vertebrae3d = vertebrae3d[all_fx_g_!=4]
    all_fx_g_ = all_fx_g_[all_fx_g_!=4]


    len_data = all_fx_g_.shape[0]
    print(len_data)
    num_p = 1920*2

    print('generating pointset and removing smaller vertebrae below number of points threshold')
    points_all = []
    too_small  = []
    for j in range(len_data):
        mesh = F.affine_grid(torch.eye(3,4).unsqueeze(0).cuda(),(1,1,96,64,80)).reshape(-1,3)
        pts = mesh[vertebrae3d[j].reshape(-1)>.99,:]
        ids = torch.cat((torch.randperm(pts.shape[0]),torch.randperm(pts.shape[0]),torch.randperm(pts.shape[0])),0)[:num_p]
        if pts.shape[0] > num_p:
            points_all.append(pts[ids])
        else: 
            too_small.append(j)

    print('remaining vertebrae: ',len_data, 'deleted: ', len(too_small))
    points_all = torch.stack(points_all,0).cuda()
    
    enc.eval()
    enc.cuda()
        
    optimizer = torch.optim.Adam(mlp.parameters(),lr=0.001)
    run_loss = torch.zeros(num_iterations)
    t0 = time.time()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,num_iterations//3,2)
    scaler = torch.cuda.amp.GradScaler()

    B=64

    print('starting training')
    with tqdm(total=num_iterations, file=sys.stdout) as pbar:
        t_run = time.time()

        for i in range(num_iterations):

            idx = torch.randperm(829)[:B]

            input0 = vertebrae3d[idx]
            target = all_fx_g_[idx].long()
            target[target == 1] = 1
            target[target == 2] = 1
            target[target == 3] = 1
            target[target > 3] = 0.5
            optimizer.zero_grad()

            #with torch.cuda.amp.autocast():
            with torch.no_grad():

                points_in = points_all[idx].cuda().unsqueeze(1).view(B,3,-1)

                q = enc(points_in)


            output = mlp(q.view(B,-1,1))

            #print(output.shape)
            #loss_f = nn.CrossEntropyLoss()(output[:,:5],coeff[target])#target2) 
            #loss_l = nn.CrossEntropyLoss()(output[:,5:7],(target>0).long())

            loss = nn.CrossEntropyLoss()(output, target.view(B,1).long())
                

            #scaler.scale(loss).backward()
            loss.backward()
            #scaler.step(optimizer)
            optimizer.step()
            #scaler.update()
            scheduler.step()

            run_loss[i] = loss.item()

        torch.save(run_loss, 'run_loss_'+f'{i}' + '_mlp.pth')
        torch.save(mlp, 'run_checkpoint'+ f'{i}' + '_mlp.pth')
        t_run2 = time.time()
        
    return

'''