
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import sys
import numpy as np
from torch.utils.checkpoint import checkpoint
from tqdm.notebook import tqdm,trange

from lossfunctions import ChamferLoss

from .training_mlp_datasplit import generate_data_split


from torch.utils.tensorboard import SummaryWriter


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


def train_imagevox2surf(data_path_img, data_path_surf, encoder, decoder, config, save_to):
    print('loading data: this may take a while')
    data_all_img = np.load(data_path_img, allow_pickle=True)['a'].item(0)
    data_all_surf = np.load(data_path_surf, allow_pickle=True)['a'].item(0)
    print('done loading data done')
    num_runs = 1#config['num_runs']
    t0 = time.time()
    len_data = data_all_surf.shape[0]
    len_data_img = data_all_img.shape[0]
    print('len_data', len_data, len_data_img)
    num_p = 1920*2
    writer = SummaryWriter('runs/imgvox2imgsurf')

    #remove vertebrae that do not fulfill num_p requirement
    points_all, too_small = point_sampling(data_all_surf, num_p, fps=False)
    data_all_surf = np.delete(data_all_surf, too_small, axis=0)
    data_all_img = np.delete(data_all_img, too_small, axis=0)
    print(points_all.shape, data_all_img.shape)
    len_data = points_all.shape[0]
    print('removed tiny vertebrae, ', points_all.shape[0])


    #split train and val
    len_data_train = int(len_data*.85)
    len_data_val = len_data - len_data_train
    print('training with train and val split', len_data_train, len_data_val)


    encoder.train()
    decoder.train()
    encoder.cuda()
    decoder.cuda() 
    print('start training')
    for run in trange(num_runs):
        print('training run: ' , run)
        t_run = time.time()
        iterations = config['its_per_run']
        run_loss = torch.zeros(iterations)
        run_loss_val = torch.zeros(iterations)
        optimizer = torch.optim.Adam(list(encoder.parameters())+list(decoder.parameters()),lr=config['learning_rate']) 
        scaler = torch.cuda.amp.GradScaler()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,\
                                                                         config['cosine_annealing_T_0'],\
                                                                            config['cosine_annealing_T_mult'])
        B = config['batch_size']

        D,H,W = config['patch_size']

        smooth1 = nn.Sequential(nn.AvgPool3d(3,stride=1,padding=1)).cuda()

        for i in range(iterations):
            optimizer.zero_grad()
            #train
            idx_ts = torch.randperm(len_data_train)[:B]
            with torch.cuda.amp.autocast():
                in_vert3d = torch.sigmoid(torch.from_numpy(data_all_img[idx_ts]).cuda().unsqueeze(1))*2-1 # no smoothing
                in_surf3d = torch.sigmoid(smooth1(torch.from_numpy(data_all_surf[idx_ts]).cuda().unsqueeze(1))*5)*2-1
            
                A = (torch.eye(3,4).unsqueeze(0)+.025*torch.randn(B,3,4)).cuda()
                vertebrae_in = F.grid_sample(in_vert3d,F.affine_grid(A,(B,1,D,H,W)), align_corners=False)
                vertebrae_surf = F.grid_sample(in_surf3d,F.affine_grid(A,(B,1,D,H,W)), align_corners=False)
                
                if len(too_small) > 0:
                    print(too_small)
                    for s in range(len(too_small)):
                        s_ind = too_small[s]
                        #points_in = torch.cat([points_in[:s], points_in[s+1:]],0)
                        vertebrae_in = torch.cat([vertebrae_in[:s], vertebrae_in[s+1:]],0)
                        vertebrae_surf = torch.cat([vertebrae_surf[:s], vertebrae_surf[s+1:]],0)

                z = encoder(vertebrae_in)
                reco = decoder(z)
                loss = nn.SmoothL1Loss()(vertebrae_surf*4, reco*4)
                #loss = ChamferLoss()(reco, points_in.permute(0,2,1))#nn.SmoothL1Loss()(vertebrae_in*4,reco*4)
                #run_loss[i] = loss.item()
            #val
            idx_ts = torch.randperm(len_data_val)[:B]
            with torch.cuda.amp.autocast():
                encoder.eval()
                decoder.eval()
                with torch.no_grad():
                    in_vert3d = torch.sigmoid(torch.from_numpy(data_all_img[idx_ts]).cuda().unsqueeze(1))*2-1 # no smoothing
                in_surf3d = torch.sigmoid(smooth1(torch.from_numpy(data_all_surf[idx_ts]).cuda().unsqueeze(1))*5)*2-1
            
                A = (torch.eye(3,4).unsqueeze(0)+.025*torch.randn(B,3,4)).cuda()
                vertebrae_in = F.grid_sample(in_vert3d,F.affine_grid(A,(B,1,D,H,W)), align_corners=False)
                vertebrae_surf = F.grid_sample(in_surf3d,F.affine_grid(A,(B,1,D,H,W)), align_corners=False)
                
                if len(too_small) > 0:
                    print(too_small)
                    for s in range(len(too_small)):
                        s_ind = too_small[s]
                        #points_in = torch.cat([points_in[:s], points_in[s+1:]],0)
                        vertebrae_in = torch.cat([vertebrae_in[:s], vertebrae_in[s+1:]],0)
                        vertebrae_surf = torch.cat([vertebrae_surf[:s], vertebrae_surf[s+1:]],0)

                z = encoder(vertebrae_in)
                reco = decoder(z)
    
                loss_val = nn.SmoothL1Loss()(vertebrae_surf*4,reco*4)
                #run_loss_val[i] = loss_val.item()



            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            run_loss[i] = 100*loss.item()
            run_loss_val[i] = 100*loss_val.item()

            writer.add_scalar('loss/smoothl1/train', run_loss[i], i)
            writer.add_scalar('loss/smoothl1/val', run_loss_val[i], i)

            if i < 10:
                print("loss: ", run_loss[i], "loss_val: ", run_loss_val[i])


            if i % 3000 == 0 and i != 0:
                print("loss: ", run_loss[i-15:i].mean())
                print("saving checkpoint")
                #torch.save(run_loss, 'run_loss_' + save_to + f'{run}'+'.pth')
                torch.save([encoder, decoder], 'run_checkpoint'+ f'{run}'  + save_to + '_' + f'{i}' + '.pth')

        t_run2 = time.time()
        #torch.save(run_loss, 'final_run_loss_'+f'{run}' + save_to + '.pth')
        torch.save([encoder, decoder], 'final_checkpoint'+ f'{run}' + save_to + '.pth')
        print('runtime for run:', t_run2-t_run)

