
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

from lossfunctions import ChamferLoss#, WassersteinLoss

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


def train_point2image(data_path, encoder, decoder, num_p,config, save_to):

    print('loading data: this may take a while')
    data_all = np.load(data_path, allow_pickle=False)['a']
    print('done loading data')
    num_runs = 1#config['num_runs']
    t0 = time.time()
    len_data = data_all.shape[0]
    #num_p = 1920*2



    points_all, too_small = point_sampling(data_all, num_p, fps=False)
    #data_all = np.delete(data_all, too_small, axis=0)
    print(points_all.shape, data_all.shape)
    len_data = points_all.shape[0]

    #split train and val
    len_data_train = int(len_data*.85)
    len_data_val = len_data - len_data_train
    print('training with train and val split', len_data_train, len_data_val)

    writer = SummaryWriter('runs/point2image')
    
    encoder.train()
    decoder.train()
    encoder.cuda()
    decoder.cuda() 
    print('start training')
    for run in trange(num_runs):
        print('training run: ' , run)
        t_run = time.time()
        B = config['batch_size']
        B_acc = 4
        B = int(B//B_acc)
        D,H,W = config['patch_size']
        iterations = config['its_per_run'] * B_acc
        run_loss = torch.zeros(iterations)
        run_loss_val = torch.zeros(iterations)
        optimizer = torch.optim.Adam(list(encoder.parameters())+list(decoder.parameters()),lr=config['learning_rate']) 
        scaler = torch.cuda.amp.GradScaler()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,\
                                                                         config['cosine_annealing_T_0'],\
                                                                            config['cosine_annealing_T_mult'])
    

        smooth1 = nn.Sequential(nn.AvgPool3d(3,stride=1,padding=1)).cuda()

        for i in range(iterations):
            #training
            idx_ts = torch.randperm(len_data_train)[:B]
            with torch.cuda.amp.autocast():
                in_vert3d = torch.sigmoid(smooth1(torch.from_numpy(data_all[idx_ts]).cuda().unsqueeze(1))*5)*2-1
                in_vert3d_unsmoothed = torch.sigmoid(torch.from_numpy(data_all[idx_ts]).cuda().unsqueeze(1))*2-1
                A = (torch.eye(3,4).unsqueeze(0)+.025*torch.randn(B,3,4)).cuda()
                target = F.grid_sample(in_vert3d,F.affine_grid(A,(B,1,D,H,W)), align_corners=False)
                in_vert3d_unsmoothed = F.grid_sample(in_vert3d_unsmoothed,F.affine_grid(A,(B,1,D,H,W)), align_corners=False)
                points_in, too_small = point_sampling(in_vert3d_unsmoothed.squeeze(1), num_p, fps=False)#[0].permute(0,2,1)
                points_in = points_in.permute(0,2,1)
                x = encoder(points_in)
                reco = decoder(x)

                if len(too_small) > 0:
                    #print(too_small, idx_val[too_small])
                    for s in range(len(too_small)):
                        s_ind = too_small[s]
                        target = torch.cat([target[:s], target[s+1:]],0)
                loss = nn.SmoothL1Loss()(target*4,reco*4)
                loss = loss/ B_acc
                


            run_loss[i] = 100*loss.item()

            #val
            idx_val = len_data_train + torch.randperm(len_data_val)[:B] 
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    encoder.eval()
                    decoder.eval()
                    in_vert3d = torch.sigmoid(smooth1(torch.from_numpy(data_all[idx_val]).cuda().unsqueeze(1))*5)*2-1
                    in_vert3d_unsmoothed = torch.sigmoid(torch.from_numpy(data_all[idx_val]).cuda().unsqueeze(1))*2-1
                    A = (torch.eye(3,4).unsqueeze(0)+.025*torch.randn(B,3,4)).cuda()
                    target = F.grid_sample(in_vert3d,F.affine_grid(A,(B,1,D,H,W)), align_corners=False)
                    in_vert3d_unsmoothed = F.grid_sample(in_vert3d_unsmoothed,F.affine_grid(A,(B,1,D,H,W)), align_corners=False)
                    points_in, too_small = point_sampling(in_vert3d_unsmoothed.squeeze(1), num_p, fps=False)#[0].permute(0,2,1)
                    points_in = points_in.permute(0,2,1)
                    x = encoder(points_in)
                    reco = decoder(x)

                    if len(too_small) > 0:
                        #print(too_small, idx_val[too_small])
                        for s in range(len(too_small)):
                            s_ind = too_small[s]
                            target = torch.cat([target[:s], target[s+1:]],0)

                    loss_val = nn.SmoothL1Loss()(target*4,reco*4)
                    loss_val = loss_val/ B_acc

            run_loss_val[i] = 100*loss_val.item()

            if ((i + 1) % B_acc == 0) or (i + 1 == iterations):
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                if i > 4:
                    writer.add_scalar('loss/SmoothL1/train/', run_loss[i-4:i].mean(), int(i//4))
                    writer.add_scalar('loss/SmoothL1/val', run_loss_val[i-4:i].mean(), int(i//4))
                    writer.add_scalar('loss/SmoothL1/train-val', run_loss[i-4:i].mean() - run_loss_val[i-4:i].mean(), int(i//4))


            if i % 5000 == 0 and i != 0:
                print("loss: ", run_loss[i-15:i].mean())
                print("saving checkpoint")
                #torch.save(run_loss, 'run_loss_' + save_to + f'{run}'+'.pth')
                torch.save([encoder, decoder], 'run_checkpoint'+ f'{run}'  + save_to + '_' + f'{i}' + '.pth')

        t_run2 = time.time()
        #torch.save(run_loss, 'run_loss_'+f'{run}' + save_to + '.pth')
        torch.save([encoder, decoder], 'final_checkpoint'+ f'{run}' + save_to + '.pth')
        print('runtime for run:', t_run2-t_run)


def train_point2point(data_path, encoder, decoder, num_p,config, save_to):

    print('loading data: this may take a while')
    data_all = np.load(data_path, allow_pickle=False)['a']
    print('done loading data')
    num_runs = 1#config['num_runs']
    t0 = time.time()
    len_data = data_all.shape[0]
    #num_p = 1920*2

    points_all, too_small = point_sampling(data_all, num_p, fps=False)
    #data_all = np.delete(data_all, too_small, axis=0)
    print(points_all.shape, data_all.shape)
    len_data = points_all.shape[0]

    #split train and val
    len_data_train = int(len_data*.85)
    len_data_val = len_data - len_data_train
    print('training with train and val split', len_data_train, len_data_val)

    writer = SummaryWriter('runs/point2point_w')
    
    encoder.train()
    decoder.train()
    encoder.cuda()
    decoder.cuda() 
    print('start training')
    for run in trange(num_runs):
        print('training run: ' , run)
        t_run = time.time()
        B = config['batch_size']
        B_acc = 32
        B = int(B//B_acc)
        D,H,W = config['patch_size']
        iterations = config['its_per_run'] * B_acc
        run_loss = torch.zeros(iterations)
        run_loss_val = torch.zeros(iterations)
        optimizer = torch.optim.Adam(list(encoder.parameters())+list(decoder.parameters()),lr=config['learning_rate']) 
        scaler = torch.cuda.amp.GradScaler()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,\
                                                                         config['cosine_annealing_T_0'],\
                                                                            config['cosine_annealing_T_mult'])


        smooth1 = nn.Sequential(nn.AvgPool3d(3,stride=1,padding=1)).cuda()

        for i in range(iterations):
            optimizer.zero_grad()
            #train
            idx_ts = torch.randperm(len_data_train)[:B]
            encoder.train()
            decoder.train()
            #with torch.cuda.amp.autocast():
            #print(data_all[idx_ts].shape)
            in_vert3d = torch.sigmoid(smooth1(torch.from_numpy(data_all[idx_ts]).cuda().unsqueeze(1))*5)*2-1
            #print(in_vert3d.shape)
            in_vert3d_unsmoothed = torch.sigmoid(torch.from_numpy(data_all[idx_ts]).cuda().unsqueeze(1))*2-1
            A = (torch.eye(3,4).unsqueeze(0)+.025*torch.randn(B,3,4)).cuda()
            target = F.grid_sample(in_vert3d,F.affine_grid(A,(B,1,D,H,W)), align_corners=False)
            in_vert3d_unsmoothed = F.grid_sample(in_vert3d_unsmoothed,F.affine_grid(A,(B,1,D,H,W)), align_corners=False)
            points_in = point_sampling(in_vert3d.squeeze(1), num_p, fps=False)[0].permute(0,2,1)
            #TODO unsmooth pc and remove 100
            points_in.requires_grad = True

            x = encoder(points_in)
            reco = decoder(x)#.permute(0,2,1)

            loss = ChamferLoss()(reco, points_in)#nn.SmoothL1Loss()(target*4,reco*4)
            #loss = WassersteinLoss()(reco.permute(0,2,1), points_in.permute(0,2,1))
            loss = loss/ B_acc

            run_loss[i] = loss.item()

            #val
            idx_val = len_data_train + torch.randperm(len_data_val)[:B] 
            #with torch.cuda.amp.autocast():
            with torch.no_grad():
                encoder.eval()
                decoder.eval()
                in_vert3d = torch.sigmoid(smooth1(torch.from_numpy(data_all[idx_val]).cuda().unsqueeze(1))*5)*2-1
                in_vert3d_unsmoothed = torch.sigmoid(torch.from_numpy(data_all[idx_val]).cuda().unsqueeze(1))*2-1
                A = (torch.eye(3,4).unsqueeze(0)+.025*torch.randn(B,3,4)).cuda()
                in_vert3d_unsmoothed = F.grid_sample(in_vert3d,F.affine_grid(A,(B,1,D,H,W)), align_corners=False)
                points_in, too_small = point_sampling(in_vert3d.squeeze(1), num_p, fps=False)
                points_in = points_in.permute(0,2,1)

                #if len(too_small) > 0:
                #    print(too_small, idx_val[too_small])
                x = encoder(points_in)
                reco = decoder(x)#.permute(0,2,1)

                #loss_val = #WassersteinLoss()(reco.permute(0,2,1), points_in.permute(0,2,1))
                loss_val = ChamferLoss()(reco, points_in)#nn.SmoothL1Loss()(target*4,reco*4)
                loss_val = loss_val/ B_acc

            
            run_loss_val[i] = loss_val.item()

            if ((i + 1) % B_acc == 0) or (i + 1 == iterations):
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                writer.add_scalar('loss/wasserstein/train', run_loss[i-B_acc:i].mean(), int(i//B_acc))
                writer.add_scalar('loss/wasserstein/val', run_loss_val[i-B_acc:i].mean(), int(i//B_acc))

            if i % 5000 == 0 and i != 0:
                print("loss: ", run_loss[i-15:i].mean())
                print("saving checkpoint")
                #torch.save(run_loss, 'run_loss_' + save_to + f'{run}'+'.pth')
                torch.save([encoder, decoder], 'run_checkpoint_wasserstein'+ f'{run}'  + save_to + '_' + f'{i}' + '.pth')

        t_run2 = time.time()
        #torch.save(run_loss, 'run_loss_'+f'{run}' + save_to + '.pth')
        torch.save([encoder, decoder], 'final_checkpoint_wasserstein'+ f'{run}' + save_to + '.pth')
        print('runtime for run:', t_run2-t_run)





def train_mlp_point2fx(path, enc, mlp, num_p, config, save_to, data_ratio=1, no_mild = False, ratio=False):
    num_iterations = 6000
    seed = torch.random.seed()
    print(seed)

    ##load data
    print('loading data')
    verse_all = np.load(path, allow_pickle=True)['arr_0'].item(0)
    vertebrae_all = verse_all['vertebrae_all']
    vertebrae3d = torch.from_numpy(vertebrae_all).cuda().flip(3)
    
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
        writer_n = 'runs_mlp/data_split_img2pb_r' + str(data_ratio)
        writer = SummaryWriter(writer_n)
        num_iterations = 6000
    else:
        writer = SummaryWriter('runs_mlp/img2point')

    len_data = all_fx_g_.shape[0]
    print(len_data)
    #num_p = 1920*2

    points_all, too_small = point_sampling(vertebrae3d.squeeze(1), num_p, fps=False)
    data_all = np.delete(vertebrae3d.cpu(), too_small, axis=0)
    #all_fx_g_ = np.delete(all_fx_g_.cpu(), too_small, axis=0)

    len_data = points_all.shape[0]
    writer = SummaryWriter('runs_mlp/point2point')
    
    enc.eval()
    enc.cuda()
        
    optimizer = torch.optim.Adam(mlp.parameters(),lr=0.0001)
    run_loss = torch.zeros(num_iterations)
    t0 = time.time()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,num_iterations//3,2)
    scaler = torch.cuda.amp.GradScaler()
    _,_,D,H,W = data_all.shape

    smooth1 = nn.Sequential(nn.AvgPool3d(3,stride=1,padding=1)).cuda()

    B=16#64
    B_acc=1 #for gradient accumulation

    print('starting training')
    with tqdm(total=num_iterations, file=sys.stdout) as pbar:
        t_run = time.time()

        for i in range(num_iterations):

            idx = torch.randperm(len_data)[:B]

            input0 = points_all[idx]
            target_fx = all_fx_g_[idx].long()
            if no_mild: 
                target_fx[target_fx == 1] = 0
            else:
                target_fx[target_fx == 1] = 1
            target_fx[target_fx == 2] = 1
            target_fx[target_fx == 3] = 1
            target_fx[target_fx > 3] = 0
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    in_vert3d = torch.sigmoid(smooth1(data_all[idx].cuda())*5)*2-1
                    in_vert3d_unsmoothed = torch.sigmoid(data_all[idx].cuda())*2-1
                    A = (torch.eye(3,4).unsqueeze(0)+.025*torch.randn(B,3,4)).cuda()
                    in_vert3d_unsmoothed = F.grid_sample(in_vert3d_unsmoothed,F.affine_grid(A,(B,1,D,H,W)), align_corners=False)
                    points_in, too_small = point_sampling(in_vert3d_unsmoothed.squeeze(1), num_p, fps=False)#[0].permute(0,2,1)
                    
                    points_in = points_all[idx].cuda().squeeze(1).permute(0,2,1)#.view(B,3,-1)
                    #points_in = points_in.permute(0,2,1)#.view(B,3,-1)
                    q = enc(points_in)


                output = mlp(q.view(B,-1,1))

            #gradient acc
            loss = nn.CrossEntropyLoss()(output, target_fx.view(B,1).long().cuda())
            #loss = loss/ B_acc
                

            #scaler.scale(loss).backward()
            loss.backward()

            #if ((i + 1) % B_acc == 0) or (i + 1 == num_iterations):
            optimizer.step()
            scheduler.step()

            run_loss[i] = loss.item()
            writer.add_scalar('loss/all_fx/train/', run_loss[i], i)
            
            if i % 3000 == 0 and i != 0:
                print("loss: ", run_loss[i-15:i].mean())

        #torch.save(run_loss, 'run_loss_' + save_to +'_mlp.pth')
        torch.save(mlp, 'run_checkpoint' + save_to + str(seed) +'_mlp.pth')
        t_run2 = time.time()
        
    return
