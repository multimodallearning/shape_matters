
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


def train_image2point(data_path, encoder, decoder, config, save_to):
    print('loading data: this may take a while')
    data_all = np.load(data_path, allow_pickle=False)['a']
    seed = torch.random.seed()
    print(seed)

    print('done loading data')
    num_runs = 1#config['num_runs']
    t0 = time.time()
    len_data = data_all.shape[0]
    num_p = 1920*2
    writer = SummaryWriter('runs/img2point')

    #remove vertebrae that do not fulfill num_p requirement
    points_all, too_small = point_sampling(data_all, num_p, fps=False)
    data_all = np.delete(data_all, too_small, axis=0)
    print(points_all.shape, data_all.shape)
    len_data = points_all.shape[0]


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
                in_vert3d = torch.sigmoid(smooth1(torch.from_numpy(data_all[idx_ts]).cuda().unsqueeze(1))*5)*2-1
                A = (torch.eye(3,4).unsqueeze(0)+.025*torch.randn(B,3,4)).cuda()
                vertebrae_in = F.grid_sample(in_vert3d,F.affine_grid(A,(B,1,D,H,W)), align_corners=False)
                
                in_vert3d = torch.sigmoid(smooth1(torch.from_numpy(data_all[idx_ts]).cuda().unsqueeze(1))*5)*2-1
                in_vert3d_unsmoothed = torch.sigmoid(torch.from_numpy(data_all[idx_ts]).cuda().unsqueeze(1))*2-1
                A = (torch.eye(3,4).unsqueeze(0)+.025*torch.randn(B,3,4)).cuda()
                vertebrae_in = F.grid_sample(in_vert3d,F.affine_grid(A,(B,1,D,H,W)), align_corners=False)
                in_vert3d_unsmoothed = F.grid_sample(in_vert3d_unsmoothed,F.affine_grid(A,(B,1,D,H,W)), align_corners=False)
                points_in, too_small = point_sampling(in_vert3d_unsmoothed.squeeze(1), num_p, fps=False)
                if len(too_small) > 0:
                    print(too_small)
                    for s in range(len(too_small)):
                        s_ind = too_small[s]
                        #points_in = torch.cat([points_in[:s], points_in[s+1:]],0)
                        vertebrae_in = torch.cat([vertebrae_in[:s], vertebrae_in[s+1:]],0)

                z = encoder(vertebrae_in)
                reco = decoder(z)

                loss = ChamferLoss()(reco, points_in.permute(0,2,1))#nn.SmoothL1Loss()(vertebrae_in*4,reco*4)
                #run_loss[i] = loss.item()
            #val
            idx_ts = torch.randperm(len_data_val)[:B]
            with torch.cuda.amp.autocast():
                encoder.eval()
                decoder.eval()
                with torch.no_grad():
                    in_vert3d = torch.sigmoid(smooth1(torch.from_numpy(data_all[idx_ts]).cuda().unsqueeze(1))*5)*2-1
                    A = (torch.eye(3,4).unsqueeze(0)+.025*torch.randn(B,3,4)).cuda()
                    vertebrae_in = F.grid_sample(in_vert3d,F.affine_grid(A,(B,1,D,H,W)), align_corners=False)
                    
                    in_vert3d = torch.sigmoid(smooth1(torch.from_numpy(data_all[idx_ts]).cuda().unsqueeze(1))*5)*2-1
                    in_vert3d_unsmoothed = torch.sigmoid(torch.from_numpy(data_all[idx_ts]).cuda().unsqueeze(1))*2-1
                    A = (torch.eye(3,4).unsqueeze(0)+.025*torch.randn(B,3,4)).cuda()
                    vertebrae_in = F.grid_sample(in_vert3d,F.affine_grid(A,(B,1,D,H,W)), align_corners=False)
                    in_vert3d_unsmoothed = F.grid_sample(in_vert3d_unsmoothed,F.affine_grid(A,(B,1,D,H,W)), align_corners=False)
                    points_in, too_small = point_sampling(in_vert3d_unsmoothed.squeeze(1), num_p, fps=False)
                    if len(too_small) > 0:
                        print(too_small)
                        for s in range(len(too_small)):
                            s_ind = too_small[s]
                            #points_in = torch.cat([points_in[:s], points_in[s+1:]],0)
                            vertebrae_in = torch.cat([vertebrae_in[:s], vertebrae_in[s+1:]],0)
                            print(vertebrae_in.shape, points_in.shape)


                    z = encoder(vertebrae_in)
                    reco = decoder(z)

                loss_val = ChamferLoss()(reco, points_in.permute(0,2,1))#nn.SmoothL1Loss()(vertebrae_in*4,reco*4)
                #run_loss_val[i] = loss_val.item()



            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            run_loss[i] = 100*loss.item()
            run_loss_val[i] = 100*loss_val.item()

            writer.add_scalar('loss/chamfer/train', run_loss[i], i)
            writer.add_scalar('loss/chamfer/val', run_loss_val[i], i)

            if i < 10:
                print("loss: ", run_loss[i], "loss_val: ", run_loss_val[i])


            if i % 3000 == 0 and i != 0:
                print("loss: ", run_loss[i-15:i].mean())
                print("saving checkpoint")
                #torch.save(run_loss, 'run_loss_' + save_to + f'{run}'+'.pth')
                torch.save([encoder, decoder], 'run_checkpoint'+ f'{run}'  + save_to + '_' + f'{i}' + '.pth')

        t_run2 = time.time()
        #torch.save(run_loss, 'final_run_loss_'+f'{run}' + save_to + '.pth')
        torch.save([encoder, decoder], 'final_checkpoint'+ f'{run}' + save_to + str(seed) + '.pth')
        print('runtime for run:', t_run2-t_run)


def train_image2image(data_path, encoder, decoder, config, save_to):

    print('loading data: this may take a while')
    data_all = np.load(data_path, allow_pickle=False)['a']
    
    print('loading verse dataset as well')
    v_path = '/home/hempe/storage/staff/hellenahempe/code/Notebooks/Segmentation/MICCAI/PointAutoEncoder/data/verse19_dataset_sparse.npz'
    verse_all = np.load(v_path, allow_pickle=True)['arr_0'].item(0)
    vertebrae_all = verse_all['vertebrae_all']
    vertebrae_val = verse_all['vertebrae_val']
    vertebrae3d = torch.from_numpy(vertebrae_all).flip(3)
    vertebrae3d_val = torch.from_numpy(vertebrae_val).flip(3)
    
    vertebrae3d = torch.cat([vertebrae3d, vertebrae3d_val],0)
    
    data_all = torch.cat([vertebrae3d.squeeze(1),torch.from_numpy(data_all)],0).numpy()
    data_all = vertebrae3d.squeeze(1).numpy()
    print('done loading data')
    num_runs = 1#config['num_runs']
    t0 = time.time()
    len_data = data_all.shape[0]
    num_p = 1920*2

    #remove vertebrae that do not fulfill num_p requirement
    points_all, too_small = point_sampling(data_all, num_p, fps=False)
    data_all = np.delete(data_all, too_small, axis=0)
    print(points_all.shape, data_all.shape)
    len_data = points_all.shape[0]
    
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
            idx_ts = torch.randperm(len_data)[:B]
            with torch.cuda.amp.autocast():
                in_vert3d = torch.sigmoid(smooth1(torch.from_numpy(data_all[idx_ts]).cuda().unsqueeze(1))*5)*2-1
                A = (torch.eye(3,4).unsqueeze(0)+.025*torch.randn(B,3,4)).cuda()
                vertebrae_in = F.grid_sample(in_vert3d,F.affine_grid(A,(B,1,D,H,W)), align_corners=False)
                
                z = encoder(vertebrae_in)
                reco = decoder(z)

                loss = nn.SmoothL1Loss()(vertebrae_in*4,reco*4)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            run_loss[i] = 100*loss.item()

            if i % 3000 == 0 and i != 0:
                print("loss: ", run_loss[i-15:i].mean())
                print("saving checkpoint")
                torch.save(run_loss, 'run_loss_' + save_to + f'{run}'+'.pth')
                torch.save([encoder, decoder], 'run_checkpoint'+ f'{run}'  + save_to + '_' + f'{i}' + '.pth')

        t_run2 = time.time()
        torch.save(run_loss, 'final_run_loss_'+f'{run}' + save_to + '.pth')
        torch.save([encoder, decoder], 'final_checkpoint'+ f'{run}' + save_to + '.pth')
        print('runtime for run:', t_run2-t_run)


def train_mlp_image2fx(path, enc, mlp, config, save_to, data_ratio=1, no_mild = False, ratio=False):
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
        #num_iterations = 6000
    else:
        writer = SummaryWriter('runs_mlp/img2')


    ###class weights
    bincount_temp = torch.zeros_like(all_fx_g_)
    print(all_fx_g_.shape, vertebrae3d.shape)
    bincount_temp[all_fx_g_ > 0] = 1
    print(torch.bincount(bincount_temp))
    weights = 1/torch.sqrt(torch.bincount(bincount_temp))
    print('WEIGHTS = ', weights)


    len_data = all_fx_g_.shape[0]
    print('len_data', len_data)
    num_p = 1920*2

    points_all, too_small = point_sampling(vertebrae3d.squeeze(1), num_p, fps=True)
    data_all = np.delete(vertebrae3d.cpu().numpy(), too_small, axis=0)

    len_data = points_all.shape[0]
    print(len_data)
    
    enc.eval()
    enc.cuda()
        
    optimizer = torch.optim.Adam(mlp.parameters(),lr=0.0001)
    run_loss = torch.zeros(num_iterations)
    t0 = time.time()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,num_iterations//3,2)
    scaler = torch.cuda.amp.GradScaler()

    B = 16

    print('starting training', num_iterations)
    with tqdm(total=num_iterations, file=sys.stdout) as pbar:

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


            with torch.cuda.amp.autocast():
                with torch.no_grad():
                        
                    affine_matrix = (0.07*torch.randn(B,3,4)+torch.eye(3,4).unsqueeze(0)).cuda()
                    augment_grid = F.affine_grid(affine_matrix,(B,1,96,64,80),align_corners=False)
                    vertebrae_in0 = F.grid_sample(input0,augment_grid,align_corners=False)
                    #print(vertebrae_in0.shape, target.shape, 'in loop')

                    q = enc(vertebrae_in0)

                output = mlp(q.view(B,-1,1))

                #print(target.shape, output.shape)
                #loss = nn.BCEWithLogitsLoss(weight=weights.cuda())(output.view(B,2), F.one_hot(target.view(B,1),2).view(B,2).float())
                loss = nn.CrossEntropyLoss()(output, target.view(B,1).long().cuda())
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
        torch.save(mlp, 'run_checkpoint' + save_to + str(seed)+'_mlp.pth')
        t_run2 = time.time()
        
    return
