

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import sys
import numpy as np
from torch.utils.checkpoint import checkpoint
from tqdm.notebook import tqdm,trange

from lossfunctions import ChamferLoss, WassersteinLoss



def train_conv_enc_pb_dec(data_path, encoder, decoder, config):
    print('loading data: this may take a while')
    data_all = np.load(data_path, allow_pickle=False)['a']
    print('done loading data')
    num_runs = config['num_runs']
    t0 = time.time()
    len_data = data_all.shape[0]
    num_p = 3840

    print('generating pointset and removing smaller vertebrae below number of points threshold')
    points_all = []
    too_small  = []
    for j in range(len(data_all)):
        mesh = F.affine_grid(torch.eye(3,4).unsqueeze(0).cuda(),(1,1,96,64,80)).reshape(-1,3)
        pts = mesh[data_all[j].reshape(-1)>.99,:]
        ids = torch.cat((torch.randperm(pts.shape[0]),torch.randperm(pts.shape[0]),torch.randperm(pts.shape[0])),0)[:num_p]
        if pts.shape[0] > num_p:
            points_all.append(pts[ids])
        else: 
            too_small.append(j)
    data_all = np.delete(data_all, too_small, axis=0)
    len_data = data_all.shape[0]
    print('remaining vertebrae: ',len_data)

    encoder.cuda()
    decoder.cuda() #todo check
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

        #TODO add augmentation
        smooth1 = nn.Sequential(nn.AvgPool3d(3,stride=1,padding=1)).cuda()

        with tqdm(total=iterations, file=sys.stdout) as pbar:
            for i in range(iterations):
                optimizer.zero_grad()
                idx_ts = torch.randperm(len_data)[:B]
                with torch.cuda.amp.autocast():
                    in_vert3d = torch.sigmoid(smooth1(torch.from_numpy(data_all[idx_ts]).cuda().unsqueeze(1))*5)*2-1
                    A = (torch.eye(3,4).unsqueeze(0)+.025*torch.randn(B,3,4)).cuda()
                    vertebrae_in = F.grid_sample(in_vert3d,F.affine_grid(A,(B,1,D,H,W)), align_corners=False)
                    target = F.grid_sample(in_vert3d,F.affine_grid(A,(B,1,D,H,W)), align_corners=False)

                    z = encoder(vertebrae_in)
                    reco = decoder(z)

                    loss = nn.SmoothL1Loss()(target*4,reco*4)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                run_loss[i] = 100*loss.item()

                if i % 100 == 99:
                    print(i, 100*loss.item())

                str1 = f"iter: {i}, loss: {'%0.3f'%(run_loss[i-3:i-1].mean())}, runtime: {'%0.3f'%(time.time()-t_run)} sec"#, GPU max/memory: {'%0.2f'%(torch.cuda.max_memory_allocated()*1e-9)} GByte"
                pbar.set_description(str1)
                pbar.update(1)
        t_run2 = time.time()
        torch.save(run_loss, 'run_loss_'+f'{run}' + '.pth')
        torch.save([encoder, decoder], 'run_checkpoint'+ f'{run}' + '.pth')
        print('runtime for run:', t_run2-t_run)



def train_conv_enc_conv_dec(data_path, encoder, decoder, config):
    print('loading data: this may take a while')
    data_all = np.load(data_path, allow_pickle=False)['a']
    print('done loading data')
    num_runs = config['num_runs']
    t0 = time.time()
    len_data = data_all.shape[0]
    num_p = 1920*2

    print('generating pointset and removing smaller vertebrae below number of points threshold')
    points_all = []
    too_small  = []
    for j in range(len(data_all)):
        mesh = F.affine_grid(torch.eye(3,4).unsqueeze(0).cuda(),(1,1,96,64,80)).reshape(-1,3)
        pts = mesh[data_all[j].reshape(-1)>.99,:]
        ids = torch.cat((torch.randperm(pts.shape[0]),torch.randperm(pts.shape[0]),torch.randperm(pts.shape[0])),0)[:num_p]
        if pts.shape[0] > num_p:
            points_all.append(pts[ids])
        else: 
            too_small.append(j)
    data_all = np.delete(data_all, too_small, axis=0)
    len_data = data_all.shape[0]
    print('remaining vertebrae: ',len_data)

    encoder.cuda()
    decoder.cuda() #todo check
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

        #TODO add augmentation
        smooth1 = nn.Sequential(nn.AvgPool3d(3,stride=1,padding=1)).cuda()

        with tqdm(total=iterations, file=sys.stdout) as pbar:
            for i in range(iterations):
                optimizer.zero_grad()
                idx_ts = torch.randperm(len_data)[:B]
                with torch.cuda.amp.autocast():
                    in_vert3d = torch.sigmoid(smooth1(torch.from_numpy(data_all[idx_ts]).cuda().unsqueeze(1))*5)*2-1
                    A = (torch.eye(3,4).unsqueeze(0)+.025*torch.randn(B,3,4)).cuda()
                    vertebrae_in = F.grid_sample(in_vert3d,F.affine_grid(A,(B,1,D,H,W)), align_corners=False)
                    target = F.grid_sample(in_vert3d,F.affine_grid(A,(B,1,D,H,W)), align_corners=False)

                    z = encoder(vertebrae_in)
                    reco = decoder(z)

                    loss = nn.SmoothL1Loss()(target*4,reco*4)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                run_loss[i] = 100*loss.item()

                str1 = f"iter: {i}, loss: {'%0.3f'%(run_loss[i-3:i-1].mean())}, runtime: {'%0.3f'%(time.time()-t_run)} sec"#, GPU max/memory: {'%0.2f'%(torch.cuda.max_memory_allocated()*1e-9)} GByte"
                pbar.set_description(str1)
                pbar.update(1)
        t_run2 = time.time()
        torch.save(run_loss, 'run_loss_'+f'{run}' + 'convenc_convdec.pth')
        torch.save([encoder, decoder], 'run_checkpoint'+ f'{run}' + 'convenc_convdec.pth')
        print('runtime for run:', t_run2-t_run)


def train_conv_enc_point_dec(data_path, encoder, decoder, config):
    print('loading data: this may take a while')
    data_all = np.load(data_path, allow_pickle=False)['a']
    print('done loading data')
    num_runs = config['num_runs']
    t0 = time.time()
    len_data = data_all.shape[0]
    num_p = 1920*2

    print('generating pointset and removing smaller vertebrae below number of points threshold')
    points_all = []
    too_small  = []
    for j in range(len(data_all)):
        mesh = F.affine_grid(torch.eye(3,4).unsqueeze(0).cuda(),(1,1,96,64,80)).reshape(-1,3)
        pts = mesh[data_all[j].reshape(-1)>.99,:]
        ids = torch.cat((torch.randperm(pts.shape[0]),torch.randperm(pts.shape[0]),torch.randperm(pts.shape[0])),0)[:num_p]
        if pts.shape[0] > num_p:
            points_all.append(pts[ids])
        else: 
            too_small.append(j)
    data_all = np.delete(data_all, too_small, axis=0)
    len_data = data_all.shape[0]
    print('remaining vertebrae: ',len_data)
    points_all = torch.stack(points_all,0).cuda()

    encoder.cuda()
    decoder.cuda() #todo check
    print('start training')
    for run in trange(num_runs):
        print('training run: ' , run)
        t_run = time.time()
        iterations = config['its_per_run']#*4
        run_loss = torch.zeros(iterations)
        optimizer = torch.optim.Adam(list(encoder.parameters())+list(decoder.parameters()),lr=config['learning_rate']) 
        scaler = torch.cuda.amp.GradScaler()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,\
                                                                         config['cosine_annealing_T_0'],\
                                                                            config['cosine_annealing_T_mult'])
        B = config['batch_size']
        B=16
        D,H,W = config['patch_size']

        #TODO add augmentation
        smooth1 = nn.Sequential(nn.AvgPool3d(3,stride=1,padding=1)).cuda()

        with tqdm(total=iterations, file=sys.stdout) as pbar:
            for i in range(iterations):
                optimizer.zero_grad()
                idx_ts = torch.randperm(len_data)[:B]
                with torch.cuda.amp.autocast():
                    in_vert3d = torch.sigmoid(smooth1(torch.from_numpy(data_all[idx_ts]).cuda().unsqueeze(1))*5)*2-1
                    A = (torch.eye(3,4).unsqueeze(0)+.025*torch.randn(B,3,4)).cuda()
                    vertebrae_in = F.grid_sample(in_vert3d,F.affine_grid(A,(B,1,D,H,W)), align_corners=False)
                    target =  points_all[idx_ts].cuda().unsqueeze(1).view(B,3,-1)

                    z = encoder(vertebrae_in)
                    reco = decoder(z)

                    loss = ChamferLoss()(reco, target.cuda())#loss = nn.SmoothL1Loss()(target*4,reco*4)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                run_loss[i] = 100*loss.item()

                if i % 500 == 0:
                    print(i, loss.item())

                str1 = f"iter: {i}, loss: {'%0.3f'%(run_loss[i-3:i-1].mean())}, runtime: {'%0.3f'%(time.time()-t_run)} sec"#, GPU max/memory: {'%0.2f'%(torch.cuda.max_memory_allocated()*1e-9)} GByte"
                pbar.set_description(str1)
                pbar.update(1)
        t_run2 = time.time()
        torch.save(run_loss, 'run_loss_'+f'{run}' + 'convenc_convdec.pth')
        torch.save([encoder, decoder], 'run_checkpoint'+ f'{run}' + 'convenc_convdec.pth')
        print('runtime for run:', t_run2-t_run)


def train_var_enc_conv_dec(data_path, encoder, decoder, config):
    print('loading data: this may take a while')
    data_all = np.load(data_path, allow_pickle=False)['a']
    print('done loading data')
    num_runs = config['num_runs']
    t0 = time.time()
    len_data = data_all.shape[0]
    num_p = 1920*2

    print('generating pointset and removing smaller vertebrae below number of points threshold')
    points_all = []
    too_small  = []
    for j in range(len(data_all)):
        mesh = F.affine_grid(torch.eye(3,4).unsqueeze(0).cuda(),(1,1,96,64,80)).reshape(-1,3)
        pts = mesh[data_all[j].reshape(-1)>.99,:]
        ids = torch.cat((torch.randperm(pts.shape[0]),torch.randperm(pts.shape[0]),torch.randperm(pts.shape[0])),0)[:num_p]
        if pts.shape[0] > num_p:
            points_all.append(pts[ids])
        else: 
            too_small.append(j)
    data_all = np.delete(data_all, too_small, axis=0)
    len_data = data_all.shape[0]
    print('remaining vertebrae: ',len_data)

    encoder.cuda()
    decoder.cuda() #todo check
    print('start training')
    for run in trange(num_runs):
        print('training run: ' , run)
        t_run = time.time()
        iterations = config['its_per_run']
        run_loss = torch.zeros(iterations)
        run_kl = torch.zeros(iterations)
        optimizer = torch.optim.Adam(list(encoder.parameters())+list(decoder.parameters()),lr=config['learning_rate']) 
        scaler = torch.cuda.amp.GradScaler()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,\
                                                                         config['cosine_annealing_T_0'],\
                                                                            config['cosine_annealing_T_mult'])
        B = config['batch_size']
        D,H,W = config['patch_size']

        #TODO add augmentation
        smooth1 = nn.Sequential(nn.AvgPool3d(3,stride=1,padding=1)).cuda()

        with tqdm(total=iterations, file=sys.stdout) as pbar:
            for i in range(iterations):
                optimizer.zero_grad()
                idx_ts = torch.randperm(len_data)[:B]
                with torch.cuda.amp.autocast():
                    in_vert3d = torch.sigmoid(smooth1(torch.from_numpy(data_all[idx_ts]).cuda().unsqueeze(1))*5)*2-1
                    A = (torch.eye(3,4).unsqueeze(0)+.025*torch.randn(B,3,4)).cuda()
                    vertebrae_in = F.grid_sample(in_vert3d,F.affine_grid(A,(B,1,D,H,W)), align_corners=False)
                    target = F.grid_sample(in_vert3d,F.affine_grid(A,(B,1,D,H,W)), align_corners=False)

                    z, kl = encoder(vertebrae_in)
                    reco = decoder(z)
            

                    loss =  nn.SmoothL1Loss()(target*4,reco*4)+ i*0.5* kl.mean()
                    
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                run_loss[i] = 100*loss.item()
                run_kl[i] = 100 * kl.mean()

                if i % 100 == 99:
                    print(i, 100*loss.item())



                str1 = f"iter: {i}, loss: {'%0.3f'%(run_loss[i-3:i-1].mean())}, runtime: {'%0.3f'%(time.time()-t_run)} sec"#, GPU max/memory: {'%0.2f'%(torch.cuda.max_memory_allocated()*1e-9)} GByte"
                pbar.set_description(str1)
                pbar.update(1)
        t_run2 = time.time()
        torch.save(run_loss, 'run_loss_'+f'{run}' + 'varenc_convdec.pth')
        torch.save(run_kl, 'run_kl_'+f'{run}' + 'varenc_convdec.pth')
        torch.save([encoder, decoder], 'run_checkpoint'+ f'{run}' + 'varenc_convdec.pth')
        print('runtime for run:', t_run2-t_run)




def train_point_enc_pb_dec(data_path, encoder, decoder, config):
    print('loading data: this may take a while')
    data_all = np.load(data_path, allow_pickle=False)['a']
    print('done loading data')
    num_runs = config['num_runs']
    t0 = time.time()
    len_data = data_all.shape[0]
    num_p = 1920*2

    print('generating pointset and removing smaller vertebrae below number of points threshold')
    points_all = []
    too_small  = []
    for j in range(len(data_all)):
        mesh = F.affine_grid(torch.eye(3,4).unsqueeze(0).cuda(),(1,1,96,64,80)).reshape(-1,3)
        pts = mesh[data_all[j].reshape(-1)>.99,:]
        ids = torch.cat((torch.randperm(pts.shape[0]),torch.randperm(pts.shape[0]),torch.randperm(pts.shape[0])),0)[:num_p]
        if pts.shape[0] > num_p:
            points_all.append(pts[ids])
        else: 
            too_small.append(j)
    data_all = np.delete(data_all, too_small, axis=0)
    len_data = data_all.shape[0]
    print('remaining vertebrae: ',len_data)
    
    points_all = torch.stack(points_all,0).cuda()

    encoder.cuda()
    decoder.cuda() #todo check
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

        #TODO add augmentation
        smooth1 = nn.Sequential(nn.AvgPool3d(3,stride=1,padding=1)).cuda()

        with tqdm(total=iterations, file=sys.stdout) as pbar:
            for i in range(iterations):
                optimizer.zero_grad()
                idx_ts = torch.randperm(len_data)[:B]
                with torch.cuda.amp.autocast():
                    in_vert3d = torch.sigmoid(smooth1(torch.from_numpy(data_all[idx_ts]).cuda().unsqueeze(1))*5)*2-1
                    #A = (torch.eye(3,4).unsqueeze(0)+.025*torch.randn(B,3,4)).cuda()
                    #vertebrae_in = F.grid_sample(in_vert3d,F.affine_grid(A,(B,1,D,H,W)), align_corners=False)
                    target = in_vert3d
                    points_in = points_all[idx_ts].cuda().unsqueeze(1).view(B,3,-1)

                    x = encoder(points_in)
                    reco = decoder(x)

                    loss = nn.SmoothL1Loss()(target*4,reco*4)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                run_loss[i] = 100*loss.item()

                str1 = f"iter: {i}, loss: {'%0.3f'%(run_loss[i-3:i-1].mean())}, runtime: {'%0.3f'%(time.time()-t_run)} sec"#, GPU max/memory: {'%0.2f'%(torch.cuda.max_memory_allocated()*1e-9)} GByte"
                pbar.set_description(str1)
                pbar.update(1)
        t_run2 = time.time()
        torch.save(run_loss, 'run_loss_'+f'{run}' + 'pointenc_pbdec.pth')
        torch.save([encoder, decoder], 'run_checkpoint'+ f'{run}' + 'pointenc_pbdec.pth')
        print('runtime for run:', t_run2-t_run)


def train_point_dgcnn_enc_pb_dec(data_path, encoder, decoder, config):
    print('loading data: this may take a while')
    data_all = np.load(data_path, allow_pickle=False)['a']
    print('done loading data')
    num_runs = 1#config['num_runs']
    t0 = time.time()
    len_data = data_all.shape[0]
    num_p = 1920*2

    print('generating pointset and removing smaller vertebrae below number of points threshold')
    points_all = []
    too_small  = []
    for j in range(len(data_all)):
        mesh = F.affine_grid(torch.eye(3,4).unsqueeze(0).cuda(),(1,1,96,64,80)).reshape(-1,3)
        pts = mesh[data_all[j].reshape(-1)>.99,:]
        ids = torch.cat((torch.randperm(pts.shape[0]),torch.randperm(pts.shape[0]),torch.randperm(pts.shape[0])),0)[:num_p]
        if pts.shape[0] > num_p:
            points_all.append(pts[ids])
        else: 
            too_small.append(j)
    data_all = np.delete(data_all, too_small, axis=0)
    len_data = data_all.shape[0]
    print('remaining vertebrae: ',len_data)

    points_all = torch.stack(points_all,0).cuda()

    encoder.cuda()
    decoder.cuda() #todo check
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
        B=16
        D,H,W = config['patch_size']

        #TODO add augmentation
        smooth1 = nn.Sequential(nn.AvgPool3d(3,stride=1,padding=1)).cuda()

        with tqdm(total=iterations, file=sys.stdout) as pbar:
            for i in range(iterations):
                optimizer.zero_grad()
                idx_ts = torch.randperm(len_data)[:B]
                with torch.cuda.amp.autocast():
                    in_vert3d = torch.sigmoid(smooth1(torch.from_numpy(data_all[idx_ts]).cuda().unsqueeze(1))*5)*2-1
                    #A = (torch.eye(3,4).unsqueeze(0)+.025*torch.randn(B,3,4)).cuda()
                    #vertebrae_in = F.grid_sample(in_vert3d,F.affine_grid(A,(B,1,D,H,W)), align_corners=False)
                    target = in_vert3d
                    points_in = points_all[idx_ts].cuda().unsqueeze(1).view(B,3,-1)

                    x = encoder(points_in.permute(0,2,1),20)

                    reco = decoder(x.squeeze(1))

                    loss = nn.SmoothL1Loss()(target*4,reco*4)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                run_loss[i] = 100*loss.item()
                
                
                if i % 3000 == 0 and i != 0:
                    print("loss: ", run_loss[i-15:i].mean())
                    print("saving checkpoint")
                    torch.save(run_loss, 'run_loss_' + 'dgcnnorig20_pbdec' + f'{run}'+'.pth')
                    torch.save([encoder, decoder], 'run_checkpoint'+ f'{run}'  + 'dgcnnorig20_pbdec' + '_' + f'{i}' + '.pth')

            t_run2 = time.time()
            torch.save(run_loss, 'run_loss_'+f'{run}' + 'dgcnnorig20_pbdec'+ '.pth')
            torch.save([encoder, decoder], 'final_checkpoint'+ f'{run}' + 'dgcnnorig20_pbdec' + '.pth')
            print('runtime for run:', t_run2-t_run)



def train_point_dgcnn_enc_point_dec(data_path, encoder, decoder, config):
    print('loading data: this may take a while')
    data_all = np.load(data_path, allow_pickle=False)['a']
    print('done loading data')
    num_runs = config['num_runs']
    t0 = time.time()
    len_data = data_all.shape[0]
    num_p = 1920*2

    print('generating pointset and removing smaller vertebrae below number of points threshold')
    points_all = []
    too_small  = []
    for j in range(len(data_all)):
        mesh = F.affine_grid(torch.eye(3,4).unsqueeze(0).cuda(),(1,1,96,64,80)).reshape(-1,3)
        pts = mesh[data_all[j].reshape(-1)>.99,:]
        ids = torch.cat((torch.randperm(pts.shape[0]),torch.randperm(pts.shape[0]),torch.randperm(pts.shape[0])),0)[:num_p]
        if pts.shape[0] > num_p:
            points_all.append(pts[ids])
        else: 
            too_small.append(j)
    data_all = np.delete(data_all, too_small, axis=0)
    len_data = data_all.shape[0]
    print('remaining vertebrae: ',len_data)
    
    

    points_all = torch.stack(points_all,0).cuda()

    encoder.cuda()
    decoder.cuda() #todo check
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
        B=16
        D,H,W = config['patch_size']

        #TODO add augmentation
        smooth1 = nn.Sequential(nn.AvgPool3d(3,stride=1,padding=1)).cuda()

        with tqdm(total=iterations, file=sys.stdout) as pbar:
            for i in range(iterations):
                optimizer.zero_grad()
                idx_ts = torch.randperm(len_data)[:B]
                with torch.cuda.amp.autocast():
                    in_vert3d = torch.sigmoid(smooth1(torch.from_numpy(data_all[idx_ts]).cuda().unsqueeze(1))*5)*2-1
                    #A = (torch.eye(3,4).unsqueeze(0)+.025*torch.randn(B,3,4)).cuda()
                    #vertebrae_in = F.grid_sample(in_vert3d,F.affine_grid(A,(B,1,D,H,W)), align_corners=False)
                    target = in_vert3d
                    points_in = points_all[idx_ts].cuda().unsqueeze(1).view(B,3,-1)

                    x = encoder(points_in.permute(0,2,1),8)

                    reco = decoder(x.squeeze(1))

                    #loss = nn.SmoothL1Loss()(target*4,reco*4)
                    loss = ChamferLoss()(reco, points_in)#nn.SmoothL1Loss()(target*4,reco*4)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                run_loss[i] = 100*loss.item()
                if i % 100 == 99:
                    print(i, 100*loss.item())

                str1 = f"iter: {i}, loss: {'%0.3f'%(run_loss[i-3:i-1].mean())}, runtime: {'%0.3f'%(time.time()-t_run)} sec"#, GPU max/memory: {'%0.2f'%(torch.cuda.max_memory_allocated()*1e-9)} GByte"
                pbar.set_description(str1)
                pbar.update(1)
        t_run2 = time.time()
        torch.save(run_loss, 'run_loss_'+f'{run}' + 'dgcnn_pointdec.pth')
        torch.save([encoder, decoder], 'run_checkpoint'+ f'{run}' + 'dgcnn_pointdec.pth')
        print('runtime for run:', t_run2-t_run)



def train_point_enc_point_dec(data_path, encoder, decoder, config):
    print('loading data: this may take a while')
    data_all = np.load(data_path, allow_pickle=False)['a']
    print('done loading data')
    num_runs = config['num_runs']
    t0 = time.time()
    len_data = data_all.shape[0]
    num_p = 1920*2

    print('generating pointset and removing smaller vertebrae below number of points threshold')
    points_all = []
    too_small  = []
    for j in range(len(data_all)):
        mesh = F.affine_grid(torch.eye(3,4).unsqueeze(0).cuda(),(1,1,96,64,80)).reshape(-1,3)
        pts = mesh[data_all[j].reshape(-1)>.99,:]
        ids = torch.cat((torch.randperm(pts.shape[0]),torch.randperm(pts.shape[0]),torch.randperm(pts.shape[0])),0)[:num_p]
        if pts.shape[0] > num_p:
            points_all.append(pts[ids])
        else: 
            too_small.append(j)
    data_all = np.delete(data_all, too_small, axis=0)
    len_data = data_all.shape[0]
    print('remaining vertebrae: ',len_data)
    
    

    points_all = torch.stack(points_all,0).cuda()

    encoder.cuda()
    decoder.cuda() #todo check
    encoder.train()
    decoder.train()
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
        B = 16
        D,H,W = config['patch_size']

        #TODO add augmentation
        smooth1 = nn.Sequential(nn.AvgPool3d(3,stride=1,padding=1)).cuda()

        with tqdm(total=iterations, file=sys.stdout) as pbar:
            for i in range(iterations):
                optimizer.zero_grad()
                idx_ts = torch.randperm(len_data)[:B]
                #with torch.cuda.amp.autocast():
                in_vert3d = torch.sigmoid(smooth1(torch.from_numpy(data_all[idx_ts]).cuda().unsqueeze(1))*5)*2-1
                #A = (torch.eye(3,4).unsqueeze(0)+.025*torch.randn(B,3,4)).cuda()
                #vertebrae_in = F.grid_sample(in_vert3d,F.affine_grid(A,(B,1,D,H,W)), align_corners=False)

                points_in = torch.tanh(points_all[idx_ts].cuda().unsqueeze(1).view(B,3,-1))
                #target = points_in.float()
                points_in.requires_grad == True

                x = encoder(points_in.float())
                reco = decoder(x).view(B,3,-1)

                loss = ChamferLoss()(reco, points_in)#nn.SmoothL1Loss()(target*4,reco*4)
                #print(loss)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                #loss.backward()
                #optimizer.step()
            
                scheduler.step()
                run_loss[i] = 100*loss.item()

                str1 = f"iter: {i}, loss: {'%0.3f'%(run_loss[i-3:i-1].mean())}, runtime: {'%0.3f'%(time.time()-t_run)} sec"#, GPU max/memory: {'%0.2f'%(torch.cuda.max_memory_allocated()*1e-9)} GByte"
                pbar.set_description(str1)
                pbar.update(1)

                if i % 500 ==499:
                    print(i, loss)

            torch.cuda.empty_cache()

        t_run2 = time.time()
        torch.save(run_loss, 'run_loss_'+f'{run}' + 'pointenc_pointdec.pth')
        torch.save([encoder, decoder], 'run_checkpoint'+ f'{run}' + 'pointenc_pointdec.pth')
        print('runtime for run:', t_run2-t_run)
