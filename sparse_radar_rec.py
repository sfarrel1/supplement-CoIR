'''
This program is used to test out different untrained NN
architectures to solve the inverse problem of sparse radar imaging.
Simulated or experimental data can be used, and MLP or CNN based
architectures can be tested.

Created by: Sean Farrell
Date: 3/16/23
'''
import os
import matplotlib.pyplot as plt

import torch
import numpy as np

from modules import utils
from methods import grad_desc_recon, dip_recon, deepDecoder_recon, convDecoder_recon, comDecoder_recon, inr_recon

if __name__ == '__main__':
    # Device will determine whether to run the training on GPU or CPU.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device count?", torch.cuda.device_count())
    print("Current device?", torch.cuda.current_device())
    print("Device name? ", torch.cuda.get_device_name(torch.cuda.current_device()))

    # Specify import parameters for radar data, and optimization
    DATA_PARAMS = {'array_style': 'mra',  # choose 'random','mra',or 'uniform'
                   'num_ant_choose': 16,  # number of antennas unmasked in array
                   'num_ant': 86,  # total number of antennas in linear array
                   'num_time': 256,  # number of time domain samples from ADC
                   'num_ang_bin': 256,  # number of angle bins in radar image
                   'num_rng_bin': 256,  # number of range bins in radar image
                   'array_gap': 76.8 * 2.54e-5,  # spacing between receiver positions in mil -> meters
                   'rng_bin_start': 10}  # range bins below this will be ignored

    # parameters for optimization
    lr = 1.0e-3  # learning rate
    niters = 2000 #2000  # max number of optimization iterations
    view_recon = False  # set true to plot intermediate reconstruction results

    # List out all methods to test out
    # possible methods: 'comDecoder', 'convDecoder', 'deepDecoder',
    # 'dip', 'gd_l1', 'gd_tv', 'inr_relu','inr_sine'
    use_methods = ['comDecoder', 'convDecoder', 'deepDecoder',
                  'dip', 'gd_l1', 'inr_relu','inr_sine']
    # use_methods = ['comDecoder']
    N_methods = len(use_methods)

    # List out all experimental and simulated scenes to test
    # Outdoor scenes
    # exp_runs = [0,0,0,1,1,1,2,2,2]
    # exp_frames = [135,390,451,182,277,438,105,160,300]
    exp_runs = [1]
    exp_frames = [182]




    N_exp = len(exp_runs)
    data_type = 'exp'  # set to 'exp' experimental or 'sim' simulated
    scene_type = 'outdoor'  # set to 'outdoor' or 'indoor' or 'outdoor_rand'

    snr = 19

    # Instantiate methods
    GD = grad_desc_recon.GradDescRecon(device)
    DIP = dip_recon.DIPRecon(device)
    DD = deepDecoder_recon.DeepDecoderRecon(device)
    convD = convDecoder_recon.ConvDecoderRecon(device)
    comD = comDecoder_recon.ComDecoderRecon(device)
    INR_relu = inr_recon.INRRecon(device)
    INR_sine = inr_recon.INRRecon(device)

    for d_idx in range(N_exp):
        run = exp_runs[d_idx]
        frame = exp_frames[d_idx]
        gt_datacube, full_datacube, sparse_datacube, gt_heatmap, full_heatmap, sparse_heatmap, mask = utils.get_data(
            run, frame, DATA_PARAMS, device, data_type, scene_type, snr)

        for m_idx in range(N_methods):
            print(f"Run: {run} | frame: {frame} | method: {use_methods[m_idx]}")
            save_dir = 'results/exp/' + scene_type + '/' + DATA_PARAMS['array_style'] + '/run' + str(
                run) + '_frame' + str(frame) + '/' + use_methods[m_idx]

            if use_methods[m_idx] == 'gd_l1':
                reg = 'l1'
                lambda_reg = 1e-2
                GD.recon(gt_datacube, full_datacube, sparse_datacube,
                         gt_heatmap, full_heatmap, sparse_heatmap,
                         mask, save_dir, DATA_PARAMS, lr, niters,
                         reg, lambda_reg, use_methods[m_idx], view_recon)

            elif use_methods[m_idx] == 'gd_tv':
                reg = 'tv'
                lambda_reg = 5e-2
                GD.recon(gt_datacube, full_datacube, sparse_datacube,
                         gt_heatmap, full_heatmap, sparse_heatmap,
                         mask, save_dir, DATA_PARAMS, lr, niters,
                         reg, lambda_reg, use_methods[m_idx], view_recon)

            elif use_methods[m_idx] == 'dip':
                reg = 'l1'
                lambda_reg = 1e-5
                DIP.recon(gt_datacube, full_datacube, sparse_datacube,
                          gt_heatmap, full_heatmap, sparse_heatmap,
                          mask, save_dir, DATA_PARAMS, lr, niters,
                          reg, lambda_reg, use_methods[m_idx], view_recon)

            elif use_methods[m_idx] == 'deepDecoder':
                reg = 'l1'
                lambda_reg = 1e-5
                DD.recon(gt_datacube, full_datacube, sparse_datacube,
                         gt_heatmap, full_heatmap, sparse_heatmap,
                         mask, save_dir, DATA_PARAMS, lr, niters,
                         reg, lambda_reg, use_methods[m_idx], view_recon)

            elif use_methods[m_idx] == 'convDecoder':
                reg = 'l1'
                lambda_reg = 1e-5
                convD.recon(gt_datacube, full_datacube, sparse_datacube,
                            gt_heatmap, full_heatmap, sparse_heatmap,
                            mask, save_dir, DATA_PARAMS, lr, niters,
                            reg, lambda_reg, use_methods[m_idx], view_recon)

            elif use_methods[m_idx] == 'comDecoder':
                reg = 'l1'
                lambda_reg = 1e-5
                comD.recon(gt_datacube, full_datacube, sparse_datacube,
                           gt_heatmap, full_heatmap, sparse_heatmap,
                           mask, save_dir, DATA_PARAMS, lr, niters,
                           reg, lambda_reg, use_methods[m_idx], view_recon)

            elif use_methods[m_idx] == 'inr_relu':
                ffenc = True
                alpha = 20  # 8 , 20
                omega0 = 30
                nonlin = 'relu'
                map_feat = 128
                reg = 'l1'
                lambda_reg = 1e-5
                INR_relu.recon(gt_datacube, full_datacube, sparse_datacube,
                               gt_heatmap, full_heatmap, sparse_heatmap,
                               mask, save_dir, DATA_PARAMS, lr, niters,
                               reg, lambda_reg, use_methods[m_idx], ffenc, alpha, omega0,
                               nonlin, map_feat, view_recon)

            elif use_methods[m_idx] == 'inr_sine':
                ffenc = False
                alpha = 1
                omega0 = 30 #60
                nonlin = 'sine'
                map_feat = 128
                reg = 'l1'
                lambda_reg = 1e-5
                INR_sine.recon(gt_datacube, full_datacube, sparse_datacube,
                               gt_heatmap, full_heatmap, sparse_heatmap,
                               mask, save_dir, DATA_PARAMS, 1e-4, niters,
                               reg, lambda_reg, use_methods[m_idx], ffenc, alpha, omega0,
                               nonlin, map_feat, view_recon)

            else:
                raise ValueError('Incorrect NN Architecture Method')

            plt.close('all')
