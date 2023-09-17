import copy
import time
import torch
import numpy as np
from scipy.io import savemat
import os
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_func
import tqdm
import cv2
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure

from modules.utils import TV, L1_reg, L2_reg, psnr, prop_plane, delay_sum, get_noise
from modules.DIP_models import *

class DIPRecon:
    def __init__(self, device):
        self.device = device

    def norm(self, x):
        # min-max normalization
        return (x - x.min())/(x.max() - x.min() + 1e-9)

    def energy(self, x):
        # compute power of tensor x
        return torch.sqrt(torch.sum(torch.abs(x) ** 2))

    def recon(self,gt_datacube,full_datacube,sparse_datacube,
              gt_heatmap, full_heatmap, sparse_heatmap,
              mask,save_dir,DATA_PARAMS,lr,niters,reg,reg_weight, model_name, view_recon=False):
        assert reg in ['none', 'l1', 'l2', 'tv']

        reg_fn = None

        if reg == 'none':
            reg_fn = lambda x: torch.tensor([0]).to(self.device).double()
        elif reg == 'l2':
            reg_fn = L2_reg
        elif reg == 'l1':
            reg_fn = L1_reg
        elif reg == 'tv':
            reg_fn = TV
        else:
            raise ValueError('Incorrect Regularization Type')

        # -----------------------------------------------------------------------------------------------------
        # Data Preprocessing Section
        # -----------------------------------------------------------------------------------------------------
        N_t = DATA_PARAMS['num_time']
        N_ant = DATA_PARAMS['num_ant']
        N_ang = DATA_PARAMS['num_ang_bin']
        N_rng = DATA_PARAMS['num_rng_bin']
        rng_st = DATA_PARAMS['rng_bin_start']

        # Crop the heatmaps
        gt_heatmap_orig = gt_heatmap
        full_heatmap_orig = full_heatmap
        sparse_heatmap_orig = sparse_heatmap
        gt_heatmap = gt_heatmap[:, :, :, rng_st:N_rng]
        full_heatmap = full_heatmap[:, :, :, rng_st:N_rng]
        sparse_heatmap = sparse_heatmap[:, :, :, rng_st:N_rng]

        # Normalize the measured datacube by its total power
        # found this to help during the optimization process
        sparse_datacube = sparse_datacube / self.energy(sparse_datacube)

        # -----------------------------------------------------------------------------------------------------
        # Optimization Section
        # -----------------------------------------------------------------------------------------------------
        H = N_ang; W = N_rng

        #### Initialize DIP model ####
        # Dmitry Ulyanov Deep Image Prior to solve inverse problems
        pad = 'replicate'
        INPUT = 'noise'
        LR = lr
        in_size = [H, W]  # network input size
        input_channels = 32
        ns = 128
        nlay = 6
        model = skip(
            num_input_channels=input_channels, num_output_channels=2,
            num_channels_down=[ns] * nlay,
            num_channels_up=[ns] * nlay,
            num_channels_skip=[4] * (nlay), #[0] * (nlay-2) + [4,4]
            filter_size_down=3, filter_size_up=3,
            filter_skip_size=1,
            need_sigmoid=False, need_bias=True,
            pad=pad, act_fun='LeakyReLU', #Swish
            upsample_mode='nearest',
            downsample_mode='stride',
            need1x1_up=True
        ).to(self.device)

        # Get noise input for the model
        inp = get_noise(input_channels, INPUT, in_size, noise_type='u', var=1/10).to(self.device)

        print('Number of parameters: ', sum(param.numel() for param in model.parameters()))
        optimizer = torch.optim.Adam(lr=LR, params=model.parameters())

        # Measure how long training is going to take
        # print("[INFO] training the network...")
        startTime = time.time()

        # initialize a dictionary to store training history
        H_out = {
            "train_loss": []
        }
        best_net = copy.deepcopy(model)
        best_loss = float('inf')
        best_im = None
        best_dc_sparse = None

        for idx in tqdm.tqdm(range(niters)):
            # Estimate image of radar heatmap
            output = model(inp)
            # reshape output into complex image [...,H,W] + 1j[...,H,W]
            pred = torch.view_as_complex(torch.permute(output, [0, 2, 3, 1]).contiguous())[None, ...]

            # compute radar datacube and then subsample
            pred_datacube = prop_plane(pred, (N_ang, N_rng))
            pred_datacube = pred_datacube[:, :, 0:N_ant, 0:N_t]  # truncate to remove zero padding

            # Normalize by the power of the subsampled datacube
            pred_datacube = pred_datacube / self.energy(mask * pred_datacube)

            # compute loss and backpropagate
            reg_term = reg_weight * reg_fn(torch.abs(pred))
            loss_real = ((sparse_datacube.real - mask * pred_datacube.real) ** 2).sum()
            loss_imag = ((sparse_datacube.imag - mask * pred_datacube.imag) ** 2).sum()
            loss = (loss_real + loss_imag) + reg_term

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Store net if loss improves by at least 1 percent
            with torch.no_grad():
                if view_recon:
                    # Plot the corresponding heatmaps
                    im_sparse = torch.abs(sparse_heatmap).detach().cpu().numpy()  # size:[H,W]
                    im_sparse = (np.squeeze(im_sparse))  # size:[H,W]
                    cv2.imshow('Sparse Recon.', self.norm(im_sparse))
                    cv2.waitKey(1)
                    im_ideal = torch.abs(gt_heatmap).detach().cpu().numpy()  # size:[H,W]
                    im_ideal = (np.squeeze(im_ideal))  # size:[H,W]
                    cv2.imshow('GT Recon.', self.norm(im_ideal))
                    cv2.waitKey(1)
                    im_pred = torch.abs(pred.detach()).cpu().numpy()  # size:[H,W]
                    im_pred = (np.squeeze(im_pred))  # size:[H,W]
                    cv2.imshow('Recon. Heatmap', self.norm(im_pred))
                    cv2.waitKey(1)

                    im_datacube = torch.abs(pred_datacube[:, :, 0:N_ant, 0:N_t]).detach().cpu().numpy()  # size:[H,W]
                    im_datacube = (np.squeeze(im_datacube))  # size:[H,W]
                    cv2.imshow('Recon. Datacube', self.norm(im_datacube))
                    cv2.waitKey(1)

                    im_val_datacube = torch.abs(gt_datacube).detach().cpu().numpy()
                    im_val_datacube = (np.squeeze(im_val_datacube))  # size:[H,W]
                    cv2.imshow('GT Datacube', self.norm(im_val_datacube))
                    cv2.waitKey(1)

                    im_sparse_datacube = torch.abs(sparse_datacube).detach().cpu().numpy()
                    im_sparse_datacube = (np.squeeze(im_sparse_datacube))  # size:[H,W]
                    cv2.imshow('Sparse Datacube', self.norm(im_sparse_datacube))
                    cv2.waitKey(1)

                if 1.05*loss.data < best_loss:
                    best_net = copy.deepcopy(model)
                    best_loss = loss.data
                    best_im = pred.detach()
                    best_dc_sparse = pred_datacube.detach()

            # update training history
            H_out["train_loss"].append(loss.data.cpu().detach().numpy())

        # finish measuring model train time
        endTime = time.time()
        totTime = endTime - startTime

        # -----------------------------------------------------------------------------------------------------
        # Save Results Section
        # -----------------------------------------------------------------------------------------------------
        # Create save directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        gt = self.norm(torch.abs(gt_heatmap)).cpu().squeeze().numpy()
        best_im_crop = best_im[:, :, :, rng_st:N_rng].detach()
        recon = self.norm(torch.abs(best_im_crop)).cpu().squeeze().numpy()

        dc_sparse = self.norm(torch.abs(mask * best_dc_sparse.detach())).cpu().squeeze().numpy()
        sparse_noise = self.norm(torch.abs(sparse_heatmap)).cpu().squeeze().numpy()
        full_noise = self.norm(torch.abs(full_heatmap)).cpu().squeeze().numpy()

        # multi-scale SSIM
        ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)

        # Compute reconstruction metrics
        psnr_m = psnr(gt, recon)
        ssim_m = ssim_func(gt, recon, data_range=gt.max())
        mae_m = np.mean(np.abs(gt - recon))
        ms_ssim_m = ms_ssim(torch.tensor(gt)[None,None,:,:], torch.tensor(recon)[None,None,:,:])

        cmap_color = 'turbo'

        # Save reconstruction metrics as a .mat file
        mdic1 = {"psnr": np.array(psnr_m), "ssim": np.array(ssim_m), "loss": np.array(H_out["train_loss"]),
                 "mae": np.array(mae_m), "ms_ssim": np.array(ms_ssim_m), "runTime": np.array(totTime)}
        savemat(os.path.join(save_dir, 'metrics.mat'), mdic1)

        # Save reconstruction metrics as numpy variables
        with open(os.path.join(save_dir, 'metrics.npy'), 'wb') as f:
            np.save(f, psnr_m)
            np.save(f, ssim_m)
            np.save(f, np.array(ms_ssim_m))
            np.save(f, mae_m)
            np.save(f, totTime)

        # Save images at torch tensors
        torch.save(best_im.detach().cpu(), os.path.join(save_dir, 'recon.pt'))
        torch.save(full_heatmap_orig.detach().cpu(), os.path.join(save_dir, 'full_heatmap.pt'))
        torch.save(gt_heatmap_orig.detach().cpu(), os.path.join(save_dir, 'gt_heatmap.pt'))
        torch.save(sparse_heatmap_orig.detach().cpu(), os.path.join(save_dir, 'sparse_heatmap.pt'))
        torch.save(best_net.state_dict(), os.path.join(save_dir, 'best_model_state.pt'))

        fig = plt.figure(figsize=(20, 5))
        plt.subplot(1, 3, 1)
        plt.imshow((gt), cmap=cmap_color)
        plt.title('Ground truth')
        plt.xticks([], [])
        plt.yticks([], [])
        plt.colorbar()

        plt.subplot(1, 3, 2)
        plt.imshow((sparse_noise), cmap=cmap_color)
        plt.title('Sparse Array Heatmap')
        plt.xticks([], [])
        plt.yticks([], [])
        plt.colorbar()

        plt.subplot(1, 3, 3)
        plt.imshow((recon), cmap=cmap_color)
        plt.title('%s | PSNR: %.1f dB | SSIM: %.2f | MS-SSIM: %.2f' % (model_name, psnr_m, ssim_m, ms_ssim_m))
        plt.xticks([], [])
        plt.yticks([], [])
        plt.colorbar()
        plt.savefig(os.path.join(save_dir, 'comp.png'), bbox_inches='tight')

        # Now save individual images
        fig2 = plt.figure(figsize=(20, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(gt, cmap=cmap_color)
        plt.title('Ground truth')
        plt.xticks([], [])
        plt.yticks([], [])
        plt.colorbar()
        plt.savefig(os.path.join(save_dir, 'gt.png'), bbox_inches='tight')

        fig3 = plt.figure(figsize=(20, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(sparse_noise, cmap=cmap_color)
        plt.title('Sparse Array Heatmap')
        plt.xticks([], [])
        plt.yticks([], [])
        plt.colorbar()
        plt.savefig(os.path.join(save_dir, 'sparse.png'), bbox_inches='tight')

        fig4 = plt.figure(figsize=(20, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(recon, cmap=cmap_color)
        plt.title('%s | PSNR: %.1f dB | SSIM: %.2f | MS-SSIM: %.2f' % (model_name, psnr_m, ssim_m, ms_ssim_m))
        plt.xticks([], [])
        plt.yticks([], [])
        plt.colorbar()
        plt.savefig(os.path.join(save_dir, 'recon.png'), bbox_inches='tight')

        fig5 = plt.figure(figsize=(20, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(dc_sparse, cmap='viridis')
        plt.title('Predicted Sparse Datacube')
        plt.xticks([], [])
        plt.yticks([], [])
        plt.colorbar()
        plt.savefig(os.path.join(save_dir, 'recon_dc_sparse.png'), bbox_inches='tight')

        fig6 = plt.figure(figsize=(20, 5))
        plt.subplot(1, 3, 1)
        plt.style.use("ggplot")
        plt.loglog(H_out["train_loss"])
        plt.title("Training Loss Single Sample")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(save_dir, 'train_loss.png'), bbox_inches='tight')

        fig7 = plt.figure(figsize=(20, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(full_noise, cmap=cmap_color)
        plt.title('Full Array Heatmap')
        plt.xticks([], [])
        plt.yticks([], [])
        plt.colorbar()
        plt.savefig(os.path.join(save_dir, 'full.png'), bbox_inches='tight')

        plt.close('all')