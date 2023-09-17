'''
This program is used to generate all the plots needed to create a
comparison figure for the reconstruction methods. Raw reconstructions
are loaded in, convert to cartesian images, stored as .svg and .png files.
Use for experimental data.

Created by: Sean Farrell
Date: 4/11/23
'''

import os

import cv2
import matplotlib.colors
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib import patches
from scipy.interpolate import RegularGridInterpolator

def norm(x):
    # min-max normalization
    return (x - x.min()) / (x.max() - x.min() + 1e-9)

def pol2cartImg(Nx,Ny, img_pol,rng_st,truncate=False):
    # img_pol is a [K,L] image in polar coordinates

    if truncate==False:
        # create grid_pol to hold polar coordinate locations
        # zero-pad polar image sqrt(2) * r , sqrt(2)*256 = ceil() = 363 at minimum
        N_rng = 256+128
        N_ang = 256
        pad_r = N_rng / 256
        pad_size = N_rng - 256
        img_pol_pad = np.zeros((N_rng,N_ang))
        img_pol_pad[128:256+128,:] = img_pol
        r = np.linspace(15.18987*pad_r,0,num=N_rng) # range bins
        r = r[rng_st:N_rng]
        a = np.arcsin(np.linspace(1,-1,num=N_ang)) # angle bins
        R, A = np.meshgrid(r,a,indexing='xy')   #size [K,L]

        # create grid_xy to hold the cartesian coordinates to query
        r_max = 15.18987
        r_min = np.min(r)
        # x = np.linspace(-r_max/np.sqrt(2), r_max/np.sqrt(2), num=Nx)
        # y = np.linspace(r_max/np.sqrt(2),r_min, num=Ny)
        x = np.linspace(-r_max , r_max, num=Nx)
        y = np.linspace(r_max, r_min, num=Ny)
        X, Y = np.meshgrid(x, y, indexing='xy')  # size [M,N]
    else:
        # create grid_pol to hold polar coordinate locations
        # polar image is cropped to fill in cartesian image completely
        N_rng = 256
        N_ang = 256
        img_pol_pad = img_pol
        r = np.linspace(15.18987, 0, num=N_rng)  # range bins
        r = r[rng_st:N_rng]
        a = np.arcsin(np.linspace(1, -1, num=N_ang))  # angle bins
        R, A = np.meshgrid(r, a, indexing='xy')  # size [K,L]

        # create grid_xy to hold the cartesian coordinates to query
        r_max = 15.18987#14.976
        r_min = np.min(r)
        x = np.linspace(-r_max/np.sqrt(2), r_max/np.sqrt(2), num=Nx)
        y = np.linspace(r_max/np.sqrt(2),r_min, num=Ny)
        X, Y = np.meshgrid(x, y, indexing='xy')  # size [M,N]
    # now convert cartesian image into polar coordinates
    R_xy = np.sqrt(X**2 + Y**2)
    A_xy = np.arctan2(Y,X) - np.pi/2

    # fit linear interpolator to polar image
    interp = RegularGridInterpolator((r,a),img_pol_pad)

    # apply interolator to query cartesian points
    img_cart = interp((R_xy,A_xy))

    return img_cart

if __name__ =='__main__':
    # setup file directoy paths and determine how many item there are
    # use_methods = ['gd_l1','inr_sine', 'dip','comDecoder']
    use_methods = ['gd_l1','inr_sine', 'dip','convDecoder','deepDecoder','inr_relu']
    # use_methods = ['comDecoder']
    N_methods = len(use_methods)
    array_style = 'mra' #'mra', 'mimo1', 'mimo2'
    data_type = 'exp'
    scene_type = 'outdoor'
    rng_st = 0
    N_rng = 256
    transImg = True  # set true to transpose the estimated reconstructed heatmaps
    coordOpt = 'cart'      # set to 'cart' or 'polar' to change plot coordinate frame
    truncate = True # set true to plot zoomed in region of polar plot
    useColorBar = False  # set true to save images with colorbar

    # List out all experimental outdoor and experimental indoor
    run = 1 #1 #2
    frame = 182 #182 #157

    # Define the plotting parameters
    cmap_color = 'turbo'

    axis_font = {'fontname': 'Times New Roman', 'size': '14'}

    # Define patch coordinates outdoor-run 1-frame 182
    z1h = 160  # bottom of zoomed in region 1
    z1w = 100  # left side of zoomed in region 1
    z2h = 182  # bottom of zoomed in region 2
    z2w = 150  # left side of zoomed in region 2
    dh = 64  # height of zoomed in regions
    dw = 64  # width fof zoomed in regions

    # # Define patch coordinates indoor-run 4-frame 367
    # z1h = 240  # bottom of zoomed in region 1
    # z1w = 20  # left side of zoomed in region 1
    # z2h = 142  # bottom of zoomed in region 2
    # z2w = 110  # left side of zoomed in region 2
    # dh = 64  # height of zoomed in regions
    # dw = 64  # width fof zoomed in regions

    # # Define patch coordinates outdoor-run 1-frame 438
    # z1h = 150  # bottom of zoomed in region 1
    # z1w = 80  # left side of zoomed in region 1
    # z2h = 80  # bottom of zoomed in region 2
    # z2w = 80  # left side of zoomed in region 2
    # dh = 64  # height of zoomed in regions
    # dw = 64  # width fof zoomed in regions

    # Create save directory
    save_dir = 'plot_results_cart/'+ data_type + '/' + scene_type + '_full_comp_run' + str(run)+'_frame'+str(frame)



    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for m_idx in range(N_methods):
        method = use_methods[m_idx]
        print(f"Run: {run} | frame: {frame} | method: {method}")
        load_dir = 'results/' + data_type + '/' + scene_type + '/' + array_style + '/run' + str(run) + '_frame' + str(
            frame) + '/' + method

        # Load in the heatmaps (reconstruction, sparse, full)
        recon_heatmap = torch.load(os.path.join(load_dir, 'recon.pt'))
        img_recon = (torch.abs(recon_heatmap[:, :, :, rng_st:N_rng])).cpu().squeeze().numpy()

        sparse_heatmap = torch.load(os.path.join(load_dir, 'sparse_heatmap.pt'))
        full_heatmap = torch.load(os.path.join(load_dir, 'full_heatmap.pt'))
        img_sparse = (torch.abs(sparse_heatmap[:, :, :, rng_st:N_rng])).cpu().squeeze().numpy()
        img_full = (torch.abs(full_heatmap[:, :, :, rng_st:N_rng])).cpu().squeeze().numpy()

        if transImg:
            img_recon = np.fliplr(np.flipud(np.transpose(img_recon)))
            img_sparse = np.fliplr(np.flipud(np.transpose(img_sparse)))
            img_full = np.fliplr(np.flipud(np.transpose(img_full)))

        if coordOpt == 'cart':
            # convert all heatmap images to cartesian coordinates
            Nx = 256
            Ny = 256
            img_recon = pol2cartImg(Nx, Ny, img_recon, rng_st, truncate=truncate)
            img_sparse = pol2cartImg(Nx, Ny, img_sparse, rng_st, truncate=truncate)
            img_full = pol2cartImg(Nx, Ny, img_full, rng_st, truncate=truncate)

        img_recon = norm(img_recon)
        img_sparse = norm(img_sparse)
        img_full = norm(img_full)

        # Plot and save Reconstruciton result
        fig1 = plt.figure(figsize=(10.0, 10.0))
        # ax1 = fig1.add_subplot(111,xticklabels=[], yticklabels=[])
        ax1 = fig1.gca()
        ax1.tick_params(left=False, bottom=False)
        ax1.set_xticks([])
        ax1.set_yticks([])
        img1 = ax1.imshow(img_recon, cmap=cmap_color)
        if useColorBar:
            cbar = fig1.colorbar(img1, ax=ax1, shrink=0.99, aspect=50)
            cbar.ax.set_ylabel('Linear Magnitude', **axis_font)
            ticklabs = cbar.ax.get_yticklabels()
            ticks_loc = cbar.ax.get_yticks().tolist()
            cbar.ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
            cbar.ax.set_yticklabels(ticklabs, fontsize=14, fontname='Times New Roman')

        plt.xticks([], [])
        plt.yticks([], [])
        plt.savefig(os.path.join(save_dir,'recon_'+method+'.png'), bbox_inches='tight',
                transparent=True,pad_inches=0.0)
        plt.savefig(os.path.join(save_dir, 'recon_' + method + '.svg'), bbox_inches='tight',
                    transparent=True, pad_inches=0.0)

        # Save the patches of interest
        patch1 = img_recon[z1h - dh:z1h, z1w:z1w + dw]
        patch2 = img_recon[z2h - dh:z2h, z2w:z2w + dw]

        # Plot and save patches
        fig1p1 = plt.figure(figsize=(5.0, 5.0))
        ax1 = fig1p1.gca()
        ax1.tick_params(left=False, bottom=False)
        ax1.set_xticks([])
        ax1.set_yticks([])
        img1p1 = ax1.imshow(patch1, cmap=cmap_color, vmin=np.min(img_recon), vmax=np.max(img_recon))
        plt.xticks([], [])
        plt.yticks([], [])
        plt.savefig(os.path.join(save_dir, 'recon_' + method + '_p1.png'), bbox_inches='tight',
                    transparent=True, pad_inches=0.0)
        plt.savefig(os.path.join(save_dir, 'recon_' + method + '_p1.svg'), bbox_inches='tight',
                    transparent=True, pad_inches=0.0)

        fig1p2 = plt.figure(figsize=(5.0, 5.0))
        ax1 = fig1p2.gca()
        ax1.tick_params(left=False, bottom=False)
        ax1.set_xticks([])
        ax1.set_yticks([])
        img1p2 = ax1.imshow(patch2, cmap=cmap_color, vmin=np.min(img_recon), vmax=np.max(img_recon))
        plt.xticks([], [])
        plt.yticks([], [])
        plt.savefig(os.path.join(save_dir, 'recon_' + method + '_p2.png'), bbox_inches='tight',
                    transparent=True, pad_inches=0.0)
        plt.savefig(os.path.join(save_dir, 'recon_' + method + '_p2.svg'), bbox_inches='tight',
                    transparent=True, pad_inches=0.0)
        plt.close('all')

    # -----------------------------------------------------------------------------------------------------------------

    # Plot and save Sparse Heatmap Results
    fig2 = plt.figure(figsize=(10.0, 10.0))
    # ax1 = fig1.add_subplot(111,xticklabels=[], yticklabels=[])
    ax2 = fig2.gca()
    ax2.tick_params(left=False, bottom=False)
    ax2.set_xticks([])
    ax2.set_yticks([])
    img2 = ax2.imshow(img_sparse, cmap=cmap_color)
    if useColorBar:
        et_yticklabelscbar = fig2.colorbar(img2, ax=ax2, shrink=0.99, aspect=50)
        cbar.ax.set_ylabel('Linear Magnitude', **axis_font)
        ticklabs = cbar.ax.get_yticklabels()
        ticks_loc = cbar.ax.get_yticks().tolist()
        cbar.ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        cbar.ax.s(ticklabs, fontsize=14, fontname='Times New Roman')

    plt.xticks([], [])
    plt.yticks([], [])
    plt.savefig(os.path.join(save_dir, 'recon_sparse.png'), bbox_inches='tight',
                transparent=True, pad_inches=0.0)
    plt.savefig(os.path.join(save_dir, 'recon_sparse.svg'), bbox_inches='tight',
                transparent=True, pad_inches=0.0)

    # Save the patches of interest
    patch1 = img_sparse[z1h - dh:z1h, z1w:z1w + dw]
    patch2 = img_sparse[z2h - dh:z2h, z2w:z2w + dw]

    # Plot and save patches
    fig2p1 = plt.figure(figsize=(5.0, 5.0))
    ax1 = fig2p1.gca()
    ax1.tick_params(left=False, bottom=False)
    ax1.set_xticks([])
    ax1.set_yticks([])
    img2p1 = ax1.imshow(patch1, cmap=cmap_color, vmin=np.min(img_sparse), vmax=np.max(img_sparse))
    plt.xticks([], [])
    plt.yticks([], [])
    plt.savefig(os.path.join(save_dir, 'recon_sparse_p1.png'), bbox_inches='tight',
                transparent=True, pad_inches=0.0)
    plt.savefig(os.path.join(save_dir, 'recon_sparse_p1.svg'), bbox_inches='tight',
                transparent=True, pad_inches=0.0)

    fig2p2 = plt.figure(figsize=(5.0, 5.0))
    ax1 = fig2p2.gca()
    ax1.tick_params(left=False, bottom=False)
    ax1.set_xticks([])
    ax1.set_yticks([])
    img2p2 = ax1.imshow(patch2, cmap=cmap_color, vmin=np.min(img_sparse), vmax=np.max(img_sparse))
    plt.xticks([], [])
    plt.yticks([], [])
    plt.savefig(os.path.join(save_dir, 'recon_sparse_p2.png'), bbox_inches='tight',
                transparent=True, pad_inches=0.0)
    plt.savefig(os.path.join(save_dir, 'recon_sparse_p2.svg'), bbox_inches='tight',
                transparent=True, pad_inches=0.0)

    # ----------------------------------------------------------------------------------------------------------------

    # Plot and save Full Heatmap Results
    fig3 = plt.figure(figsize=(10.0, 10.0))
    # ax1 = fig1.add_subplot(111,xticklabels=[], yticklabels=[])
    ax3 = fig3.gca()
    ax3.tick_params(left=False, bottom=False)
    ax3.set_xticks([])
    ax3.set_yticks([])
    img3 = ax3.imshow(img_full, cmap=cmap_color)
    if useColorBar:
        cbar = fig3.colorbar(img3, ax=ax3, shrink=0.99, aspect=50)
        cbar.ax.set_ylabel('Linear Magnitude', **axis_font)
        ticklabs = cbar.ax.get_yticklabels()
        ticks_loc = cbar.ax.get_yticks().tolist()
        cbar.ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        cbar.ax.set_yticklabels(ticklabs, fontsize=14, fontname='Times New Roman')

    plt.xticks([], [])
    plt.yticks([], [])
    plt.savefig(os.path.join(save_dir, 'recon_full.png'), bbox_inches='tight',
                transparent=True, pad_inches=0.0)
    plt.savefig(os.path.join(save_dir, 'recon_full.svg'), bbox_inches='tight',
                transparent=True, pad_inches=0.0)

    # Save the patches of interest
    patch1 = img_full[z1h - dh:z1h, z1w:z1w + dw]
    patch2 = img_full[z2h - dh:z2h, z2w:z2w + dw]

    # Plot and save patches
    fig3p1 = plt.figure(figsize=(5.0, 5.0))
    ax1 = fig3p1.gca()
    ax1.tick_params(left=False, bottom=False)
    ax1.set_xticks([])
    ax1.set_yticks([])
    img3p1 = ax1.imshow(patch1, cmap=cmap_color, vmin=np.min(img_full), vmax=np.max(img_full))
    plt.xticks([], [])
    plt.yticks([], [])
    plt.savefig(os.path.join(save_dir, 'recon_full_p1.png'), bbox_inches='tight',
                transparent=True, pad_inches=0.0)
    plt.savefig(os.path.join(save_dir, 'recon_full_p1.svg'), bbox_inches='tight',
                transparent=True, pad_inches=0.0)

    fig3p2 = plt.figure(figsize=(5.0, 5.0))
    ax1 = fig3p2.gca()
    ax1.tick_params(left=False, bottom=False)
    ax1.set_xticks([])
    ax1.set_yticks([])
    img3p2 = ax1.imshow(patch2, cmap=cmap_color, vmin=np.min(img_full), vmax=np.max(img_full))
    plt.xticks([], [])
    plt.yticks([], [])
    plt.savefig(os.path.join(save_dir, 'recon_full_p2.png'), bbox_inches='tight',
                transparent=True, pad_inches=0.0)
    plt.savefig(os.path.join(save_dir, 'recon_full_p2.svg'), bbox_inches='tight',
                transparent=True, pad_inches=0.0)