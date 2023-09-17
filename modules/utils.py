'''
This program contains misc. functions needed
to model radar propagation and preprocess data.
'''
import cv2
import torch
import numpy as np
import scipy


def psnr(x, xhat):
    ''' Compute Peak Signal to Noise Ratio in dB

        Inputs:
            x: Ground truth signal
            xhat: Reconstructed signal

        Outputs:
            snrval: PSNR in dB
    '''
    err = x - xhat
    denom = np.mean(pow(err, 2))

    snrval = 10 * np.log10(np.max(x) / denom)

    return snrval


def is_mat_file(filename):
    return any(filename.endswith(extension) for extension in ['.mat'])


def awgn(s, SNRdB, device):
    '''
    Add complex noise to a signal
    Inputs:
        :param s: input signal (real or complex)
        :param SNRdB: noise level in decibels
        :param device: hardware to store tensors
    :return:
        signal with AWGN
    '''
    gamma = 10 ** (SNRdB / 10)  # SNR to linear scale
    P = torch.sum(torch.pow(torch.abs(s), 2)) / torch.numel(s)  # signal power
    N0 = P / gamma  # find the noise spectral density
    if torch.any(torch.isreal(s)):
        n = torch.sqrt(N0 / 2) * torch.normal(0, 1, size=s.shape, device=device)
    else:
        n = torch.sqrt(N0 / 2) * (
                torch.normal(0, 1, size=s.shape, device=device) + 1j * torch.normal(0, 1, size=s.shape,
                                                                                    device=device))
    r = s + n  # received signal
    return r


def coord2image(reflect_cart, rho, N_ang, N_rng, device):
    # Define query point cartesian locations
    # width = range dimension
    # height = angle dimension

    # Based on cascade mmWave parameters
    # range resolution: 0.059335 m
    # max range: 15.18987 m infront of radar

    r = torch.linspace(0, 15.18987, steps=N_rng, device=device)  # dependent on the width of scene
    a = torch.arcsin_(torch.linspace(-1, 1, steps=N_ang, device=device))  # dependent on the height of scene

    R, A = torch.meshgrid(r, a, indexing='xy')

    coords_sph = torch.hstack((R.reshape(-1, 1), A.reshape(-1, 1)))[None, ...]

    # Convert from spherical to cartesian coordinates
    # coords_cart_x = (coords_sph[:,:, 0] * torch.cos(coords_sph[:,:, 1])).reshape(-1,)
    # coords_cart_y = (coords_sph[:,:, 0] * torch.sin(coords_sph[:,:, 1])).reshape(-1,)

    image = torch.zeros((N_ang, N_rng), device=device)
    reflect_cart2 = torch.zeros_like(reflect_cart, device=device)

    for kp in range(0, reflect_cart.shape[0]):
        ang_tmp = torch.atan2(reflect_cart[kp, 1], reflect_cart[kp, 0])
        rng_tmp = torch.sqrt(reflect_cart[kp, 0] ** 2 + reflect_cart[kp, 1] ** 2)

        ang_idx = torch.argmin(torch.abs(a - ang_tmp))
        rng_idx = torch.argmin(torch.abs(r - rng_tmp))

        # Convert closest polar to cart coordinates
        x_tmp = r[rng_idx] * torch.cos(a[ang_idx])
        y_tmp = r[rng_idx] * torch.sin(a[ang_idx])
        reflect_cart2[kp, 0] = x_tmp
        reflect_cart2[kp, 1] = y_tmp

        image[ang_idx, rng_idx] = image[ang_idx, rng_idx] + rho[kp]

    return image, reflect_cart2


def prop_sph(reflect_cart, array_cart, tx_cart, rho, device):
    '''
    Function generates a synthetic radar datacube given antenna positions, trasmitter
    position, reflector positions, reflectivity of each reflector
    Inputs:
        :param reflect_cart:
        :param array_cart:
        :param tx_cart:
        :param rho:
        :param device:
    :return:
        complex radar datacube
    '''
    # radar parameters taken from ColoRadar dataset cascade mmWave radar
    RADAR_PARAMS = {'samples': 256,  # number of samples per chirp
                    'Rs': 8000000,  # ADC sampling frequency [Hz]
                    'As': 7.90000010527e+13,  # FMCW frequency sweep slope
                    'f_start': 76999999488.0,  # FMCW start frequency [Hz]
                    'idle_time': 4.99999987369e-06,  # FMCW delay between chirps [s]
                    'adc_start_time': 6.0,  # time delay from chirp start to sampling start
                    'ramp_end_time': 3.99999998402e-11}  # encompasses time ramp is started to when it ends
    c = 299792458  # wave speed [m/s]
    fc = RADAR_PARAMS['f_start']  # radar center frequency [Hz]
    lambda_c = c / fc  # radar wavelength [m]

    N_t = torch.tensor(RADAR_PARAMS['samples']).to(device)  # number of time samples
    N_ant = array_cart.shape[0]  # number of antennas in linear array
    N_ref = reflect_cart.shape[0]  # number of reflectors in the scene

    # Generate time sample vector
    t_ax = (1 / RADAR_PARAMS['Rs']) * torch.arange(0, N_t, 1, dtype=torch.float, device=device).unsqueeze(0)  # [1,N_t]
    t_ax = torch.repeat_interleave(t_ax, N_ant, dim=0)  # [N_ant,N_t]

    datacube = torch.zeros((N_ant, N_t), dtype=torch.cfloat, device=device)
    range_mat = torch.zeros((N_ant, N_ref), dtype=torch.float, device=device)
    # B = torch.zeros((N_ant, N_t, N_ref), dtype=torch.cfloat, device=device)

    # Cycle through all reflectors in the scene
    for kp in range(0, N_ref):
        # Compute the round trip time of flight
        d_TX2reflector = torch.sqrt((tx_cart[0] - reflect_cart[kp, 0]) ** 2 +
                                    (tx_cart[1] - reflect_cart[kp, 1]) ** 2 +
                                    (tx_cart[2] - reflect_cart[kp, 2]) ** 2)
        d_RX2reflector = torch.sqrt((array_cart[:, 0] - reflect_cart[kp, 0]) ** 2 +
                                    (array_cart[:, 1] - reflect_cart[kp, 1]) ** 2 +
                                    (array_cart[:, 2] - reflect_cart[kp, 2]) ** 2)
        d_tgt = d_TX2reflector + d_RX2reflector  # [N_ant]

        d_tgt = torch.repeat_interleave(d_tgt.unsqueeze(1), N_t, dim=1)  # [N_ant,N_t]
        tau = d_tgt / c  # time of flight for signal

        pt_signal = rho[kp] * torch.exp(1j * 2 * torch.pi * fc * tau) * torch.exp(
            1j * 2 * torch.pi * RADAR_PARAMS['As'] * t_ax * tau)

        datacube = datacube + pt_signal

    return datacube


def prop_plane(data, fftSize):
    # Zero frequency needs to be in the top left corner for 2DFFT, radar datacube
    # has zero frequency in middle row and column zero

    data_shift = torch.fft.ifftshift(data, dim=(-2))
    data_rd = torch.fft.ifft2(data_shift, s=fftSize, dim=(-2, -1), norm="backward")

    return data_rd


def delay_sum(data, fftSize):
    # Zero frequency needs to be in the top left corner for 2DFFT, radar datacube
    # has zero frequency in middle row and column zero, converts radar datacube
    # into a radar heatmap image

    data_sp = torch.fft.fft2(data, s=fftSize, dim=(-2, -1), norm="backward")
    data_sp = torch.fft.fftshift(data_sp, dim=(-2))
    return data_sp


def get_inp(tensize, const= 1 / 10.0, dtype=torch.cfloat, device='cpu'):
    '''
    Get initialization variable
    :param tensize: dimensions for initialization variable
    :param const: scalar to reduce random number generator magnitude
    :return: input that will be optimized
    '''

    inp = torch.rand(tensize) * const
    # inp = torch.randn(tensize) * const
    inp = torch.autograd.Variable(inp, requires_grad=True).to(device).to(dtype)
    inp = torch.nn.Parameter(inp)

    return inp


def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)[None, :]


def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_()
    else:
        assert False


# Deep Image Prior Noise input method
def get_noise(input_depth, method, spatial_size, noise_type='u', var=1. / 10):
    """Returns a pytorch.Tensor of size
    (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`)
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid`
        for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is
        standard deviation scaler.
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)

        fill_noise(net_input, noise_type)
        net_input *= var
    elif method == 'meshgrid':
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1]) / float(spatial_size[1] - 1),
                           np.arange(0, spatial_size[0]) / float(spatial_size[0] - 1))
        meshgrid = np.concatenate([X[None, :], Y[None, :]])
        net_input = np_to_torch(meshgrid).to(dtype=torch.float)
    elif method == 'fourfeature':
        scale = 5
        shape = [1, input_depth // 2, spatial_size[0], spatial_size[1]]
        x = torch.randn(shape, dtype=torch.float) * scale
        x = 2 * torch.pi * x
        net_input = torch.cat([torch.sin(x), torch.cos(x)], dim=1)
    else:
        assert False

    return net_input


def get_scale_factor(net, num_channels, in_size, gt_datacube, N_ang, N_rng, nonlin, device):
    # get the norm of the deep decoder output and scale measurement
    shape = [1, num_channels, in_size[0], in_size[1]]
    ni = torch.autograd.Variable(torch.zeros(shape, device=device))
    ni.data.uniform_()

    # generate random image for the above net input
    if nonlin == 'convDecoder':
        output = net(ni)
        out_img = torch.view_as_complex(torch.permute(output, [0, 2, 3, 1]).contiguous())[None, ...]
    else:
        # Build the input query coordinates
        x = torch.linspace(-1, 1, N_ang).cuda()
        y = torch.linspace(-1, 1, N_rng).cuda()

        X, Y = torch.meshgrid(x, y, indexing='xy')

        coords = torch.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))[None, ...]
        out_img = torch.view_as_complex(net(coords)).reshape(-1, N_ang, N_rng)[None, ...]

    out_img_tt = torch.sqrt(torch.real(out_img) ** 2 + torch.imag(out_img) ** 2)

    # get norm of zero padded image
    orig_img = delay_sum(gt_datacube, (N_ang, N_rng))
    orig_img_tt = torch.sqrt(torch.real(orig_img) ** 2 + torch.imag(orig_img) ** 2)

    # compute the scale factor
    s = np.linalg.norm(out_img_tt.detach().cpu().numpy()) / np.linalg.norm(orig_img_tt.detach().cpu().numpy())
    return s, ni


def get_data(run_idx, frame_idx, DATA_PARAMS, device, data_type='exp', scene_type='outdoor', snr=20):
    # Extract data parameters
    array_style = DATA_PARAMS['array_style']
    N_choose = DATA_PARAMS['num_ant_choose']
    N_ant = DATA_PARAMS['num_ant']
    N_t = DATA_PARAMS['num_time']
    N_ang = DATA_PARAMS['num_ang_bin']
    N_rng = DATA_PARAMS['num_rng_bin']
    array_gap = DATA_PARAMS['array_gap']
    rng_st = DATA_PARAMS['rng_bin_start']

    # Define transmitter and receiver positions (units meters)
    tx_cart = torch.tensor([0, 0, 0])
    array_y = ((torch.arange(0, N_ant, 1, device=device)) * array_gap).unsqueeze(1)
    array_x = torch.zeros(array_y.shape, device=device)
    array_z = torch.zeros(array_y.shape, device=device)
    array_cart = torch.cat([array_x, array_y, array_z], dim=1)  # [N_ant,3]

    if (array_style == 'mra') & (N_choose == 16):
        array_y_root = torch.tensor([0, 1, 4, 6])
        array_y_idx = torch.cat((array_y_root, 46 + array_y_root, 59 + array_y_root, 79 + array_y_root))
        array_y_idx = array_y_idx.sort().values
    elif (array_style == 'mra') & (N_choose == 20):
        array_y_root = torch.tensor([0, 1, 4, 6])
        array_y_idx = torch.cat((array_y_root, 17 + array_y_root,
                                 37 + array_y_root, 53 + array_y_root, 79 + array_y_root)).to(device)  # 36
        array_y_idx = array_y_idx.sort().values
    elif array_style == 'random':
        array_y_idx = torch.randperm(N_ant - 2)[:N_choose] + 1  # make sure first index is not selected
        array_y_idx = array_y_idx.sort().values
        array_y_idx = torch.cat((torch.tensor(0).reshape(1), array_y_idx, torch.tensor(N_ant - 1).reshape(1))).to(
            device=device)
    elif array_style == 'uniform':
        # Select sequential measurement subset indicies to keep
        array_y_idx = torch.arange(0, N_choose, 1, device=device)
    elif array_style =='mimo1':
        # This is the classic mimo array design with uniform dense sampling
        array_y_idx = torch.linspace(0,15,16,dtype=torch.int)
        array_y_idx = array_y_idx.sort().values
    elif array_style == 'mimo2':
        # This is the classic mimo array design maximizing aperture size
        array_y_idx = torch.linspace(0, 75, 16,dtype=torch.int)
        array_y_idx = array_y_idx.sort().values
    else:
        raise ValueError('Incorrect Antenna Sampling')
    # print(f"Selected Antennas: {array_y_idx}")

    # Create datacube sampling mask
    mask = torch.zeros((1, 1, N_ant, N_t), device=device)
    mask[:, :, array_y_idx, :] = 1  # set selected antennas to a mask level of 1

    with torch.no_grad():
        # Load in experimental or generate simulated radar datacube
        if data_type == 'exp':
            if scene_type == 'outdoor':
                dataset_dir = 'coloRadarRiceADC_outdoor'
            elif scene_type == 'indoor':
                dataset_dir = 'C:/Users/farre/PycharmProjects/CoIRv3/coloRadarRiceADC_indoor'
            elif scene_type == 'outdoor_rand':
                dataset_dir = 'C:/Users/farre/PycharmProjects/CoIRv3/coloRadarRiceADC_outdoor_rand'
            else:
                raise ValueError('Incorrect Experimental Scene Type')

            frame_dir = dataset_dir + '/' + str(run_idx) + '/' + str(frame_idx) + '/' + 'datacube.mat'
            # print(frame_dir)
            mat = scipy.io.loadmat(frame_dir)
            exp_datacube = torch.from_numpy(mat['datacube']).to(dtype=torch.cfloat)
            # Keep elevation = 0 and chirp = 0
            full_datacube = (exp_datacube[:, 0, 0, 0:N_t]).to(device=device)[None, None, ...]  # [1,1,86,256]

        elif data_type == 'sim':
            # Load in presaved coordinates and reflectivities
            dataset_dir = 'C:/Users/farre/PycharmProjects/CoIRv3/sim_scenes'
            frame_dir = dataset_dir + '/' + str(run_idx) + '/' + str(frame_idx) + '/' + 'scene.mat'
            mat = scipy.io.loadmat(frame_dir)
            scene = torch.from_numpy(mat['scene']).to(dtype=torch.float)
            rho = torch.from_numpy(mat['reflect'].astype('float')).to(dtype=torch.float).to(device=device)
            rho = rho / torch.max(rho)

            # Remove points close to the detector
            range_calc = torch.sqrt(scene[:, 0] ** 2 + scene[:, 1] ** 2)
            reflect_cart_orig = scene[range_calc > 2, :]
            rho = rho[range_calc > 2]

            rho_img, reflect_cart = coord2image(reflect_cart_orig, rho, N_ang, N_rng, device)

            full_datacube_nf = prop_plane(rho_img,[N_ang,N_rng])
            full_datacube_nf = full_datacube_nf[0:N_ant,0:N_t]

            full_datacube = awgn(full_datacube_nf, snr, device)[None, None, :, :]

        else:
            raise ValueError('Incorrect Experimental Data Type')

    # -----------------------------------------------------------------------------------------------------
    # Data Preprocessing Section
    # -----------------------------------------------------------------------------------------------------

    if data_type == 'sim':
        # Have access to the ground truth noise free full datacube
        # Normalize ground truth datacube
        #full_datacube_nf = full_datacube_nf[None, None, :, :]
        #max_vals = torch.max(torch.abs(full_datacube_nf), dim=3)
        #scale = max_vals.values
        #gt_datacube = full_datacube_nf / torch.repeat_interleave(scale.unsqueeze(3), N_t, dim=3)
        gt_datacube = full_datacube_nf[None, None, :, :]
        # gt_heatmap = torch.flipud(rho_img)[None, None, :, :]
        gt_heatmap = rho_img[None, None, :, :]
        # gt_heatmap = delay_sum(gt_datacube, (N_ang, N_rng))

        # # Apply guassian blur to reflectivity image
        # rho_img = rho_img.cpu().numpy()
        # rho_blur = scipy.ndimage.gaussian_filter(rho_img,sigma=[1.127,0.099])
        # gt_heatmap = torch.from_numpy(rho_blur).to(device=device)[None,None,:,:]

    else:
        # Normalize input datacube
        max_vals = torch.max(torch.abs(full_datacube), dim=3)
        scale = max_vals.values
        full_datacube = full_datacube / torch.repeat_interleave(scale.unsqueeze(3), N_t, dim=3)
        # experimental data we do not have access to ground truth so use full datacube as proxy ground truth
        gt_datacube = full_datacube
        gt_heatmap = delay_sum(gt_datacube, (N_ang, N_rng))

    # Apply mask to full datacube
    sparse_datacube = mask * full_datacube

    # Create radar heatmaps
    full_heatmap = delay_sum(full_datacube, (N_ang, N_rng))
    sparse_heatmap = delay_sum(sparse_datacube, (N_ang, N_rng))


    return gt_datacube,full_datacube,sparse_datacube,\
           gt_heatmap,full_heatmap,sparse_heatmap, mask

# Regularization functions
def TV(x):
    grad_x = x[..., 1:, 1:] - x[..., 1:, :-1]
    grad_y = x[..., 1:, 1:] - x[..., :-1, 1:]
    loss = torch.mean(torch.abs(grad_x)) + torch.mean(torch.abs(grad_y))
    return loss

def L1_reg(x):
    loss = torch.sum(torch.abs(x))
    return loss

def L2_reg(x):
    loss = torch.sum(x**2)
    return loss

def grad_reg(x):
  if x.dim() == 2:
    x = x[None, None, ...]

  kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0,
    -1]]).float().to(x.device)[None, None, ...]
  ky = torch.tensor([[1, 2 ,1], [0, 0, 0], [-1, -2,
    -1]]).float().to(x.device)[None, None, ...]

  eps = 1e-7

  x_d = torch.nn.functional.conv2d(x, kx, padding=1)
  y_d = torch.nn.functional.conv2d(x, ky, padding=1)

  xy_d = torch.sqrt(x_d**2 + y_d**2 + eps).squeeze()

  return torch.mean(xy_d)

