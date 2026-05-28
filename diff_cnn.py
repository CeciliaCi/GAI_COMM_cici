"""
Train and test script for the DMCE.
"""
from DMCE import utils, DiffusionModel, Trainer, Tester, CNN
import os
import os.path as path
import argparse
import modules.utils as ut
import datetime
import csv
import matplotlib.pyplot as plt
import numpy as np
import torch
from DMCE.utils import cmplx2real
from loaders import Channels

CUDA_DEFAULT_ID = 0


def load_leo_files(files):
    return Channels(files=files, strict_dtype=True).channels


def convert_complex_channels(data):
    data = torch.from_numpy(np.asarray(data[:, None, :]))
    return cmplx2real(data, dim=1, new_dim=False).float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', '-d', default='cpu', type=str)
    parser.add_argument('--channel-type', '--ch-type', default='3gpp', type=str)
    parser.add_argument('--data-dir', default='dataset', type=str)
    parser.add_argument('--scenario', default=None, type=str)
    parser.add_argument('--los', default=None, type=str)
    parser.add_argument('--height-km', default=None, type=int)
    parser.add_argument('--elevation', default=None, type=str, help='Elevation interval such as 30-90')
    parser.add_argument('--paths', default=None, type=int)
    parser.add_argument('--seeds', nargs='*', default=None, type=int)
    parser.add_argument('--train-samples', default=None, type=int)
    parser.add_argument('--val-samples', default=None, type=int)
    parser.add_argument('--test-samples', default=None, type=int)
    parser.add_argument('--split-ratios', nargs=3, default=(0.8, 0.1, 0.1), type=float)
    parser.add_argument('--rx-antennas', default=None, type=int)
    parser.add_argument('--tx-antennas', default=None, type=int)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--snr-min-db', default=-15, type=float)
    parser.add_argument('--snr-max-db', default=20, type=float)
    parser.add_argument('--snr-step-db', default=5, type=float)
    parser.add_argument('--train-files', nargs='+', default=None, type=str)
    parser.add_argument('--val-files', nargs='+', default=None, type=str)
    parser.add_argument('--test-files', nargs='+', default=None, type=str)

    # get the used device
    args = parser.parse_args()
    device = args.device

    date_time_now = datetime.datetime.now()
    date_time = date_time_now.strftime('%Y-%m-%d_%H-%M-%S')  # convert to str compatible with all OSs

    ch_type = args.channel_type
    ch_type_lower = ch_type.lower()
    is_leo = ch_type_lower.startswith('leo')
    n_dim = args.rx_antennas or (16 if is_leo else 64) # RX antennas
    n_dim2 = args.tx_antennas or (144 if is_leo else 16) # TX antennas
    num_train_samples = args.train_samples if args.train_samples is not None else (None if is_leo else 100_000)
    num_val_samples = args.val_samples if args.val_samples is not None else (None if is_leo else 10_000)
    num_test_samples = args.test_samples if args.test_samples is not None else (None if is_leo else 10_000)
    seed = 453451

    return_all_timesteps = False # evaluates all intermediate MSEs
    fft_pre = True # learn channel distribution in angular domain through Fourier transform

    # set data params
    n_path = args.paths if args.paths is not None else (None if is_leo else 3)
    if n_dim2 > 1:
        mode = '2D'
    else:
        mode = '1D'
    complex_data = True

    file_split_mode = any([args.train_files, args.val_files, args.test_files])
    if file_split_mode:
        if not is_leo:
            raise ValueError('Explicit --train-files/--val-files/--test-files are only supported for LEO data.')
        if not args.train_files or not args.val_files:
            raise ValueError('Explicit file split mode requires both --train-files and --val-files.')
        train_files = args.train_files
        val_files = args.val_files
        test_uses_val_files = args.test_files is None
        test_files = args.test_files if args.test_files is not None else val_files
        data_train = load_leo_files(train_files)
        data_val = load_leo_files(val_files)
        data_test = data_val if test_uses_val_files else load_leo_files(test_files)
    else:
        train_files = None
        val_files = None
        test_files = None
        test_uses_val_files = False
        data_train, data_val, data_test = ut.load_or_create_data(ch_type=ch_type, n_path=n_path, n_antennas_rx=n_dim,
                                         n_antennas_tx=n_dim2, n_train_ch=num_train_samples, n_val_ch=num_val_samples,
                                         n_test_ch=num_test_samples, return_toep=False, data_dir=args.data_dir,
                                         scenario=args.scenario, los=args.los, height_km=args.height_km,
                                         elevation=args.elevation, seeds=args.seeds,
                                         split_ratios=tuple(args.split_ratios), random_seed=seed)
    num_train_samples = data_train.shape[0]
    num_val_samples = data_val.shape[0]
    num_test_samples = data_test.shape[0]
    if ch_type_lower.startswith('3gpp') and n_dim2 > 1:
        data_train = np.reshape(data_train, (-1, n_dim, n_dim2), 'F')
        data_test = np.reshape(data_test, (-1, n_dim, n_dim2), 'F')
        data_val = np.reshape(data_val, (-1, n_dim, n_dim2), 'F')

    power_normalization_factor = None
    if file_split_mode:
        avg_train_power = np.mean(np.abs(data_train) ** 2)
        if avg_train_power <= 0:
            raise ValueError('Cannot normalize channels with zero average train power.')
        power_normalization_factor = float(np.sqrt(avg_train_power))
        data_train = data_train / power_normalization_factor
        data_val = data_val / power_normalization_factor
        data_test = data_val if test_uses_val_files else data_test / power_normalization_factor
        print(f'Normalize train/val/test channels by train power factor {power_normalization_factor:.6e}.')

    data_train = convert_complex_channels(data_train)
    data_val = convert_complex_channels(data_val)
    data_test = convert_complex_channels(data_test)
    if ch_type_lower.startswith('3gpp'):
        ch_type += f'_path={n_path}'
    elif is_leo:
        if file_split_mode:
            train_label = path.splitext(path.basename(train_files[0]))[0].replace('LEO_', '')
            val_label = path.splitext(path.basename(val_files[0]))[0].replace('LEO_', '')
            if test_uses_val_files:
                label_parts = ['leo', train_label, 'train', val_label, 'valtest']
            else:
                test_label = path.splitext(path.basename(test_files[0]))[0].replace('LEO_', '')
                label_parts = ['leo', train_label, 'train', val_label, 'val', test_label, 'test']
        else:
            label_parts = ['leo']
            if args.scenario:
                label_parts.append(args.scenario)
            if args.los:
                label_parts.append(args.los)
            if args.height_km is not None:
                label_parts.append(f'h{args.height_km}km')
            if args.elevation:
                label_parts.append(f'el{args.elevation}')
            if n_path is not None:
                label_parts.append(f'path={n_path}')
        ch_type = '_'.join(label_parts)

    # set data params
    cwd = os.getcwd()
    bin_dir = path.join(cwd, 'bin')
    data_shape = tuple(data_train.shape[1:])

    # data parameter dictionary, which is saved in 'sim_params.json'
    data_dict = {
        'bin_dir': str(bin_dir),
        'num_train_samples': num_train_samples,
        'num_val_samples': num_val_samples,
        'num_test_samples': num_test_samples,
        'train_dataset': ch_type,
        'test_dataset': ch_type,
        'file_split_mode': file_split_mode,
        'train_files': train_files,
        'val_files': val_files,
        'test_files': test_files,
        'test_uses_val_files': test_uses_val_files,
        'power_normalization_factor': power_normalization_factor,
        'n_antennas': n_dim,
        'n_antennas_tx': n_dim2,
        'mode': mode,
        'data_shape': data_shape,
        'complex_data': complex_data
    }

    # set Diffusion model params
    num_timesteps = 100 #int(np.random.choice([100, 300, 500, 1_000, 2_000]))
    loss_type = 'l2'
    which_schedule = 'linear'

    max_snr_dB = 40
    beta_start = 1 - 10**(max_snr_dB/10) / (1 + 10**(max_snr_dB/10))
    if num_timesteps == 5:
        beta_end = 0.95  # -22.5dB
    elif num_timesteps == 10:
        beta_end = 0.7  # -22.5dB
    elif num_timesteps == 50:
        beta_end = 0.2  # -22.5dB
    elif num_timesteps == 100:
        beta_end = 0.1 # -22.5dB
    elif num_timesteps == 300:
        beta_end = 0.035  # -23dB
    elif num_timesteps == 500:
        beta_end = 0.02 #-22dB
    elif num_timesteps == 1_000:
        beta_end = 0.01 #-22dB
    elif num_timesteps == 10_000:
        beta_end = 0.001 #-24dB
    else:
        beta_end = 0.035
    objective = 'pred_noise'  # one of 'pred_noise' (L_n), 'pred_x_0' (L_h), 'pred_post_mean' (L_mu)
    loss_weighting = False # bool(np.random.choice([True, False]))
    clipping = False
    reverse_method = 'reverse_mean'  # either 'reverse_mean' or 'ground_truth'
    reverse_add_random = False  # True: PDF Sampling method | False: Reverse Mean Forwarding method

    # diffusion model parameter dictionary, which is saved in 'sim_params.json'
    diff_model_dict = {
        'data_shape': data_shape,
        'complex_data': complex_data,
        'loss_type': loss_type,
        'which_schedule': which_schedule,
        'num_timesteps': num_timesteps,
        'beta_start': beta_start,
        'beta_end': beta_end,
        'objective': objective,
        'loss_weighting': loss_weighting,
        'clipping': clipping,
        'reverse_method': reverse_method,
        'reverse_add_random': reverse_add_random
    }

    kernel_size = (3, 3)
    n_layers_pre = 2
    max_filter = 64
    ch_layers_pre = np.linspace(start=1, stop=max_filter, num=n_layers_pre+1, dtype=int)
    ch_layers_pre[0] = 2
    ch_layers_pre = tuple(ch_layers_pre)
    ch_layers_pre = tuple(int(x) for x in ch_layers_pre)
    n_layers_post = 3
    ch_layers_post = np.linspace(start=1, stop=max_filter, num=n_layers_post+1, dtype=int)
    ch_layers_post[0] = 2
    ch_layers_post = ch_layers_post[::-1]
    ch_layers_post = tuple(ch_layers_post)
    ch_layers_post = tuple(int(x) for x in ch_layers_post)
    n_layers_time = 1
    ch_init_time = 16
    batch_norm = False
    downsamp_fac = 1

    # batch_norm = True
    cnn_dict = {
        'data_shape': data_shape,
        'n_layers_pre': n_layers_pre,
        'n_layers_post': n_layers_post,
        'ch_layers_pre': ch_layers_pre,
        'ch_layers_post': ch_layers_post,
        'n_layers_time': n_layers_time,
        'ch_init_time': ch_init_time,
        'kernel_size': kernel_size,
        'mode': mode,
        'batch_norm': batch_norm,
        'downsamp_fac': downsamp_fac,
        'device': device,
    }

    # set Trainer params
    batch_size = 128
    lr_init = 1e-4
    lr_step_multiplier = 1.0
    epochs_until_lr_step = 150
    num_epochs = args.epochs
    val_every_n_batches = 2000
    num_min_epochs = 50
    num_epochs_no_improve = 20
    track_val_loss = True
    track_fid_score = False
    track_mmd = False
    use_fixed_gen_noise = True
    use_ray = False
    save_mode = 'best' # newest, all
    dir_result = path.join(cwd, 'results')
    timestamp = utils.get_timestamp()
    dir_result = path.join(dir_result, timestamp)

    # Trainer parameter dictionary, which is saved in 'sim_params.json'
    trainer_dict = {
        'batch_size': batch_size,
        'lr_init': lr_init,
        'lr_step_multiplier': lr_step_multiplier,
        'epochs_until_lr_step': epochs_until_lr_step,
        'num_epochs': num_epochs,
        'val_every_n_batches': val_every_n_batches,
        'track_val_loss': track_val_loss,
        'track_fid_score': track_fid_score,
        'track_mmd': track_mmd,
        'use_fixed_gen_noise': use_fixed_gen_noise,
        'save_mode': save_mode,
        'mode': mode,
        'dir_result': str(dir_result),
        'use_ray': use_ray,
        'complex_data': complex_data,
        'num_min_epochs': num_min_epochs,
        'num_epochs_no_improve': num_epochs_no_improve,
        'fft_pre': fft_pre,
    }

    # set Tester params
    batch_size_test = 512
    criteria = ['nmse']

    # Tester parameter dictionary, which is saved in 'sim_params.json'
    tester_dict = {
        'batch_size': batch_size_test,
        'criteria': criteria,
        'complex_data': complex_data,
        'return_all_timesteps': return_all_timesteps,
        'fft_pre': fft_pre,
        'mode': mode,
        'snr_min_db': args.snr_min_db,
        'snr_max_db': args.snr_max_db,
        'snr_step_db': args.snr_step_db,
    }

    # create result directory
    os.makedirs(dir_result, exist_ok=True)

    # instantiate CNN, DiffusionModel, Trainer and Tester
    cnn = CNN(**cnn_dict)
    diffusion_model = DiffusionModel(cnn, **diff_model_dict)
    trainer = Trainer(diffusion_model, data_train, data_val, **trainer_dict)
    tester = Tester(diffusion_model, data_test, **tester_dict)

    # Print number of trainable parameters
    print(f'Number of trainable model parameters: {diffusion_model.num_parameters}')

    # other parameters dictionary, which is saved in 'sim_params.json'
    misc_dict = {'num_parameters': diffusion_model.num_parameters}

    # save the simulation parameters as a JSON file
    sim_dict = {
        'data_dict': data_dict,
        'diff_model_dict': diff_model_dict,
        'unet_dict': cnn_dict,
        'trainer_dict': trainer_dict,
        'tester_dict': tester_dict,
        'misc_dict': misc_dict
    }

    utils.save_params(dir_result=dir_result, filename='sim_params', params=sim_dict)

    # run training routine
    train_dict = trainer.train()
    utils.save_params(dir_result=dir_result, filename='train_results', params=train_dict)

    params = dict()
    params['dim'] = n_dim
    params['dim2'] = n_dim2
    params['data_train'] = num_train_samples
    params['data_test'] = num_test_samples
    params['data_val'] = num_val_samples
    params['epochs'] = num_epochs
    params['batch_size'] = batch_size
    params['lr_start'] = lr_init
    params['lr_step_mult'] = lr_step_multiplier
    params['epochs_until_lr_step'] = epochs_until_lr_step
    params['timesteps'] = num_timesteps
    params['beta_start'] = beta_start
    params['beta_end'] = beta_end
    params['snr_low'] = diffusion_model.snrs_db.cpu().detach().numpy()[-1]
    params['snr_high'] = diffusion_model.snrs_db.cpu().detach().numpy()[0]
    params['eval_snr_min_db'] = args.snr_min_db
    params['eval_snr_max_db'] = args.snr_max_db
    params['eval_snr_step_db'] = args.snr_step_db
    params['dataset_train'] = ch_type
    params['dataset_test'] = ch_type
    params['file_split_mode'] = file_split_mode
    params['train_files'] = train_files
    params['val_files'] = val_files
    params['test_files'] = test_files
    params['test_uses_val_files'] = test_uses_val_files
    params['power_normalization_factor'] = power_normalization_factor
    params['schedule'] = which_schedule
    params['kernel_size'] = kernel_size
    params['timestamp'] = timestamp
    params['trained_epochs'] = train_dict['trained_epochs']
    params['num_min_epochs'] = num_min_epochs
    params['num_epochs_no_improve'] = num_epochs_no_improve
    params['loss_weighting'] = loss_weighting
    params['n_layers_pre'] = n_layers_pre
    params['ch_layers_pre'] = ch_layers_pre
    params['n_layers_post'] = n_layers_post
    params['ch_layers_post'] = ch_layers_post
    params['n_layers_time'] = n_layers_time
    params['ch_init_time'] = ch_init_time
    params['num_learnable_params'] = diffusion_model.num_parameters
    params['fft_pre'] = fft_pre
    params['batch_norm'] = batch_norm
    params['downsamp_fac'] = downsamp_fac

    params['seed'] = seed
    os.makedirs('./results/dm_est/', exist_ok=True)
    file_name = f'./results/dm_est/{date_time}_{ch_type}_dim={n_dim}x{n_dim2}_valdata={num_val_samples}_' \
                f'T={num_timesteps}_params.csv'
    with open(file_name, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in params.items():
           writer.writerow([key, value])


    file_name = f'./results/dm_est/{date_time}_{ch_type}_dim={n_dim}x{n_dim2}_valdata={num_val_samples}_' \
                f'T={num_timesteps}_loss.png'
    plt.figure()
    plt.semilogy(range(1, len(train_dict['train_losses'])+1), train_dict['train_losses'], label='train-loss')
    plt.semilogy(range(1, len(train_dict['val_losses'])+1), train_dict['val_losses'], label='val-loss')
    #plt.plot(range(1, params['epochs'] + 1), losses_all_test, label='val-loss')
    plt.legend(['train-loss', 'val-loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(file_name)

    # run testing routine
    test_dict = tester.test()

    if return_all_timesteps:
        # plot all curves
        file_name = f'./results/dm_est/{date_time}_{ch_type}_dim={n_dim}x{n_dim2}_valdata={num_val_samples}_' \
                    f'T={num_timesteps}_perstep.png'
        plt.figure()
        lines = []
        for isnr in range(len(test_dict[criteria[0]]['NMSEs_total_power'])):
            mse_list_allsteps = test_dict[criteria[0]]['NMSEs_total_power'][isnr]
            snr_now = test_dict[criteria[0]]['SNRs'][isnr]
            n_timesteps_eval = len(mse_list_allsteps)
            lines += plt.semilogy(range(num_timesteps-n_timesteps_eval+1, num_timesteps+1), mse_list_allsteps, label=f'SNR = {int(snr_now)}')
            #plt.legend([f'SNR = {int(snr_now)}'])
            plt.xlabel('Timesteps')
            plt.ylabel('nMSE')
        labels = [l.get_label() for l in lines]
        plt.legend(lines, labels)
        plt.savefig(file_name)

        # save all mses
        mse_list = list()
        mse_list.append(test_dict[criteria[0]]['SNRs'].copy())
        mse_list[-1].insert(0, 'SNR')
        mse_list.append(test_dict[criteria[0]]['NMSEs_total_power'].copy())
        mse_list[-1].insert(0, 'nmse_dm')
        mse_list = [list(i) for i in zip(*mse_list)]
        print(mse_list)
        file_name = f'./results/dm_est/{date_time}_{ch_type}_dim={n_dim}x{n_dim2}_valdata={num_val_samples}_T={num_timesteps}_perstep.csv'
        with open(file_name, 'w') as myfile:
            wr = csv.writer(myfile, lineterminator='\n')
            wr.writerows(mse_list)

        # remove all mses except last to save it later
        for isnr in range(len(test_dict[criteria[0]]['NMSEs_total_power'])):
            test_dict[criteria[0]]['NMSEs_total_power'][isnr] = test_dict[criteria[0]]['NMSEs_total_power'][isnr][-1]

    mse_list = list()
    mse_list.append(test_dict[criteria[0]]['SNRs'].copy())
    mse_list[-1].insert(0, 'SNR')
    mse_list.append(test_dict[criteria[0]]['NMSEs_total_power'].copy())
    mse_list[-1].insert(0, 'nmse_dm')
    mse_list = [list(i) for i in zip(*mse_list)]
    print(mse_list)
    file_name = f'./results/dm_est/{date_time}_{ch_type}_dim={n_dim}x{n_dim2}_valdata={num_val_samples}_T={num_timesteps}.csv'
    with open(file_name, 'w') as myfile:
        wr = csv.writer(myfile, lineterminator='\n')
        wr.writerows(mse_list)

    utils.save_params(dir_result=dir_result, filename='test_results', params=test_dict)


if __name__ == '__main__':
    main()
