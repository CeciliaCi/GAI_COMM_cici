import modules.utils as ut
import csv
import datetime
from estimators.lmmse import LMMSE, mp_eval
import numpy as np
import multiprocessing as mp
import os
import argparse


def mp_gmm(obj, *args):
    return obj.estimate_from_y(*args)

def mp_omp(obj, *args):
    return obj.estimate(*args)


def nmse(x, x_hat):
    return np.sum(np.abs(x - x_hat) ** 2) / np.sum(np.abs(x) ** 2)


def angular_diag_lmmse(y, angular_variance, sigma2):
    y_ang = np.fft.fft2(y, axes=(-2, -1), norm='ortho')
    h_hat_ang = (angular_variance / (angular_variance + sigma2))[None, :, :] * y_ang
    return np.fft.ifft2(h_hat_ang, axes=(-2, -1), norm='ortho')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--channel-type', '--ch-type', default='leo', type=str)
    parser.add_argument('--data-dir', default='dataset', type=str)
    parser.add_argument('--scenario', default='Rural', type=str)
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
    parser.add_argument('--processes', default=None, type=int)
    parser.add_argument('--lmmse-mode', default='angular_diag', choices=['angular_diag', 'global_full', 'none'],
                        help='LMMSE baseline variant. angular_diag is the stable default for LEO channels.')
    parser.add_argument('--no-normalize-power', action='store_true',
                        help='Use raw channel amplitudes instead of normalizing average element power to 1.')
    args = parser.parse_args()

    ch_type = args.channel_type
    ch_type_lower = ch_type.lower()
    is_leo = ch_type_lower.startswith('leo')
    n_processes = args.processes or min(2, max(1, int(mp.cpu_count() / 2)))
    pool = None
    if args.lmmse_mode == 'global_full':
        print('Uses ' + str(n_processes) + ' processes')
        pool = mp.Pool(processes=n_processes)

    date_time_now = datetime.datetime.now()
    date_time = date_time_now.strftime('%Y-%m-%d_%H-%M-%S')  # convert to str compatible with all OSs

    n_antennas_rx = args.rx_antennas or (16 if is_leo else 64)
    n_antennas_tx = args.tx_antennas or (144 if is_leo else 16)
    n_train_ch = args.train_samples if args.train_samples is not None else (800 if is_leo else 100_000)
    n_val_ch = args.val_samples if args.val_samples is not None else (100 if is_leo else 10_000)
    n_test_ch = args.test_samples if args.test_samples is not None else (100 if is_leo else 10_000)
    snrs = [-15, -10, -5, 0, 5, 10, 15, 20]
    n_path = args.paths if args.paths is not None else (None if is_leo else 3)

    eval_LS = True
    eval_lmmse_genie = ch_type == '3gpp' and args.lmmse_mode == 'global_full'

    channels_train, toep_train, channels_val, _, channels_test, toep_test = ut.load_or_create_data(ch_type=ch_type,
                            n_path=n_path, n_antennas_rx=n_antennas_rx, n_antennas_tx=n_antennas_tx,
                            n_train_ch=n_train_ch, n_val_ch=n_val_ch, n_test_ch=n_test_ch, return_toep=True,
                            data_dir=args.data_dir, scenario=args.scenario, los=args.los,
                            height_km=args.height_km, elevation=args.elevation, seeds=args.seeds,
                            split_ratios=tuple(args.split_ratios))

    n_antennas = n_antennas_rx * n_antennas_tx
    matrix_channels = channels_train.ndim == 3

    avg_element_power = np.mean(np.abs(channels_train) ** 2)
    print(f'Average train channel element power before normalization: {avg_element_power:.6e}')
    if not args.no_normalize_power:
        if avg_element_power <= 0:
            raise ValueError('Cannot normalize channels with zero average power.')
        norm_factor = np.sqrt(avg_element_power)
        channels_train = channels_train / norm_factor
        channels_test = channels_test / norm_factor
        print('Normalize train/test channels to unit average element power for SNR-based AWGN baselines.')

    if matrix_channels:
        channels_train_matrix = channels_train
        channels_test_matrix = channels_test
        channels_train_vector = np.reshape(channels_train, (-1, n_antennas), 'F')
        channels_test_vector = np.reshape(channels_test, (-1, n_antennas), 'F')
    else:
        channels_train_matrix = None
        channels_test_matrix = None
        channels_train_vector = channels_train
        channels_test_vector = channels_test

    mse_list = list()
    mse_list.append(snrs.copy())
    mse_list[-1].insert(0, 'SNR')

    if args.lmmse_mode == 'angular_diag':
        if channels_train_matrix is None or channels_test_matrix is None:
            raise ValueError('angular_diag LMMSE requires matrix channels with shape [samples, rx, tx].')
        mse_list.append(['lmmse_ang_diag'])
        train_ang = np.fft.fft2(channels_train_matrix, axes=(-2, -1), norm='ortho')
        angular_variance = np.mean(np.abs(train_ang) ** 2, axis=0)
        for snr in snrs:
            y = ut.get_observation(channels_test_matrix, snr)
            sigma2 = 10 ** (-snr / 10)
            h_hat = angular_diag_lmmse(y, angular_variance, sigma2)
            mse_list[-1].append(nmse(channels_test_matrix, h_hat))
    elif args.lmmse_mode == 'global_full':
        mse_list.append(['lmmse_glob'])
        cov = np.zeros([n_antennas, n_antennas], dtype=complex)
        for i in range(channels_train_vector.shape[0]):
            cov = cov + np.expand_dims(channels_train_vector[i, :], 1) @ np.expand_dims(channels_train_vector[i, :].conj(), 0)
        cov = cov / channels_train_vector.shape[0]
        eval_list_glob = list()
        for snr in snrs:
            y = ut.get_observation(channels_test_vector, snr)
            eval_list_glob.append([LMMSE(snr), y, cov, False])
        res_glob_lmmse = pool.starmap(mp_eval, eval_list_glob)
        for it, res in enumerate(res_glob_lmmse):
            mse_act = nmse(channels_test_vector, res)
            mse_list[-1].append(mse_act)

    if eval_LS:
        mse_list.append(['LS'])
        channels_test_ls = channels_test_matrix if channels_test_matrix is not None else channels_test_vector
        for snr in snrs:
            y = ut.get_observation(channels_test_ls, snr)
            mse_act = nmse(channels_test_ls, y)
            mse_list[-1].append(mse_act)


    if ch_type == '3gpp' and eval_lmmse_genie:
        mse_list.append(['lmmse_genie'])
        eval_list_genie = list()
        for snr in snrs:
            y = ut.get_observation(channels_test_vector, snr)
            eval_list_genie.append([LMMSE(snr), y, toep_test, True])
        res_genie_lmmse = pool.starmap(mp_eval, eval_list_genie)
        for it, res in enumerate(res_genie_lmmse):
            mse_act = nmse(channels_test_vector, res)
            mse_list[-1].append(mse_act)


    mse_list = [list(i) for i in zip(*mse_list)]
    print(mse_list)
    os.makedirs('./results/baselines/', exist_ok=True)
    label_parts = [ch_type]
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
    file_label = '_'.join(label_parts)
    file_name = f'./results/baselines/{date_time}_{file_label}_ant={n_antennas_rx}x{n_antennas_tx}_' \
                f'testdata={channels_test.shape[0]}.csv'
    with open(file_name, 'w') as myfile:
        wr = csv.writer(myfile, lineterminator='\n')
        wr.writerows(mse_list)
    if pool is not None:
        pool.close()
        pool.join()


if __name__ == '__main__':
    main()
