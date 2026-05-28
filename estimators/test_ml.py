#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 14:32:31 2021

@author: marius
"""

import argparse
import copy
import csv
import os
import sys

import torch
from matplotlib import pyplot as plt
import numpy as np

BASELINE_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_DIR = os.path.dirname(BASELINE_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

num_threads = 2
os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ["OMP_DYNAMIC"] = "false"
os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
os.environ["MKL_NUM_THREADS"] = str(num_threads)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)
torch.set_num_threads(num_threads)

from tqdm import tqdm as tqdm
from loaders import Channels, expand_scenarios, infer_channel_image_size
from torch.utils.data import DataLoader


def format_snr_value(snr):
    if float(snr).is_integer():
        return int(snr)
    return float(snr)


# Config args
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--train', type=str, default='Rural')
parser.add_argument('--test', type=str, default='Rural')
parser.add_argument('--spacing', nargs='+', type=float, default=[0.5])
parser.add_argument('--pilot_alpha', type=float, default=0.6)
parser.add_argument('--snr_values', nargs='+', type=float, default=[-15, -10, -5, 0, 5, 10, 15, 20])
parser.add_argument('--num_test_sample', type=int, default=256)
args = parser.parse_args()

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

seed = 10
torch.manual_seed(seed)
np.random.seed(seed)

if args.train == 'O1_28':
    from configs.ve import CE_ncsnpp_deep_continuous as configs
else:
    from configs.ve import CE_ncsnpp_deep_continuous_norm as configs

config = configs.get_config()

config.data.spacing_list = args.spacing

####################################### Prepare datasets #######################################
train_seed, val_seed = 1111, 2222

# Number of samples
num_test_sample = args.num_test_sample

# Load training dataset
config.data.channel = args.train
config.data.scenario_list = expand_scenarios(args.train)
config.data.image_size = infer_channel_image_size(
    config.data.scenario_list,
    seed=train_seed,
    num_paths=config.data.num_paths,
)
config.data.num_pilots = max(1, int(np.floor(config.data.image_size[0] * args.pilot_alpha)))
dataset = Channels(train_seed, config, norm='global')
n_tx, n_rx = dataset.n_tx, dataset.n_rx
args.antennas = [n_tx, n_rx]

# Load validation dataset
val_config = copy.deepcopy(config)
val_config.purpose = 'val'
val_config.data.channel = args.test
val_config.data.scenario_list = expand_scenarios(args.test)
val_config.data.image_size = infer_channel_image_size(
    val_config.data.scenario_list,
    seed=val_seed,
    num_paths=val_config.data.num_paths,
)
if val_config.data.image_size != config.data.image_size:
    raise ValueError(
        f"Train/test channel dimensions do not match: "
        f"train image_size={config.data.image_size}, "
        f"test image_size={val_config.data.image_size}."
    )
val_config.data.spacing_list = args.spacing
val_config.data.num_pilots = config.data.num_pilots
val_dataset = Channels(val_seed, val_config, norm=[dataset.mean, dataset.std])
if len(val_dataset) < num_test_sample:
    raise ValueError(
        f"Only {len(val_dataset)} validation samples are available, "
        f"but num_test_sample={num_test_sample}."
    )
val_loader = DataLoader(val_dataset, batch_size=num_test_sample, shuffle=True, num_workers=0, drop_last=True)
val_iter = iter(val_loader)

print(f"Loaded ML train data from: {dataset.filenames}")
print(f"Loaded ML test data from: {val_dataset.filenames}")
print(f"N_t={n_tx}, N_r={n_rx}, pilots={config.data.num_pilots}, test_samples={num_test_sample}")

# Noise power
snr_range = np.asarray(args.snr_values, dtype=float)
noise_range = 10 ** (-snr_range / 10.)

# Results logging
nmse_all = np.zeros((len(snr_range), num_test_sample))
result_dir = os.path.join(BASELINE_DIR, f'results/ml_baseline/train{args.train}_test{args.test}')
os.makedirs(result_dir, exist_ok=True)

# Get a batch of sample
val_sample = next(val_iter)
del val_iter, val_loader
val_P = val_sample['P']
val_P = torch.conj(torch.transpose(val_P, -1, -2))
val_H_herm = val_sample['H_herm']
val_H = val_H_herm[:, 0] + 1j * val_H_herm[:, 1]
# Convert to numpy vectors
val_P = val_P.resolve_conj().numpy()
val_H = val_H.resolve_conj().numpy()

# For each SNR value
for snr_idx, local_noise in tqdm(enumerate(noise_range), total=len(noise_range)):
    val_Y = np.matmul(val_P, val_H)
    val_Y = val_Y + \
            np.sqrt(local_noise) / np.sqrt(2.) * \
            (np.random.normal(size=val_Y.shape) + \
             1j * np.random.normal(size=val_Y.shape))

    # For each sample
    for sample_idx in tqdm(range(val_Y.shape[0])):
        # Normal equation
        normal_P = np.matmul(val_P[sample_idx].T.conj(), val_P[sample_idx]) + \
                   local_noise * np.eye(val_P[sample_idx].shape[-1])
        normal_Y = np.matmul(val_P[sample_idx].T.conj(), val_Y[sample_idx])
        # Single-shot solve
        est_H, _, _, _ = np.linalg.lstsq(normal_P, normal_Y)
        # Estimate error
        nmse_all[snr_idx, sample_idx] = \
            (np.sum(np.square(np.abs(est_H - val_H[sample_idx])), axis=(-1, -2))) / \
            np.sum(np.square(np.abs(val_H[sample_idx])), axis=(-1, -2))

avg_nmse = np.mean(nmse_all, axis=-1)

csv_rows = [['SNR', 'ML']]
for snr, nmse in zip(snr_range, avg_nmse):
    csv_rows.append([format_snr_value(snr), float(nmse)])

with open(os.path.join(result_dir, 'results.csv'), 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['SNR', 'ml'])
    for snr, nmse in zip(snr_range, avg_nmse):
        writer.writerow([format_snr_value(snr), float(nmse)])

# Plot results
plt.rcParams['font.size'] = 14
plt.figure(figsize=(10, 10))
#plt.plot(snr_range, avg_nmse[0, 0], linewidth=4, label=f'{args.test}, Maximum likelihood')
plt.plot(snr_range, avg_nmse, linewidth=4, label=f'{args.test}, Maximum likelihood')
plt.grid()
plt.legend()
plt.title('Channel estimation')
plt.xlabel('SNR [dB]')
plt.ylabel('NMSE')
plt.tight_layout()
plt.savefig(os.path.join(result_dir, 'results_mse.png'), dpi=300,
            bbox_inches='tight')
plt.close()

# Save to file
# torch.save({'snr_range': snr_range,
#             'spacing': args.spacing,
#             'pilot_alpha': args.pilot_alpha,
#             'nmse_all': nmse_all,
#             'avg_nmse': avg_nmse
#             }, result_dir + f'/results_Nt{args.data.image_size[0]}_Nr{args.data.image_size[1]}.pt')
torch.save({'snr_range': snr_range,
            'spacing': args.spacing,
            'pilot_alpha': args.pilot_alpha,
            'num_test_sample': num_test_sample,
            'nmse_all': nmse_all,
            'avg_nmse': avg_nmse,
            'train_files': dataset.filenames,
            'test_files': val_dataset.filenames,
            'n_tx': n_tx,
            'n_rx': n_rx,
            'num_pilots': config.data.num_pilots,
            }, result_dir + f'/results_Nt{args.antennas[0]}_Nr{args.antennas[1]}.pt')

print(csv_rows)
