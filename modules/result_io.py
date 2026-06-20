import csv
import json
import os
import os.path as path
from glob import glob

import numpy as np


RUN_INDEX_FIELDS = [
    'run_name',
    'run_dir',
    'timestamp',
    'status',
    'channel_type',
    'train_dataset',
    'test_dataset',
    'scenario',
    'los',
    'height_km',
    'elevation',
    'paths',
    'seeds',
    'split_ratios',
    'file_split_mode',
    'train_files',
    'val_files',
    'test_files',
    'test_uses_val_files',
    'num_train_samples',
    'num_val_samples',
    'num_test_samples',
    'power_normalization_factor',
    'n_antennas_rx',
    'n_antennas_tx',
    'mode',
    'data_shape',
    'fft_pre',
    'num_timesteps',
    'beta_start',
    'beta_end',
    'schedule',
    'objective',
    'reverse_method',
    'reverse_add_random',
    'epochs_requested',
    'trained_epochs',
    'batch_size',
    'lr_init',
    'num_parameters',
    'eval_snr_min_db',
    'eval_snr_max_db',
    'eval_snr_step_db',
    'test_nmse_by_snr',
    'best_checkpoint',
    'sim_params_path',
    'train_results_path',
    'test_results_path',
    'dm_est_csv_path',
    'dm_est_params_csv_path',
    'loss_plot_path',
]


def sanitize_run_component(value, *, max_len=80):
    text = str(value)
    chars = [char if char.isalnum() or char in {'_', '-', '.', '='} else '_' for char in text]
    text = ''.join(chars).strip('_')
    while '__' in text:
        text = text.replace('__', '_')
    if not text:
        return 'run'
    return text[:max_len].rstrip('_')


def parse_leo_file_label(file_path):
    stem = path.splitext(path.basename(str(file_path)))[0]
    if stem.startswith('LEO_'):
        stem = stem[len('LEO_'):]

    info = {
        'scenario': None,
        'seed': None,
        'pilot': None,
        'los': None,
        'height': None,
        'elevation': None,
        'paths': None,
        'fallback': sanitize_run_component(stem, max_len=32),
    }
    scenario_parts = []
    for token in stem.split('_'):
        token_lower = token.lower()
        if token_lower.startswith('seed'):
            info['seed'] = token[4:] or None
        elif token in {'LOS', 'NLOS'}:
            info['los'] = token
        elif token_lower.startswith('h') and token_lower.endswith('km'):
            info['height'] = token
        elif token_lower.startswith('el'):
            info['elevation'] = token
        elif token_lower.startswith('path'):
            info['paths'] = token
        elif token_lower.startswith('p') and len(token) > 1:
            info['pilot'] = token
        else:
            scenario_parts.append(token)
    if scenario_parts:
        info['scenario'] = '_'.join(scenario_parts)
    return info


def build_run_name(timestamp, *, ch_type, is_leo, file_split_mode, args,
                   train_files, val_files, test_files, test_uses_val_files, n_path):
    if is_leo:
        parts = ['leo']
        if file_split_mode:
            train_info = parse_leo_file_label(train_files[0])
            val_info = parse_leo_file_label(val_files[0])
            test_info = parse_leo_file_label(test_files[0])

            for key in ('scenario', 'pilot', 'los', 'height', 'elevation', 'paths'):
                value = train_info.get(key) or test_info.get(key)
                if value:
                    parts.append(value)

            parts.append(f"tr{train_info['seed']}" if train_info.get('seed') else f"tr{train_info['fallback']}")
            if test_uses_val_files:
                parts.append(
                    f"valtest{val_info['seed']}" if val_info.get('seed') else f"valtest{val_info['fallback']}"
                )
            else:
                if val_info.get('seed'):
                    parts.append(f"val{val_info['seed']}")
                parts.append(f"te{test_info['seed']}" if test_info.get('seed') else f"te{test_info['fallback']}")
        else:
            if args.scenario:
                parts.append(args.scenario)
            if args.los:
                parts.append(args.los)
            if args.height_km is not None:
                parts.append(f'h{args.height_km}km')
            if args.elevation:
                parts.append(f'el{args.elevation}')
            if n_path is not None:
                parts.append(f'path={n_path}')
            if args.seeds:
                parts.append('seed=' + '-'.join(str(seed) for seed in args.seeds))
            parts.append('split')
    else:
        parts = [ch_type]

    label = sanitize_run_component('_'.join(parts), max_len=120)
    return f'{timestamp}_{label}'


def csv_cell(value):
    if value is None:
        return ''
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_key_value_csv(file_path, params):
    os.makedirs(path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in params.items():
            writer.writerow([key, value])


def write_rows_csv(file_path, rows):
    os.makedirs(path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as csv_file:
        writer = csv.writer(csv_file, lineterminator='\n')
        writer.writerows(rows)


def write_run_index(index_path, rows):
    os.makedirs(path.dirname(index_path), exist_ok=True)
    with open(index_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=RUN_INDEX_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_cell(row.get(field)) for field in RUN_INDEX_FIELDS})


def load_json_file(file_path, default=None):
    if not path.isfile(file_path):
        return {} if default is None else default
    with open(file_path, 'r') as handle:
        return json.load(handle)


def build_dm_est_params(
    *,
    data_dict,
    diff_model_dict,
    cnn_dict,
    trainer_dict,
    tester_dict,
    train_dict,
    ch_type,
    timestamp,
    run_name,
    dir_result,
    cwd,
    seed,
    snrs_db,
    num_parameters,
):
    snrs_db = np.asarray(snrs_db)
    return {
        'dim': data_dict['n_antennas'],
        'dim2': data_dict['n_antennas_tx'],
        'data_train': data_dict['num_train_samples'],
        'data_test': data_dict['num_test_samples'],
        'data_val': data_dict['num_val_samples'],
        'epochs': trainer_dict['num_epochs'],
        'batch_size': trainer_dict['batch_size'],
        'lr_start': trainer_dict['lr_init'],
        'lr_step_mult': trainer_dict['lr_step_multiplier'],
        'epochs_until_lr_step': trainer_dict['epochs_until_lr_step'],
        'timesteps': diff_model_dict['num_timesteps'],
        'beta_start': diff_model_dict['beta_start'],
        'beta_end': diff_model_dict['beta_end'],
        'snr_low': snrs_db[-1],
        'snr_high': snrs_db[0],
        'eval_snr_min_db': tester_dict['snr_min_db'],
        'eval_snr_max_db': tester_dict['snr_max_db'],
        'eval_snr_step_db': tester_dict['snr_step_db'],
        'dataset_train': ch_type,
        'dataset_test': ch_type,
        'file_split_mode': data_dict['file_split_mode'],
        'train_files': data_dict['train_files'],
        'val_files': data_dict['val_files'],
        'test_files': data_dict['test_files'],
        'test_uses_val_files': data_dict['test_uses_val_files'],
        'power_normalization_factor': data_dict['power_normalization_factor'],
        'schedule': diff_model_dict['which_schedule'],
        'kernel_size': cnn_dict['kernel_size'],
        'timestamp': timestamp,
        'run_name': run_name,
        'run_dir': path.relpath(dir_result, cwd),
        'trained_epochs': train_dict['trained_epochs'],
        'num_min_epochs': trainer_dict['num_min_epochs'],
        'num_epochs_no_improve': trainer_dict['num_epochs_no_improve'],
        'loss_weighting': diff_model_dict['loss_weighting'],
        'n_layers_pre': cnn_dict['n_layers_pre'],
        'ch_layers_pre': cnn_dict['ch_layers_pre'],
        'n_layers_post': cnn_dict['n_layers_post'],
        'ch_layers_post': cnn_dict['ch_layers_post'],
        'n_layers_time': cnn_dict['n_layers_time'],
        'ch_init_time': cnn_dict['ch_init_time'],
        'num_learnable_params': num_parameters,
        'fft_pre': trainer_dict['fft_pre'],
        'batch_norm': cnn_dict['batch_norm'],
        'downsamp_fac': cnn_dict['downsamp_fac'],
        'seed': seed,
    }


def nmse_table(test_dict, criterion='nmse'):
    criterion_dict = test_dict[criterion]
    rows = [
        ['SNR', *criterion_dict['SNRs']],
        ['nmse_dm', *criterion_dict['NMSEs_total_power']],
    ]
    return [list(row) for row in zip(*rows)]


def nmse_curve_for_index(test_dict, criterion='nmse'):
    criterion_dict = test_dict.get(criterion, {})
    snrs = criterion_dict.get('SNRs', [])
    nmse_values = criterion_dict.get('NMSEs_total_power', [])
    curve = {}
    for snr, nmse in zip(snrs, nmse_values):
        snr_key = f'{float(snr):g}'
        if isinstance(nmse, (list, tuple)):
            curve[snr_key] = [float(value) for value in nmse]
        else:
            curve[snr_key] = float(nmse)
    return curve


def latest_checkpoint_name(dir_result):
    model_dir = path.join(dir_result, 'train_models')
    if not path.isdir(model_dir):
        return ''
    checkpoints = sorted(file for file in os.listdir(model_dir) if file.endswith('.pt'))
    return checkpoints[-1] if checkpoints else ''


def dm_est_params_path(output_dir, *, date_time, ch_type, n_dim, n_dim2, num_val_samples, num_timesteps):
    return path.join(
        output_dir,
        f'{date_time}_{ch_type}_dim={n_dim}x{n_dim2}_valdata={num_val_samples}_T={num_timesteps}_params.csv',
    )


def dm_est_csv_path(output_dir, *, date_time, ch_type, n_dim, n_dim2, num_val_samples, num_timesteps, perstep=False):
    suffix = '_perstep.csv' if perstep else '.csv'
    return path.join(
        output_dir,
        f'{date_time}_{ch_type}_dim={n_dim}x{n_dim2}_valdata={num_val_samples}_T={num_timesteps}{suffix}',
    )


def append_run_index(index_path, row):
    os.makedirs(path.dirname(index_path), exist_ok=True)
    write_header = not path.exists(index_path) or path.getsize(index_path) == 0
    with open(index_path, 'a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=RUN_INDEX_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: csv_cell(row.get(field)) for field in RUN_INDEX_FIELDS})


def infer_channel_type(train_dataset):
    if not train_dataset:
        return ''
    train_dataset = str(train_dataset)
    if train_dataset.startswith('leo'):
        return 'leo'
    if train_dataset.startswith('3gpp'):
        return '3gpp'
    return train_dataset.split('_path=')[0]


def infer_dataset_metadata(data_dict):
    train_dataset = data_dict.get('train_dataset', '')
    train_files = data_dict.get('train_files') or []
    val_files = data_dict.get('val_files') or []
    test_files = data_dict.get('test_files') or []
    all_files = [*train_files, *val_files, *test_files]

    metadata = {
        'scenario': '',
        'los': '',
        'height_km': '',
        'elevation': '',
        'paths': '',
        'seeds': '',
    }
    if all_files:
        parsed = [parse_leo_file_label(file_path) for file_path in all_files]
        first = parsed[0]
        metadata.update({
            'scenario': first.get('scenario') or '',
            'los': first.get('los') or '',
            'height_km': (first.get('height') or '').lstrip('h').removesuffix('km'),
            'elevation': (first.get('elevation') or '').removeprefix('el'),
            'paths': (first.get('paths') or '').removeprefix('path'),
            'seeds': [item['seed'] for item in parsed if item.get('seed')],
        })
        return metadata

    if str(train_dataset).startswith('leo_'):
        parts = str(train_dataset)[len('leo_'):].split('_')
        if parts:
            metadata['scenario'] = parts[0]
        seeds = [part[4:] for part in parts if part.startswith('seed')]
        metadata['seeds'] = seeds
        pilot_or_path = [part for part in parts if part.startswith('path=') or part.startswith('path')]
        if pilot_or_path:
            metadata['paths'] = pilot_or_path[0].replace('path=', '').replace('path', '')
    elif '_path=' in str(train_dataset):
        metadata['paths'] = str(train_dataset).split('_path=', maxsplit=1)[1].split('_', maxsplit=1)[0]
    return metadata


def find_dm_est_file(results_root, data_dict, diff_model_dict, *, suffix):
    dm_est_dir = path.join(results_root, 'dm_est')
    if not path.isdir(dm_est_dir):
        return ''
    train_dataset = str(data_dict.get('train_dataset', ''))
    n_rx = data_dict.get('n_antennas')
    n_tx = data_dict.get('n_antennas_tx')
    n_val = data_dict.get('num_val_samples')
    timesteps = diff_model_dict.get('num_timesteps')
    patterns = [
        f'*{train_dataset}*dim={n_rx}x{n_tx}*valdata={n_val}*T={timesteps}{suffix}',
        f'*{train_dataset}*T={timesteps}{suffix}',
    ]
    matches = []
    for pattern in patterns:
        matches.extend(glob(path.join(dm_est_dir, pattern)))
    matches = sorted(set(matches))
    if suffix == '.csv':
        matches = [
            match for match in matches
            if not match.endswith('_params.csv') and not match.endswith('_perstep.csv')
        ]
    return matches[-1] if matches else ''


def existing_run_index_row(run_dir, *, cwd, results_root):
    sim_params_path = path.join(run_dir, 'sim_params.json')
    train_results_path = path.join(run_dir, 'train_results.json')
    test_results_path = path.join(run_dir, 'test_results.json')

    params = load_json_file(sim_params_path)
    train_dict = load_json_file(train_results_path)
    test_dict = load_json_file(test_results_path)

    data_dict = params.get('data_dict', {})
    diff_model_dict = params.get('diff_model_dict', {})
    trainer_dict = params.get('trainer_dict', {})
    tester_dict = params.get('tester_dict', {})
    misc_dict = params.get('misc_dict', {})
    metadata = infer_dataset_metadata(data_dict)
    criterion = (tester_dict.get('criteria') or ['nmse'])[0]
    dm_est_csv = find_dm_est_file(results_root, data_dict, diff_model_dict, suffix='.csv')
    dm_est_params_csv = find_dm_est_file(results_root, data_dict, diff_model_dict, suffix='_params.csv')
    loss_plot = find_dm_est_file(results_root, data_dict, diff_model_dict, suffix='_loss.png')

    completed = bool(path.isfile(train_results_path) and path.isfile(test_results_path))
    return {
        'run_name': path.basename(path.normpath(run_dir)),
        'run_dir': path.relpath(run_dir, cwd),
        'timestamp': misc_dict.get('timestamp') or path.basename(path.normpath(run_dir)),
        'status': 'completed' if completed else 'incomplete',
        'channel_type': infer_channel_type(data_dict.get('train_dataset', '')),
        'train_dataset': data_dict.get('train_dataset', ''),
        'test_dataset': data_dict.get('test_dataset', ''),
        'scenario': metadata['scenario'],
        'los': metadata['los'],
        'height_km': metadata['height_km'],
        'elevation': metadata['elevation'],
        'paths': metadata['paths'],
        'seeds': metadata['seeds'],
        'split_ratios': '',
        'file_split_mode': data_dict.get('file_split_mode', ''),
        'train_files': data_dict.get('train_files', ''),
        'val_files': data_dict.get('val_files', ''),
        'test_files': data_dict.get('test_files', ''),
        'test_uses_val_files': data_dict.get('test_uses_val_files', ''),
        'num_train_samples': data_dict.get('num_train_samples', ''),
        'num_val_samples': data_dict.get('num_val_samples', ''),
        'num_test_samples': data_dict.get('num_test_samples', ''),
        'power_normalization_factor': data_dict.get('power_normalization_factor', ''),
        'n_antennas_rx': data_dict.get('n_antennas', ''),
        'n_antennas_tx': data_dict.get('n_antennas_tx', ''),
        'mode': data_dict.get('mode', ''),
        'data_shape': data_dict.get('data_shape', ''),
        'fft_pre': trainer_dict.get('fft_pre', ''),
        'num_timesteps': diff_model_dict.get('num_timesteps', ''),
        'beta_start': diff_model_dict.get('beta_start', ''),
        'beta_end': diff_model_dict.get('beta_end', ''),
        'schedule': diff_model_dict.get('which_schedule', ''),
        'objective': diff_model_dict.get('objective', ''),
        'reverse_method': diff_model_dict.get('reverse_method', ''),
        'reverse_add_random': diff_model_dict.get('reverse_add_random', ''),
        'epochs_requested': trainer_dict.get('num_epochs', ''),
        'trained_epochs': train_dict.get('trained_epochs', ''),
        'batch_size': trainer_dict.get('batch_size', ''),
        'lr_init': trainer_dict.get('lr_init', ''),
        'num_parameters': misc_dict.get('num_parameters', ''),
        'eval_snr_min_db': tester_dict.get('snr_min_db', ''),
        'eval_snr_max_db': tester_dict.get('snr_max_db', ''),
        'eval_snr_step_db': tester_dict.get('snr_step_db', ''),
        'test_nmse_by_snr': nmse_curve_for_index(test_dict, criterion),
        'best_checkpoint': latest_checkpoint_name(run_dir),
        'sim_params_path': path.relpath(sim_params_path, cwd),
        'train_results_path': path.relpath(train_results_path, cwd) if path.isfile(train_results_path) else '',
        'test_results_path': path.relpath(test_results_path, cwd) if path.isfile(test_results_path) else '',
        'dm_est_csv_path': path.relpath(dm_est_csv, cwd) if dm_est_csv else '',
        'dm_est_params_csv_path': path.relpath(dm_est_params_csv, cwd) if dm_est_params_csv else '',
        'loss_plot_path': path.relpath(loss_plot, cwd) if loss_plot else '',
    }


def append_completed_run_index(
    *,
    index_path,
    cwd,
    run_name,
    dir_result,
    timestamp,
    args,
    n_path,
    data_dict,
    diff_model_dict,
    trainer_dict,
    tester_dict,
    train_dict,
    test_dict,
    criterion,
    num_parameters,
    dm_est_csv_path_value,
    dm_est_params_csv_path_value,
    loss_plot_path,
):
    metadata = infer_dataset_metadata(data_dict)
    append_run_index(
        index_path,
        {
            'run_name': run_name,
            'run_dir': path.relpath(dir_result, cwd),
            'timestamp': timestamp,
            'status': 'completed',
            'channel_type': args.channel_type,
            'train_dataset': data_dict['train_dataset'],
            'test_dataset': data_dict['test_dataset'],
            'scenario': args.scenario or metadata['scenario'],
            'los': args.los or metadata['los'],
            'height_km': args.height_km if args.height_km is not None else metadata['height_km'],
            'elevation': args.elevation or metadata['elevation'],
            'paths': n_path if n_path is not None else metadata['paths'],
            'seeds': args.seeds or metadata['seeds'],
            'split_ratios': tuple(args.split_ratios),
            'file_split_mode': data_dict['file_split_mode'],
            'train_files': data_dict['train_files'],
            'val_files': data_dict['val_files'],
            'test_files': data_dict['test_files'],
            'test_uses_val_files': data_dict['test_uses_val_files'],
            'num_train_samples': data_dict['num_train_samples'],
            'num_val_samples': data_dict['num_val_samples'],
            'num_test_samples': data_dict['num_test_samples'],
            'power_normalization_factor': data_dict['power_normalization_factor'],
            'n_antennas_rx': data_dict['n_antennas'],
            'n_antennas_tx': data_dict['n_antennas_tx'],
            'mode': data_dict['mode'],
            'data_shape': data_dict['data_shape'],
            'fft_pre': trainer_dict['fft_pre'],
            'num_timesteps': diff_model_dict['num_timesteps'],
            'beta_start': diff_model_dict['beta_start'],
            'beta_end': diff_model_dict['beta_end'],
            'schedule': diff_model_dict['which_schedule'],
            'objective': diff_model_dict['objective'],
            'reverse_method': diff_model_dict['reverse_method'],
            'reverse_add_random': diff_model_dict['reverse_add_random'],
            'epochs_requested': trainer_dict['num_epochs'],
            'trained_epochs': train_dict['trained_epochs'],
            'batch_size': trainer_dict['batch_size'],
            'lr_init': trainer_dict['lr_init'],
            'num_parameters': num_parameters,
            'eval_snr_min_db': tester_dict['snr_min_db'],
            'eval_snr_max_db': tester_dict['snr_max_db'],
            'eval_snr_step_db': tester_dict['snr_step_db'],
            'test_nmse_by_snr': nmse_curve_for_index(test_dict, criterion),
            'best_checkpoint': latest_checkpoint_name(dir_result),
            'sim_params_path': path.relpath(path.join(dir_result, 'sim_params.json'), cwd),
            'train_results_path': path.relpath(path.join(dir_result, 'train_results.json'), cwd),
            'test_results_path': path.relpath(path.join(dir_result, 'test_results.json'), cwd),
            'dm_est_csv_path': path.relpath(dm_est_csv_path_value, cwd),
            'dm_est_params_csv_path': path.relpath(dm_est_params_csv_path_value, cwd),
            'loss_plot_path': path.relpath(loss_plot_path, cwd),
        },
    )
