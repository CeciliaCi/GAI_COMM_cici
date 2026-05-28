# Diffusion-Based Channel Estimation 

Source code of the paper 
>B. Fesl, M. Baur, F. Strasser, M. Joham, and W. Utschick,
>"Diffusion-Based Generative Prior for Low-Complexity MIMO Channel Estimation," in IEEE Wireless Communications Letters, 2024.

[[IEEE](https://ieeexplore.ieee.org/document/10705115)] [[arXiv](https://arxiv.org/abs/2403.03545)]

---

## Abstract

This letter proposes a novel channel estimator based on diffusion models (DMs), one of the currently top-rated generative models, with provable convergence to the mean square error (MSE)-optimal estimator. A lightweight convolutional neural network (CNN) with positional embedding of the signal-to-noise ratio (SNR) information is designed to learn the channel distribution in the sparse angular domain. Combined with an estimation strategy that avoids stochastic resampling and truncates reverse diffusion steps that account for lower SNR than the given pilot observation, the resulting DM estimator unifies low complexity and memory overhead. Numerical results exhibit better performance than state-of-the-art estimators.

---

## DMSE Scheduler Package

A standalone implementation of the DMSE scheduler is available as a separate package:

```bash
pip install diffusers-dmse
```

[[PyPI](https://pypi.org/project/diffusers-dmse/)] [[GitHub](https://github.com/benediktfesl/diffusers-dmse)]

This repository contains an application-specific implementation. The standalone package provides a reusable and diffusers-compatible version of the scheduler.

---

## Requirements

The code is tested with `Python 3.10` and `Pytorch 2.1.1`. For further details, see `environment.yml`.

---

## Instructions

1. Load channel data from  
   https://syncandshare.lrz.de/getlink/fi93y1AnwmsvHrAGNqq5zX/  
   (password: Diffusion2024)  
   and move it into folder `bin`.

2. LEO DeepMIMO MATLAB 7.3 datasets are loaded from `dataset` by default. Files may use either
   `LEO_<Scenario>_seed<Seed>.mat` or detailed names such as
   `LEO_DenseUrban_LOS_h1000km_el30-90_path10_seed1111.mat`. Each file contains top-level
   `channels`, `dataset_params`, and `sample_info` variables. The loader validates complex
   `[num_users, 16, 144]` channels, keeps the training interface in the spatial domain, and exposes
   unitary 2-D DFT angular-domain feature maps as `[num_users, 2, 16, 144]`.

   To inspect and split LEO files without starting training, run:

```
python loaders.py --data-dir dataset --scenario DenseUrban
```

   To train on matching LEO files, run:

```
python diff_cnn.py -d cuda:0 --channel-type leo --data-dir dataset --scenario DenseUrban
```

   To keep Urban seeds separated at the file level, use explicit train/validation files. If
   `--test-files` is omitted, the validation file is also used for the final NMSE evaluation.
   The default Diffusion NMSE evaluation SNR range is `[-15, 20]` dB with a 5 dB step:

```
python diff_cnn.py -d cuda:0 --channel-type leo \
  --train-files dataset/LEO_Urban_seed1111.mat \
  --val-files dataset/LEO_Urban_seed2222.mat \
  --epochs 500
```

3. To evaluate the pre-trained models used for the plots in the paper, run:

```
python load_and_eval_dm.py -d cuda:0
```

4. To train a DM from scratch on the default 3GPP data and evaluate the performance afterward, run:

```
python diff_cnn.py -d cuda:0
```

5. To evaluate the baseline estimators on the default LEO Rural dataset in `dataset`, run:

```
python baselines.py
```

   The default LMMSE baseline is angular-domain diagonal LMMSE, which is more stable than full covariance LMMSE for
   the 16x144 LEO channels with limited training samples. The script normalizes channel average element power before
   adding AWGN, so the SNR values match the normalized-channel assumption. Pass `--lmmse-mode global_full` to run the
   older full covariance LMMSE, or `--no-normalize-power` to evaluate raw physical amplitudes directly.

   To select another LEO scenario, pass `--scenario`, for example:

```
python baselines.py --scenario Urban
```

## Related repositories

- https://github.com/benediktfesl/Diffusion_MSE
- https://github.com/benediktfesl/diffusers-dmse
