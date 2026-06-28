# Diffusion Channel Estimation

本仓库用于训练、评估和可视化 **Diffusion Model Channel Estimator (DMCE)**。项目基于扩散模型学习 MIMO 信道的先验分布，并在含噪观测下生成低复杂度信道估计结果。当前工程同时支持原论文中的 3GPP 数据流程，以及面向 LEO DeepMIMO 信道数据的训练、基线评估和结果绘图。

原始论文：

> B. Fesl, M. Baur, F. Strasser, M. Joham, and W. Utschick,  
> "Diffusion-Based Generative Prior for Low-Complexity MIMO Channel Estimation,"  
> IEEE Wireless Communications Letters, 2024.

链接：[IEEE](https://ieeexplore.ieee.org/document/10705115) / [arXiv](https://arxiv.org/abs/2403.03545)

## 工程结构

```text
DMCE/                         核心扩散模型、CNN 网络和工具函数
estimators/                   传统估计器实现，例如 LMMSE
modules/                      数据生成、AWGN、FFT 等辅助函数
dataset/                      LEO DeepMIMO .mat 数据文件
results/                      训练、评估和绘图结果
diff_cnn.py                   训练并评估 Diffusion Model 的主入口
load_and_eval_dm.py           加载预训练模型并评估
baselines.py                  LS / LMMSE 等传统基线评估
loaders.py                    LEO 数据加载、筛选、切分和元数据解析
plot/                         结果图绘制脚本
  plot_dm_channel_heatmaps.py         信道热力图可视化
  plot_robustness_nmse.py             Doppler / elevation 分组 NMSE 鲁棒性图
  plot_offgrid_doppler_magnitude.py   Doppler frequency shift - Magnitude 单图
```

## 环境

推荐使用仓库中的 `environment.yml` 创建 Conda 环境：

```bash
conda env create -f environment.yml
conda activate dmce
```

主要依赖包括：

- Python 3.10
- PyTorch
- NumPy / SciPy
- h5py
- Matplotlib
- scikit-learn
- tqdm

## 数据

LEO 数据默认放在 `dataset/` 下，文件为 MATLAB `.mat` 格式。每个 LEO 数据文件应包含：

- `channels`：复数信道张量，形状通常为 `[num_users, 16, 144]`
- `dataset_params`：数据集级参数，例如载频、子载波间隔、采样间隔等
- `sample_info`：样本级物理信息，例如 `elevation_deg`、`max_doppler_hz`、路径损耗等

可以直接检查数据文件、信道维度和角域变换误差：

```bash
conda run -n dmce python loaders.py --data-dir dataset --scenario Rural
```

## 训练 Diffusion Model

使用默认 3GPP 配置训练：

```bash
conda run -n dmce python diff_cnn.py -d cuda:0
```

使用 LEO 数据训练：

```bash
conda run -n dmce python diff_cnn.py \
  -d cuda:0 \
  --channel-type leo \
  --data-dir dataset \
  --scenario Rural \
  --epochs 500
```

使用显式 train / validation 文件划分：

```bash
conda run -n dmce python diff_cnn.py \
  -d cuda:0 \
  --channel-type leo \
  --train-files dataset/LEO_Rural_seed1111_p0625.mat \
  --val-files dataset/LEO_Rural_seed2222_p0625.mat \
  --epochs 500
```

使用 `dataset/p006` 预设验证 DMCE：

```bash
conda run -n dmce python diff_cnn.py \
  -d cuda:0 \
  --data-preset p006 \
  --epochs 500 \
  --snr-min-db -15 \
  --snr-max-db 20 \
  --snr-step-db 5
```

训练结果会保存到：

```text
results/<timestamp>_<short_experiment_label>/
```

其中包含：

- `sim_params.json`：训练、模型、数据和评估参数
- `train_models/*.pt`：模型 checkpoint
- `train_results.json`：训练记录
- `test_results.json`：测试结果

目录名只保留少量关键信息，例如：

```text
results/2026-06-09-12h00m00s_leo_Rural_p0625_tr1111_valtest2222/
```

完整的训练集、验证集、测试集、样本数、模型参数和 NMSE 曲线会追加写入：

```text
results/runs_index.csv
```

如果需要把已有训练结果重新汇总到该 CSV，可以运行：

```bash
conda run -n dmce python scripts/build_runs_index.py
```

默认只扫描 `results/*/sim_params.json` 对应的训练 run；如需同时纳入论文预训练模型目录，可加：

```bash
conda run -n dmce python scripts/build_runs_index.py --include-best-models
```

聚合后的 DM 评估 CSV 和 loss 图会保存到：

```text
results/dm_est/
```

## 传统基线

运行 LS 和 LMMSE 基线：

```bash
conda run -n dmce python baselines.py
```

默认基线面向 LEO Rural 数据，使用角域对角 LMMSE。可以切换场景：

```bash
conda run -n dmce python baselines.py --scenario Urban
```

基线结果保存到：

```text
results/baselines/
```

## 绘图工具

### 信道热力图

绘制原始信道、含噪信道和 DM 去噪结果：

```bash
conda run -n dmce python plot/plot_dm_channel_heatmaps.py \
  --model-dir results/2026-05-24-16h55m47s \
  --sample-index 0 \
  --snr-db 10 \
  --device cuda:0
```

输出目录：

```text
results/channel_heatmaps/
```

### Doppler / Elevation 鲁棒性 NMSE

按物理条件分组绘制 `NMSE vs Doppler / Elevation`：

```bash
conda run -n dmce python plot/plot_robustness_nmse.py \
  --model-dir results/2026-05-24-16h55m47s \
  --snr-db 0 \
  --device cuda:0
```

输出目录：

```text
results/robustness_nmse/
```

### Doppler Frequency Shift - Magnitude 单图

绘制一张横坐标为 Doppler frequency shift、纵坐标为 Magnitude 的一维谱切片图：

```bash
conda run -n dmce python plot/plot_offgrid_doppler_magnitude.py \
  --model-dir results/2026-05-24-16h55m47s \
  --sample-index 0 \
  --snr-db 0 \
  --offset 0.5 \
  --device cuda:0
```

如果要绘制最新的 `p006` 训练 run，可直接使用预设：

```bash
conda run -n dmce python plot/plot_offgrid_doppler_magnitude.py \
  --data-preset p006 \
  --sample-index 0 \
  --snr-db 0 \
  --offset 0.5 \
  --device cuda:0
```

默认叠加：

- Ground truth
- Noisy observation
- DM estimate

可选叠加 LMMSE / LS：

```bash
conda run -n dmce python plot/plot_offgrid_doppler_magnitude.py \
  --model-dir results/2026-05-24-16h55m47s \
  --sample-index 0 \
  --snr-db 0 \
  --offset 0.5 \
  --methods dm lmmse ls \
  --device cuda:0
```

输出目录：

```text
results/offgrid_doppler_magnitude/
```

### 角域能量集中度分析

比较 Urban 与 DenseUrban 信道在 2D FFT 角域中的能量集中程度：

```bash
conda run -n dmce python scripts/analyze_angular_concentration.py
```

脚本默认报告 top-k 能量占比、effective bins，以及达到 90%/95% 能量所需的角域 bin 数。若需要保存汇总 CSV：

```bash
conda run -n dmce python scripts/analyze_angular_concentration.py \
  --output results/angular_concentration/urban_denseurban_summary.csv
```

## 常用结果目录

```text
results/<timestamp>_<label>/          单次训练目录
results/runs_index.csv               训练 run 汇总索引
results/dm_est/                      DM 的 NMSE-SNR 聚合结果
results/baselines/                   LS / LMMSE 基线结果
results/channel_heatmaps/            信道热力图
results/robustness_nmse/             Doppler / elevation 鲁棒性图
results/offgrid_doppler_magnitude/   Doppler-Magnitude 单图
```

## 说明

- 信道以复数矩阵表示，进入 CNN 前转换为实部/虚部双通道张量。
- 当 `fft_pre=True` 时，训练和推理会在角域中进行，并在评估或绘图时变换回对应域。
- LEO 数据中的 `sample_info` 和 `dataset_params` 可用于按仰角、Doppler、链路几何等物理条件分析模型表现。
- `p001/p0625/p125/p15625` 文件表示不同导频比例数据集，不是 Doppler 或 elevation 横轴变量。

## 相关仓库

- https://github.com/benediktfesl/Diffusion_MSE
- https://github.com/benediktfesl/diffusers-dmse
