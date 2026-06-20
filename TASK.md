# 任务背景

当前项目已经能够训练和评估 Diffusion Model 信道估计器，并且已经具备面向离格 Doppler 现象的 `Doppler frequency shift vs Magnitude` 单图绘制入口。

当前阶段需要使用 `dataset/p006` 下的两个 LEO Rural LOS 数据集验证 DMCE：

```text
dataset/p006/LEO_Rural_LOS_h1000km_el30-90_path10_seed1111.mat
dataset/p006/LEO_Rural_LOS_h1000km_el30-90_path10_seed2222.mat
```

这两个文件均为 MATLAB v7.3/HDF5 `.mat` 数据文件，每个文件包含 `10000 x 16 x 144` 个 complex128 信道样本。默认 Python 环境缺少 `h5py`，因此所有读取、训练和绘图命令均应通过 `conda run -n dmce` 执行。

本次验证采用文件级划分：`seed1111` 用作训练集，`seed2222` 用作验证集，并同时作为最终测试集。验证结果需要同时包含量化 NMSE 结果和一张直观的 Doppler-Magnitude 谱形图。

# 不足之处

当前问题：已有训练结果主要面向 `p001/p0625/p125/p15625` 等导频比例数据集，还没有一个明确面向 `dataset/p006` 两个数据集的 DMCE 训练、测试和图形审查流程。

引入新需求：

1. 使用 `dataset/p006` 下两个 `.mat` 文件完成一次 DMCE 训练和测试。
2. 严格按文件划分训练/验证/测试数据，不将两个 seed 混合随机切分。
3. 训练完成后保留标准 NMSE 评估结果，用于判断不同 SNR 下的估计误差。
4. 使用新训练 run 的 `sim_params.json` 和 `train_models/*.pt` 生成一张 `Doppler frequency shift vs Magnitude` 图。
5. 单图默认叠加 `Ground truth`、`Noisy observation` 和 `DM estimate`。
6. 图中使用 `0 dB` SNR 和 `0.5` 分数 Doppler-bin offset 作为默认离格验证设置。
7. 不修改 `diff_cnn.py`、`DMCE/diffusion_model.py`、模型 checkpoint、数据集文件和已有历史 `sim_params.json`。

# 任务

## 验证流程

* 步骤1：**数据可读性检查**：确认 `dataset/p006` 中两个文件可由 `loaders.py` 正确读取，并确认总样本数、信道维度和角域变换往返误差。
* 步骤2：**DMCE 训练与测试**：使用 `seed1111` 作为训练文件，使用 `seed2222` 作为验证和测试文件，运行 `diff_cnn.py` 完成训练、测试和结果落盘。
* 步骤3：**新 run 定位**：训练完成后定位新生成的 `results/<run>/` 目录，并确认其中包含 `sim_params.json`、`train_models/*.pt`、`train_results.json` 和 `test_results.json`。
* 步骤4：**NMSE 结果检查**：从 `test_results.json`、`results/dm_est/*.csv` 或 `results/runs_index.csv` 中检查各 SNR 下的 DMCE NMSE。
* 步骤5：**Doppler-Magnitude 单图生成**：使用新 run 作为 `--model-dir`，调用 `plot/plot_offgrid_doppler_magnitude.py` 生成一张 Doppler 频移谱幅度图。
* 步骤6：**图形审查**：检查输出 PNG 是否非空，曲线是否包含真实信道、含噪观测和 DM 估计，谱峰位置和能量泄漏是否清晰可见。

## 运行命令

数据检查命令：

```bash
conda run -n dmce python loaders.py \
  --data-dir dataset/p006 \
  --scenario Rural
```

预期数据检查结果：

```text
matched files: 2
channels size: [20000 16 144]
channels dtype: complex128
angular feature size: [20000 2 16 144]
split sizes: train=[16000 16 144], val=[2000 16 144], test=[2000 16 144]
```

训练与测试命令：

```bash
conda run -n dmce python diff_cnn.py \
  -d cuda:0 \
  --channel-type leo \
  --train-files dataset/p006/LEO_Rural_LOS_h1000km_el30-90_path10_seed1111.mat \
  --val-files dataset/p006/LEO_Rural_LOS_h1000km_el30-90_path10_seed2222.mat \
  --epochs 500 \
  --snr-min-db -15 \
  --snr-max-db 20 \
  --snr-step-db 5
```

训练完成后定位新模型目录：

```bash
MODEL_DIR=$(ls -td results/*leo_Rural_LOS_h1000km_el30-90_path10_tr1111_valtest2222 | head -1)
echo "$MODEL_DIR"
```

绘图脚本语法检查：

```bash
conda run -n dmce python -m py_compile plot/plot_offgrid_doppler_magnitude.py
```

生成 Doppler-Magnitude 单图：

```bash
conda run -n dmce python plot/plot_offgrid_doppler_magnitude.py \
  --model-dir "$MODEL_DIR" \
  --sample-index 0 \
  --snr-db 0 \
  --offset 0.5 \
  --device cuda:0
```

## 输出文件

预期训练输出：

```text
results/<run>/sim_params.json
results/<run>/train_models/*.pt
results/<run>/train_results.json
results/<run>/test_results.json
results/dm_est/*.csv
results/runs_index.csv
```

预期绘图输出：

```text
results/offgrid_doppler_magnitude/*_doppler_magnitude.png
```

**注意**：

* 作用对象：本任务主要验证 `dataset/p006` 数据集上的 DMCE 训练、测试和单图绘制流程。
* 默认数据划分：`seed1111` 训练，`seed2222` 验证和测试；不使用两个文件混合后的随机切分作为主流程。
* 默认训练轮数为 `500`；如需快速冒烟验证，可临时降低 `--epochs`，但正式结果应使用完整训练设置。
* 默认测试 SNR 范围为 `[-15, 20] dB`，步长为 `5 dB`。
* 默认绘图 SNR 为 `0 dB`，默认分数 Doppler 偏移为 `0.5`。
* 默认只画一个样本，`--sample-index 0`；不做 per-sample 批量统计。
* 不做 Git 操作。
* 不新增额外 CSV、CDF、boxplot、SNR stress 曲线或 3D 曲面图。

## 性能定义

验证目标：确认 DMCE 能够在 `dataset/p006` 的 LEO Rural LOS 信道数据上完成训练和测试，并通过 NMSE 指标与 Doppler-Magnitude 单图共同观察模型估计效果。

量化指标：

* `test_results.json` 中应包含 `-15 dB` 到 `20 dB` 的 NMSE 评估结果。
* `results/runs_index.csv` 应追加本次 p006 run 的记录。
* `0 dB` 附近的 NMSE 可作为本次图形审查的主要数值参考。

图形指标：

* 横轴：Doppler frequency shift，建议按 kHz 显示。
* 纵轴：Magnitude，即 Doppler 谱幅度 `|H(f_D)|`，默认使用线性幅度。
* 曲线：默认显示 `Ground truth`、`Noisy observation`、`DM estimate`。
* 标注：使用虚线标出真实 Doppler frequency shift，并可标注估计峰值位置。
* 审查重点：能否清楚看到离格 Doppler 造成的谱泄漏，以及 DM 估计谱峰是否比含噪观测更接近 ground truth。

# 补充说明

* `p006` 数据集中包含 `subcarrier_spacing_hz=60000` 和 `sampling_interval_s=3.26e-08`，绘图脚本可自动推断 Doppler 频率轴。
* 对于 `16 x 144` 信道宽度，默认 Doppler bin 间隔可由 `subcarrier_spacing_hz / 144` 推得，约为 `416.67 Hz/bin`。
* 如果 CUDA 不可用，可将命令中的 `cuda:0` 改为 `cpu`，但完整训练会明显变慢。
* 如果需要叠加传统基线，可在绘图命令中追加 `--methods dm lmmse ls`；本次默认只显示 `Ground truth / Noisy / DM`。
* 验证完成标准：训练 run 完整落盘，NMSE 结果可读取，且 `results/offgrid_doppler_magnitude/` 下生成一张非空 PNG。
