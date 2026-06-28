# 任务背景

当前项目已经能够训练和评估 Diffusion Model 信道估计器，并且已经具备面向 LEO Rural、Urban 和 DenseUrban 场景的文件级训练、验证和测试流程。

现有实验中，Rural LOS 场景的 NMSE 高于 Urban Mixed 和 DenseUrban Mixed。角域集中度分析表明，Rural 测试集的 top-10 角域 bin 平均包含约 `76%` 的信道能量，能量比 Urban 和 DenseUrban 更集中。因此，Rural 的较高 NMSE 不应简单归因于信道角域结构更加复杂。

当前 DMCE 使用 `fft_pre=True` 在角域学习信道分布，但训练目标仅为扩散过程中的噪声预测 MSE。该目标没有直接约束最终信道重构误差，也没有专门保护 Rural LOS 信道中的少数高能量角域主峰。

当前阶段需要使用 `dataset/p002` 下的 Rural LOS 数据集验证角域峰值联合损失：

```text
dataset/p002/LEO_Rural_LOS_h1000km_el30-90_path10_seed1111.mat
dataset/p002/LEO_Rural_LOS_h1000km_el30-90_path10_seed2222.mat
```

采用文件级划分：`seed1111` 作为训练集，`seed2222` 作为验证集和最终测试集。默认 Python 环境可能缺少 `h5py`，因此所有读取、训练和测试命令均应通过 `conda run -n dmce` 执行。

# 不足之处

当前问题：DMCE 使用统一的 `pred_noise` L2 损失训练。该损失能够学习整体信道先验，但没有直接优化最终的 total-power NMSE、per-sample NMSE 和 Rural 高能量角域峰值误差。

引入新需求：

1. 保留现有 `pred_noise`、CNN 网络、`T=100`、batch size、早停机制和当前 `5e-5` 学习率。
2. 从预测噪声反推出角域信道 `x0_pred`，不改变现有反向扩散估计流程。
3. 在原始噪声 MSE 基础上新增 total-power NMSE、per-sample NMSE 和 top-k 主峰损失。
4. 默认联合损失为：

   ```text
   L = L_noise
     + 0.1 * L_total_nmse
     + 0.1 * L_sample_nmse
     + 0.05 * L_top10_peak
   ```

5. 新增命令行参数 `--angular-total-weight`、`--angular-sample-weight`、`--angular-peak-weight` 和 `--angular-peak-k`。
6. 所有辅助损失权重默认设为 `0`，保证不传新参数时与历史训练行为兼容。
7. 使用 `alpha_bar[t]` 对辅助重构损失加权，避免高噪声扩散时间步产生过大的重构梯度。
8. 测试阶段同时输出 `NMSEs_total_power` 和 `NMSEs_per_sample`。
9. 保留评估噪声模式 `--eval-noise-mode dataset/sample`，公平评估默认使用 `sample`。
10. 保留只打印冒烟测试模式 `--smoke-print-only`，冒烟结果不得写入正式输出目录。
11. 不修改数据集文件、已有 checkpoint、已有历史结果和历史 `sim_params.json`。

# 任务

## 实现流程

* 步骤1：**联合损失参数扩展**：在 `diff_cnn.py` 中新增四个角域辅助损失参数，并将参数写入 `diff_model_dict` 和新 run 的 `sim_params.json`。
* 步骤2：**角域信道重构**：在 `pred_noise` 训练目标下，根据 `x_t`、时间步 `t` 和预测噪声计算 `x0_pred`；辅助权重全部为零时不得执行额外损失计算。
* 步骤3：**双 NMSE 损失**：计算 batch total-power NMSE 和逐样本归一化 NMSE，使训练目标同时覆盖两种测试口径。
* 步骤4：**主峰保持损失**：根据真实角域信道的复数能量选择 top-k bin，并对这些 bin 的实部和虚部重构误差进行归一化约束。默认 `k=10`。
* 步骤5：**时间步稳定化**：使用对应样本的 `alpha_bar[t]` 缩放辅助损失，确保首尾扩散时间步的损失和梯度均为有限值。
* 步骤6：**公平 NMSE 输出**：测试阶段同时计算 `NMSEs_total_power` 和 `NMSEs_per_sample`，并写入正式 JSON、CSV 和 runs index。
* 步骤7：**只打印冒烟测试**：使用 `--smoke-print-only` 时打印训练摘要、联合损失配置和两种 NMSE，不写 checkpoint、JSON、CSV、loss 图、run 目录或 runs index。
* 步骤8：**三组消融实验**：在完全相同的数据划分、学习率、batch size、扩散参数、早停和测试 SNR 下，依次运行原始基线、双 NMSE 辅助损失和完整联合损失。

## 运行命令

语法检查命令：

```bash
conda run -n dmce python -m py_compile \
  DMCE/functional.py \
  DMCE/diffusion_model.py \
  diff_cnn.py \
  modules/result_io.py
```

完整联合损失冒烟测试命令：

```bash
conda run -n dmce python diff_cnn.py \
  -d cuda:0 \
  --channel-type leo \
  --train-files dataset/p002/LEO_Rural_LOS_h1000km_el30-90_path10_seed1111.mat \
  --val-files dataset/p002/LEO_Rural_LOS_h1000km_el30-90_path10_seed2222.mat \
  --epochs 1 \
  --snr-min-db 0 \
  --snr-max-db 0 \
  --snr-step-db 5 \
  --eval-noise-mode sample \
  --angular-total-weight 0.1 \
  --angular-sample-weight 0.1 \
  --angular-peak-weight 0.05 \
  --angular-peak-k 10 \
  --smoke-print-only
```

冒烟测试预期行为：

```text
print only
angular loss weights: total=0.1, sample=0.1, peak=0.05, top-k=10
NMSEs_total_power: ...
NMSEs_per_sample: ...
no results/<run>/ directory written
no checkpoint/JSON/CSV/loss plot written
no results/runs_index.csv update
```

实验1，原始噪声 MSE 基线：

```bash
conda run -n dmce python diff_cnn.py \
  -d cuda:0 \
  --channel-type leo \
  --train-files dataset/p002/LEO_Rural_LOS_h1000km_el30-90_path10_seed1111.mat \
  --val-files dataset/p002/LEO_Rural_LOS_h1000km_el30-90_path10_seed2222.mat \
  --epochs 5000 \
  --snr-min-db -15 \
  --snr-max-db 20 \
  --snr-step-db 5 \
  --eval-noise-mode sample
```

实验2，噪声 MSE 加双 NMSE 辅助损失：

```bash
conda run -n dmce python diff_cnn.py \
  -d cuda:0 \
  --channel-type leo \
  --train-files dataset/p002/LEO_Rural_LOS_h1000km_el30-90_path10_seed1111.mat \
  --val-files dataset/p002/LEO_Rural_LOS_h1000km_el30-90_path10_seed2222.mat \
  --epochs 5000 \
  --snr-min-db -15 \
  --snr-max-db 20 \
  --snr-step-db 5 \
  --eval-noise-mode sample \
  --angular-total-weight 0.1 \
  --angular-sample-weight 0.1
```

实验3，完整角域峰值联合损失：

```bash
conda run -n dmce python diff_cnn.py \
  -d cuda:0 \
  --channel-type leo \
  --train-files dataset/p002/LEO_Rural_LOS_h1000km_el30-90_path10_seed1111.mat \
  --val-files dataset/p002/LEO_Rural_LOS_h1000km_el30-90_path10_seed2222.mat \
  --epochs 5000 \
  --snr-min-db -15 \
  --snr-max-db 20 \
  --snr-step-db 5 \
  --eval-noise-mode sample \
  --angular-total-weight 0.1 \
  --angular-sample-weight 0.1 \
  --angular-peak-weight 0.05 \
  --angular-peak-k 10
```

## 输出文件

正式训练预期输出：

```text
results/<run>/sim_params.json
results/<run>/train_models/*.pt
results/<run>/train_results.json
results/<run>/test_results.json
results/dm_est/*.csv
results/runs_index.csv
```

新 run 的 `sim_params.json` 应记录：

```text
angular_total_weight
angular_sample_weight
angular_peak_weight
angular_peak_k
```

`test_results.json` 中应同时包含：

```text
NMSEs_total_power
NMSEs_per_sample
```

`results/dm_est/*.csv` 中应同时包含：

```text
nmse_dm
nmse_dm_per_sample
```

`results/runs_index.csv` 中应记录联合损失配置，并同时保存两种 NMSE 曲线：

```text
test_nmse_by_snr
test_nmse_per_sample_by_snr
```

**注意**：

* 作用对象：本任务只优化 `dataset/p002` Rural LOS 场景的 DMCE 训练目标。
* 默认数据划分：`seed1111` 训练，`seed2222` 验证和测试；不得混合两个 seed 后随机划分。
* 三组消融实验必须使用完全一致的数据、训练和测试配置，只有辅助损失权重不同。
* 当前学习率固定为 `5e-5`，本任务不修改学习率调度方式。
* `--epochs 5000` 是最大训练轮数；保留现有早停机制，因此实际训练轮数可能小于 5000。
* 默认测试 SNR 范围为 `[-15, 20] dB`，步长为 `5 dB`。
* 公平比较必须使用 `--eval-noise-mode sample`。
* `--smoke-print-only` 不得写入正式 `results/` 输出。
* 不做 Git 操作。
* 不修改数据集、历史 checkpoint、已有历史结果或历史 `sim_params.json`。

## 性能定义

验证目标：在保持现有 DMCE 网络和推理过程不变的情况下，通过直接约束最终角域信道重构和 Rural LOS 主峰，使 Rural 场景的两种 NMSE 相对原始噪声 MSE 基线同时下降。

损失定义：

* `L_noise`：现有 `pred_noise` L2 损失。
* `L_total_nmse`：batch 内总重构误差能量除以总真实信道能量。
* `L_sample_nmse`：每个样本先按自身真实信道能量归一化，再对 batch 求平均。
* `L_top10_peak`：对真实角域信道能量最高的 10 个 bin 计算归一化复数重构误差。
* 辅助损失仅用于训练，不改变测试阶段的反向扩散步骤。

量化指标：

* `NMSEs_total_power`：用于与历史结果兼容。
* `NMSEs_per_sample`：用于检查每个样本的相对估计误差。
* 主要检查 `0 dB` 结果和 `[-15, 20] dB` 全部测试点的平均结果。

验收标准：

* 辅助权重全部为 `0` 时，固定输入、噪声和时间步下的损失必须与原始实现一致。
* 完美重构时，三个辅助损失均应为 `0` 或数值精度范围内接近 `0`。
* 相同大小的误差施加到主峰 bin 时，`L_top10_peak` 应高于施加到非主峰 bin 时的结果。
* 在首个和最后一个扩散时间步上，联合损失及模型梯度均不得出现 `NaN` 或 `Inf`。
* 完整联合损失在 `0 dB` 和全 SNR 平均上应同时改善 `NMSEs_total_power` 与 `NMSEs_per_sample`。
* 任一测试 SNR 下，相对原始基线的 NMSE 退化不得超过 `5%`。
* 如果完整联合损失未满足上述数值标准，应保留真实结果并调整权重，不得人为修改数据集或结果文件。

# 补充说明

* top-k bin 必须根据真实角域信道的复数能量 `real^2 + imag^2` 选择，不能分别对实部和虚部选择。
* top-k 掩码只由真实信道生成，不需要对 bin 选择过程反向传播；选中 bin 内的重构误差必须保持可微。
* `alpha_bar[t]` 加权只用于抑制高噪声时间步反推 `x0_pred` 时的梯度放大，不改变原始噪声预测损失。
* 由于使用正交归一化 FFT，角域和空间域的总误差能量保持一致，但 top-k 主峰损失只在角域中定义。
* 本任务的目标是降低 Rural 相对自身基线的估计误差，不要求 Rural 的 NMSE 必须低于 Urban 或 DenseUrban。
* 如果 CUDA 不可用，可将命令中的 `cuda:0` 改为 `cpu`，但完整训练会明显变慢。
* 验证完成标准：冒烟测试只打印且不落盘；三组消融实验配置可追溯；正式结果同时包含两种 NMSE；完整联合损失满足稳定性和精度验收要求。
