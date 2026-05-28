# 任务背景

当前项目完成了：基于扩散模型的低复杂度 MIMO 信道估计框架（`Diffusion_channel_est` / `GAI_COMM_baseline`），已经支持 LEO DeepMIMO 窄带等效信道数据集的读取、结构体字段解析、`num_users × 16 × 144` 复数信道张量维度自检，以及角域 2D DFT 变换流程。
环境假设：现有 LEO 数据加载逻辑能够按场景或 seed 聚合多个 `.mat` 文件，并通过比例切分形成 Train/Val/Test 数据集；但该方式会打乱并混合不同 seed 的样本，无法严格保证某一个指定文件只用于训练，另一个指定文件只用于验证或评估。

# 不足之处

缺少：面向指定 LEO `.mat` 文件的确定性训练/验证/评估入口。
当前问题：如果通过 `--scenario Urban --seeds 1111 2222` 一类参数加载数据，代码会先聚合两个 seed 文件，再按比例随机切分，导致 `LEO_Urban_seed1111.mat` 与 `LEO_Urban_seed2222.mat` 的样本可能同时出现在训练、验证或测试集合中。

引入新需求：需要严格按照文件级别划分 Diffusion Model 的数据来源：

1. `dataset/LEO_Urban_seed1111.mat`：仅用于 Diffusion Model 训练。
2. `dataset/LEO_Urban_seed2222.mat`：用于训练过程中的验证集。
3. `dataset/LEO_Urban_seed2222.mat`：同时作为最终 NMSE 评估数据集，输出 Diffusion Model 在该验证 seed 上的 NMSE 曲线。
4. 数据维度保持为 `[num_users, 16, 144]`，进入 CNN 前仍映射为实部/虚部双通道表示。

# 任务

## 重构 / 新增模块

* 模块1：**文件级 LEO 数据入口 (`diff_cnn.py` 优化)**：新增支持显式指定训练、验证和测试 `.mat` 文件路径的命令行参数，避免通过 seed 聚合后随机切分。
* 模块2：**确定性 Train/Val/Test 构造模块**：当用户传入指定文件时，直接从对应文件读取完整信道张量，不再执行多文件聚合打乱与比例切分。
* 模块3：**Diffusion Model NMSE 评估接口保持兼容**：保持现有 `Trainer` 和 `Tester` 流程不变，训练后使用 `Tester(diffusion_model, data_test, ...)` 输出 `SNR,nmse_dm` 结果。
* 模块4：**结果元数据记录模块**：在 `sim_params.json` 和 `results/dm_est/*_params.csv` 中记录实际使用的训练、验证、测试文件路径，确保后续结果可追溯。

## 实现新方法

参考：当前 LEO 数据集文件位于 `dataset/` 目录，Urban 场景包含两个 seed 文件。
新增方法：

1. 方法A：**显式文件参数机制**：在 `diff_cnn.py` 中新增 `--train-files`、`--val-files`、`--test-files` 参数，支持传入一个或多个 `.mat` 文件路径。
2. 方法B：**验证集兼作测试集机制**：若用户只传入 `--train-files` 和 `--val-files`，未传入 `--test-files`，则默认将 `--val-files` 同时作为最终 NMSE 评估数据。
3. 方法C：**训练集功率归一化机制**：使用训练集 `mean(abs(H_train)^2)` 计算归一化因子，并用同一因子缩放训练、验证和测试数据，使 Diffusion Model 中 AWGN 的 SNR 定义与归一化信道假设一致。
4. 方法D：**样本数量自动记录机制**：从文件实际读取样本数，并写入 `num_train_samples`、`num_val_samples`、`num_test_samples`，避免手动参数与真实文件样本数不一致。
5. 方法E：**结果命名增强机制**：结果标签中体现 `leo_Urban_seed1111_train_seed2222_valtest` 或等价信息，方便区分不同实验。

**注意**：

* 作用对象：主要修改 `diff_cnn.py` 的数据载入入口，以及必要的数据加载辅助函数；不对 `networks.py`、`diffusion_model.py` 进行破坏性重构。
* 假设条件：`LEO_Urban_seed1111.mat` 和 `LEO_Urban_seed2222.mat` 均包含顶层 `channels`、`dataset_params` 和 `sample_info`，且信道张量可归一化为 `[num_users, 16, 144]`。
* 不做的内容：不新增冒烟测试，不新增单独测试脚本，不进行宽带 OFDM 频率选择性衰落适配，不修改 CNN 网络结构。
* 数据划分约束：指定文件模式下不允许再对 train/val/test 进行随机混合切分，保证 seed1111 与 seed2222 的职责固定。

## 性能定义

优化目标：确保 Diffusion Model 可以严格使用 `LEO_Urban_seed1111.mat` 训练，并在 `LEO_Urban_seed2222.mat` 上完成验证和最终 NMSE 评估。
评估输出：训练结束后，`results/dm_est/` 中应生成包含 `SNR,nmse_dm` 的 CSV 文件，其中 `SNR` 单位为 dB，`nmse_dm` 为线性 NMSE。
预期运行命令：

```bash
python diff_cnn.py -d cuda:0 \
  --channel-type leo \
  --train-files dataset/LEO_Urban_seed1111.mat \
  --val-files dataset/LEO_Urban_seed2222.mat \
  --epochs 500
```

等价显式测试文件命令：

```bash
python diff_cnn.py -d cuda:0 \
  --channel-type leo \
  --train-files dataset/LEO_Urban_seed1111.mat \
  --val-files dataset/LEO_Urban_seed2222.mat \
  --test-files dataset/LEO_Urban_seed2222.mat \
  --epochs 500
```

不同方法预期：相比原先按 `scenario/seeds/split_ratios` 随机切分的方式，指定文件模式应当提供完全可复现、可解释的 seed 级实验结果，便于比较 Urban 场景下跨 seed 泛化能力。

# 补充说明

* 方法1说明：`--train-files`、`--val-files`、`--test-files` 应支持相对路径和绝对路径；对于本任务，默认使用相对路径 `dataset/LEO_Urban_seed1111.mat` 与 `dataset/LEO_Urban_seed2222.mat`。
* 方法2说明：若 `--test-files` 未指定，则 `data_test = data_val`，即最终 NMSE 与验证 seed 一致；结果记录中必须明确标注该行为。
* 方法3公式：训练集功率归一化因子为 `sqrt(mean(abs(H_train)^2))`，训练、验证和测试信道统一除以该因子。
* 方法4说明：输出 NMSE 为线性值；若需要 dB 表示，后处理公式为 `NMSE_dB = 10 * log10(NMSE)`。
* 量化方式：最终 CSV 应包含 Diffusion Model 在多个 SNR 点上的 NMSE，例如表头为 `SNR,nmse_dm`。
