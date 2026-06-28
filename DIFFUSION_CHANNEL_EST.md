# 扩散模型信道估计设计
本项目针对稀疏角域多输入多输出（MIMO）信道，训练并评估基于扩散模型的信道估计器。信道矩阵首先表示为复数值空间域张量，随后转换为实部/虚部双通道张量，以供卷积神经网络（CNN）使用。当`fft_pre=True`时，训练器在扩散训练前对天线轴执行酉快速傅里叶变换（FFT），并在评估阶段使用逆变换还原。

## 低轨卫星（LEO）DeepMIMO数据层
低轨卫星DeepMIMO数据集通过`loaders.py`加载。加载器支持读取顶层包含`channels`（信道）、`dataset_params`（数据集参数）和`sample_info`（样本信息）变量的MATLAB `.mat`格式文件，同时通过`h5py`库兼容MATLAB 7.3/HDF5格式文件。文件名过滤支持两种命名格式：简洁命名（如`LEO_DenseUrban_seed1111.mat`）和详细命名（如`LEO_DenseUrban_LOS_h1000km_el30-90_path10_seed1111.mat`）。

归一化后的信道张量维度为`[用户数量, 16, 144]`，数据类型为`complex128`（双精度复数）。其中接收端维度对应4×4均匀平面天线阵列（UPA），发射端维度对应12×12均匀平面天线阵列（UPA）。角域映射采用酉二维离散傅里叶变换（DFT），最终输出为`[用户数量, 2, 16, 144]`的实数值特征图。通过逆变换验证原始信道与逆变换后信道的**弗罗贝尼乌斯范数平方**`||原始信道 - 逆变换信道||_F²`在数值上趋近于零。

`sample_info`中的用户级数值字段会被整合为可选的几何特征矩阵。默认情况下，这些特征不输入卷积神经网络，仅作为元数据保留，用于后续基于几何条件的扩散模型实验。

## 程序入口
`diff_cnn.py`是训练与评估的主入口文件。默认配置沿用论文中的3GPP标准设置。传入参数`--channel-type leo`可启用低轨卫星数据加载器，默认使用`16×144`的天线维度，自动聚合所有匹配文件、打乱数据顺序，并按照`--split-ratios`参数划分数据集；若手动指定`--train-samples`（训练样本数）、`--val-samples`（验证样本数）和`--test-samples`（测试样本数），则优先使用手动配置。

若需要严格按文件划分训练、验证和评估数据，可传入`--train-files`、`--val-files`和`--test-files`。例如Urban双seed实验中，`dataset/LEO_Urban_seed1111.mat`只用于训练，`dataset/LEO_Urban_seed2222.mat`用于验证；如果省略`--test-files`，验证文件会同时作为最终NMSE评估数据。指定文件模式下不会再执行多文件聚合后的随机混合切分，并会使用训练集平均元素功率因子统一归一化训练、验证和测试信道；实际文件路径、样本数量、归一化因子和NMSE评估SNR范围会写入`sim_params.json`与`results/dm_est/*_params.csv`。`--data-preset p006`提供`dataset/p006`的固定文件级验证配置：`seed1111`用于训练，`seed2222`用于验证和测试。Diffusion NMSE默认评估范围为`[-15, 20] dB`，步长为`5 dB`，可通过`--snr-min-db`、`--snr-max-db`和`--snr-step-db`调整。

`loaders.py`也可直接独立运行，用于检索匹配的低轨卫星数据文件、打印信道维度和解析后的物理元数据，并验证角域变换的往返误差。

`scripts/analyze_angular_concentration.py`用于比较不同LEO数据集在角域中的能量集中程度。脚本对空间域信道执行酉二维FFT，按每个样本的角域能量占比计算top-k能量比例、effective bins，以及达到90%/95%能量所需的角域bin数量，用于判断Urban与DenseUrban等场景的角域结构差异。

`baselines.py`默认从`dataset`目录加载低轨卫星Rural场景数据，并使用`16×144`天线维度评估LS与角域对角LMMSE基线。角域对角LMMSE先对信道执行酉二维FFT，再按每个角域bin的训练集功率方差进行逐元素LMMSE收缩，适合大规模天线和有限训练样本的LEO信道。基线评估前会按训练集平均元素功率对信道进行归一化，使AWGN噪声的SNR定义与归一化信道假设一致；如需旧的全协方差LMMSE，可传入`--lmmse-mode global_full`，如需直接使用物理幅度，可传入`--no-normalize-power`。可通过`--scenario Urban`或`--scenario DenseUrban`切换到其他低轨卫星场景。

---

### 翻译说明
1. **专业术语标准化**：严格遵循通信领域通用译法（MIMO=多输入多输出、UPA=均匀平面天线阵列、FFT/DFT=傅里叶变换）
2. **代码变量保留**：所有Python文件名、参数、张量维度、数据类型**原样保留**，方便你对照代码使用
3. **句式优化**：将英文长句拆分为符合中文技术文档习惯的短句，保证可读性
4. **关键公式标注**：保留数学公式原文，补充中文释义
