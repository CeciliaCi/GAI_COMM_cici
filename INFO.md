# LEO DeepMIMO 数据集框架说明

## 文件结构

每个生成的 LEO DeepMIMO 数据集保存为一个 `.mat` 文件，包含以下三个顶层变量：

| 变量名 | 说明 |
|--------|------|
| `channels` | 窄带等效信道张量，`complex double` 类型 |
| `dataset_params` | 数据集的完整参数结构体 |
| `sample_info` | 用户几何与链路信息结构体 |

---

## 1. `channels` — 信道张量

### 维度

```
size(channels) = [num_users, num_rx_antennas, num_tx_antennas]
```

| 维度 | 含义 | 典型值（当前配置） |
|------|------|-------------------|
| dim 1 | 用户样本数 | `500` |
| dim 2 | 接收天线数（用户终端 UPA） | `16` (4×4) |
| dim 3 | 发射天线数（卫星 UPA） | `144` (12×12) |

### 数值含义

- **窄带等效中心子载波**：将 QuaDRiGa 的多径延迟域在中心频点处合并为单值
- `channels(i, :, :)` 表示第 `i` 个用户的 `16×144` MIMO 信道矩阵
- 每个元素为复数值（`complex double`）

### 示例：提取第 1 个用户的信道矩阵

```matlab
H = squeeze(channels(1, :, :));  % 16×144 complex double
```

---

## 2. `dataset_params` — 参数结构体

### 场景与链路

| 字段 | 类型 | 说明 | 当前值 |
|------|------|------|--------|
| `scenario` | `char` | 完整场景描述 | `'3GPP 38.811 DenseUrban LOS'` |
| `scenario_type` | `char` | 场景类型 | `'DenseUrban'` |
| `link_state` | `char` | 链路状态 | `'LOS'` |
| `channel_model` | `char` | QuaDRiGa 信道模型标识 | 依场景而定 |
| `channel_model_reference` | `char` | 模型标准引用 | `'3GPP 38.811'` |
| `path_count` | `double` | 多径路径数（簇数） | `10` |
| `quadriga_num_clusters` | `double` | QuaDRiGa 簇数 | `10` |
| `quadriga_num_subpaths` | `double` | 每簇子径数 | `1` |

### 轨道与几何

| 字段 | 类型 | 说明 | 当前值 |
|------|------|------|--------|
| `earth_radius_km` | `double` | 地球半径 (km) | `6378` |
| `orbit_height_km` | `double` | 轨道高度 (km) | `1000` |
| `min_elevation_deg` | `double` | 最小仰角 (°) | `30` |
| `max_elevation_deg` | `double` | 最大仰角 (°) | `90` |
| `min_nadir_angle_deg` | `double` | 最小星下点角 (°) | `0` (=90-90) |
| `max_nadir_angle_deg` | `double` | 最大星下点角 (°) | `60` (=90-30) |

### 频率与带宽

| 字段 | 类型 | 说明 | 当前值 |
|------|------|------|--------|
| `carrier_frequency_hz` | `double` | 载波频率 (Hz) | `2e9` |
| `speed_of_light_mps` | `double` | 光速 (m/s) | `299792458` |
| `wavelength_m` | `double` | 波长 (m) | `0.149896` |
| `bandwidth_hz` | `double` | 信号带宽 (Hz) | `20e6` |
| `noise_temperature_k` | `double` | 噪声温度 (K) | `290` |

### OFDM 参数（元数据 / 后续可能扩展用）

| 字段 | 类型 | 说明 | 当前值 |
|------|------|------|--------|
| `num_subcarriers` | `double` | 总子载波数 | `512` |
| `num_pilot_subcarriers` | `double` | 导频子载波数 | `128` |
| `pilot_ratio` | `double` | 导频占比 | `0.25` |
| `subcarrier_spacing_hz` | `double` | 子载波间隔 (Hz) | `60e3` |
| `sampling_interval_s` | `double` | 采样间隔 (s) | `32.6e-9` |
| `cyclic_prefix_length` | `double` | CP 长度 | `36` |

### 天线配置

| 字段 | 类型 | 说明 | 当前值 |
|------|------|------|--------|
| `num_rx_antennas` | `double` | 用户终端天线数 | `16` |
| `num_tx_antennas` | `double` | 卫星天线数 | `144` |
| `satellite_array` | `struct` | 卫星天线阵列参数 | `12×12` UPA |
| `user_array` | `struct` | 用户天线阵列参数 | `4×4` UPA |
| `antenna_spacing_wavelength` | `double` | 天线间距 (波长倍数) | `1` |
| `antenna_spacing_m` | `double` | 天线间距 (m) | `≈0.1499` |
| `satellite_antenna_gain_dbi` | `double` | 卫星天线增益 (dBi) | `7` |
| `user_antenna_gain_dbi` | `double` | 用户天线增益 (dBi) | `0` |

### 功率与用户

| 字段 | 类型 | 说明 | 当前值 |
|------|------|------|--------|
| `num_users` | `double` | 用户数 | `500` |
| `user_terminal_height_m` | `double` | 用户终端高度 (m) | `1.5` |
| `tx_power_range_dbw` | `double[2]` | 发射功率范围 (dBW) | `[0, 20]` |

### 标识信息

| 字段 | 类型 | 说明 | 示例值 |
|------|------|------|--------|
| `dataset_name` | `char` | 数据集名称 | `'LEO_DenseUrban_LOS_h1000km_el30-90_path10_seed1111'` |
| `output_mat_file` | `char` | .mat 文件名 | `'LEO_DenseUrban_LOS_h1000km_el30-90_path10_seed1111.mat'` |
| `output_directory` | `char` | 输出目录路径 | 自动生成 |
| `seed` | `double` | 随机种子 | `1111` |
| `dataset_type` | `char` | 数据集类型 | `'channel_estimation'` |
| `framework` | `char` | 框架标识 | `'QuaDRiGa'` |
| `deepmimo_style` | `logical` | 是否为 DeepMIMO 风格 | `true` |
| `compatibility_mode` | `char` | 兼容模式 | `'deepmimo_style_leo_ntn_sweep'` |

---

## 3. `sample_info` — 用户几何与链路信息

### 字段列表

| 字段 | 维度 | 说明 |
|------|------|------|
| `user_position_m` | `[3, num_users]` | 用户位置 (x,y,z) 米，以星下点为原点 |
| `satellite_position_m` | `[3, num_users]` | 卫星位置 (x,y,z) 米 |
| `range_km` | `[1, num_users]` | 星地距离 (km) |
| `elevation_deg` | `[1, num_users]` | 仰角 (°) |
| `azimuth_deg` | `[1, num_users]` | 方位角 (°) |
| `nadir_angle_deg` | `[1, num_users]` | 星下点角 (°) |
| `path_loss_db` | `[1, num_users]` | 路径损耗 (dB) |
| `large_scale_gain` | `[1, num_users]` | 大尺度增益（线性） |
| `los_flag` | `[1, num_users]` | LOS 标记（逻辑值，当前全为 `true`） |
| `link_state` | `cell[1, num_users]` 或 `char` | 链路状态字符串（当前全为 `'LOS'`） |
| `orbit_height_km` | `[1, num_users]` | 轨道高度 (km) |
| `path_count` | `[1, num_users]` | 路径数 |
| `tx_power_dbw` | `[1, num_users]` | 各用户的发射功率 (dBW) |
| `max_doppler_hz` | `[1, num_users]` | 最大多普勒频移 (Hz) |
| `coverage_radius_m` | `double` | 覆盖外半径 (m) |
| `coverage_inner_radius_m` | `double` | 覆盖内半径 (m) |

### 几何关系示意

```
         卫星 (0, 0, H)
        /|
       / | 星下点角 = 90° - 仰角
      /  |
     /   |
    /θ   | H = 轨道高度 - 用户高度
   /     |
  /______|
    d    地面用户 (r, 0, h)
```

---

## 典型数据加载示例

```matlab
%% 加载一个数据集
load('LEO_DenseUrban_LOS_h1000km_el30-90_path10_seed1111.mat');

%% 查看基本信息
whos channels dataset_params sample_info
fprintf('channels size: [%s]\n', num2str(size(channels)));
fprintf('num_users = %d, num_rx = %d, num_tx = %d\n', ...
    size(channels,1), size(channels,2), size(channels,3));
fprintf('scenario: %s\n', dataset_params.scenario);
fprintf('orbit height: %.0f km\n', dataset_params.orbit_height_km);
fprintf('path count: %d\n', dataset_params.path_count);

%% 提取第 i 个用户的 MIMO 信道矩阵 (rx×tx)
i = 1;
H_i = squeeze(channels(i, :, :));  % [16×144] complex double

%% 查看第 i 个用户的几何信息
fprintf('User %d: range=%.2f km, elevation=%.1f°, azimuth=%.1f°\n', ...
    i, sample_info.range_km(i), ...
    sample_info.elevation_deg(i), ...
    sample_info.azimuth_deg(i));
```

## 补充文件

每生成一个数据集，输出目录下还会自动生成：

| 文件 | 说明 |
|------|------|
| `dataset_params.mat` | `dataset_params` 结构体的独立备份 |
| `dataset_params.json` | `dataset_params` 的 JSON 格式可读版本 |
| `dataset_summary.txt` | 文本格式的摘要信息（含功率统计等） |
| `channel_geometry.csv` | 每个用户的几何信息 CSV |
| `channel_power_histogram.png` | 信道功率分布直方图 |
| `beamspace_example.png` | 第 1 个用户的波束域信道示例图 |
