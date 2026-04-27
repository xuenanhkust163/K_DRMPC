# K-DRMPC K-DRMPC - 分布鲁棒Koopman模型预测控制实验

## 项目简介

本项目实现了基于Koopman算子的分布鲁棒模型预测控制（K-DRMPC）算法，用于车辆动力学控制和避障。

**核心特性**:
- Deep Koopman网络：数据驱动的非线性系统线性化
- 分布鲁棒MPC：基于Wasserstein模糊集和CVaR约束
- 多种MPC控制器对比：LMPC、NMPC、K-MPC、K-DRMPC
- 完整仿真环境：赛道建模、干扰生成、性能评估

## 快速开始

### 1. 创建虚拟环境

```bash
# 创建conda环境
conda create -n koopman_experiment python=3.11
conda activate koopman_experiment

# 或使用venv
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 运行完整流水线

```bash
# 运行所有实验（训练+仿真+分析）
python run_all.py

# 或分步运行
python run_training.py      # 训练Koopman模型
python run_simulation.py    # 运行仿真
python run_analysis.py      # 生成图表和表格
```

### 4. 单次仿真示例（全参数显式）

在 Lusail Short 赛道跑 100 步、开启障碍物，并将所有参数显式指定：

```bash
cd /Users/nevinxue/source/K_DRMPC && .venv/bin/python run_simulation.py \
  --track lusail-short \
  --steps 100 \
  --cost-profile default \
  --control-every 1 \
  --obstacle-strategy robust \
  --delta-max 2.5 \
  --delta-rate-max 0.05 \
  --speed-scale 0.18 \
  --horizon 40 \
  --theta 0.05 \
  --epsilon 0.1 \
  --obstacles \
  --verbose
```

常用赛道命令：

```bash
# Lusail Short（100步，带障碍，详细日志）
cd /Users/nevinxue/source/K_DRMPC && .venv/bin/python run_simulation.py \
  --track lusail-short --steps 100 --obstacles --verbose

# Sprint Oval（200步，带障碍，详细日志）
cd /Users/nevinxue/source/K_DRMPC && .venv/bin/python run_simulation.py \
  --track sprint-oval --steps 200 --obstacles --verbose

# 原始 Lusail（200步，带障碍，详细日志）
cd /Users/nevinxue/source/K_DRMPC && .venv/bin/python run_simulation.py \
  --track lusail --steps 200 --obstacles --verbose

# 全赛道（慎用，耗时更长）
cd /Users/nevinxue/source/K_DRMPC && .venv/bin/python run_simulation.py \
  --track all --steps 200 --obstacles --verbose
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--steps` | 最大仿真步数 | 200（示例中显式设为 100） |
| `--track` | 选择赛道（sprint-oval / lusail-short / lusail / custom / all） | sprint-oval |
| `--cost-profile` | 代价模板（default / tracking-first / progress-first / mpcc-paper / mpcc-paper-cvar / stabilize-first） | default |
| `--control-every` | 控制降频，每 N 步求解一次 MPC | 1 |
| `--obstacle-strategy` | 障碍约束策略（robust / non-robust） | robust |
| `--delta-max` | 最大转向角 [度] | 2.5 |
| `--delta-rate-max` | 最大转向角速率 [弧度/秒] | 0.05 |
| `--speed-scale` | 全局参考速度缩放因子 | 0.18 |
| `--horizon` | MPC 预测时域步数 | 40 |
| `--theta` | Wasserstein 球半径 | 0.05 |
| `--epsilon` | CVaR 风险水平 | 0.1 |
| `--obstacles` | 开启赛道障碍物 | 关闭 |
| `--disturbance` | 开启在线扰动 | 关闭 |
| `--verbose` | 逐步详细日志打印到控制台 | 关闭 |
| `--fast` | 快速验证模式（固定 300 步） | 关闭 |

## 项目结构

```
K-DRMPC/
├── controllers/          # MPC控制器实现
│   ├── lmpc_controller.py      # 线性MPC
│   ├── nmpc_controller.py      # 非线性MPC
│   ├── kmpc_controller.py      # Koopman MPC
│   └── kdrmpc_controller.py    # 分布鲁棒Koopman MPC
├── model/                # Koopman模型
│   ├── koopman_network.py      # 网络架构
│   ├── koopman_trainer.py      # 训练逻辑
│   └── projection.py           # 投影矩阵
├── simulation/           # 仿真引擎
│   ├── simulator.py            # 闭环仿真
│   └── metrics.py              # 性能指标
├── tracks/               # 赛道模型
│   ├── base_track.py           # 基础赛道类
│   ├── lusail_track.py         # 卢赛尔赛道
│   └── custom_track.py         # 自定义赛道
├── vehicle/              # 车辆模型
│   └── bicycle_model.py        # 自行车模型
├── disturbance/          # 干扰生成
│   ├── disturbance_generator.py # 高斯混合干扰
│   └── wasserstein.py          # Wasserstein模糊集
├── visualization/        # 可视化
│   ├── plot_trajectories.py    # 轨迹绘制
│   ├── plot_tsne.py            # t-SNE可视化
│   └── plot_tables.py          # 表格生成
├── data/                 # 数据加载
│   └── data_loader.py
├── _output/              # 输出目录
│   ├── models/                 # 训练好的模型
│   ├── results/                # 仿真结果
│   ├── figures/                # 图表
│   └── tables/                 # 表格
├── config.py             # 配置文件
├── run_all.py            # 完整流水线
├── run_training.py       # 训练脚本
├── run_simulation.py     # 仿真脚本
└── run_analysis.py       # 分析脚本
```

## 控制器方法

| 方法 | 描述 | 特点 |
|------|------|------|
| **LMPC** | 线性MPC | 快速、稳定、基准方法 |
| **NMPC** | 非线性MPC | 精确、使用完整非线性模型 |
| **K-MPC** | Koopman MPC | 数据驱动、Koopman线性化 |
| **K-DRMPC** | 分布鲁棒Koopman MPC | 鲁棒性强、处理不确定性 |

## 配置参数

主要参数在 `config.py` 中配置：

```python
# 车辆参数
DT = 0.1                   # 采样时间 [s]
L_WHEELBASE = 2.6          # 轴距 [m]
V_MAX = 40.0               # 最大速度 [m/s]

# MPC参数
T_HORIZON = 40             # 预测时域
Q_WEIGHTS = np.diag([1.5, 3.0])  # 状态权重
R_WEIGHTS = np.diag([1.5, 3.0])  # 控制权重

# 分布鲁棒参数
THETA_WASSERSTEIN = 0.05   # Wasserstein半径
EPSILON_CVAR = 0.1         # CVaR风险水平
```

## 输出结果

### 仿真结果
- `_output/results/*.pkl` - 各方法的仿真数据
- 包含：状态轨迹、控制输入、求解时间等

### 图表
- `_output/figures/*.pdf` - 高质量PDF图表
- 轨迹对比、状态曲线、性能分析等

### 表格
- `_output/tables/*.txt` - 文本表格
- `_output/tables/*.tex` - LaTeX表格

## 性能指标

系统计算以下性能指标：
- **圈速时间**：完成一圈所需时间
- **跟踪误差**：与参考轨迹的偏差（RMS）
- **速度统计**：最大、平均、最小速度
- **约束违反**：障碍物距离违反百分比
- **CVaR安全裕度**：风险价值安全裕度
- **求解时间**：MPC优化耗时统计
- **控制努力度**：控制输入RMS值

## 依赖要求

- Python >= 3.10
- PyTorch >= 2.8.0
- NumPy >= 1.26.4
- CasADi >= 3.7.2
- SciPy >= 1.13.1
- Matplotlib >= 3.9.4

完整依赖列表见 `requirements.txt`

## 引用

如果本工作对您的研究有帮助，请引用：

```bibtex
@article{koopman_drpc,
  title={Distributionally Robust Model Predictive Control with Disturbance Uncertainty Adaptiveness for Obstacle Avoidance},
  journal={...},
  year={2026}
}
```

## 许可证

本项目基于Eclipse Public License (EPL)发布。

## 联系方式

如有问题或建议，请提交Issue或Pull Request。
