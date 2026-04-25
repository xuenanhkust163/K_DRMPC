# K-DRMPC独立项目设置指南

## 概述

本指南说明如何将K-DRMPC文件夹作为独立的Python项目运行，不依赖父项目。

## 前提条件

- Python 3.10 或更高版本
- pip 或 conda
- Git（可选，用于版本控制）

## 设置步骤

### 方法1：使用conda（推荐）

```bash
# 1. 进入K-DRMPC目录
cd K-DRMPC

# 2. 创建conda环境
conda create -n koopman_exp python=3.11 -y

# 3. 激活环境
conda activate koopman_exp

# 4. 安装依赖
pip install -r requirements.txt

# 5. 验证安装
python -c "import torch; import numpy; import casadi; print('✓ 所有核心库安装成功')"
```

### 方法2：使用venv

```bash
# 1. 进入K-DRMPC目录
cd K-DRMPC

# 2. 创建虚拟环境
python3 -m venv venv

# 3. 激活环境
# macOS/Linux:
source venv/bin/activate
# Windows:
# venv\Scripts\activate

# 4. 安装依赖
pip install -r requirements.txt

# 5. 验证安装
python -c "import torch; import numpy; import casadi; print('✓ 所有核心库安装成功')"
```

## 数据准备

K-DRMPC需要访问训练数据。确保以下文件存在：

```bash
# 从父项目复制数据（如果需要完全独立）
cp -r ../_output/_data_process ./_output/_data_process

# 或者保持符号链接到父项目
ln -s ../_output/_data_process ./_output/_data_process
```

## 运行项目

### 完整流水线

```bash
# 运行所有步骤（训练+仿真+分析）
python run_all.py
```

### 分步运行

```bash
# 1. 训练Koopman模型
python run_training.py

# 2. 运行仿真实验
python run_simulation.py

# 3. 生成图表和表格
python run_analysis.py
```

### 单独运行特定仿真

```python
# 在Python中
from run_simulation import main
main()
```

## 项目结构说明

```
K-DRMPC/                          # 项目根目录
├── controllers/                     # 控制器实现
├── model/                          # Koopman模型
├── simulation/                     # 仿真引擎
├── tracks/                         # 赛道模型
├── vehicle/                        # 车辆模型
├── disturbance/                    # 干扰生成
├── visualization/                  # 可视化工具
├── data/                           # 数据加载
├── _output/                        # 输出目录
│   ├── models/                     # 训练好的模型
│   ├── results/                    # 仿真结果(.pkl)
│   ├── figures/                    # 图表(PDF)
│   └── tables/                     # 表格(TXT/TEX)
├── config.py                       # 配置文件
├── requirements.txt                # Python依赖
├── run_all.py                      # 完整流水线
├── run_training.py                 # 训练脚本
├── run_simulation.py              # 仿真脚本
├── run_analysis.py                # 分析脚本
└── README.md                      # 项目文档
```

## 配置修改

主要参数在 `config.py` 中：

```python
# 车辆参数
DT = 0.1                   # 采样时间 [s]
V_MAX = 40.0               # 最大速度 [m/s]

# MPC参数
T_HORIZON = 40             # 预测时域

# 分布鲁棒参数
THETA_WASSERSTEIN = 0.05   # Wasserstein半径
EPSILON_CVAR = 0.1         # CVaR风险水平
```

## 输出文件

### 仿真结果
- `_output/results/*.pkl` - 包含完整仿真数据
- 可用pickle加载：
  ```python
  import pickle
  with open('_output/results/NMPC_LusailTrack.pkl', 'rb') as f:
      result = pickle.load(f)
  ```

### 图表
- `_output/figures/*.pdf` - 高质量PDF图表
- 轨迹对比、状态曲线、性能分析

### 表格
- `_output/tables/*.txt` - 文本格式
- `_output/tables/*.tex` - LaTeX格式

## 常见问题

### Q: 找不到数据文件
A: 确保 `_output/_data_process/` 目录存在并包含训练数据

### Q: 导入错误
A: 确保在K-DRMPC目录下运行，或将其添加到PYTHONPATH

### Q: CUDA不可用
A: 代码支持CPU运行，如需GPU支持请安装CUDA版本的PyTorch

### Q: 仿真速度慢
A: 
- 减少 `MAX_SIM_STEPS` (config.py)
- 减少 `T_HORIZON` (预测时域)
- 使用更快的求解器选项

## 独立部署

如需将K-DRMPC部署到其他机器：

```bash
# 1. 打包项目
tar -czf koopman_experiment.tar.gz \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='_output/results/*.pkl' \
    K-DRMPC/

# 2. 在目标机器上
tar -xzf koopman_experiment.tar.gz
cd K-DRMPC
pip install -r requirements.txt
```

## 性能优化建议

1. **使用GPU**（如果可用）
   ```python
   # 在config.py中确保
   # PyTorch会自动使用CUDA
   ```

2. **并行仿真**
   ```bash
   # 修改run_simulation.py使用多进程
   ```

3. **减少预测时域**
   ```python
   T_HORIZON = 20  # 从40减少到20
   ```

4. **使用更快的求解器选项**
   ```python
   IPOPT_MAX_ITER = 200  # 从500减少
   IPOPT_TOL = 1e-4      # 从1e-6放宽
   ```

## 技术支持

遇到问题？
1. 检查README.md
2. 查看config.py中的参数说明
3. 查看输出日志中的错误信息
4. 提交Issue

## 版本信息

- Python: >= 3.10
- PyTorch: 2.8.0
- NumPy: 1.26.4
- CasADi: 3.7.2

完整依赖见 `requirements.txt`
