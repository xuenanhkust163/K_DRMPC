# K-DRMPC 快速参考卡

## 🚀 快速开始

```bash
# 1. 进入目录
cd K-DRMPC

# 2. 激活环境（如果已创建）
source venv/bin/activate  # 或 conda activate koopman_exp

# 3. 运行
python run_all.py
```

## 📦 首次设置

```bash
# 方法1: 使用自动脚本
./setup.sh

# 方法2: 手动设置
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 🎯 常用命令

### 运行实验

```bash
# 完整流水线
python run_all.py

# 分步运行
python run_training.py       # 训练模型
python run_simulation.py     # 运行仿真
python run_analysis.py       # 生成图表
```

### 查看结果

```python
# Python中加载结果
import pickle

# 加载仿真结果
with open('_output/results/NMPC_LusailTrack.pkl', 'rb') as f:
    result = pickle.load(f)

# 查看基本信息
print(f'方法: {result.method_name}')
print(f'圈速: {result.lap_time}')
print(f'步数: {result.total_steps}')
```

## 📊 输出文件位置

| 类型 | 路径 | 说明 |
|------|------|------|
| 仿真数据 | `_output/results/*.pkl` | 完整仿真结果 |
| 图表 | `_output/figures/*.pdf` | 高质量PDF图 |
| 表格 | `_output/tables/*.txt` | 文本表格 |
| 模型 | `_output/models/*.pth` | 训练好的模型 |

## ⚙️ 常用配置

编辑 `config.py`:

```python
# 减少仿真时间（测试用）
MAX_SIM_STEPS = 500  # 默认2000

# 减少预测时域（加速）
T_HORIZON = 20  # 默认40

# 放宽求解器（加速）
IPOPT_MAX_ITER = 200  # 默认500
IPOPT_TOL = 1e-4      # 默认1e-6
```

## 🔍 故障排查

### 问题：找不到模块

```bash
# 确保在K-DRMPC目录
cd /path/to/koopman_mpc/K-DRMPC

# 重新激活环境
deactivate
source venv/bin/activate
```

### 问题：找不到数据

```bash
# 检查数据文件
ls -la ../_output/_data_process/

# 如果不存在，需要从父项目生成
cd ..
python data_preprocess/data_preprocessor.py
```

### 问题：CUDA错误

```python
# 代码自动使用CPU（如果没有GPU）
# 强制使用CPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
```

## 📈 性能指标

从结果中提取：

```python
import numpy as np

states = np.array(result.states)
controls = np.array(result.controls)

# 速度统计
avg_speed = np.mean(states[:, 2])
max_speed = np.max(states[:, 2])

# 控制努力度
control_effort = np.sqrt(np.mean(controls**2, axis=0))

# 求解时间
avg_solve_time = np.mean(result.solve_times)
```

## 🎨 可视化

```bash
# 生成所有图表
python visualization/plot_trajectories.py

# t-SNE可视化
python visualization/plot_tsne.py

# 生成表格
python visualization/plot_tables.py
```

## 📝 控制器对比

| 方法 | 速度 | 精度 | 鲁棒性 | 完成赛道 |
|------|------|------|--------|----------|
| LMPC | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ✗ |
| NMPC | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✓ |
| K-MPC | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ✗ |
| K-DRMPC | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ? |

## 🔧 环境管理

```bash
# 查看已安装包
pip list

# 导出环境
pip freeze > my_requirements.txt

# 更新包
pip install --upgrade torch numpy

# 重建环境
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 📚 文档

- `README.md` - 项目概述
- `SETUP.md` - 详细设置指南
- `docs/` - 数据结构文档

## 💡 提示

1. **测试时**：减少 `MAX_SIM_STEPS` 到 100-500
2. **调试时**：设置 `verbose=True` 查看详细信息
3. **加速**：使用较小的 `T_HORIZON`
4. **质量**：使用默认参数获得最佳结果

## 🆘 获取帮助

1. 查看错误日志
2. 检查 `config.py` 参数
3. 阅读 `SETUP.md`
4. 提交Issue

---

**版本**: 1.0  
**Python**: >= 3.10  
**最后更新**: 2026-04-13
