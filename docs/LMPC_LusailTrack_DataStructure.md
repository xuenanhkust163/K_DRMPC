# LMPC_LusailTrack.pkl 数据结构详解

## 文件概述

**文件路径**: `Experiment/_output/results/LMPC_LusailTrack.pkl`

**文件类型**: Python Pickle 序列化文件

**数据对象**: `SimResult` 类实例（定义在 `simulation/simulator.py`）

**用途**: 存储 LMPC（Linear Model Predictive Control）控制器在卢赛尔赛道（LusailTrack）上的完整闭环仿真结果。

**控制器说明**: LMPC 使用线性化的车辆动力学模型进行预测控制，是传统的基准方法。

---

## 对象基本信息

| 属性 | 值 | 说明 |
|------|-----|------|
| **对象类型** | `simulation.simulator.SimResult` | 仿真结果容器类 |
| **method_name** | `"LMPC"` | 控制器方法名称：Linear MPC |
| **track_name** | `"LusailTrack"` | 赛道名称：卢赛尔赛道 |
| **lap_completed** | `False` | 未完成完整一圈 |
| **lap_time** | `None` | 无圈速记录（未完成） |
| **total_steps** | `1705` | 总仿真步数（170.4秒） |

---

## 数据结构详解

### 1. states（状态轨迹）

**数据类型**: `list` of `numpy.ndarray`

**尺寸**: 1706 个状态向量，每个形状为 `(5,)`

**转换为数组后**: `(1706, 5)` 的二维数组

**含义**: 记录仿真过程中每个时刻的车辆状态，包括初始状态（T+1个时间点）

#### 状态向量组成 `[px, py, v, psi, omega]`

| 索引 | 变量名 | 物理含义 | 单位 | 最小值 | 最大值 | 平均值 |
|------|--------|----------|------|--------|--------|--------|
| 0 | **px** | 位置X坐标 | 米 (m) | 0.00 | 2091.88 | 1488.60 |
| 1 | **py** | 位置Y坐标 | 米 (m) | -31.92 | 746.21 | 383.18 |
| 2 | **v** | 纵向速度 | 米/秒 (m/s) | 16.88 | 40.00 | 28.16 |
| 3 | **psi** | 航向角 | 弧度 (rad) | -0.87 | 13.41 | 3.61 |
| 4 | **omega** | 横摆角速度 | 弧度/秒 (rad/s) | -0.20 | 0.24 | 0.08 |

**示例数据**（初始状态）:
```python
states[0] = [0.0, 0.0, 16.896, -0.871, 0.237]
# 位置: (0, 0)m, 速度: 16.9m/s, 航向角: -0.87rad (-49.9°), 角速度: 0.24rad/s
```

**关键观察**:
- ✓ 车辆从原点 (0, 0) 开始，沿X轴正向行驶
- ✓ 位置X范围: 0 → 2092m，行驶距离约2.1km
- ✓ 位置Y范围: -32 → 746m，表示有横向运动
- ✓ 速度范围 16.9-40.0 m/s（约 61-144 km/h），平均 28.2 m/s（101 km/h）
- ✓ 航向角从 -0.87 rad 增加到 13.41 rad（768°），表示多个弯道
- ✓ 横摆角速度较小（-0.2 至 0.24 rad/s），说明转弯较为平缓

---

### 2. controls（控制输入轨迹）

**数据类型**: `list` of `numpy.ndarray`

**尺寸**: 1705 个控制向量，每个形状为 `(2,)`

**转换为数组后**: `(1705, 2)` 的二维数组

**含义**: 记录每个控制时刻施加的控制输入（T个时间步，比状态少1个）

#### 控制向量组成 `[a, delta]`

| 索引 | 变量名 | 物理含义 | 单位 | 最小值 | 最大值 | 平均值 | RMS | 标准差 |
|------|--------|----------|------|--------|--------|--------|-----|--------|
| 0 | **a** | 加速度 | 米/秒² (m/s²) | -5.00 | 3.00 | 0.05 | 1.37 | 1.37 |
| 1 | **delta** | 前轮转向角 | 弧度 (rad) | -0.026 | 0.037 | 0.009 | 0.016 | 0.014 |

**示例数据**（第一个控制输入）:
```python
controls[0] = [-0.093, 0.035]
# 加速度: -0.09m/s²（轻微减速）, 转向角: 0.035rad (2.0°)
```

**约束限制**（来自 config.py）:
- 加速度: `A_MIN=-5.0` 至 `A_MAX=3.0` m/s²
- 转向角: `-DELTA_MAX=-π/4` 至 `DELTA_MAX=π/4` rad（-45° 至 45°）

**关键观察**:
- ✓ **加速度充分利用**: -5.0 至 3.0 m/s²，说明控制器积极加减速
- ✓ **平均加速度接近0** (0.054 m/s²)，表示加速和减速基本平衡
- ✓ **转向角非常小**: -0.026 至 0.037 rad（-1.49° 至 2.13°）
- ✓ 转向角平均值 0.009 rad（0.52°），表明赛道相对平缓
- ✓ 转向角RMS仅 0.016 rad，说明转向操作温和

---

### 3. solve_times（求解时间）

**数据类型**: `list` of `float`

**尺寸**: 1705 个时间值

**单位**: 毫秒 (ms)

**含义**: 每个时间步MPC优化问题的求解耗时

| 统计量 | 值 | 说明 |
|--------|-----|------|
| **平均值** | 0.47 ms | 平均求解时间（非常快） |
| **标准差** | 0.13 ms | 波动很小 |
| **最小值** | 0.31 ms | 最快求解 |
| **最大值** | 3.00 ms | 最慢求解 |
| **中位数** | 0.45 ms | 典型求解时间 |

**关键观察**:
- ✓ **求解速度极快**: 平均 0.47ms，远小于实时性要求（<100ms）
- ✓ **稳定性好**: 标准差仅 0.13ms，波动小
- ✓ **最大值仅 3ms**: 线性MPC计算效率非常高
- ✓ 适合实时控制应用（10Hz控制频率，每步可用100ms）

**与K-MPC对比**:
- LMPC平均: 0.47ms vs K-MPC平均: 74.84ms
- LMPC快约 **160倍**

---

### 4. solve_statuses（求解状态）

**数据类型**: `list` of `str`

**尺寸**: 1705 个状态字符串

**含义**: IPOPT求解器的返回状态，表明优化是否成功

#### 状态分布

| 状态 | 数量 | 百分比 | 说明 |
|------|------|--------|------|
| **optimal** | 1705 | 100.0% | 全部找到最优解 |

**关键观察**:
- ✓ **100%成功率**: 所有1705步都成功求解
- ✓ **数值稳定性好**: 线性MPC凸优化问题性质良好
- ✓ 没有遇到不可行或求解失败的情况

---

### 5. ref_states（参考状态轨迹）

**数据类型**: `list` of `numpy.ndarray`

**尺寸**: 1705 个参考状态，每个形状为 `(5,)` 或 `(T_HORIZON, 5)`

**含义**: 每个时刻MPC控制器跟踪的参考状态（来自赛道的参考轨迹）

**示例数据**:
```python
ref_states[0] = [0.0, 0.0, 16.896, -0.871, 0.237]
ref_states[1] = [1.55, -1.78, 16.87, -0.84, 0.24]
```

**用途**: 
- 用于计算跟踪误差
- 评估控制器性能
- 可视化实际轨迹 vs 参考轨迹

---

### 6. timestamps（时间戳）

**数据类型**: `list` of `float`

**尺寸**: 1705 个时间点

**单位**: 秒 (s)

**含义**: 每个控制时刻的仿真时间

**时间序列**: `[0.0, 0.1, 0.2, 0.3, ..., 170.4]`

**采样时间**: DT = 0.1 秒（10Hz控制频率）

**总时长**: 170.4 秒（约2分50秒）

---

## 控制平滑度分析

### 控制输入变化率

| 控制量 | 最小变化 | 最大变化 | RMS变化 | 说明 |
|--------|----------|----------|---------|------|
| **加速度变化 (da)** | -8.00 | 3.43 | 0.30 | 存在突变（饱和切换） |
| **转向角变化 (ddelta)** | -0.003 | 0.003 | 0.0003 | 非常平滑 |

**关键观察**:
- ⚠️ 加速度最大变化 -8.0 m/s²，说明存在从最大加速到最大减速的突变
- ✓ 转向角变化非常小（RMS 0.0003 rad），控制平滑
- 加速度突变可能是由于达到约束边界导致的饱和现象

---

## 数据关系图

```
时间步:     0        1        2       ...    1704
         ┌────────┬────────┬────────┬───────┬────────┐
states   │ state0 │ state1 │ state2 │ ...   │ state1705│  (1706个)
         └────────┴────────┴────────┴───────┴────────┘
                    ↓        ↓               ↓
         ┌────────┬────────┬────────┬───────┬────────┐
controls │ ctrl0  │ ctrl1  │ ctrl2  │ ...   │ ctrl1704│  (1705个)
         └────────┴────────┴────────┴───────┴────────┘
         
         state[t] + control[t] → state[t+1]  (动力学传播)
```

---

## 典型使用场景

### 1. 加载数据

```python
import pickle
import numpy as np

# 加载仿真结果
with open('LMPC_LusailTrack.pkl', 'rb') as f:
    result = pickle.load(f)

# 转换为NumPy数组
states = np.array(result.states)        # (1706, 5)
controls = np.array(result.controls)    # (1705, 2)
solve_times = np.array(result.solve_times)  # (1705,)
```

### 2. 提取轨迹

```python
# 位置轨迹
positions = states[:, :2]  # (1706, 2) [px, py]

# 速度曲线
velocities = states[:, 2]  # (1706,)

# 控制输入
accelerations = controls[:, 0]  # (1705,)
steering_angles = controls[:, 1]  # (1705,)
```

### 3. 性能分析

```python
# 跟踪误差计算
tracking_error = np.sqrt(
    (states[:-1, 0] - ref_states[:, 0])**2 + 
    (states[:-1, 1] - ref_states[:, 1])**2
)
rms_error = np.sqrt(np.mean(tracking_error**2))

# 控制努力度
control_effort_a = np.sqrt(np.mean(controls[:, 0]**2))  # 1.37 m/s²
control_effort_delta = np.sqrt(np.mean(controls[:, 1]**2))  # 0.016 rad

# 控制平滑度
du = np.diff(controls, axis=0)
smoothness = np.sqrt(np.mean(du**2, axis=0))

# 求解时间统计
avg_solve_time = np.mean(solve_times)  # 0.47 ms
max_solve_time = np.max(solve_times)   # 3.00 ms
```

### 4. 可视化

```python
import matplotlib.pyplot as plt

# 绘制轨迹
plt.figure(figsize=(10, 8))
plt.plot(states[:, 0], states[:, 1], 'b-', linewidth=1.5, label='LMPC Trajectory')
plt.xlabel('X Position (m)', fontsize=12)
plt.ylabel('Y Position (m)', fontsize=12)
plt.title('LMPC on Lusail Track', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()

# 绘制速度曲线
plt.figure(figsize=(10, 4))
plt.plot(timestamps, states[:, 2], 'g-', linewidth=1.5)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Velocity (m/s)', fontsize=12)
plt.title('Velocity Profile - LMPC', fontsize=14)
plt.grid(True, alpha=0.3)
plt.show()

# 绘制控制输入
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(timestamps, controls[:, 0], 'r-', linewidth=1)
ax1.set_ylabel('Acceleration (m/s²)', fontsize=11)
ax1.set_title('Control Inputs - LMPC', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='k', linewidth=0.5)

ax2.plot(timestamps, np.degrees(controls[:, 1]), 'b-', linewidth=1)
ax2.set_xlabel('Time (s)', fontsize=12)
ax2.set_ylabel('Steering Angle (deg)', fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='k', linewidth=0.5)

plt.tight_layout()
plt.show()

# 绘制求解时间
plt.figure(figsize=(10, 4))
plt.plot(timestamps, solve_times, 'purple', linewidth=0.8, alpha=0.7)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Solve Time (ms)', fontsize=12)
plt.title('MPC Solve Time - LMPC', fontsize=14)
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 与其他方法的比较

在完整的仿真实验中，会生成以下结果文件：

| 文件名 | 控制器 | 求解时间(平均) | 成功率 | 特点 |
|--------|--------|----------------|--------|------|
| `LMPC_LusailTrack.pkl` | **线性MPC** | **0.47 ms** | **100%** | 快速、稳定、基准 |
| `NMPC_LusailTrack.pkl` | 非线性MPC | ~50-500 ms | 高 | 精确但慢 |
| `K-MPC_LusailTrack.pkl` | Koopman MPC | ~75 ms | 较高 | 数据驱动 |
| `K-DRMPC_LusailTrack.pkl` | 分布鲁棒MPC | ~100-200 ms | 高 | 鲁棒性强 |

**LMPC优势**:
- ✓ 计算速度最快（0.47ms）
- ✓ 100%求解成功率
- ✓ 数值稳定性好
- ✓ 适合实时应用

**LMPC局限**:
- ⚠️ 线性模型在非线性区域精度有限
- ⚠️ 可能无法充分利用车辆动力学极限
- ⚠️ 跟踪误差可能较大（相比非线性方法）

---

## 仿真结果分析

### 性能指标估算

| 指标 | 值 | 说明 |
|------|-----|------|
| **仿真时长** | 170.4 s | 约2分50秒 |
| **行驶距离** | ~6308 m | 估算值（平均速度×时间） |
| **赛道完成度** | < 100% | 未完成完整一圈（赛道长5426m） |
| **平均速度** | 28.16 m/s | 约101 km/h |
| **最大速度** | 40.00 m/s | 达到速度上限（144 km/h） |
| **求解成功率** | 100% | 所有步骤成功 |
| **实时性** | ✓ 满足 | 0.47ms << 100ms |

### 未完成原因分析

尽管LMPC运行了170.4秒（超过最大圈速预期），但未完成一圈，可能原因：
1. **跟踪误差过大**: 横向偏离赛道中心线超过阈值
2. **速度策略保守**: 平均速度较低（28 m/s vs 最大40 m/s）
3. **线性模型局限**: 在高速弯道时模型失配

---

## 注意事项

1. **状态与控制的数量差异**: states 有 T+1 个（包含初始状态），controls 有 T 个
2. **单位统一**: 所有物理量使用国际单位制（SI）
3. **角度表示**: 航向角使用弧度制，持续累加（可能超过 2π）
4. **未完成圈**: 该仿真未完成完整一圈（lap_completed=False）
5. **Pickle版本**: 使用Python pickle序列化，需相同Python版本和类定义才能加载
6. **求解器**: 使用IPOPT求解线性MPC优化问题

---

## 相关文件

- **类定义**: `Experiment/simulation/simulator.py` - `SimResult` 类
- **配置文件**: `Experiment/config.py` - 参数配置
- **控制器**: `Experiment/controllers/lmpc_controller.py` - LMPC控制器实现
- **车辆模型**: `Experiment/vehicle/bicycle_model.py` - 自行车模型
- **可视化**: `Experiment/visualization/plot_trajectories.py` - 轨迹绘制

---

## 数据生成时间

根据仿真日志和文件修改时间确定。

---

**文档创建日期**: 2026-04-13

**分析工具**: Python 3.11.14, NumPy 1.26.4

**控制器类型**: Linear Model Predictive Control (LMPC)

**求解器**: IPOPT (Interior Point Optimizer)
