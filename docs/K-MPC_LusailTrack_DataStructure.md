# K-MPC_LusailTrack.pkl 数据结构详解

## 文件概述

**文件路径**: `Experiment/_output/results/K-MPC_LusailTrack.pkl`

**文件类型**: Python Pickle 序列化文件

**数据对象**: `SimResult` 类实例（定义在 `simulation/simulator.py`）

**用途**: 存储 K-MPC（Koopman Model Predictive Control）控制器在卢赛尔赛道（LusailTrack）上的完整闭环仿真结果。

---

## 对象基本信息

| 属性 | 值 | 说明 |
|------|-----|------|
| **对象类型** | `simulation.simulator.SimResult` | 仿真结果容器类 |
| **method_name** | `"K-MPC"` | 控制器方法名称：Koopman MPC |
| **track_name** | `"LusailTrack"` | 赛道名称：卢赛尔赛道 |
| **lap_completed** | `False` | 未完成完整一圈 |
| **lap_time** | `None` | 无圈速记录（未完成） |
| **total_steps** | `391` | 总仿真步数（39.0秒） |

---

## 数据结构详解

### 1. states（状态轨迹）

**数据类型**: `list` of `numpy.ndarray`

**尺寸**: 392 个状态向量，每个形状为 `(5,)`

**转换为数组后**: `(392, 5)` 的二维数组

**含义**: 记录仿真过程中每个时刻的车辆状态，包括初始状态（T+1个时间点）

#### 状态向量组成 `[px, py, v, psi, omega]`

| 索引 | 变量名 | 物理含义 | 单位 | 最小值 | 最大值 | 平均值 |
|------|--------|----------|------|--------|--------|--------|
| 0 | **px** | 位置X坐标 | 米 (m) | -406.70 | 65.32 | -119.93 |
| 1 | **py** | 位置Y坐标 | 米 (m) | -20.25 | 983.31 | 340.82 |
| 2 | **v** | 纵向速度 | 米/秒 (m/s) | 16.90 | 40.00 | 37.71 |
| 3 | **psi** | 航向角 | 弧度 (rad) | -0.87 | 33.55 | 28.32 |
| 4 | **omega** | 横摆角速度 | 弧度/秒 (rad/s) | -0.02 | 4.94 | 0.88 |

**示例数据**（初始状态）:
```python
states[0] = [0.0, 0.0, 16.896, -0.871, 0.237]
# 位置: (0, 0)m, 速度: 16.9m/s, 航向角: -0.87rad (-49.9°), 角速度: 0.24rad/s
```

**关键观察**:
- 车辆从原点 (0, 0) 开始
- 速度范围 16.9-40.0 m/s（约 61-144 km/h），最终达到最大速度 40 m/s
- 航向角持续增加（0 → 33.55 rad ≈ 1922°），表示车辆在赛道上持续转弯
- 横摆角速度最大 4.94 rad/s，表明有急转弯操作

---

### 2. controls（控制输入轨迹）

**数据类型**: `list` of `numpy.ndarray`

**尺寸**: 391 个控制向量，每个形状为 `(2,)`

**转换为数组后**: `(391, 2)` 的二维数组

**含义**: 记录每个控制时刻施加的控制输入（T个时间步，比状态少1个）

#### 控制向量组成 `[a, delta]`

| 索引 | 变量名 | 物理含义 | 单位 | 最小值 | 最大值 | 平均值 | RMS |
|------|--------|----------|------|--------|--------|--------|-----|
| 0 | **a** | 加速度 | 米/秒² (m/s²) | 3.00 | 3.00 | 3.00 | 3.00 |
| 1 | **delta** | 前轮转向角 | 弧度 (rad) | -0.0015 | 0.437 | 0.072 | 0.139 |

**示例数据**（第一个控制输入）:
```python
controls[0] = [3.0, 0.05]
# 加速度: 3.0m/s²（最大加速）, 转向角: 0.05rad (2.9°)
```

**约束限制**（来自 config.py）:
- 加速度: `A_MIN=-5.0` 至 `A_MAX=3.0` m/s²
- 转向角: `-DELTA_MAX=-π/4` 至 `DELTA_MAX=π/4` rad（-45° 至 45°）

**关键观察**:
- ⚠️ **加速度始终为 3.0 m/s²**（最大值），说明控制器一直在全力加速
- 转向角范围 -0.0015 至 0.437 rad（-0.09° 至 25.03°）
- 转向角 RMS = 0.139 rad，说明有一定转向操作

---

### 3. solve_times（求解时间）

**数据类型**: `list` of `float`

**尺寸**: 391 个时间值

**单位**: 毫秒 (ms)

**含义**: 每个时间步MPC优化问题的求解耗时

| 统计量 | 值 | 说明 |
|--------|-----|------|
| **平均值** | 74.84 ms | 平均求解时间 |
| **标准差** | 237.99 ms | 波动较大 |
| **最小值** | 1.47 ms | 最快求解 |
| **最大值** | 1116.96 ms | 最慢求解（1.1秒） |
| **中位数** | 3.69 ms | 典型求解时间 |

**关键观察**:
- 中位数 3.69ms 远小于平均值 74.84ms，说明存在少量极慢的求解
- 最大值 1117ms 可能是遇到了复杂的优化情况
- 大部分求解时间很短（<10ms），满足实时性要求（<100ms）

---

### 4. solve_statuses（求解状态）

**数据类型**: `list` of `str`

**尺寸**: 391 个状态字符串

**含义**: IPOPT求解器的返回状态，表明优化是否成功

**常见状态值**:
- `"optimal"`: 找到最优解
- `"infeasible"`: 问题不可行
- `"error: ..."`: 求解出错

**用途**: 用于诊断MPC求解器的可靠性和鲁棒性

---

### 5. ref_states（参考状态轨迹）

**数据类型**: `list` of `numpy.ndarray`

**尺寸**: 391 个参考状态，每个形状为 `(5,)` 或 `(T_HORIZON, 5)`

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

**尺寸**: 391 个时间点

**单位**: 秒 (s)

**含义**: 每个控制时刻的仿真时间

**时间序列**: `[0.0, 0.1, 0.2, 0.3, ..., 39.0]`

**采样时间**: DT = 0.1 秒（10Hz控制频率）

**总时长**: 39.0 秒

---

## 数据关系图

```
时间步:     0        1        2       ...    390
         ┌────────┬────────┬────────┬───────┬────────┐
states   │ state0 │ state1 │ state2 │ ...   │ state391 │  (392个)
         └────────┴────────┴────────┴───────┴────────┘
                    ↓        ↓               ↓
         ┌────────┬────────┬────────┬───────┬────────┐
controls │ ctrl0  │ ctrl1  │ ctrl2  │ ...   │ ctrl390│  (391个)
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
with open('K-MPC_LusailTrack.pkl', 'rb') as f:
    result = pickle.load(f)

# 转换为NumPy数组
states = np.array(result.states)        # (392, 5)
controls = np.array(result.controls)    # (391, 2)
solve_times = np.array(result.solve_times)  # (391,)
```

### 2. 提取轨迹

```python
# 位置轨迹
positions = states[:, :2]  # (392, 2) [px, py]

# 速度曲线
velocities = states[:, 2]  # (392,)

# 控制输入
accelerations = controls[:, 0]  # (391,)
steering_angles = controls[:, 1]  # (391,)
```

### 3. 性能分析

```python
# 跟踪误差计算
tracking_error = np.sqrt(
    (states[:-1, 0] - ref_states[:, 0])**2 + 
    (states[:-1, 1] - ref_states[:, 1])**2
)

# 控制努力度
control_effort = np.sqrt(np.mean(controls**2, axis=0))

# 求解时间统计
avg_solve_time = np.mean(solve_times)
max_solve_time = np.max(solve_times)
```

### 4. 可视化

```python
import matplotlib.pyplot as plt

# 绘制轨迹
plt.figure()
plt.plot(states[:, 0], states[:, 1], 'b-', label='Actual Trajectory')
plt.plot(ref_states[:, 0], ref_states[:, 1], 'r--', label='Reference')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.legend()
plt.title('K-MPC on Lusail Track')
plt.axis('equal')
plt.show()

# 绘制速度曲线
plt.figure()
plt.plot(timestamps, states[:, 2])
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Velocity Profile')
plt.show()
```

---

## 与其他方法的比较

在完整的仿真实验中，会生成以下结果文件：

| 文件名 | 控制器 | 用途 |
|--------|--------|------|
| `LMPC_LusailTrack.pkl` | 线性MPC | 基准对比 |
| `NMPC_LusailTrack.pkl` | 非线性MPC | 基准对比 |
| `K-MPC_LusailTrack.pkl` | Koopman MPC | 本文方法 |
| `K-DRMPC_LusailTrack.pkl` | 分布鲁棒MPC | 改进方法 |

通过比较这些文件，可以评估不同控制器的性能差异。

---

## 注意事项

1. **状态与控制的数量差异**: states 有 T+1 个（包含初始状态），controls 有 T 个
2. **单位统一**: 所有物理量使用国际单位制（SI）
3. **角度表示**: 航向角使用弧度制，可能超过 2π（持续累加）
4. **未完成圈**: 该仿真未完成完整一圈（lap_completed=False），可能因发散或其他原因终止
5. **Pickle版本**: 使用Python pickle序列化，需相同Python版本和类定义才能加载

---

## 相关文件

- **类定义**: `Experiment/simulation/simulator.py` - `SimResult` 类
- **配置文件**: `Experiment/config.py` - 参数配置
- **控制器**: `Experiment/controllers/kmpc_controller.py` - K-MPC控制器实现
- **可视化**: `Experiment/visualization/plot_trajectories.py` - 轨迹绘制

---

## 数据生成时间

根据文件修改时间和仿真日志确定。

---

**文档创建日期**: 2026-04-13

**分析工具**: Python 3.11.14, NumPy 1.26.4
