"""
K-DRMPC论文实现的中心配置文件。
所有论文中的超参数都在此定义。
"""

import os  # 导入操作系统接口模块，用于路径操作
import numpy as np  # 导入NumPy库，用于数值计算和数组操作

# ============================================================
# 项目路径配置
# ============================================================
# K-DRMPC目录（当前文件所在目录）
EXPERIMENT_ROOT = os.path.dirname(os.path.abspath(__file__))
# 独立项目，K-DRMPC即为根目录
PROJECT_ROOT = EXPERIMENT_ROOT

# 已有数据文件路径（从父项目）
DATA_NPZ_PATH = os.path.join(
    EXPERIMENT_ROOT, "..", "_output", "_data_process",
    "training_data.npz"
)
NORM_JSON_PATH = os.path.join(
    EXPERIMENT_ROOT, "..", "_output", "_data_process",
    "training_data_norm_params.json"
)

# 输出目录路径
OUTPUT_DIR = os.path.join(EXPERIMENT_ROOT, "_output")  # 总输出目录
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")  # 模型保存目录
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")  # 仿真结果保存目录
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")  # 图表保存目录
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")  # 表格保存目录

# 结果导出模式：后续默认采用 simulation + animate，暂不导出静态图
EXPORT_STATIC_FIGURES = False
EXPORT_ANIMATION = True
ANIMATION_FPS = 12

# ============================================================
# 状态和控制维度
# ============================================================
N_X = 5       # 状态维度：[px, py, psi, v, omega] = [位置x, 位置y, 航向角, 速度, 角速度]
N_U = 2       # 控制输入维度：[a, delta] = [加速度, 转向角]
N_Z = 32      # Koopman隐空间维度（提升后的线性系统状态维度）
N_W = 5       # 干扰维度（与状态维度相同）

# 统一状态索引（全仓库使用，避免硬编码）
IDX_PX = 0
IDX_PY = 1
IDX_PSI = 2
IDX_V = 3
IDX_OMEGA = 4

# ============================================================
# Deep Koopman网络架构（论文第3.3节）
# ============================================================
ENCODER_LAYERS = [N_X, 64, 128, 64, N_Z]    # 编码器层结构：5 -> 64 -> 128 -> 64 -> 32
DECODER_LAYERS = [N_Z, 64, 32, N_X]          # 解码器层结构：32 -> 64 -> 32 -> 5
ACTIVATION = "relu"  # 激活函数：ReLU

# ============================================================
# 训练超参数（论文第3.4节）
# ============================================================
BATCH_SIZE = 256  # 批次大小：每次训练使用的样本数
EPOCHS = 500  # 训练轮数：完整遍历数据集的次数
LEARNING_RATE = 1e-3  # 初始学习率：0.001
LR_PATIENCE = 20  # 学习率调度器耐心值：验证损失20轮不改善则降低学习率
LR_FACTOR = 0.5  # 学习率衰减因子：每次降低为原来的50%
EARLY_STOP_PATIENCE = 50  # 早停耐心值：验证损失50轮不改善则停止训练
VAL_SPLIT = 0.1  # 验证集比例：10%的数据用作验证集

# 损失函数权重（论文第3.3.5节）
LAMBDA_RECON = 1.0  # 重构损失权重：保证编解码器能准确重构原始状态
LAMBDA_LINEAR = 1.0  # 线性演化损失权重：保证Koopman空间中的线性演化准确性
LAMBDA_PRED = 0.5  # 多步预测损失权重：提高长期预测能力

# 训练损失的多步预测时域
K_PRED = 10  # 前向预测步数：计算10步的预测误差

# D矩阵的岭回归正则化参数
GAMMA_RIDGE = 1e-4  # 正则化强度：防止过拟合，提高数值稳定性

# 数据子采样（原始Ts=0.01s -> 论文dt=0.1s）
ORIGINAL_TS = 0.01  # 原始数据采样时间间隔：0.01秒
SUBSAMPLE_RATE = 10  # 子采样率：每10个样本取1个，得到有效采样时间0.1秒

# ============================================================
# 车辆参数（论文第2节，表5）
# ============================================================
DT = 0.1                   # 采样时间 [秒]
CONTROL_UPDATE_INTERVAL = 1  # 控制更新间隔：每N个仿真步求解一次MPC，1表示每步求解
L_WHEELBASE = 2.6          # 车辆轴距 [米]
V_MIN = 0.0                # 最小速度 [米/秒]
V_MAX = 4.0               # 最大速度 [米/秒]
A_MIN = 0.0               # 允许轻微制动，给航向回正留出恢复能力 [米/秒^2]
A_MAX = 0.5                # 小幅放开纵向加速度上界，抑制长距离提速发散 [米/秒^2]
DELTA_MAX = np.deg2rad(4.5)  # 最大转向角 [弧度]
DELTA_RATE_MAX = 2.0        # 最大转向速率 [弧度/秒]
D_SAFE = 0.5               # 安全裕度 [米]（避障时的额外安全距离）
TRACK_HALF_WIDTH = 12.0    # 赛道半宽 W/2 [米]（调整为原配置的1/4）
TRACK_BOUNDARY_SLACK_PENALTY = 5.0   # 赛道边界松弛惩罚（显著提高，避免“贴边换进度”）

# 横向加速度限制（用于速度曲线规划）
A_LAT_MAX = 4.0            # 最大横向加速度 [米/秒^2]（限制过弯速度）
REF_SPEED_SCALE = 0.18     # 进一步降低全局参考速度，明显降低速度追求
REF_ACCEL_MAX = 0.4        # 参考速度曲线的最大加速度 [米/秒^2]
REF_DECEL_MAX = 1.2        # 参考速度曲线的最大减速度 [米/秒^2]

# ============================================================
# MPC参数（论文第6节，表5）
# ============================================================
T_HORIZON = 20             # 预测时域：MPC向前预测20步（2秒）

# 参考前瞻时域配置（当前关闭，保持固定时域求解）
ADAPTIVE_REF_HORIZON = False
REF_PREVIEW_DISTANCE_M = 80.0   # 目标前瞻距离 [m]
REF_HORIZON_MIN = 20            # 自适应时域下限 [步]
REF_HORIZON_MAX = 80            # 自适应时域上限 [步]

# 代价函数权重（论文第6.1.2节）
Q_WEIGHTS = np.diag([0.25, 3.5])    # Q矩阵：显著降低v追踪，保留/略增omega稳定
R_WEIGHTS = np.diag([1.5, 3.0])     # R矩阵：[a, delta]控制输入平滑性的权重对角阵

# 轨迹跟踪调优参数（供K-MPC/K-DRMPC统一读取）
Q_PSI_TRACK = 14.0                  # 航向误差权重（显著提高）
Q_PROGRESS_TRACK = 3.0             # 前向进度权重（显著降低）
Q_POS_TRACK = 20.0                  # 位置误差权重（显著提高）
POSITION_TERM_INTERVAL = 1          # 位置项加入频率（1=每步）
R_ABS_A = 2.5                       # 绝对加速度惩罚（降低激进提速）
R_ABS_DELTA = 2.0                   # 绝对转角惩罚
Q_TERMINAL_HEADING = 28.0           # 终端航向误差权重
Q_TERMINAL_POS = 30.0               # 终端位置误差权重

# ============================================================
# 分布鲁棒参数（论文第6.1.3节）
# ============================================================
ENABLE_DISTURBANCE = False    # 全局噪声开关：False 表示关闭在线扰动和经验扰动样本
N_DISTURBANCE_SAMPLES = 100    # 历史干扰样本数量：用于构建Wasserstein模糊集
THETA_WASSERSTEIN = 0.05       # Wasserstein球半径：控制模糊集大小（不确定性程度）
EPSILON_CVAR = 0.1             # CVaR风险水平：0.1表示考虑最坏的10%情况

# ============================================================
# 敏感性分析范围（论文第6.7节）
# ============================================================
THETA_VALUES = [0.00, 0.02, 0.05, 0.10, 0.20]  # Wasserstein半径的测试值列表
EPSILON_VALUES = [0.01, 0.05, 0.10, 0.20, 0.30]  # CVaR风险水平的测试值列表
SIGMA_VALUES = [0.01, 0.05, 0.10, 0.15]  # 噪声标准差的测试值列表

# ============================================================
# 仿真参数
# ============================================================
MAX_SIM_STEPS = 3000       # 每圈最大仿真步数（3000步 × 0.1秒 = 300秒）
IPOPT_MAX_ITER = 500       # IPOPT求解器最大迭代次数
IPOPT_TOL = 1e-6           # IPOPT求解器收敛容差
IPOPT_PRINT_LEVEL = 0      # IPOPT输出级别：0表示抑制输出（不打印详细信息）

# ============================================================
# 障碍物配置
# ============================================================
ENABLE_OBSTACLES = False  # 全局障碍物开关：False 表示仿真与绘图都不使用任何障碍物
OBSTACLE_RADIUS = 3.0      # 默认障碍物半径 [米]
VEHICLE_RADIUS = 2.0       # 车辆近似足迹半径 [米]（用于碰撞检测）

# 预定义的障碍物位置列表（赛道上的具体坐标，避开起点附近）
# SprintOvalTrack: 椭圆形赛道，起点约 (72.3, -99.1)，总长 798m
OBSTACLE_POSITIONS = [
    (100.0, 50.0),    # 赛道右侧上方（进度 ~25%）
    (-50.0, 120.0),    # 赛道上侧（进度 ~50%）
    (-120.0, -50.0),   # 赛道左侧下方（进度 ~75%）
]

# ============================================================
# 绘图配置
# ============================================================
PLOT_TRACK_HALF_WIDTH = 12.0  # 轨迹边界可视化半宽 [米]（与约束半宽一致；仅用于画图，不进入约束）
FIGURE_DPI = 300  # 图像分辨率：300 DPI（高质量打印）
FIGURE_FORMAT = "pdf"  # 图像格式：PDF（矢量图，适合论文）
# 方法颜色映射（用于绘图区分不同方法）
METHOD_COLORS = {
    "LMPC": "#1f77b4",    # 蓝色：线性MPC
    "NMPC": "#ff7f0e",    # 橙色：非线性MPC
    "K-MPC": "#2ca02c",   # 绿色：Koopman MPC
    "K-DRMPC": "#d62728", # 红色：Koopman分布鲁棒MPC（本文方法）
}
# 方法标签映射（用于图例显示）
METHOD_LABELS = {
    "LMPC": "LMPC",                    # 线性MPC
    "NMPC": "NMPC",                    # 非线性MPC
    "K-MPC": "K-MPC",                  # Koopman MPC
    "K-DRMPC": "K-DRMPC (Ours)",      # Koopman分布鲁棒MPC（本文方法）
}
