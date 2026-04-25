"""
基于Koopman的分布鲁棒MPC (K-DRMPC)控制器。
这是论文的主要贡献（第5节）。

在K-MPC基础上扩展：
- 使用Wasserstein模糊集建模干扰不确定性
- 基于CVaR的安全约束（公式4.10-4.11）
- 通过拉格朗日对偶实现可处理的凸重构

核心创新：
1. 分布鲁棒性：不假设干扰的具体分布，而是考虑一组可能的分布
2. Wasserstein模糊集：以经验分布为中心的概率分布集合
3. CVaR约束：条件风险价值，保证在最坏情况下的安全性
4. 凸重构：将无穷维优化问题转化为有限维凸优化问题
"""

import numpy as np  # 导入NumPy库，用于高效的数值计算和多维数组操作
import casadi as ca  # 导入CasADi库，用于符号计算和非线性优化（IPOPT求解器）
import os  # 导入操作系统接口模块，用于文件和路径操作
import sys  # 导入系统模块，用于修改Python路径
import time  # 导入时间模块，用于计算求解耗时

# 将父目录添加到系统路径，确保可以导入同级别的模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 从配置文件导入MPC和分布鲁棒优化相关参数
from config import (
    N_X,  # 物理状态维度，默认5 [px, py, psi, v, omega]
    N_U,  # 控制输入维度，默认2 [加速度a, 转向角delta]
    N_Z,  # Koopman空间维度，默认8
    N_W,  # 干扰维度，用于采样历史扰动
    T_HORIZON,  # 预测时域长度，默认20步
    DT,  # 时间步长，默认0.1秒
    Q_WEIGHTS,  # 状态跟踪权重向量
    R_WEIGHTS,  # 控制输入权重向量
    V_MIN,  # 最小速度 [米/秒]
    V_MAX,  # 最大速度 [米/秒]
    A_MIN,  # 最小加速度 [米/秒^2]
    A_MAX,  # 最大加速度 [米/秒^2]
    DELTA_MAX,  # 最大转向角 [弧度]
    DELTA_RATE_MAX,  # 最大转向角速率 [弧度/秒]
    D_SAFE,  # 安全距离余量 [米]
    VEHICLE_RADIUS,  # 车辆半径 [米]
    N_DISTURBANCE_SAMPLES,  # 干扰样本数量，默认100
    THETA_WASSERSTEIN,  # Wasserstein球半径，默认0.1
    EPSILON_CVAR,  # CVaR风险水平，默认0.05（95%置信度）
    IPOPT_MAX_ITER,  # IPOPT求解器最大迭代次数
    IPOPT_PRINT_LEVEL,  # IPOPT求解器打印级别
    IDX_V,
    IDX_OMEGA
)

# 从MPC公共模块导入编解码器转换函数
from controllers.mpc_common import (
    pytorch_to_casadi_encoder,  # PyTorch编码器转CasADi函数
    pytorch_to_casadi_decoder  # PyTorch解码器转CasADi函数
)

# 障碍物接近阈值：只有车辆在这个距离内才添加障碍物约束
OBSTACLE_PROXIMITY = 200.0  # 单位：米

# 优化中使用的最大干扰样本数（为了保证可处理性）
# 如果样本太多，会使优化问题过于复杂，因此进行子采样
MAX_OPT_SAMPLES = 20

# 障碍物松弛变量的惩罚权重
# 较大的值会强制满足障碍物约束，但可能导致问题不可解
OBSTACLE_SLACK_PENALTY = 1000.0


class KDRMPCController:
    """
    带有CVaR约束的分布鲁棒Koopman MPC控制器。

    核心特性：
    1. 线性Koopman动力学: z_{t+1} = A*z + B*u
       - A: 状态转移矩阵（学习得到）
       - B: 控制矩阵（学习得到）

    2. 以经验干扰分布为中心的Wasserstein模糊集
       - 模糊集: P ∈ {Q : W(P, P_N) <= theta}
       - W: Wasserstein距离
       - P_N: 经验分布（从N个样本构建）
       - theta: 模糊集半径（控制鲁棒性程度）

    3. CVaR安全约束重构（定理，第5.5节）
       - CVaR_epsilon[l(z, w)] <= 0
       - 将概率约束转化为确定性约束
       - 引入辅助变量Lambda和S

    优化问题形式（论文公式20）：
        min_u Σ_t ||D*z_t - y_ref_t||_Q^2 + ||u_t||_R^2
        s.t.  z_{t+1} = A*z_t + B*u_t  (标称动力学)
              u_min <= u_t <= u_max
              CVaR约束: lambda*theta + (1/(epsilon*N))*Σs_i <= slack
                        s_i >= l_nom + lambda*||w_i||
    """

    def __init__(self, koopman_model, D_matrix, norm_params,
                 disturbance_samples=None, theta=THETA_WASSERSTEIN,
                 epsilon=EPSILON_CVAR):
        """
        初始化K-DRMPC控制器。

        参数:
            koopman_model: 训练好的DeepKoopmanPaper模型
                          包含编码器、解码器和Koopman矩阵A, B
            D_matrix: numpy数组，形状为(2, n_z)
                     投影矩阵，用于从Koopman状态提取[v, omega]
            norm_params: 字典，包含归一化参数
                        {'px_mean', 'px_std', 'py_mean', 'py_std'}
            disturbance_samples: numpy数组，形状为(N, 5)
                                用于构建模糊集的干扰样本
                                如果为None，则使用随机生成的样本
            theta: 浮点数，Wasserstein球半径
                  默认使用config.py中的THETA_WASSERSTEIN（0.1）
                  theta越大，考虑的分布范围越广，控制器越保守
            epsilon: 浮点数，CVaR风险水平
                    默认使用config.py中的EPSILON_CVAR（0.05）
                    epsilon=0.05表示95%置信度下的风险约束
        """
        # 控制器名称，用于日志和结果标识
        self.name = "K-DRMPC"

        # 初始化上一次控制输入为零向量
        self.u_prev = np.zeros(N_U)

        # 热启动信息，用于加速下一次求解
        self._warm_start = None

        # ================================================================
        # Koopman模型组件
        # ================================================================
        # 获取Koopman线性动力学矩阵
        # A: 状态转移矩阵 (n_z x n_z)
        # B: 控制矩阵 (n_z x n_u)
        self.A, self.B = koopman_model.get_matrices()

        # 保存投影矩阵D (2 x n_z)
        self.D = D_matrix

        # 保存归一化参数，用于状态编码和解码
        self.norm_params = norm_params

        # ================================================================
        # 分布鲁棒优化参数
        # ================================================================
        # Wasserstein球半径
        self.theta = theta

        # CVaR风险水平
        self.epsilon = epsilon

        # ================================================================
        # 干扰样本处理（子采样以保证可处理性）
        # ================================================================
        if disturbance_samples is not None:
            # 如果样本数量超过最大值，进行随机子采样
            if len(disturbance_samples) > MAX_OPT_SAMPLES:
                # 使用固定种子保证可复现性
                rng = np.random.RandomState(42)
                # 随机选择MAX_OPT_SAMPLES个样本（不重复）
                idx = rng.choice(
                    len(disturbance_samples),
                    MAX_OPT_SAMPLES,
                    replace=False
                )
                self.w_samples = disturbance_samples[idx]
            else:
                # 样本数量合适，直接使用
                self.w_samples = disturbance_samples
        else:
            # 如果没有提供样本，生成随机样本作为默认值
            # 使用零均值、标准差0.05的正态分布
            self.w_samples = np.random.randn(MAX_OPT_SAMPLES, N_W) * 0.05

        # 保存实际使用的样本数量
        self.N_samples = len(self.w_samples)

        # ================================================================
        # 预计算CVaR约束所需的范数（提高优化效率）
        # ================================================================
        # 预计算||w_i||：干扰本身的范数
        # 这用于Wasserstein距离的计算
        self.w_norms = np.array([
            np.linalg.norm(self.w_samples[i])
            for i in range(self.N_samples)
        ])

        # 将矩阵存储为CasADi DM（密集矩阵）类型
        # 这样可以在CasADi符号计算中直接使用
        self._A_ca = ca.DM(self.A)  # CasADi格式的A矩阵
        self._B_ca = ca.DM(self.B)  # CasADi格式的B矩阵
        self._D_ca = ca.DM(self.D)  # CasADi格式的D矩阵

        # ================================================================
        # 构建CasADi函数（编解码器）
        # ================================================================
        # 获取神经网络权重
        weights = koopman_model.get_network_weights()

        # 将PyTorch编码器转换为CasADi函数
        self._ca_encode = pytorch_to_casadi_encoder(weights)

        # 将PyTorch解码器转换为CasADi函数
        # 用于障碍物避免和位置跟踪
        self._ca_decode = pytorch_to_casadi_decoder(weights)

    def _encode_state(self, x_physical):
        """将物理状态编码为Koopman潜在状态（带归一化）。"""
        # 复制物理状态，避免修改原始数据
        x_norm = x_physical.copy()

        # 对px（索引0）进行归一化
        x_norm[0] = (
            x_norm[0] - self.norm_params['px_mean']
        ) / self.norm_params['px_std']

        # 对py（索引1）进行归一化
        x_norm[1] = (
            x_norm[1] - self.norm_params['py_mean']
        ) / self.norm_params['py_std']

        # 将NumPy数组转换为CasADi列向量
        x_ca = ca.DM(x_norm.reshape(-1, 1))

        # 使用CasADi编码器计算潜在状态
        z_ca = self._ca_encode(x_ca)

        # 转换回NumPy数组并展平
        return np.array(z_ca).flatten()

    def solve(self, x_current, ref_trajectory, obstacles, u_prev=None):
        """
        求解K-DRMPC优化问题（论文公式20）。

        优化包含：
        1. 通过D投影在潜在空间中的跟踪代价
        2. 控制平滑性代价
        3. 输入/状态约束
        4. 分布鲁棒的CVaR障碍物避免约束

        参数:
            x_current: numpy数组，形状为(5,)
                      当前物理状态 [px, py, psi, v, omega]
            ref_trajectory: numpy数组，形状为(T, 5)
                          参考轨迹（物理坐标）
            obstacles: 列表，每个元素为(ox, oy, radius)
                      障碍物位置和半径
            u_prev: numpy数组，形状为(2,)
                   上一次的控制输入 [a, delta]

        返回:
            u_opt: numpy数组，形状为(2,)
                  最优控制输入 [加速度a, 转向角delta]
            solve_info: 字典，包含求解信息和DR参数
        """
        if u_prev is not None:
            self.u_prev = u_prev

        # 获取预测时域长度（默认20步）
        T = T_HORIZON

        # 记录求解开始时间
        t_start = time.time()

        # 步骤1：编码当前状态到Koopman空间
        z0 = self._encode_state(x_current)

        # 步骤2：构建参考轨迹（在[v, omega]空间）
        y_ref = np.zeros((T, 2))
        for t in range(T):
            ref_t = ref_trajectory[min(t, len(ref_trajectory) - 1)]
            y_ref[t] = [ref_t[IDX_V], ref_t[IDX_OMEGA]]  # [v, omega]

        # 步骤3：过滤附近的障碍物
        px, py = x_current[0], x_current[1]
        nearby_obstacles = []
        for obs in obstacles:
            ox, oy, r = obs
            # 计算车辆与障碍物的距离
            dist = np.sqrt((px - ox)**2 + (py - oy)**2)
            # 只关注在接近阈值内的障碍物（200米）
            if dist < OBSTACLE_PROXIMITY:
                nearby_obstacles.append(obs)

        # 障碍物数量和检查步骤
        n_obs = len(nearby_obstacles)
        # 检查步骤：从第4步开始，每隔8步检查一次
        # 比K-MPC更稀疏，因为CVaR约束计算成本更高
        check_steps = list(range(4, T + 1, 8))
        n_check = len(check_steps)

        opti = ca.Opti()

        # Decision variables: controls
        U = opti.variable(N_U, T)

        # Propagate latent dynamics (nominal, no disturbance in prediction)
        Z = [ca.DM(z0)]
        for t in range(T):
            z_next = ca.mtimes(self._A_ca, Z[-1]) + \
                     ca.mtimes(self._B_ca, U[:, t])
            Z.append(z_next)

        # === Tracking Cost ===
        cost = 0
        Q = ca.DM(Q_WEIGHTS)
        R = ca.DM(R_WEIGHTS)

        # Position tracking weight (decoded position vs reference)
        Q_pos = 0.5

        # Normalize reference positions for decoder comparison
        ref_px_norm = np.array([(ref_trajectory[min(t, len(ref_trajectory)-1), 0]
                                 - self.norm_params['px_mean']) / self.norm_params['px_std']
                                for t in range(T)])
        ref_py_norm = np.array([(ref_trajectory[min(t, len(ref_trajectory)-1), 1]
                                 - self.norm_params['py_mean']) / self.norm_params['py_std']
                                for t in range(T)])

        for t in range(T):
            y_t = ca.mtimes(self._D_ca, Z[t])
            y_ref_t = ca.DM(y_ref[t])
            cost += ca.mtimes([(y_t - y_ref_t).T, Q, (y_t - y_ref_t)])

            # Position tracking via decoder (every 4th step)
            if t % 4 == 0:
                x_dec = self._ca_decode(Z[t])
                pos_err_x = x_dec[0] - ref_px_norm[t]
                pos_err_y = x_dec[1] - ref_py_norm[t]
                cost += Q_pos * (pos_err_x**2 + pos_err_y**2)

            if t == 0:
                du = U[:, t] - ca.DM(self.u_prev)
            else:
                du = U[:, t] - U[:, t - 1]
            cost += ca.mtimes([du.T, R, du])

        # === Input Constraints ===
        for t in range(T):
            opti.subject_to(opti.bounded(A_MIN, U[0, t], A_MAX))
            opti.subject_to(opti.bounded(-DELTA_MAX, U[1, t], DELTA_MAX))
            if t == 0:
                opti.subject_to(opti.bounded(
                    -DELTA_RATE_MAX * DT,
                    U[1, t] - self.u_prev[1],
                    DELTA_RATE_MAX * DT))
            else:
                opti.subject_to(opti.bounded(
                    -DELTA_RATE_MAX * DT,
                    U[1, t] - U[1, t - 1],
                    DELTA_RATE_MAX * DT))

        # === Distributionally Robust CVaR Constraints (Eq. 4.10-4.11) ===
        if n_obs > 0 and n_check > 0:
            # CVaR auxiliary variables
            Lambda = {}
            S_vars = {}
            for j in range(n_obs):
                for tc_idx, tc in enumerate(check_steps):
                    key = (j, tc)
                    Lambda[key] = opti.variable()
                    opti.subject_to(Lambda[key] >= 0)
                    S_vars[key] = opti.variable(self.N_samples)
                    opti.subject_to(S_vars[key] >= 0)

            # Add obstacle slack for robustness
            n_obs_slack = n_obs * n_check
            obs_slack = opti.variable(n_obs_slack)
            opti.subject_to(obs_slack >= 0)
            cost += OBSTACLE_SLACK_PENALTY * ca.dot(obs_slack, obs_slack)

            obs_slack_idx = 0
            for j, obs in enumerate(nearby_obstacles):
                ox, oy, r = obs
                d_min = r + VEHICLE_RADIUS + D_SAFE

                ox_norm = (ox - self.norm_params['px_mean']) / self.norm_params['px_std']
                oy_norm = (oy - self.norm_params['py_mean']) / self.norm_params['py_std']

                for tc_idx, tc in enumerate(check_steps):
                    key = (j, tc)
                    lam = Lambda[key]
                    s = S_vars[key]

                    # Decode nominal predicted state at time tc
                    x_dec = self._ca_decode(Z[tc])
                    px_pred = x_dec[0]
                    py_pred = x_dec[1]

                    # Nominal safety margin (in physical space)
                    dx_phys = (px_pred - ox_norm) * self.norm_params['px_std']
                    dy_phys = (py_pred - oy_norm) * self.norm_params['py_std']
                    dist_physical = ca.sqrt(dx_phys**2 + dy_phys**2 + 1e-6)
                    l_nom = d_min - dist_physical  # l <= 0 is safe

                    # CVaR constraint for each disturbance sample
                    for i in range(self.N_samples):
                        Cw_norm_i = float(self.Cw_norms[i])
                        w_norm_i = float(self.w_norms[i])
                        opti.subject_to(
                            s[i] >= l_nom + Cw_norm_i + lam * w_norm_i
                        )

                    # CVaR aggregate constraint (softened with slack)
                    sum_s = ca.sum1(s)
                    opti.subject_to(
                        lam * self.theta +
                        (1.0 / (self.epsilon * self.N_samples)) * sum_s
                        <= obs_slack[obs_slack_idx]
                    )
                    obs_slack_idx += 1

        opti.minimize(cost)

        # === Solver ===
        opts = {
            'ipopt.max_iter': IPOPT_MAX_ITER,
            'ipopt.print_level': IPOPT_PRINT_LEVEL,
            'print_time': False,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.tol': 1e-4,
            'ipopt.acceptable_tol': 1e-3,
            'ipopt.acceptable_iter': 5,
        }
        opti.solver('ipopt', opts)

        # Warm start
        if self._warm_start is not None:
            try:
                opti.set_initial(U, self._warm_start['U'])
            except Exception:
                pass

        # Solve
        try:
            sol = opti.solve()
            u_opt = np.array(sol.value(U[:, 0])).flatten()
            status = "optimal"

            # Store warm start
            self._warm_start = {'U': sol.value(U)}

        except Exception as e:
            # Try suboptimal solution from debug
            try:
                u_opt = np.array(opti.debug.value(U[:, 0])).flatten()
                u_opt[0] = np.clip(u_opt[0], A_MIN, A_MAX)
                u_opt[1] = np.clip(u_opt[1], -DELTA_MAX, DELTA_MAX)
                status = "suboptimal"
            except Exception:
                u_opt = self.u_prev.copy()
                status = f"failed: {str(e)[:50]}"

        solve_time = time.time() - t_start
        self.u_prev = u_opt.copy()

        return u_opt, {
            'solve_time': solve_time,
            'status': status,
            'method': self.name,
            'theta': self.theta,
            'epsilon': self.epsilon,
        }

    def update_disturbance_samples(self, new_samples):
        """Update the disturbance sample set (e.g., from online data)."""
        if len(new_samples) > MAX_OPT_SAMPLES:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(new_samples), MAX_OPT_SAMPLES, replace=False)
            self.w_samples = new_samples[idx]
        else:
            self.w_samples = new_samples
        self.N_samples = len(self.w_samples)
        self.Cw_norms = np.array([
            np.linalg.norm(self.C_mat @ self.w_samples[i])
            for i in range(self.N_samples)
        ])
        self.w_norms = np.array([
            np.linalg.norm(self.w_samples[i])
            for i in range(self.N_samples)
        ])

    def reset(self):
        self.u_prev = np.zeros(N_U)
        self._warm_start = None
