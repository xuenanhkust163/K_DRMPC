"""
入口脚本：在两条赛道上运行所有4种方法的仿真。
生成用于分析的仿真结果。

本脚本是K-DRMPC项目的核心仿真入口，负责：
1. 加载训练好的Deep Koopman模型
2. 在两条赛道（卢赛尔赛道、自定义弯道路）上运行4种MPC方法
3. 执行鲁棒性分析（不同干扰水平sigma）
4. 执行敏感性分析（Wasserstein半径theta、CVaR风险水平epsilon）
5. 保存所有仿真结果供后续分析和可视化使用

4种对比方法：
- LMPC: 线性模型预测控制（基准方法）
- NMPC: 非线性模型预测控制（使用完整非线性模型）
- K-MPC: Koopman模型预测控制（数据驱动线性化）
- K-DRMPC: Koopman分布鲁棒模型预测控制（本文方法）
"""

import os  # 导入操作系统接口模块，用于文件和路径操作
import sys  # 导入系统模块，用于修改Python路径
import json  # 导入JSON模块，用于读取模型参数文件
import argparse  # 导入命令行参数解析模块
import shutil  # 导入文件操作模块，用于清理旧输出目录
import torch  # noqa: F401  # 导入PyTorch深度学习框架（模型推理时需要）
import numpy as np  # noqa: F401  # 导入NumPy库（数值计算需要）

# 将当前目录添加到系统路径，以便导入同目录下的其他模块
# os.path.dirname(os.path.abspath(__file__)) 获取当前脚本所在目录的绝对路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入config模块本身，供运行时覆盖使用
import config as _config_module  # noqa: F401

# 从配置文件导入各种常量和参数
from config import (  # noqa: F401  # THETA_WASSERSTEIN和EPSILON_CVAR在其他地方使用
    MODEL_DIR,  # 模型保存目录路径，包含训练好的模型文件
    RESULTS_DIR,  # 仿真结果保存目录路径
    FIGURES_DIR,  # 图表保存目录路径
    N_DISTURBANCE_SAMPLES,  # 干扰样本数量，用于构建Wasserstein模糊集
    SIGMA_VALUES,  # 噪声标准差测试值列表，用于鲁棒性分析
    THETA_VALUES,  # Wasserstein半径测试值列表，用于敏感性分析
    EPSILON_VALUES,  # CVaR风险水平测试值列表，用于敏感性分析
    THETA_WASSERSTEIN,  # 默认Wasserstein球半径（分布鲁棒参数）
    EPSILON_CVAR,  # 默认CVaR风险水平（分布鲁棒参数）
    MAX_SIM_STEPS  # 最大仿真步数（防止无限循环）
)
# 从模型模块导入Deep Koopman网络和相关工具
from model.koopman_network import DeepKoopmanPaper  # noqa: F401  # Deep Koopman网络架构类
from model.koopman_trainer import load_trained_model  # 加载训练好的Koopman模型函数
from model.projection import load_projection_matrix  # 加载投影矩阵D函数
# 从赛道模块导入两条赛道
from tracks.lusail_track import LusailTrack  # 卢赛尔赛道类（真实赛道，用于主要实验）
from tracks.lusail_short_track import LusailShortTrack  # 短圈Lusail风格赛道（快速验证）
from tracks.custom_track import CustomWindingTrack  # 自定义弯道赛道类（用于验证泛化能力）
from tracks.sprint_oval_track import SprintOvalTrack  # 超短平缓测试赛道（用于快速闭环验证）
from tracks.straight_track import StraightTrack  # 笔直长直道测试赛道（用于基础跟踪验证）
# 从控制器模块导入4种MPC控制器
from controllers.lmpc_controller import LMPCController  # 线性MPC控制器（基于线性化模型）
from controllers.nmpc_controller import NMPCController  # 非线性MPC控制器（基于完整非线性模型）
from controllers.kmpc_controller import KMPCController  # Koopman MPC控制器（基于Koopman线性化）
from controllers.kdrmpc_controller import KDRMPCController  # Koopman分布鲁棒MPC控制器（本文方法，处理不确定性）
# 从干扰模块导入干扰生成器
from disturbance.disturbance_generator import DisturbanceGenerator  # 高斯混合干扰生成器（模拟真实环境不确定性）
# 从仿真模块导入仿真器
from simulation.simulator import Simulator  # 闭环仿真器（执行MPC闭环仿真）
from simulation.metrics import compute_all_metrics  # 计算并打印中线跟随指标

# 分析开关：默认关闭耗时的鲁棒性/敏感性分析，仅运行主对比仿真
ENABLE_ROBUSTNESS_ANALYSIS = False
ENABLE_SENSITIVITY_ANALYSIS = False
# 方法开关：默认仅运行本文方法 K-DRMPC
ENABLE_BASELINE_METHODS = False
ENABLE_OUR_METHOD_KDRMPC = True
# 赛道开关：默认仅运行超短Lusail验证赛道
ENABLE_CUSTOM_TRACK = False
# 主赛道开关：默认改跑超短Lusail
ENABLE_SHORT_LUSAIL_TRACK = True
ENABLE_ORIGINAL_LUSAIL_TRACK = False
# 逐步详细日志默认开关（恢复为默认开启，可被命令行参数覆盖）
ENABLE_DETAILED_STEP_LOG = True
# 逐步详细日志最大步数（None表示打印全程每一步）
DETAILED_STEP_LOG_MAX_STEPS = None
COST_PROFILE_CHOICES = (
    "default",
    "tracking-first",
    "progress-first",
    "mpcc-paper",
    "mpcc-paper-cvar",
    "stabilize-first",
)
TRACK_CHOICES = (
    "sprint-oval",
    "lusail-short",
    "lusail",
    "custom",
    "straight",
    "all",
)


def parse_cli_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Run K-DRMPC simulations with optional debug verbosity and fast mode."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印逐步详细日志（每一步状态/参考/控制/求解信息）。",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="快速验证模式：最大仿真步数固定为300。",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="指定最大仿真步数（默认200）。若提供该参数，则优先于--fast。",
    )
    parser.add_argument(
        "--cost-profile",
        type=str,
        choices=COST_PROFILE_CHOICES,
        default="default",
        help="K系控制器代价模板：default / tracking-first / progress-first / mpcc-paper / mpcc-paper-cvar / stabilize-first。",
    )
    parser.add_argument(
        "--control-every",
        type=int,
        default=1,
        help="控制降频：每N个仿真步求解一次MPC，中间保持上次控制。",
    )
    parser.add_argument(
        "--obstacle-strategy",
        type=str,
        choices=("robust", "non-robust"),
        default="robust",
        help="障碍约束策略：robust=CVaR鲁棒约束，non-robust=确定性距离约束。",
    )
    parser.add_argument(
        "--delta-max",
        type=float,
        default=None,
        metavar="DEG",
        help="最大转向角 [度]，覆盖config.DELTA_MAX（例：--delta-max 3.0）。",
    )
    parser.add_argument(
        "--delta-rate-max",
        type=float,
        default=None,
        metavar="RAD_S",
        help="最大转向角速率 [弧度/秒]，覆盖config.DELTA_RATE_MAX（例：--delta-rate-max 0.1）。",
    )
    parser.add_argument(
        "--speed-scale",
        type=float,
        default=None,
        metavar="SCALE",
        help="全局参考速度缩放因子，覆盖config.REF_SPEED_SCALE（例：--speed-scale 0.3）。",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        metavar="N",
        help="MPC预测时域步数，覆盖config.T_HORIZON（例：--horizon 20）。",
    )
    parser.add_argument(
        "--disturbance",
        action="store_true",
        help="开启在线扰动（覆盖config.ENABLE_DISTURBANCE=True）。",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=None,
        metavar="THETA",
        help="Wasserstein球半径，覆盖config.THETA_WASSERSTEIN（例：--theta 0.05）。",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        metavar="EPS",
        help="CVaR风险水平，覆盖config.EPSILON_CVAR（例：--epsilon 0.1）。",
    )
    parser.add_argument(
        "--obstacles",
        action="store_true",
        help="开启赛道上的障碍物（覆盖config.ENABLE_OBSTACLES=True）。",
    )
    parser.add_argument(
        "--track",
        type=str,
        choices=TRACK_CHOICES,
        default="sprint-oval",
        help="选择运行赛道：sprint-oval / lusail-short / lusail / custom / straight / all。",
    )
    return parser.parse_args()


def _apply_config_overrides(args):
    """将CLI参数覆盖到config模块及所有已导入的依赖模块中。

    由于Python的 `from config import X` 在导入时绑定值，
    覆盖config模块属性后还需同步修补各控制器/赛道模块的命名空间。
    """
    overrides = {}

    if args.delta_max is not None:
        overrides['DELTA_MAX'] = np.deg2rad(args.delta_max)
    if args.delta_rate_max is not None:
        overrides['DELTA_RATE_MAX'] = float(args.delta_rate_max)
    if args.horizon is not None:
        if args.horizon <= 0:
            raise ValueError("--horizon 必须为正整数")
        overrides['T_HORIZON'] = args.horizon
    if args.speed_scale is not None:
        if args.speed_scale <= 0:
            raise ValueError("--speed-scale 必须为正数")
        overrides['REF_SPEED_SCALE'] = float(args.speed_scale)
    if args.disturbance:
        overrides['ENABLE_DISTURBANCE'] = True
    if args.obstacles:
        overrides['ENABLE_OBSTACLES'] = True

    if not overrides:
        return

    # 1. 修补 config 模块本身
    for key, val in overrides.items():
        setattr(_config_module, key, val)

    # 2. 修补所有已导入且含这些名称的子模块
    _PATCH_TARGETS = [
        'controllers.kdrmpc_controller',
        'controllers.kmpc_controller',
        'controllers.lmpc_controller',
        'controllers.mpc_common',
        'tracks.base_track',
        'tracks.sprint_oval_track',
        'tracks.lusail_track',
        'tracks.custom_track',
        'disturbance.disturbance_generator',
    ]
    for mod_name in _PATCH_TARGETS:
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        for key, val in overrides.items():
            if hasattr(mod, key):
                setattr(mod, key, val)

    # 3. 打印覆盖摘要
    print("[Config Override]", ", ".join(
        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
        for k, v in overrides.items()
    ))


def print_centerline_following_report(method_name, result, track):
    """
    打印“是否沿赛道中线行驶”的关键判据。

    判据（工程阈值）:
    - |e_y| P95 <= 2.0m
    - 中线±2m覆盖率 >= 90%
    """
    metrics = compute_all_metrics(result, track)

    mean_abs = metrics.get('tracking_error_mean_abs', float('nan'))
    p95_abs = metrics.get('tracking_error_p95_abs', float('nan'))
    max_abs = metrics.get('tracking_error_max', float('nan'))
    heading_mean_deg = metrics.get('heading_error_mean_abs_deg', float('nan'))
    heading_p95_deg = metrics.get('heading_error_p95_abs_deg', float('nan'))
    heading_max_deg = metrics.get('heading_error_max_abs_deg', float('nan'))
    within_1m = metrics.get('tracking_within_1m_pct', float('nan'))
    within_2m = metrics.get('tracking_within_2m_pct', float('nan'))
    offcenter_max_steps = metrics.get('tracking_offcenter_max_steps', 0)

    follows_centerline = (p95_abs <= 2.0) and (within_2m >= 90.0)

    print(f"\n[Centerline Report] {method_name} on {track.__class__.__name__}")
    print(f"  Mean |e_y| = {mean_abs:.3f} m")
    print(f"  P95  |e_y| = {p95_abs:.3f} m")
    print(f"  Max  |e_y| = {max_abs:.3f} m")
    print(f"  Mean |e_psi| = {heading_mean_deg:.2f} deg")
    print(f"  P95  |e_psi| = {heading_p95_deg:.2f} deg")
    print(f"  Max  |e_psi| = {heading_max_deg:.2f} deg")
    print(f"  Within ±1m = {within_1m:.1f}%")
    print(f"  Within ±2m = {within_2m:.1f}%")
    print(f"  Longest consecutive |e_y|>2m steps = {offcenter_max_steps}")
    print(f"  Verdict: {'YES (follows centerline)' if follows_centerline else 'NO (deviates from centerline)'}")


def load_koopman_components():
    """
    加载训练好的Koopman模型、投影矩阵和归一化参数。

    该函数从磁盘加载所有必要的模型组件，用于K-MPC和K-DRMPC控制器。
    这些组件包括：
    1. Deep Koopman网络（编码器、解码器、线性演化矩阵A、B、C）
    2. 投影矩阵D（将干扰从原始空间映射到Koopman空间）
    3. 归一化参数（用于状态和控制输入的标准化/反标准化）

    返回:
        model: DeepKoopmanPaper对象，训练好的Koopman模型
        D: numpy数组，形状为(N_Z, N_W)的投影矩阵
        norm_params: 字典，包含状态和控制输入的均值和标准差
    """
    # 加载训练好的Deep Koopman模型（包含编码器、解码器和线性演化矩阵）
    model = load_trained_model()

    # 加载投影矩阵D（用于将干扰从原始状态空间映射到Koopman隐空间）
    # D矩阵通过岭回归计算得到，形状为(N_Z, N_W) = (32, 5)
    D = load_projection_matrix()

    # 读取归一化参数文件（用于数据的标准化处理）
    # 归一化参数在训练阶段计算并保存，确保模型输入的数值稳定性
    norm_path = os.path.join(MODEL_DIR, 'norm_params.json')  # 构建归一化参数文件的完整路径
    with open(norm_path, 'r') as f:
        norm_params = json.load(f)  # 从JSON文件加载归一化参数（均值和标准差）

    return model, D, norm_params  # 返回模型、投影矩阵和归一化参数


def run_single_method(method_name, track, controller, dist_gen, max_steps,
                      detailed_step_log=False,
                      detailed_step_log_max_steps=None,
                      control_update_interval=1):
    """
    在单个赛道上运行单个方法的仿真。

    该函数执行一次完整的闭环MPC仿真，从初始状态开始，
    在每个时间步求解MPC优化问题，施加控制输入，
    直到完成赛道或达到最大步数。

    参数:
        method_name: 字符串，方法名称（如'LMPC', 'NMPC', 'K-MPC', 'K-DRMPC'）
        track: 赛道对象，包含赛道几何、参考轨迹、障碍物信息
        controller: 控制器对象，实现MPC优化算法
        dist_gen: 干扰生成器对象，在每个时间步生成扰动
        max_steps: 整数，最大仿真步数（防止无限循环）

    返回:
        result: 仿真结果对象，包含：
            - states: 状态轨迹列表 [[px, py, psi, v, omega], ...]
            - controls: 控制输入列表 [[a, delta], ...]
            - references: 参考轨迹列表
            - solve_times: 每次MPC求解时间列表
            - total_steps: 总仿真步数
            - lap_time: 完成赛道的时间（如果完成）
            - 其他性能指标
    """
    # 创建仿真器实例，将赛道、控制器和干扰生成器组合在一起
    # 仿真器负责执行闭环仿真循环
    simulator = Simulator(track, controller, dist_gen)

    # 运行闭环仿真
    # verbose=True表示打印详细的仿真信息（每步的状态、求解时间等）
    result = simulator.run(
        max_steps=max_steps,
        verbose=True,
        detailed_step_log=detailed_step_log,
        detailed_step_log_max_steps=detailed_step_log_max_steps,
        control_update_interval=control_update_interval,
    )

    # 保存仿真结果到PKL文件（Python pickle格式）
    # 文件名格式："{方法名}_{赛道类名}.pkl"，例如"LMPC_LusailTrack.pkl"
    # 结果保存在RESULTS_DIR目录下，供后续分析和可视化使用
    Simulator.save_result(result, f"{method_name}_{track.__class__.__name__}.pkl")

    # 打印中线跟随情况，明确回答“是否沿赛道中线在开”
    print_centerline_following_report(method_name, result, track)

    return result  # 返回仿真结果对象


def run_all_methods_on_track(track, model, D, norm_params, dist_gen,
                             max_steps=MAX_SIM_STEPS,
                             detailed_step_log=False,
                             detailed_step_log_max_steps=None,
                             cost_profile="default",
                             control_update_interval=1,
                             obstacle_strategy="robust",
                             theta=None,
                             epsilon=None):
    """
    在单个赛道上运行所有4种MPC方法。

    该函数是仿真流程的核心，依次在指定赛道上运行：
    1. LMPC（线性MPC）- 基准方法
    2. NMPC（非线性MPC）- 精确但计算量大
    3. K-MPC（Koopman MPC）- 数据驱动线性化
    4. K-DRMPC（Koopman分布鲁棒MPC）- 本文方法，处理不确定性

    所有方法使用相同的初始条件、干扰序列和赛道，
    以确保公平对比。

    参数:
        track: 赛道对象（LusailTrack或CustomWindingTrack）
        model: 训练好的Deep Koopman模型（用于K-MPC和K-DRMPC）
        D: 投影矩阵，形状(N_Z, N_W)，用于干扰映射
        norm_params: 归一化参数字典，包含均值和标准差
        dist_gen: 干扰生成器对象，生成环境扰动
        max_steps: 整数，最大仿真步数，默认使用配置中的MAX_SIM_STEPS

    返回:
        results: 字典，键为方法名称，值为仿真结果对象
            {'LMPC': result1, 'NMPC': result2, 'K-MPC': result3, 'K-DRMPC': result4}
    """
    results = {}  # 初始化结果字典，用于存储4种方法的仿真结果

    # 获取经验干扰样本，用于分布鲁棒方法（K-DRMPC）
    # 这些样本用于构建Wasserstein模糊集，表征干扰的不确定性分布
    # N_DISTURBANCE_SAMPLES通常设置为100，平衡计算效率和分布表征准确性
    w_empirical = dist_gen.get_empirical_samples(N_DISTURBANCE_SAMPLES)

    if ENABLE_BASELINE_METHODS:
        # ================================================================
        # 1. LMPC（线性MPC）
        # ================================================================
        print(f"\n{'='*60}")
        print(f"在 {track.__class__.__name__} 上运行LMPC")
        print(f"{'='*60}")
        lmpc = LMPCController()
        results['LMPC'] = run_single_method('LMPC', track, lmpc, dist_gen,
                                            max_steps,
                                            detailed_step_log,
                                            detailed_step_log_max_steps,
                                            control_update_interval)

        # ================================================================
        # 2. NMPC（非线性MPC）
        # ================================================================
        print(f"\n{'='*60}")
        print(f"在 {track.__class__.__name__} 上运行NMPC")
        print(f"{'='*60}")
        nmpc = NMPCController()
        results['NMPC'] = run_single_method('NMPC', track, nmpc, dist_gen,
                                            max_steps,
                                            detailed_step_log,
                                            detailed_step_log_max_steps,
                                            control_update_interval)

        # ================================================================
        # 3. K-MPC（Koopman MPC）
        # ================================================================
        print(f"\n{'='*60}")
        print(f"在 {track.__class__.__name__} 上运行K-MPC")
        print(f"{'='*60}")
        kmpc = KMPCController(
            model,
            D,
            norm_params,
            cost_profile=cost_profile,
        )
        results['K-MPC'] = run_single_method('K-MPC', track, kmpc, dist_gen,
                                             max_steps,
                                             detailed_step_log,
                                             detailed_step_log_max_steps,
                                             control_update_interval)
    else:
        print("\n[Skip] LMPC/NMPC/K-MPC 已禁用（ENABLE_BASELINE_METHODS=False）")

    # ================================================================
    # 4. K-DRMPC（Koopman分布鲁棒MPC）
    # ================================================================
    # K-DRMPC是本文提出的方法，在K-MPC基础上加入分布鲁棒性
    # 使用Wasserstein模糊集和CVaR约束处理干扰不确定性
    # 在安全性和鲁棒性方面优于其他方法
    if ENABLE_OUR_METHOD_KDRMPC:
        print(f"\n{'='*60}")
        print(f"在 {track.__class__.__name__} 上运行K-DRMPC")
        print(f"{'='*60}")
        # 创建K-DRMPC控制器，额外传入经验干扰样本用于构建Wasserstein模糊集
        _theta = theta if theta is not None else _config_module.THETA_WASSERSTEIN
        _epsilon = epsilon if epsilon is not None else _config_module.EPSILON_CVAR
        kdrmpc = KDRMPCController(model, D, norm_params,
                                  disturbance_samples=w_empirical,
                                  theta=_theta,
                                  epsilon=_epsilon,
                                  cost_profile=cost_profile,
                                  obstacle_strategy=obstacle_strategy)
        # 运行K-DRMPC仿真并保存结果到"K-DRMPC_{赛道名}.pkl"
        results['K-DRMPC'] = run_single_method('K-DRMPC', track, kdrmpc,
                                               dist_gen, max_steps,
                                               detailed_step_log,
                                               detailed_step_log_max_steps,
                                               control_update_interval)
    else:
        print("[Skip] K-DRMPC 已禁用（ENABLE_OUR_METHOD_KDRMPC=False）")

    return results  # 返回包含4种方法结果的字典


def run_robustness_analysis(track, model, D, norm_params, max_steps=500,
                            cost_profile="default",
                            obstacle_strategy="robust"):
    """
    运行K-DRMPC在不同干扰水平下的鲁棒性分析（论文表11）。

    该函数测试K-DRMPC在不同噪声标准差(sigma)下的性能，
    验证算法对干扰强度的鲁棒性。sigma值越大，干扰越强。

    测试的sigma值在config.py中的SIGMA_VALUES定义，
    通常为[0.01, 0.05, 0.10, 0.15]。

    参数:
        track: 赛道对象，用于仿真的赛道
        model: Deep Koopman模型，训练好的Koopman网络
        D: 投影矩阵，形状(N_Z, N_W)
        norm_params: 归一化参数字典
        max_steps: 整数，最大仿真步数，默认500步

    返回:
        robustness_results: 字典，键为"sigma_X.XX"，值为仿真结果对象
            {'sigma_0.01': result1, 'sigma_0.05': result2, ...}
    """
    robustness_results = {}  # 初始化鲁棒性结果字典

    # 遍历不同的噪声标准差值（在config.py中定义）
    for sigma in SIGMA_VALUES:
        print(f"\n--- 鲁棒性测试: sigma={sigma} ---")  # 显示当前测试参数
        # 创建对应sigma的干扰生成器
        # sigma越大，高斯混合干扰的方差越大，干扰越强
        dist_gen = DisturbanceGenerator(sigma=sigma)
        # 生成经验样本（用于K-DRMPC的Wasserstein模糊集构建）
        w_empirical = dist_gen.get_empirical_samples(N_DISTURBANCE_SAMPLES)

        # 创建K-DRMPC控制器，使用当前sigma的经验样本
        controller = KDRMPCController(model, D, norm_params,
                                       disturbance_samples=w_empirical,
                                       cost_profile=cost_profile,
                                       obstacle_strategy=obstacle_strategy)
        # 创建仿真器
        simulator = Simulator(track, controller, dist_gen)
        # 运行仿真，verbose=False减少输出（因为要测试多个sigma值）
        result = simulator.run(max_steps=max_steps, verbose=False)

        key = f"sigma_{sigma}"  # 构建结果键名，如"sigma_0.01"
        # 保存结果到"robustness_sigma_X.XX.pkl"
        Simulator.save_result(result, f"robustness_{key}.pkl")
        robustness_results[key] = result  # 存储到结果字典

    return robustness_results


def run_sensitivity_theta(track, model, D, norm_params, dist_gen, max_steps=500,
                          cost_profile="default",
                          obstacle_strategy="robust"):
    """
    运行K-DRMPC在不同theta值下的敏感性分析（论文表14）。

    theta（Wasserstein半径）控制分布鲁棒优化中模糊集的大小：
    - theta越大，模糊集越大，考虑更多的不确定性分布
    - theta越大，解越保守（安全性高但性能可能下降）
    - theta=0时退化为普通MPC（不考虑分布鲁棒性）

    测试的theta值在config.py中的THETA_VALUES定义，
    通常为[0.00, 0.02, 0.05, 0.10, 0.20]。

    参数:
        track: 赛道对象
        model: Deep Koopman模型
        D: 投影矩阵
        norm_params: 归一化参数字典
        dist_gen: 干扰生成器（使用固定的干扰分布）
        max_steps: 整数，最大仿真步数

    返回:
        results: 字典，键为"theta_X.XX"，值为仿真结果对象
            {'theta_0.0': result1, 'theta_0.02': result2, ...}
    """
    results = {}  # 初始化结果字典

    # 遍历不同的Wasserstein半径值（在config.py中定义）
    for theta in THETA_VALUES:
        print(f"\n--- 敏感性测试: theta={theta} ---")  # 显示当前测试参数
        # 生成经验样本（使用固定的干扰生成器）
        w_empirical = dist_gen.get_empirical_samples(N_DISTURBANCE_SAMPLES)

        # 创建K-DRMPC控制器，设置特定的theta值
        # theta参数控制Wasserstein模糊集的大小，影响鲁棒性
        controller = KDRMPCController(model, D, norm_params,
                                       disturbance_samples=w_empirical,
                                       theta=theta,
                                       cost_profile=cost_profile,
                                       obstacle_strategy=obstacle_strategy)
        # 创建仿真器
        simulator = Simulator(track, controller, dist_gen)
        # 运行仿真
        result = simulator.run(max_steps=max_steps, verbose=False)

        key = f"theta_{theta}"  # 构建结果键名，如"theta_0.05"
        # 保存结果到"sensitivity_theta_X.XX.pkl"
        Simulator.save_result(result, f"sensitivity_{key}.pkl")
        results[key] = result  # 存储到结果字典

    return results


def run_sensitivity_epsilon(track, model, D, norm_params, dist_gen, max_steps=500,
                            cost_profile="default",
                            obstacle_strategy="robust"):
    """
    运行K-DRMPC在不同epsilon值下的敏感性分析（论文表15）。

    epsilon（CVaR风险水平）控制风险约束的保守程度：
    - epsilon越小，风险约束越严格（只考虑最坏情况）
    - epsilon越大，风险约束越宽松（允许更多风险）
    - epsilon=0.1表示考虑最坏的10%情况

    测试的epsilon值在config.py中的EPSILON_VALUES定义，
    通常为[0.01, 0.05, 0.10, 0.20, 0.30]。

    参数:
        track: 赛道对象
        model: Deep Koopman模型
        D: 投影矩阵
        norm_params: 归一化参数字典
        dist_gen: 干扰生成器
        max_steps: 整数，最大仿真步数

    返回:
        results: 字典，键为"epsilon_X.XX"，值为仿真结果对象
            {'epsilon_0.01': result1, 'epsilon_0.05': result2, ...}
    """
    results = {}  # 初始化结果字典

    # 遍历不同的CVaR风险水平值（在config.py中定义）
    for epsilon in EPSILON_VALUES:
        print(f"\n--- 敏感性测试: epsilon={epsilon} ---")  # 显示当前测试参数
        # 生成经验样本
        w_empirical = dist_gen.get_empirical_samples(N_DISTURBANCE_SAMPLES)

        # 创建K-DRMPC控制器，设置特定的epsilon值
        # epsilon参数控制CVaR风险约束的严格程度
        controller = KDRMPCController(model, D, norm_params,
                                       disturbance_samples=w_empirical,
                                       epsilon=epsilon,
                                       cost_profile=cost_profile,
                                       obstacle_strategy=obstacle_strategy)
        # 创建仿真器
        simulator = Simulator(track, controller, dist_gen)
        # 运行仿真
        result = simulator.run(max_steps=max_steps, verbose=False)

        key = f"epsilon_{epsilon}"  # 构建结果键名，如"epsilon_0.1"
        # 保存结果到"sensitivity_epsilon_X.XX.pkl"
        Simulator.save_result(result, f"sensitivity_{key}.pkl")
        results[key] = result  # 存储到结果字典

    return results


def main():
    """
    主函数：执行完整的仿真活动。

    该函数是整个仿真流程的入口点，按以下顺序执行：
    1. 加载训练好的Deep Koopman模型
    2. 创建干扰生成器（默认sigma=0.05）
    3. 创建两条赛道（卢赛尔赛道、自定义弯道路）
    4. 在卢赛尔赛道上运行4种MPC方法对比实验
    5. 在自定义弯道路上运行4种MPC方法对比实验
    6. 运行鲁棒性分析（不同sigma值，论文表11）
    7. 运行敏感性分析-theta（不同Wasserstein半径，论文表14）
    8. 运行敏感性分析-epsilon（不同CVaR风险水平，论文表15）

    所有仿真结果保存到_output/results/目录，用于后续的
    分析和可视化（生成图表和表格）。
    """

    def cleanup_previous_outputs():
        """启动仿真前清理旧的结果和图表文件。"""
        cleanup_targets = (RESULTS_DIR, FIGURES_DIR)
        print("\n--- 清理旧输出（results/figures） ---")
        for target in cleanup_targets:
            if os.path.exists(target):
                shutil.rmtree(target)
                print(f"已删除目录: {target}")
            os.makedirs(target, exist_ok=True)
            print(f"已重建目录: {target}")

    args = parse_cli_args()
    _apply_config_overrides(args)
    cleanup_previous_outputs()
    if args.control_every <= 0:
        raise ValueError("--control-every 必须为正整数")
    detailed_step_log = ENABLE_DETAILED_STEP_LOG or args.verbose
    if args.steps is not None:
        if args.steps <= 0:
            raise ValueError("--steps 必须为正整数")
        sim_max_steps = args.steps
    elif args.fast:
        sim_max_steps = 300
    else:
        sim_max_steps = 200
    sim_max_steps = min(sim_max_steps, MAX_SIM_STEPS)

    # 解析theta/epsilon（优先CLI，其次config默认值）
    theta_val = args.theta if args.theta is not None else _config_module.THETA_WASSERSTEIN
    epsilon_val = args.epsilon if args.epsilon is not None else _config_module.EPSILON_CVAR

    # 打印标题分隔线
    print("=" * 60)
    print("K-DRMPC仿真实验")
    print("=" * 60)
    print(
        f"配置: detailed_step_log={detailed_step_log}, "
        f"max_steps={sim_max_steps}, cost_profile={args.cost_profile}, "
        f"control_every={args.control_every}, "
        f"obstacle_strategy={args.obstacle_strategy}, "
        f"theta={theta_val}, epsilon={epsilon_val}, track={args.track}"
    )

    # ================================================================
    # 步骤1：加载训练好的模型
    # ================================================================
    print("\n--- 加载训练好的Koopman模型 ---")
    # 加载模型组件：
    # - model: Deep Koopman网络（编码器和解码器）
    # - D: 投影矩阵（干扰映射到Koopman空间）
    # - norm_params: 归一化参数（均值和标准差）
    model, D, norm_params = load_koopman_components()

    # ================================================================
    # 步骤2：创建干扰生成器
    # ================================================================
    # 创建默认干扰生成器，sigma=0.05
    # 这个sigma值用于主要的4种方法对比实验
    dist_gen = DisturbanceGenerator(sigma=0.05)

    # ================================================================
    # 步骤3：创建赛道
    # ================================================================
    print("\n--- 创建赛道 ---")
    run_sprint_oval = args.track in ("sprint-oval", "all")
    run_lusail_short = args.track in ("lusail-short", "all")
    run_original_lusail = args.track in ("lusail", "all")
    run_custom = args.track in ("custom", "all")
    run_straight = args.track in ("straight", "all")

    sprint_oval = SprintOvalTrack() if run_sprint_oval else None
    lusail_short = LusailShortTrack() if run_lusail_short else None
    lusail = LusailTrack() if run_original_lusail else None
    custom = CustomWindingTrack() if run_custom else None
    straight = StraightTrack() if run_straight else None

    # ================================================================
    # 步骤4：在卢赛尔赛道上运行所有4种方法
    # ================================================================
    # 这部分生成论文中的主要对比结果（表9、表10）
    # 对比4种方法：LMPC、NMPC、K-MPC、K-DRMPC
    if sprint_oval is not None:
        print("\n" + "#" * 60)
        print("# 冲刺椭圆短赛道仿真")
        print("#" * 60)
        run_all_methods_on_track(
            sprint_oval, model, D, norm_params, dist_gen,
            max_steps=sim_max_steps,
            detailed_step_log=detailed_step_log,
            detailed_step_log_max_steps=DETAILED_STEP_LOG_MAX_STEPS,
            cost_profile=args.cost_profile,
            control_update_interval=args.control_every,
            obstacle_strategy=args.obstacle_strategy,
            theta=theta_val,
            epsilon=epsilon_val,
        )
    else:
        print("[Skip] 冲刺椭圆短赛道未选择（--track）")

    if lusail_short is not None:
        print("\n" + "#" * 60)
        print("# Lusail Short赛道仿真")
        print("#" * 60)
        run_all_methods_on_track(
            lusail_short, model, D, norm_params, dist_gen,
            max_steps=sim_max_steps,
            detailed_step_log=detailed_step_log,
            detailed_step_log_max_steps=DETAILED_STEP_LOG_MAX_STEPS,
            cost_profile=args.cost_profile,
            control_update_interval=args.control_every,
            obstacle_strategy=args.obstacle_strategy,
            theta=theta_val,
            epsilon=epsilon_val,
        )
    else:
        print("[Skip] Lusail Short赛道未选择（--track）")

    if lusail is not None:
        print("\n" + "#" * 60)
        print("# 原始Lusail赛道仿真")
        print("#" * 60)
        run_all_methods_on_track(
            lusail, model, D, norm_params, dist_gen,
            max_steps=sim_max_steps,
            detailed_step_log=detailed_step_log,
            detailed_step_log_max_steps=DETAILED_STEP_LOG_MAX_STEPS,
            cost_profile=args.cost_profile,
            control_update_interval=args.control_every,
            obstacle_strategy=args.obstacle_strategy,
            theta=theta_val,
            epsilon=epsilon_val,
        )

    # ================================================================
    # 步骤5：在自定义弯道路上运行所有4种方法
    # ================================================================
    # 这部分验证算法在不同赛道上的泛化能力（表13）
    if custom is not None:
        print("\n" + "#" * 60)
        print("# 自定义弯道赛道仿真")
        print("#" * 60)
        # 运行4种方法
        # 结果保存到：LMPC_CustomWindingTrack.pkl, NMPC_CustomWindingTrack.pkl,
        #           K-MPC_CustomWindingTrack.pkl, K-DRMPC_CustomWindingTrack.pkl
        custom_results = run_all_methods_on_track(
            custom, model, D, norm_params, dist_gen,
            max_steps=sim_max_steps,
            detailed_step_log=detailed_step_log,
            detailed_step_log_max_steps=DETAILED_STEP_LOG_MAX_STEPS,
            cost_profile=args.cost_profile,
            control_update_interval=args.control_every,
            obstacle_strategy=args.obstacle_strategy,
            theta=theta_val,
            epsilon=epsilon_val,
        )
    else:
        print("[Skip] 自定义弯道赛道未选择（--track）")

    if straight is not None:
        print("\n" + "#" * 60)
        print("# 笔直赛道仿真")
        print("#" * 60)
        run_all_methods_on_track(
            straight, model, D, norm_params, dist_gen,
            max_steps=sim_max_steps,
            detailed_step_log=detailed_step_log,
            detailed_step_log_max_steps=DETAILED_STEP_LOG_MAX_STEPS,
            cost_profile=args.cost_profile,
            control_update_interval=args.control_every,
            obstacle_strategy=args.obstacle_strategy,
            theta=theta_val,
            epsilon=epsilon_val,
        )
    else:
        print("[Skip] 笔直赛道未选择（--track）")

    # ================================================================
    # 步骤6：可选分析（默认关闭）
    # ================================================================
    if ENABLE_ROBUSTNESS_ANALYSIS:
        analysis_track = sprint_oval or lusail_short or lusail or custom or straight or SprintOvalTrack()
        # 鲁棒性分析（论文表11）
        print("\n" + "#" * 60)
        print("# 鲁棒性分析（表11）")
        print("#" * 60)
        run_robustness_analysis(analysis_track, model, D, norm_params,
                    max_steps=min(500, sim_max_steps),
                    cost_profile=args.cost_profile,
                    obstacle_strategy=args.obstacle_strategy)
    else:
        print("\n[Skip] 鲁棒性分析已禁用（ENABLE_ROBUSTNESS_ANALYSIS=False）")

    if ENABLE_SENSITIVITY_ANALYSIS:
        analysis_track = sprint_oval or lusail_short or lusail or custom or straight or SprintOvalTrack()
        # 敏感性分析 - theta（论文表14）
        print("\n" + "#" * 60)
        print("# 敏感性分析 - THETA（表14）")
        print("#" * 60)
        run_sensitivity_theta(analysis_track, model, D, norm_params, dist_gen,
                              max_steps=min(500, sim_max_steps),
                              cost_profile=args.cost_profile,
                              obstacle_strategy=args.obstacle_strategy)

        # 敏感性分析 - epsilon（论文表15）
        print("\n" + "#" * 60)
        print("# 敏感性分析 - EPSILON（表15）")
        print("#" * 60)
        run_sensitivity_epsilon(analysis_track, model, D, norm_params, dist_gen,
                                max_steps=min(500, sim_max_steps),
                                cost_profile=args.cost_profile,
                                obstacle_strategy=args.obstacle_strategy)
    else:
        print("[Skip] 敏感性分析已禁用（ENABLE_SENSITIVITY_ANALYSIS=False）")

    # ================================================================
    # 仿真完成
    # ================================================================
    print("\n" + "=" * 60)
    print("所有仿真完成。")
    print(f"结果保存到: {RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    # 当直接运行此脚本时（而非作为模块导入），执行主函数
    main()
