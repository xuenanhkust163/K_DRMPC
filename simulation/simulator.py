"""
闭环仿真引擎，用于在赛道上运行MPC控制器。
"""

import numpy as np  # 导入NumPy库，用于数值计算
import time  # 导入时间模块，用于计时
import os  # 导入操作系统接口模块
import sys  # 导入系统模块，用于路径操作
import pickle  # 导入pickle模块，用于对象的序列化和反序列化
import subprocess
from collections import Counter

# 将父目录添加到系统路径，以便导入项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 从配置文件导入常量和参数
from config import (
    DT, MAX_SIM_STEPS, V_MIN, V_MAX, A_MIN, A_MAX,
    DELTA_MAX, RESULTS_DIR, FIGURES_DIR,
    IDX_PX, IDX_PY, IDX_PSI, IDX_V, IDX_OMEGA,
    CONTROL_UPDATE_INTERVAL,
    TRACK_HALF_WIDTH, VEHICLE_RADIUS,
    EXPORT_STATIC_FIGURES, EXPORT_ANIMATION, ANIMATION_FPS,
)
# 从自行车模型模块导入离散时间步长函数
from vehicle.bicycle_model import discrete_step


class SimResult:
    """仿真结果容器类，用于存储仿真过程中产生的各种数据。"""

    def __init__(self, method_name, track_name):
        """
        初始化仿真结果对象。
        
        Args:
            method_name: 控制器/方法名称
            track_name: 赛道名称
        """
        self.method_name = method_name  # 存储控制器或方法的名称
        self.track_name = track_name  # 存储赛道的名称
        self.states = []           # 状态列表，每个元素是形状为(5,)的状态向量
        self.controls = []         # 控制输入列表，每个元素是形状为(2,)的控制向量
        self.solve_times = []      # MPC求解时间列表
        self.solve_statuses = []   # 求解器状态/返回值列表
        self.solve_debug = []      # 每步求解透视日志（cost分项/活跃约束等）
        self.ref_states = []       # 参考状态列表
        self.timestamps = []       # 仿真时间戳列表
        self.lap_completed = False  # 标志位，表示是否完成了一圈
        self.lap_time = None       # 完成一圈所需的时间
        self.total_steps = 0       # 总仿真步数
        self.crashed = False       # 是否撞到赛道边界而提前终止
        self.crash_step = None     # 撞车时的步数
        self.crash_time = None     # 撞车时刻 [s]
        self.crash_reason = None   # 撞车原因描述

    def to_arrays(self):
        """将列表转换为NumPy数组格式，便于后续处理和计算。"""
        return {
            'states': np.array(self.states),  # 将所有状态转换为二维数组
            'controls': np.array(self.controls),  # 将所有控制输入转换为二维数组
            'solve_times': np.array(self.solve_times),  # 将求解时间转换为一维数组
            'timestamps': np.array(self.timestamps),  # 将时间戳转换为一维数组
            'solve_debug': list(self.solve_debug),
            'crashed': self.crashed,
            'crash_step': self.crash_step,
            'crash_time': self.crash_time,
            'crash_reason': self.crash_reason,
        }


class Simulator:
    """
    在赛车赛道上对MPC控制器进行闭环仿真的类。
    该类负责整合赛道、控制器和干扰生成器，执行完整的闭环仿真流程。
    """

    def __init__(self, track, controller, disturbance_gen=None):
        """
        初始化仿真器。
        
        Args:
            track: BaseTrack实例，表示赛道对象
            controller: 控制器对象，需要具有solve(x, ref, obstacles)方法
            disturbance_gen: DisturbanceGenerator实例（可选），用于生成干扰信号
        """
        self.track = track  # 存储赛道对象
        self.controller = controller  # 存储控制器对象
        self.disturbance_gen = disturbance_gen  # 存储干扰生成器（可能为None）

    def run(self, x0=None, max_steps=MAX_SIM_STEPS, lap_fraction=0.95,
            verbose=True, detailed_step_log=False,
            detailed_step_log_max_steps=None,
            control_update_interval=CONTROL_UPDATE_INTERVAL):
        """
        运行闭环仿真。

        Args:
            x0: 形状为(5,)的初始状态向量。如果为None，则从赛道起点开始。
            max_steps: 最大仿真步数
            lap_fraction: 完成"一圈"所需的赛道完成比例（默认0.95即95%）
            verbose: 是否打印进度信息
            detailed_step_log: 是否逐步打印详细数值
            detailed_step_log_max_steps: 逐步日志最大打印步数（None表示不限制）

        Returns:
            result: SimResult实例，包含仿真结果数据
        """
        if control_update_interval <= 0:
            raise ValueError("control_update_interval 必须为正整数")

        track = self.track  # 获取赛道对象的引用
        controller = self.controller  # 获取控制器对象的引用
        obstacles = track.get_obstacles()  # 获取赛道上的障碍物列表

        # 初始化起始状态
        if x0 is None:
            cx, cy = track.get_centerline()  # 获取赛道中心线的x和y坐标
            heading = track.get_heading()  # 获取赛道中心线各点的航向角
            # 从索引0处开始，使用参考速度初始化
            ref_init = track.get_reference_v_omega(0, 1)  # 获取初始参考速度和角速度
            # 构建初始状态向量：[x位置, y位置, 航向角, 速度, 角速度]
            x0 = np.array([cx[0], cy[0], heading[0], ref_init[0, 0], ref_init[0, 1]])

        result = SimResult(controller.name, track.__class__.__name__)  # 创建仿真结果对象
        controller.reset()  # 重置控制器状态

        x = x0.copy()  # 复制初始状态，避免修改原始数据
        result.states.append(x.copy())  # 将初始状态添加到结果列表

        # 获取车辆在赛道上的初始位置信息
        start_idx, start_s, _ = track.closest_point(x[0], x[1])  # 找到最近点的索引、弧长和横向误差
        max_s = track.total_length()  # 获取赛道总长度
        cumulative_s = 0.0  # 初始化累积行驶距离
        prev_s = start_s  # 记录上一步的弧长位置
        crash_lat_limit = max(TRACK_HALF_WIDTH - VEHICLE_RADIUS, 0.0)

        if verbose:
            print(f"\n正在仿真 {controller.name} 在 {track.__class__.__name__} 赛道上...")
            print(f"  赛道长度: {max_s:.0f}m, 障碍物数量: {len(obstacles)}")
            print(f"  控制更新间隔: 每 {control_update_interval} 步求解一次MPC")

        held_u = np.zeros(2)
        held_info = {'solve_time': 0.0, 'status': 'hold', 'debug': None}

        for step in range(max_steps):  # 开始主仿真循环
            t_sim = step * DT  # 计算当前仿真时间（步数乘以时间步长）

            # 查找当前车辆在赛道上的位置
            idx, current_s, lat_err = track.closest_point(x[0], x[1])  # idx:最近点索引, current_s:当前弧长, lat_err:横向误差

            # 更新累积弧长（行驶距离）
            ds = current_s - prev_s  # 计算弧长变化量
            if ds < -max_s / 2:  # 处理环绕情况（负方向跨越起点）
                ds += max_s  # 加上赛道总长度进行修正
            elif ds > max_s / 2:  # 处理环绕情况（正方向跨越起点）
                ds -= max_s  # 减去赛道总长度进行修正
            cumulative_s += abs(ds)  # 累加绝对弧长变化
            prev_s = current_s  # 更新上一步的弧长位置

            # 获取参考轨迹
            from config import T_HORIZON  # 导入预测时域
            ref = track.get_reference_trajectory(idx, T_HORIZON, current_speed=x[IDX_V])  # 获取从当前索引开始的预测时域内的参考轨迹

            # 求解MPC优化问题；若开启降频，则在中间步保持上一次控制
            should_solve = (step % control_update_interval == 0)
            if should_solve:
                try:
                    # 将“当前实际保持控制”作为u_prev传入，避免降频模式下控制器内部状态漂移
                    try:
                        u_opt, info = controller.solve(x, ref, obstacles, u_prev=held_u)
                    except TypeError:
                        u_opt, info = controller.solve(x, ref, obstacles)
                except Exception as e:
                    if verbose:
                        print(f"  步骤 {step}: 控制器错误: {e}")
                    # 发生错误时保持上一次控制，避免因单步求解失败触发控制跳变
                    u_opt = held_u.copy()
                    info = {'solve_time': 0, 'status': f'error: {str(e)[:30]}', 'debug': None}
            else:
                u_opt = held_u.copy()
                info = held_info.copy()
                info['solve_time'] = 0.0
                info['status'] = f"hold({held_info.get('status', 'unknown')})"

            # 对控制输入进行限幅处理，确保在物理约束范围内
            u_opt[0] = np.clip(u_opt[0], A_MIN, A_MAX)  # 限制加速度在[A_MIN, A_MAX]范围内
            u_opt[1] = np.clip(u_opt[1], -DELTA_MAX, DELTA_MAX)  # 限制转向角在[-DELTA_MAX, DELTA_MAX]范围内

            # 在限幅后更新保持控制，保证下一步复用的就是实际施加到车辆的控制量
            if should_solve:
                held_u = u_opt.copy()
                held_info = info.copy()

            # 向被控对象添加干扰信号
            noise = np.zeros(5)  # 初始化5维噪声向量（对应5个状态变量）
            if self.disturbance_gen is not None:  # 如果存在干扰生成器
                w = self.disturbance_gen.sample_single()  # 采样单个干扰样本
                # 对干扰进行适当缩放以符合物理状态的实际范围
                noise[0] = w[0] * 0.1   # 位置x噪声 [米]
                noise[1] = w[1] * 0.1   # 位置y噪声 [米]
                noise[IDX_PSI] = w[3] * 0.01   # 航向角噪声 [弧度]
                noise[IDX_V] = w[2] * 0.05     # 速度噪声 [米/秒]
                noise[IDX_OMEGA] = w[4] * 0.01  # 横摆角速度噪声 [弧度/秒]

            # 传播被控对象动力学（应用控制输入和干扰）
            x_next = discrete_step(x, u_opt) + noise  # 计算下一时刻状态并叠加干扰

            # 强制执行状态边界约束
            x_next[IDX_V] = np.clip(x_next[IDX_V], V_MIN, V_MAX)  # 限制速度在[V_MIN, V_MAX]范围内

            # 记录当前步的数据到结果对象
            result.controls.append(u_opt.copy())  # 保存控制输入
            result.solve_times.append(info.get('solve_time', 0))  # 保存求解时间，默认为0
            result.solve_statuses.append(info.get('status', 'unknown'))  # 保存求解状态，默认为'unknown'
            result.solve_debug.append(info.get('debug'))
            result.ref_states.append(ref[0].copy())  # 保存第一个参考状态
            result.timestamps.append(t_sim)  # 保存当前时间戳

            x = x_next  # 更新当前状态为下一时刻状态
            result.states.append(x.copy())  # 将新状态添加到结果列表

            # 逐步详细日志：用于排查速度归零等问题
            if detailed_step_log and (
                detailed_step_log_max_steps is None or step < detailed_step_log_max_steps
            ):
                x_str = np.array2string(result.states[-2], precision=4, suppress_small=True)
                ref_str = np.array2string(ref[0], precision=4, suppress_small=True)
                u_str = np.array2string(u_opt, precision=4, suppress_small=True)
                noise_str = np.array2string(noise, precision=4, suppress_small=True)
                x_next_str = np.array2string(x_next, precision=4, suppress_small=True)
                print(
                    f"[Step {step:04d}] t={t_sim:7.2f}s "
                    f"idx={idx:4d} s={current_s:8.2f}m ds={ds:7.3f}m "
                    f"cum={cumulative_s:8.2f}m prog={cumulative_s/max_s*100:6.2f}% lat={lat_err:8.4f}m"
                )
                print(f"  x      = {x_str}")
                print(f"  ref[0] = {ref_str}")
                print(f"  u_opt  = {u_str}")
                print(
                    f"  solve  = status={info.get('status','unknown')} "
                    f"time={info.get('solve_time', 0.0) * 1000:.2f}ms"
                )
                debug = info.get('debug')
                if debug:
                    step0 = debug.get('step0', {})
                    horizon = debug.get('horizon', {})
                    active = ','.join(debug.get('active_constraints', [])) or 'none'
                    key_parts = []
                    for key in (
                        'cost_track_vomega', 'cost_contour', 'cost_lag',
                        'cost_heading', 'cost_heading_mpcc', 'cost_speed',
                        'cost_progress', 'cost_progress_mpcc', 'cost_du',
                        'cost_abs_u'
                    ):
                        if key in step0:
                            key_parts.append(f"{key}={step0[key]:.3f}")
                    if 'cost_cvar' in horizon:
                        key_parts.append(f"cost_cvar={horizon['cost_cvar']:.3f}")
                    if 'risk_eta' in horizon:
                        key_parts.append(f"risk_eta={horizon['risk_eta']:.3f}")
                    key_parts.append(f"v_slack_max={debug.get('v_slack_max', 0.0):.3f}")
                    key_parts.append(f"obs_slack_max={debug.get('obs_slack_max', 0.0):.3f}")
                    print(f"  diag   = {'; '.join(key_parts)}")
                    print(f"  active = {active}")
                print(f"  noise  = {noise_str}")
                print(f"  x_next = {x_next_str}")
                print(
                    f"  speed  = v:{result.states[-2][IDX_V]:.4f} -> {x_next[IDX_V]:.4f} m/s, "
                    f"omega:{result.states[-2][IDX_OMEGA]:.4f} -> {x_next[IDX_OMEGA]:.4f} rad/s"
                )

            # 进度报告（每100步打印一次）
            if verbose and (step + 1) % 100 == 0:
                print(f"  步骤 {step+1}/{max_steps}: "
                      f"速度={x[IDX_V]:.1f}m/s, "
                      f"横向误差={lat_err:.1f}m, "
                      f"进度={cumulative_s/max_s*100:.1f}%, "
                      f"求解时间={info.get('solve_time',0)*1000:.1f}ms")

            # 检查是否完成一圈
            if cumulative_s >= max_s * lap_fraction:  # 如果累积行驶距离达到赛道长度的指定比例
                result.lap_completed = True  # 标记为已完成一圈
                result.lap_time = (step + 1) * DT  # 记录完成时间
                if verbose:
                    print(f"  在第 {step+1} 步完成一圈, "
                          f"时间={result.lap_time:.1f}s")
                break  # 跳出仿真循环

            # 检查是否撞到赛道边界
            if abs(lat_err) >= crash_lat_limit:
                result.crashed = True
                result.crash_step = step + 1
                result.crash_time = (step + 1) * DT
                result.crash_reason = (
                    f"track boundary hit: |lat_err|={abs(lat_err):.3f}m >= {crash_lat_limit:.3f}m"
                )
                if verbose:
                    print(
                        f"  车辆在第 {step+1} 步撞到赛道边界 "
                        f"(横向误差={lat_err:.2f}m, 阈值={crash_lat_limit:.2f}m)"
                    )
                break

            # 检查是否发散（车辆偏离赛道过远）
            if abs(lat_err) > 500:  # 如果横向误差超过500米
                if verbose:
                    print(f"  车辆在第 {step+1} 步发散 "
                          f"(横向误差={lat_err:.0f}m)")
                break  # 跳出仿真循环

        result.total_steps = len(result.controls)  # 计算总步数（等于控制输入的数量）

        if verbose:
            avg_solve = np.mean(result.solve_times) if result.solve_times else 0  # 计算平均求解时间
            print(f"  仿真完成: {result.total_steps} 步, "
                  f"平均求解时间={avg_solve*1000:.1f}ms")

        return result  # 返回包含所有仿真结果的对象

    @staticmethod
    def _export_result_to_step_log(result, output_path):
        """Export a simulation result to an aligned step-by-step text log."""
        data = result.to_arrays()
        states = data['states']
        controls = data['controls']
        solve_times = data['solve_times']
        timestamps = data['timestamps']
        solve_debug = list(data.get('solve_debug', []))
        ref_states = np.array(result.ref_states)
        solve_statuses = list(result.solve_statuses)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            header = (
                f"{'step':>5} {'t(s)':>8} {'status':<12} {'solve_ms':>9} "
                f"{'x':>9} {'y':>9} {'psi':>9} {'v':>8} {'omega':>8} "
                f"{'ref_v':>8} {'ref_omega':>10} {'a':>8} {'delta':>8} "
                f"{'next_v':>8} {'next_omega':>10}"
            )
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")
            f.write(f"# method={result.method_name}\n")
            f.write(f"# track={result.track_name}\n")
            f.write(f"# lap_completed={result.lap_completed}\n")
            f.write(f"# lap_time={result.lap_time}\n")
            f.write(f"# total_steps={result.total_steps}\n")
            f.write(f"# crashed={result.crashed}\n")
            f.write(f"# crash_step={result.crash_step}\n")
            f.write(f"# crash_time={result.crash_time}\n")
            f.write(f"# crash_reason={result.crash_reason}\n")

            if len(states) > 0:
                f.write(
                    "# init_state="
                    f"[{states[0, 0]:.6f}, {states[0, 1]:.6f}, {states[0, 2]:.6f}, "
                    f"{states[0, 3]:.6f}, {states[0, 4]:.6f}]\n"
                )
            f.write("\n")

            for step in range(len(controls)):
                x_t = states[step]
                x_next = states[step + 1]
                u_t = controls[step]
                ref_t = ref_states[step] if step < len(ref_states) else np.full(5, np.nan)
                solve_time_ms = solve_times[step] * 1000.0 if step < len(solve_times) else float('nan')
                status = solve_statuses[step] if step < len(solve_statuses) else 'unknown'
                debug = solve_debug[step] if step < len(solve_debug) else None
                t_sim = timestamps[step] if step < len(timestamps) else float(step)

                f.write(
                    f"{step:5d} {t_sim:8.3f} {status:<12.12} {solve_time_ms:9.3f} "
                    f"{x_t[0]:9.3f} {x_t[1]:9.3f} {x_t[2]:9.4f} {x_t[3]:8.3f} {x_t[4]:8.4f} "
                    f"{ref_t[3]:8.3f} {ref_t[4]:10.4f} {u_t[0]:8.4f} {u_t[1]:8.4f} "
                    f"{x_next[3]:8.3f} {x_next[4]:10.4f}\n"
                )
                if debug:
                    step0 = debug.get('step0', {})
                    horizon = debug.get('horizon', {})
                    active = ','.join(debug.get('active_constraints', [])) or 'none'
                    f.write(
                        f"  # debug step0={step0} horizon={horizon} active={active} "
                        f"v_slack_max={debug.get('v_slack_max', 0.0):.6f} "
                        f"obs_slack_max={debug.get('obs_slack_max', 0.0):.6f}\n"
                    )

    @staticmethod
    def _export_result_to_compact_log(result, output_path):
        """Export a compact one-line-per-step log for quick inspection."""
        data = result.to_arrays()
        states = data['states']
        controls = data['controls']
        solve_times = data['solve_times']
        timestamps = data['timestamps']
        solve_debug = list(data.get('solve_debug', []))
        ref_states = np.array(result.ref_states)
        solve_statuses = list(result.solve_statuses)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            f.write("# step t(s) v ref_v omega ref_omega a delta solve_ms status\n")

            for step in range(len(controls)):
                x_t = states[step]
                u_t = controls[step]
                ref_t = ref_states[step] if step < len(ref_states) else np.full(5, np.nan)
                solve_time_ms = solve_times[step] * 1000.0 if step < len(solve_times) else float('nan')
                status = solve_statuses[step] if step < len(solve_statuses) else 'unknown'
                debug = solve_debug[step] if step < len(solve_debug) else None
                t_sim = timestamps[step] if step < len(timestamps) else float(step)

                diag_excerpt = ""
                if debug:
                    step0 = debug.get('step0', {})
                    important = []
                    for key in ('cost_track_vomega', 'cost_contour', 'cost_lag', 'cost_progress', 'cost_cvar'):
                        if key in step0:
                            important.append(f"{key}={step0[key]:.3f}")
                    if 'horizon' in debug and 'cost_cvar' in debug['horizon']:
                        important.append(f"cost_cvar={debug['horizon']['cost_cvar']:.3f}")
                    diag_excerpt = (" " + " ".join(important)) if important else ""

                f.write(
                    f"{step:04d} "
                    f"{t_sim:8.3f} "
                    f"{x_t[3]:9.4f} "
                    f"{ref_t[3]:9.4f} "
                    f"{x_t[4]:9.4f} "
                    f"{ref_t[4]:9.4f} "
                    f"{u_t[0]:9.4f} "
                    f"{u_t[1]:9.4f} "
                    f"{solve_time_ms:9.3f} "
                    f"{status}{diag_excerpt}\n"
                )

    @staticmethod
    def _summarize_debug_diagnostics(result, top_k=5):
        """Summarize per-step debug diagnostics into dominant terms and active constraints."""
        debug_rows = [row for row in getattr(result, 'solve_debug', []) if row]
        if not debug_rows:
            return None

        step_acc = {}
        horizon_acc = {}
        active_counter = Counter()

        for row in debug_rows:
            for key, value in row.get('step0', {}).items():
                if isinstance(value, (int, float)):
                    step_acc.setdefault(key, []).append(float(value))
            for key, value in row.get('horizon', {}).items():
                if isinstance(value, (int, float)):
                    horizon_acc.setdefault(key, []).append(float(value))
            active_counter.update(row.get('active_constraints', []))

        def build_stats(acc):
            stats = {}
            for key, values in acc.items():
                if values:
                    stats[key] = {
                        'mean': float(np.mean(values)),
                        'max': float(np.max(values)),
                    }
            return stats

        step_stats = build_stats(step_acc)
        horizon_stats = build_stats(horizon_acc)
        dominant = sorted(
            [
                (key, vals['mean'])
                for key, vals in step_stats.items()
                if key.startswith('cost_')
            ],
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]

        return {
            'step_stats': step_stats,
            'horizon_stats': horizon_stats,
            'dominant_costs': dominant,
            'active_constraints': dict(active_counter.most_common()),
        }

    @staticmethod
    def _export_result_debug_summary(result, output_path, top_k=5):
        """Append a concise debug summary to the end of an existing log."""
        summary = Simulator._summarize_debug_diagnostics(result, top_k=top_k)
        if summary is None:
            return False

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'a') as f:
            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write("DEBUG SUMMARY\n")
            f.write("=" * 80 + "\n")
            f.write(f"method={result.method_name}\n")
            f.write(f"track={result.track_name}\n")
            f.write(f"total_steps={result.total_steps}\n\n")
            f.write(f"crashed={result.crashed}\n")
            f.write(f"crash_step={result.crash_step}\n")
            f.write(f"crash_time={result.crash_time}\n")
            f.write(f"crash_reason={result.crash_reason}\n\n")

            f.write("[Top Dominant Cost Terms]\n")
            for key, mean_val in summary['dominant_costs']:
                max_val = summary['step_stats'][key]['max']
                f.write(f"  {key}: mean={mean_val:.6f}, max={max_val:.6f}\n")

            f.write("\n[Active Constraints Frequency]\n")
            if summary['active_constraints']:
                for key, count in summary['active_constraints'].items():
                    f.write(f"  {key}: {count}\n")
            else:
                f.write("  none\n")

            f.write("\n[Horizon Diagnostics]\n")
            for key, vals in sorted(summary['horizon_stats'].items()):
                f.write(f"  {key}: mean={vals['mean']:.6f}, max={vals['max']:.6f}\n")

        return True

    @staticmethod
    def _build_track_for_result(result):
        """Reconstruct a track instance from the saved result metadata."""
        track_name = result.track_name
        if track_name == 'LusailShortTrack':
            from tracks.lusail_short_track import LusailShortTrack
            return LusailShortTrack()
        if track_name == 'LusailTrack':
            from tracks.lusail_track import LusailTrack
            return LusailTrack()
        if track_name == 'CustomWindingTrack':
            from tracks.custom_track import CustomWindingTrack
            return CustomWindingTrack()
        if track_name == 'SprintOvalTrack':
            from tracks.sprint_oval_track import SprintOvalTrack
            return SprintOvalTrack()
        return None

    @staticmethod
    def _export_result_figures(result, base_name):
        """Export per-result PDF figures and return created paths."""
        track = Simulator._build_track_for_result(result)
        if track is None:
            return []

        from visualization.plot_trajectories import (
            plot_trajectory_comparison,
            plot_state_comparison,
            plot_control_comparison,
        )

        results = {result.method_name: result}
        trajectory_name = f"{base_name}_trajectory.pdf"
        states_name = f"{base_name}_states.pdf"
        controls_name = f"{base_name}_controls.pdf"

        plot_trajectory_comparison(
            results,
            track,
            title=f"{result.method_name} Trajectory on {result.track_name}",
            filename=trajectory_name,
        )
        plot_state_comparison(
            results,
            track,
            filename=states_name,
        )
        plot_control_comparison(
            results,
            filename=controls_name,
        )

        return [
            os.path.join(FIGURES_DIR, trajectory_name),
            os.path.join(FIGURES_DIR, states_name),
            os.path.join(FIGURES_DIR, controls_name),
        ]

    @staticmethod
    def _export_result_animation(result_path, base_name):
        """Export per-result animation GIF using visualization/animate_simulation.py."""
        output_path = os.path.join(FIGURES_DIR, f"{base_name}_animation.gif")
        script_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'visualization',
            'animate_simulation.py',
        )

        cmd = [
            sys.executable,
            script_path,
            '--result',
            result_path,
            '--fps',
            str(int(ANIMATION_FPS)),
            '--save',
            output_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"[Warn] 动画导出失败: {proc.stderr.strip() or proc.stdout.strip()}")
            return None
        return output_path

    @staticmethod
    def save_result(result, filename=None, save_dir=RESULTS_DIR):
        """
        将仿真结果保存到磁盘。
        
        Args:
            result: SimResult实例，要保存的仿真结果
            filename: 文件名（可选），默认格式为"{方法名}_{赛道名}.pkl"
            save_dir: 保存目录，默认为RESULTS_DIR
        """
        os.makedirs(save_dir, exist_ok=True)  # 创建保存目录（如果不存在）
        if filename is None:
            filename = f"{result.method_name}_{result.track_name}.pkl"  # 生成默认文件名
        path = os.path.join(save_dir, filename)  # 拼接完整路径
        with open(path, 'wb') as f:
            pickle.dump(result, f)  # 使用pickle序列化并保存结果对象
        base_name = os.path.splitext(os.path.basename(path))[0]
        log_dir = os.path.join(save_dir, 'logs')
        step_log_path = os.path.join(log_dir, f"{base_name}.log")
        Simulator._export_result_to_step_log(result, step_log_path)
        Simulator._export_result_debug_summary(result, step_log_path)
        figure_paths = Simulator._export_result_figures(result, base_name) if EXPORT_STATIC_FIGURES else []
        animation_path = Simulator._export_result_animation(path, base_name) if EXPORT_ANIMATION else None
        print(f"结果已保存到 {path}")
        print(f"报表链接: {step_log_path}")
        for figure_path in figure_paths:
            print(f"图片链接: {figure_path}")
        if animation_path:
            print(f"动画链接: {animation_path}")

    @staticmethod
    def load_result(filepath):
        """
        从磁盘加载仿真结果。
        
        Args:
            filepath: 文件路径，包含要加载的仿真结果
            
        Returns:
            反序列化后的SimResult实例
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)  # 使用pickle反序列化并返回结果对象
