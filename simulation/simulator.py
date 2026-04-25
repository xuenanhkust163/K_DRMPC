"""
闭环仿真引擎，用于在赛道上运行MPC控制器。
"""

import numpy as np  # 导入NumPy库，用于数值计算
import time  # 导入时间模块，用于计时
import os  # 导入操作系统接口模块
import sys  # 导入系统模块，用于路径操作
import pickle  # 导入pickle模块，用于对象的序列化和反序列化

# 将父目录添加到系统路径，以便导入项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 从配置文件导入常量和参数
from config import (
    DT, MAX_SIM_STEPS, V_MIN, V_MAX, A_MIN, A_MAX,
    DELTA_MAX, RESULTS_DIR
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
        self.ref_states = []       # 参考状态列表
        self.timestamps = []       # 仿真时间戳列表
        self.lap_completed = False  # 标志位，表示是否完成了一圈
        self.lap_time = None       # 完成一圈所需的时间
        self.total_steps = 0       # 总仿真步数

    def to_arrays(self):
        """将列表转换为NumPy数组格式，便于后续处理和计算。"""
        return {
            'states': np.array(self.states),  # 将所有状态转换为二维数组
            'controls': np.array(self.controls),  # 将所有控制输入转换为二维数组
            'solve_times': np.array(self.solve_times),  # 将求解时间转换为一维数组
            'timestamps': np.array(self.timestamps),  # 将时间戳转换为一维数组
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
            verbose=True):
        """
        运行闭环仿真。

        Args:
            x0: 形状为(5,)的初始状态向量。如果为None，则从赛道起点开始。
            max_steps: 最大仿真步数
            lap_fraction: 完成"一圈"所需的赛道完成比例（默认0.95即95%）
            verbose: 是否打印进度信息

        Returns:
            result: SimResult实例，包含仿真结果数据
        """
        track = self.track  # 获取赛道对象的引用
        controller = self.controller  # 获取控制器对象的引用
        obstacles = track.get_obstacles()  # 获取赛道上的障碍物列表

        # 初始化起始状态
        if x0 is None:
            cx, cy = track.get_centerline()  # 获取赛道中心线的x和y坐标
            heading = track.get_heading()  # 获取赛道中心线各点的航向角
            # 从索引0处开始，使用参考速度初始化
            ref_init = track.get_reference_v_omega(0, 1)  # 获取初始参考速度和角速度
            # 构建初始状态向量：[x位置, y位置, 速度, 航向角, 角速度]
            x0 = np.array([cx[0], cy[0], ref_init[0, 0], heading[0], ref_init[0, 1]])

        result = SimResult(controller.name, track.__class__.__name__)  # 创建仿真结果对象
        controller.reset()  # 重置控制器状态

        x = x0.copy()  # 复制初始状态，避免修改原始数据
        result.states.append(x.copy())  # 将初始状态添加到结果列表

        # 获取车辆在赛道上的初始位置信息
        start_idx, start_s, _ = track.closest_point(x[0], x[1])  # 找到最近点的索引、弧长和横向误差
        max_s = track.total_length()  # 获取赛道总长度
        cumulative_s = 0.0  # 初始化累积行驶距离
        prev_s = start_s  # 记录上一步的弧长位置

        if verbose:
            print(f"\n正在仿真 {controller.name} 在 {track.__class__.__name__} 赛道上...")
            print(f"  赛道长度: {max_s:.0f}m, 障碍物数量: {len(obstacles)}")

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
            ref = track.get_reference_trajectory(idx, T_HORIZON)  # 获取从当前索引开始的预测时域内的参考轨迹

            # 求解MPC优化问题
            try:
                u_opt, info = controller.solve(x, ref, obstacles)  # 调用控制器求解最优控制输入
            except Exception as e:
                if verbose:
                    print(f"  步骤 {step}: 控制器错误: {e}")
                u_opt = np.zeros(2)  # 发生错误时使用零控制输入
                info = {'solve_time': 0, 'status': f'error: {str(e)[:30]}'}  # 记录错误信息

            # 对控制输入进行限幅处理，确保在物理约束范围内
            u_opt[0] = np.clip(u_opt[0], A_MIN, A_MAX)  # 限制加速度在[A_MIN, A_MAX]范围内
            u_opt[1] = np.clip(u_opt[1], -DELTA_MAX, DELTA_MAX)  # 限制转向角在[-DELTA_MAX, DELTA_MAX]范围内

            # 向被控对象添加干扰信号
            noise = np.zeros(5)  # 初始化5维噪声向量（对应5个状态变量）
            if self.disturbance_gen is not None:  # 如果存在干扰生成器
                w = self.disturbance_gen.sample_single()  # 采样单个干扰样本
                # 对干扰进行适当缩放以符合物理状态的实际范围
                noise[0] = w[0] * 0.1   # 位置x噪声 [米]
                noise[1] = w[1] * 0.1   # 位置y噪声 [米]
                noise[2] = w[2] * 0.05  # 速度噪声 [米/秒]
                noise[3] = w[3] * 0.01  # 航向角噪声 [弧度]
                noise[4] = w[4] * 0.01  # 横摆角速度噪声 [弧度/秒]

            # 传播被控对象动力学（应用控制输入和干扰）
            x_next = discrete_step(x, u_opt) + noise  # 计算下一时刻状态并叠加干扰

            # 强制执行状态边界约束
            x_next[2] = np.clip(x_next[2], V_MIN, V_MAX)  # 限制速度在[V_MIN, V_MAX]范围内

            # 记录当前步的数据到结果对象
            result.controls.append(u_opt.copy())  # 保存控制输入
            result.solve_times.append(info.get('solve_time', 0))  # 保存求解时间，默认为0
            result.solve_statuses.append(info.get('status', 'unknown'))  # 保存求解状态，默认为'unknown'
            result.ref_states.append(ref[0].copy())  # 保存第一个参考状态
            result.timestamps.append(t_sim)  # 保存当前时间戳

            x = x_next  # 更新当前状态为下一时刻状态
            result.states.append(x.copy())  # 将新状态添加到结果列表

            # 进度报告（每100步打印一次）
            if verbose and (step + 1) % 100 == 0:
                print(f"  步骤 {step+1}/{max_steps}: "
                      f"速度={x[2]:.1f}m/s, "
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
        print(f"结果已保存到 {path}")

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
