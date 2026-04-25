"""
主入口脚本：运行完整的实验流水线。
    1. 训练Deep Koopman模型
    2. 在两条赛道上运行仿真
    3. 生成所有图表和表格
"""

import os  # 导入操作系统接口模块
import sys  # 导入系统模块，用于路径操作
import time  # 导入时间模块，用于计算运行时间

# 将当前目录添加到系统路径，以便导入同目录下的其他模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    """主函数：按顺序执行完整的实验流水线。"""
    total_start = time.time()  # 记录流水线开始时间

    print("=" * 70)  # 打印分隔线（70个等号）
    print(" K-DRMPC论文实现 - 完整流水线")
    print(" 基于分布鲁棒模型预测控制的")
    print(" 具有干扰不确定性自适应能力的避障控制")
    print("=" * 70)

    # 第一阶段：模型训练
    print("\n" + "=" * 70)  # 打印阶段标题分隔线
    print(" 第一阶段：训练Deep Koopman模型")
    print("=" * 70)
    from run_training import main as run_training  # 导入训练脚本的主函数
    run_training()  # 执行模型训练

    # 第二阶段：闭环仿真
    print("\n" + "=" * 70)
    print(" 第二阶段：运行仿真")
    print("=" * 70)
    from run_simulation import main as run_simulation  # 导入仿真脚本的主函数
    run_simulation()  # 执行仿真实验

    # 第三阶段：结果分析与可视化
    print("\n" + "=" * 70)
    print(" 第三阶段：生成图表和表格")
    print("=" * 70)
    from run_analysis import main as run_analysis  # 导入分析脚本的主函数
    run_analysis()  # 执行结果分析和可视化

    total_time = time.time() - total_start  # 计算流水线总运行时间
    print("\n" + "=" * 70)
    print(f" 流水线执行完成")
    print(f" 总耗时: {total_time/60:.1f} 分钟")  # 将秒转换为分钟显示
    print("=" * 70)


if __name__ == "__main__":
    # 当直接运行此脚本时（而非作为模块导入），执行主函数
    main()
