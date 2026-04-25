"""
论文精确的Deep Koopman网络架构。

实现了论文第3.3节中描述的编码器-解码器结构，
以及线性动力学矩阵A、B、C。

网络架构：
    编码器: x ∈ R^5 -> [64, 128, 64] -> z ∈ R^32 (ReLU激活函数)
    解码器: z ∈ R^32 -> [64, 32] -> x ∈ R^5 (ReLU激活函数)
    线性动力学: z_{t+1} = A @ z_t + B @ u_t + C @ w_t

Koopman算子理论：
    Koopman算子是一种无限维线性算子，可以精确描述非线性系统的演化。
    Deep Koopman使用神经网络学习一个有限维的近似，将非线性系统
    提升到高维线性空间（Koopman空间），在该空间中使用线性模型
    进行预测和控制。
"""

import torch  # 导入PyTorch深度学习框架
import torch.nn as nn  # 导入PyTorch神经网络模块
import numpy as np  # 导入NumPy库，用于数值计算


class DeepKoopmanPaper(nn.Module):
    """
    与论文规格完全匹配的Deep Koopman模型（表3）。

    该模型实现了Koopman算子理论的核心思想：
    通过编码器将非线性系统状态提升到高维线性空间，
    在该空间中使用线性动力学进行演化，然后通过解码器
    映射回原始物理空间。

    组件:
        - 编码器phi: R^5 -> R^32
          MLP（多层感知机），3个隐藏层 [64, 128, 64]
          将物理状态x提升到Koopman空间z

        - 线性动力学矩阵:
          A ∈ R^{32x32}: 状态演化矩阵（自主动力学）
          B ∈ R^{32x2}:  控制输入矩阵（控制对系统的影响）
          C ∈ R^{32x5}:  干扰矩阵（干扰对系统的影响）

        - 解码器psi: R^32 -> R^5
          MLP，2个隐藏层 [64, 32]
          从Koopman空间z重构物理状态x

    前向过程:
        1. 编码: z_t = phi(x_t)
        2. 线性演化: z_{t+1} = A @ z_t + B @ u_t + C @ w_t
        3. 解码: x_{t+1} = psi(z_{t+1})
    """

    def __init__(self, n_x=5, n_u=2, n_z=32, n_w=5):
        """
        初始化Deep Koopman网络。

        参数:
            n_x: 整数，物理状态维度，默认5 [px, py, v, psi, omega]
            n_u: 整数，控制输入维度，默认2 [a, delta]
            n_z: 整数，Koopman空间维度，默认32
            n_w: 整数，干扰维度，默认5（与状态维度相同）
        """
        super().__init__()  # 调用父类nn.Module的初始化函数
        # 保存维度参数，供后续使用
        self.n_x = n_x  # 物理状态维度
        self.n_u = n_u  # 控制输入维度
        self.n_z = n_z  # Koopman空间维度
        self.n_w = n_w  # 干扰维度

        # ============================================================
        # 编码器: x -> z （表1）
        # ============================================================
        # 输入: n_x=5，隐藏层: 64->128->64，输出: n_z=32
        # 使用nn.Sequential按顺序堆叠层
        self.encoder = nn.Sequential(
            nn.Linear(n_x, 64),    # 线性层: 5 -> 64
            nn.ReLU(),              # ReLU激活函数: 引入非线性
            nn.Linear(64, 128),    # 线性层: 64 -> 128
            nn.ReLU(),              # ReLU激活函数
            nn.Linear(128, 64),    # 线性层: 128 -> 64
            nn.ReLU(),              # ReLU激活函数
            nn.Linear(64, n_z),    # 线性层: 64 -> 32 (Koopman空间)
        )

        # ============================================================
        # 解码器: z -> x （表2）
        # ============================================================
        # 输入: n_z=32，隐藏层: 64->32，输出: n_x=5
        self.decoder = nn.Sequential(
            nn.Linear(n_z, 64),    # 线性层: 32 -> 64
            nn.ReLU(),              # ReLU激活函数
            nn.Linear(64, 32),    # 线性层: 64 -> 32
            nn.ReLU(),              # ReLU激活函数
            nn.Linear(32, n_x),    # 线性层: 32 -> 5 (物理空间)
        )

        # ============================================================
        # 线性动力学矩阵（第3.3.3节）
        # ============================================================
        # Koopman空间的线性演化方程: z_{t+1} = A @ z_t + B @ u_t + C @ w_t
        # 使用nn.Parameter使矩阵成为可训练参数
        self.A = nn.Parameter(torch.empty(n_z, n_z))  # 状态演化矩阵 32x32
        self.B = nn.Parameter(torch.empty(n_z, n_u))  # 控制输入矩阵 32x2
        self.C = nn.Parameter(torch.empty(n_z, n_w))  # 干扰矩阵 32x5

        # 初始化网络权重和动力学矩阵
        self._init_weights()

    def _init_weights(self):
        """
        初始化网络权重和动力学矩阵。

        权重初始化策略对训练稳定性和收敛速度至关重要：
        1. 编码器/解码器：使用Xavier初始化，保持方差稳定
        2. 矩阵A：初始化为单位矩阵+小噪声，确保初始稳定性
        3. 矩阵B：使用Xavier初始化
        4. 矩阵C：初始化为很小的值，因为干扰影响应该较小
        """
        # 使用Xavier均匀分布初始化编码器和解码器的权重
        # Xavier初始化可以保持每层的输出方差一致，避免梯度消失/爆炸
        for module in [self.encoder, self.decoder]:
            for layer in module:
                if isinstance(layer, nn.Linear):  # 只初始化线性层
                    nn.init.xavier_uniform_(layer.weight)  # Xavier初始化权重
                    nn.init.zeros_(layer.bias)  # 偏置初始化为0

        # 将矩阵A初始化为单位矩阵+小噪声
        # 这确保初始状态下Koopman空间演化接近恒等变换，提高稳定性
        nn.init.eye_(self.A)  # 初始化为单位矩阵
        self.A.data += torch.randn_like(self.A) * 0.01  # 添加小噪声(标准差0.01)

        # 使用Xavier初始化矩阵B和C
        nn.init.xavier_uniform_(self.B)  # 控制输入矩阵
        nn.init.xavier_uniform_(self.C)  # 干扰矩阵
        self.C.data *= 0.01  # 将C缩小100倍，初始干扰影响很小

    def encode(self, x):
        """
        编码器：将物理状态提升到Koopman空间。

        该函数实现非线性变换 phi: R^5 -> R^32，
        将原始物理状态x映射到高维线性空间z，
        在该空间中系统动态可以用线性模型近似。

        参数:
            x: 形状为(batch, n_x)的张量，物理状态
               batch: 批次大小
               n_x: 状态维度（5）

        返回:
            z: 形状为(batch, n_z)的张量，Koopman空间状态
               n_z: Koopman空间维度（32）
        """
        return self.encoder(x)  # 通过编码器网络前向传播

    def decode(self, z):
        """
        解码器：从Koopman空间重构物理状态。

        该函数实现非线性变换 psi: R^32 -> R^5，
        将Koopman空间状态z映射回原始物理空间x，
        用于重构和预测。

        参数:
            z: 形状为(batch, n_z)的张量，Koopman空间状态

        返回:
            x_hat: 形状为(batch, n_x)的张量，重构的物理状态
        """
        return self.decoder(z)  # 通过解码器网络前向传播

    def linear_step(self, z, u, w=None):
        """
        Koopman空间中的线性动力学演化。

        实现线性状态转移方程：
            z_{t+1} = A @ z_t + B @ u_t + C @ w_t

        这是Koopman理论的核心：在高维空间中使用线性模型
        描述非线性系统的演化。矩阵A捕获自主动力学，
        B捕获控制输入的影响，C捕获外部干扰的影响。

        参数:
            z: 形状为(batch, n_z)的张量，当前Koopman空间状态
            u: 形状为(batch, n_u)的张量，控制输入
            w: 形状为(batch, n_w)的张量，干扰（可选，默认为零）

        返回:
            z_next: 形状为(batch, n_z)的张量，预测的下一时刻Koopman空间状态
        """
        # 计算z_{t+1} = A @ z_t + B @ u_t
        # 注意：这里使用 z @ A.T 而不是 A @ z.T，是因为batch在第一个维度
        # 数学上等价于对每个样本计算 A @ z_i
        z_next = z @ self.A.T + u @ self.B.T

        # 如果提供了干扰，添加干扰项 C @ w_t
        if w is not None:
            z_next = z_next + w @ self.C.T

        return z_next

    def forward(self, x, u, x_next=None, w=None):
        """
        完整的前向传播，返回损失计算所需的所有中间变量。

        该函数执行Koopman模型的完整前向过程：
        1. 编码当前状态
        2. 重构当前状态（用于重构损失）
        3. 线性动力学预测（用于线性动力学损失）
        4. 解码预测结果（用于多步预测损失）

        参数:
            x: 形状为(batch, n_x)的张量，当前状态
            u: 形状为(batch, n_u)的张量，控制输入
            x_next: 形状为(batch, n_x)的张量，真实下一状态（可选）
                   如果提供，用于计算线性动力学损失
            w: 形状为(batch, n_w)的张量，干扰（可选）

        返回:
            result: 字典，包含以下键：
                'z': 编码后的当前状态 (batch, n_z)
                'z_next_linear': 线性预测的下一Koopman状态 (batch, n_z)
                'z_next_true': 编码的真实下一状态 (batch, n_z)，如果提供了x_next
                'x_recon': 解码的当前状态重构 (batch, n_x)
                'x_next_recon': 解码的真实下一状态重构 (batch, n_x)，如果提供了x_next
                'x_next_pred': 从线性动力学解码的预测 (batch, n_x)
        """
        # 步骤1：编码当前状态 x_t -> z_t
        z = self.encode(x)

        # 步骤2：重构当前状态 z_t -> x_t_recon
        # 用于计算重构损失 L_recon = ||x - psi(phi(x))||^2
        x_recon = self.decode(z)

        # 步骤3：在Koopman空间中进行线性动力学预测
        # z_{t+1} = A @ z_t + B @ u_t + C @ w_t
        z_next_linear = self.linear_step(z, u, w)

        # 步骤4：解码线性预测结果到物理空间
        # x_{t+1}_pred = psi(z_{t+1})
        x_next_pred = self.decode(z_next_linear)

        # 构建结果字典
        result = {
            'z': z,                      # Koopman空间当前状态
            'z_next_linear': z_next_linear,  # 线性预测的Koopman下一状态
            'x_recon': x_recon,          # 当前状态重构
            'x_next_pred': x_next_pred,  # 下一状态预测
        }

        # 如果提供了真实下一状态，编码它用于线性动力学损失
        if x_next is not None:
            # 编码真实下一状态：x_{t+1} -> z_{t+1}_true
            z_next_true = self.encode(x_next)
            # 重构真实下一状态：z_{t+1}_true -> x_{t+1}_recon
            x_next_recon = self.decode(z_next_true)
            # 添加到结果字典
            result['z_next_true'] = z_next_true
            result['x_next_recon'] = x_next_recon

        return result

    def multi_step_predict(self, x0, u_seq, w_seq=None):
        """
        Koopman空间中的多步预测。

        该函数使用线性动力学模型进行K步前向预测，
        用于计算多步预测损失 L_pred，帮助模型学习
        长期动态演化特性，而不仅是单步预测。

        预测过程：
            z_0 = phi(x_0)
            z_1 = A @ z_0 + B @ u_0 + C @ w_0
            z_2 = A @ z_1 + B @ u_1 + C @ w_1
            ...
            z_K = A @ z_{K-1} + B @ u_{K-1} + C @ w_{K-1}

            x_k = psi(z_k), for k = 1, 2, ..., K

        参数:
            x0: 形状为(batch, n_x)的张量，初始状态
            u_seq: 形状为(batch, K, n_u)的张量，控制序列
                  K是预测时域（步数）
            w_seq: 形状为(batch, K, n_w)的张量，干扰序列（可选）

        返回:
            x_preds: 形状为(batch, K, n_x)的张量，预测的状态序列 t=1..K
            z_preds: 形状为(batch, K, n_z)的张量，预测的Koopman状态序列 t=1..K
        """
        K = u_seq.shape[1]  # 获取预测时域（步数）

        # 编码初始状态：x_0 -> z_0
        z = self.encode(x0)  # 形状: (batch, n_z)

        # 初始化预测结果列表
        x_preds = []  # 存储物理空间预测
        z_preds = []  # 存储Koopman空间预测

        # 逐步进行K步预测
        for k in range(K):
            # 提取第k步的控制输入
            u_k = u_seq[:, k, :]  # 形状: (batch, n_u)
            # 提取第k步的干扰（如果提供）
            w_k = w_seq[:, k, :] if w_seq is not None else None

            # 线性动力学演化：z_{k+1} = A @ z_k + B @ u_k + C @ w_k
            z = self.linear_step(z, u_k, w_k)
            # 解码到物理空间：x_{k+1} = psi(z_{k+1})
            x_pred = self.decode(z)

            # 存储预测结果
            z_preds.append(z)
            x_preds.append(x_pred)

        # 将列表转换为张量
        # stack along dim=1，得到形状 (batch, K, n_x) 和 (batch, K, n_z)
        x_preds = torch.stack(x_preds, dim=1)
        z_preds = torch.stack(z_preds, dim=1)

        return x_preds, z_preds

    def get_matrices(self):
        """
        提取A、B、C矩阵为NumPy数组。

        该函数用于将训练好的动力学矩阵导出，
        以便在MPC控制器中使用。

        返回:
            A: numpy数组，形状(n_z, n_z)，状态演化矩阵
            B: numpy数组，形状(n_z, n_u)，控制输入矩阵
            C: numpy数组，形状(n_z, n_w)，干扰矩阵
        """
        # 使用detach()从计算图中分离，然后转移到CPU并转换为NumPy
        A = self.A.detach().cpu().numpy()
        B = self.B.detach().cpu().numpy()
        C = self.C.detach().cpu().numpy()
        return A, B, C

    def get_network_weights(self):
        """
        提取所有编码器/解码器权重，用于CasADi重构。

        该函数将PyTorch网络的权重导出为NumPy数组，
        以便在CasADi中重建相同的网络结构，用于
        NMPC控制器中的非线性优化。

        返回:
            dict: 包含以下键的字典：
                'encoder_weights': 编码器权重列表，每个元素是一个线性层的权重矩阵
                'encoder_biases': 编码器偏置列表
                'decoder_weights': 解码器权重列表
                'decoder_biases': 解码器偏置列表
        """
        # 提取编码器权重和偏置
        enc_weights, enc_biases = [], []
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):  # 只处理线性层
                # 将权重和偏置转移到CPU并转换为NumPy
                enc_weights.append(layer.weight.detach().cpu().numpy())
                enc_biases.append(layer.bias.detach().cpu().numpy())

        # 提取解码器权重和偏置
        dec_weights, dec_biases = [], []
        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                dec_weights.append(layer.weight.detach().cpu().numpy())
                dec_biases.append(layer.bias.detach().cpu().numpy())

        return {
            'encoder_weights': enc_weights,  # 编码器权重列表
            'encoder_biases': enc_biases,    # 编码器偏置列表
            'decoder_weights': dec_weights,  # 解码器权重列表
            'decoder_biases': dec_biases,    # 解码器偏置列表
        }


def koopman_loss(model, x_windows, u_windows, lambda_recon=1.0,
                 lambda_linear=1.0, lambda_pred=0.5):
    """
    计算论文的三部分损失函数（第3.3.5节）。

    Deep Koopman的训练使用三个损失项的加权和：
    1. 重构损失 (L_recon): 确保编解码器能准确重构原始状态
    2. 线性动力学损失 (L_linear): 确保Koopman空间中的线性演化准确性
    3. 多步预测损失 (L_pred): 提高长期预测能力

    总损失：L_total = λ_recon * L_recon + λ_linear * L_linear + λ_pred * L_pred

    参数:
        model: DeepKoopmanPaper实例，训练中的Koopman模型
        x_windows: 形状为(batch, K+1, n_x)的张量，状态序列
                  [x_0, x_1, ..., x_K]，包含K+1个连续状态
        u_windows: 形状为(batch, K, n_u)的张量，控制序列
                  [u_0, u_1, ..., u_{K-1}]，包含K个连续控制
        lambda_recon: 浮点数，重构损失权重，默认1.0
        lambda_linear: 浮点数，线性动力学损失权重，默认1.0
        lambda_pred: 浮点数，多步预测损失权重，默认0.5

    返回:
        total_loss: 张量，总损失值
        loss_dict: 字典，包含各分项损失值：
            - 'total': 总损失
            - 'recon': 重构损失
            - 'linear': 线性动力学损失
            - 'pred': 多步预测损失
    """
    # 获取批次大小和预测时域（步数）
    # batch = x_windows.shape[0]  # 未使用，但保留作为参考
    # K = u_windows.shape[1]      # 未使用，但保留作为参考

    # 提取初始状态、下一状态和第一个控制输入
    x_0 = x_windows[:, 0, :]       # 形状: (batch, n_x)，初始状态 x_t
    x_1 = x_windows[:, 1, :]       # 形状: (batch, n_x)，下一状态 x_{t+1}
    u_0 = u_windows[:, 0, :]       # 形状: (batch, n_u)，第一个控制 u_t

    # ================================================================
    # 1. 重构损失 (L_recon)
    # ================================================================
    # 公式: L_recon = (1/M) * Σ(||x_t - psi(phi(x_t))||^2 +
    #                            ||x_{t+1} - psi(phi(x_{t+1}))||^2)
    #
    # 重构损失确保编解码器能够准确地重构输入状态，
    # 这保证了Koopman空间保留了原始状态的所有重要信息。

    # 编码当前状态：z_0 = phi(x_0)
    z_0 = model.encode(x_0)
    # 重构当前状态：x_0_recon = psi(z_0)
    x_0_recon = model.decode(z_0)

    # 编码下一状态：z_1 = phi(x_1)
    z_1 = model.encode(x_1)
    # 重构下一状态：x_1_recon = psi(z_1)
    x_1_recon = model.decode(z_1)

    # 计算重构损失（使用MSE损失）
    # 同时计算x_t和x_{t+1}的重构误差
    loss_recon = (
        nn.functional.mse_loss(x_0_recon, x_0) +  # ||x_0 - psi(phi(x_0))||^2
        nn.functional.mse_loss(x_1_recon, x_1)    # ||x_1 - psi(phi(x_1))||^2
    )

    # ================================================================
    # 2. 线性动力学损失 (L_linear)
    # ================================================================
    # 公式: L_linear = (1/M) * Σ(||phi(x_{t+1}) - A*phi(x_t) - B*u_t||^2)
    #
    # 线性动力学损失确保Koopman空间中的线性演化准确，
    # 即 z_{t+1} ≈ A @ z_t + B @ u_t

    # 使用线性模型预测下一Koopman状态：z_1_pred = A @ z_0 + B @ u_0
    z_1_pred = model.linear_step(z_0, u_0)
    # 计算线性预测与真实编码状态的误差
    loss_linear = nn.functional.mse_loss(z_1_pred, z_1)

    # ================================================================
    # 3. 多步预测损失 (L_pred)
    # ================================================================
    # 公式: L_pred = (1/MK) * Σ_k(||x_{t+k} - psi(A^k*phi(x_t) + Σ...))||^2)
    #
    # 多步预测损失确保模型能够准确预测长期演化，
    # 而不仅是单步预测。这提高了模型的预测能力和
    # 在MPC中的控制性能。

    # 提取目标状态序列：x_1, x_2, ..., x_K
    x_targets = x_windows[:, 1:, :]   # 形状: (batch, K, n_x)，t=1..K的目标状态
    # 控制序列：u_0, u_1, ..., u_{K-1}
    u_seq = u_windows                  # 形状: (batch, K, n_u)

    # 使用多步预测：从x_0开始，使用控制序列u_seq进行K步预测
    x_preds, _ = model.multi_step_predict(x_0, u_seq)
    # x_preds形状: (batch, K, n_x)，预测的t=1..K的状态

    # 计算多步预测损失（所有K步的MSE）
    loss_pred = nn.functional.mse_loss(x_preds, x_targets)

    # ================================================================
    # 4. 总损失
    # ================================================================
    # 加权求和：L_total = λ_recon * L_recon + λ_linear * L_linear + λ_pred * L_pred
    total_loss = (lambda_recon * loss_recon +
                  lambda_linear * loss_linear +
                  lambda_pred * loss_pred)

    # 构建损失字典，用于日志记录
    loss_dict = {
        'total': total_loss.item(),    # 总损失值
        'recon': loss_recon.item(),    # 重构损失值
        'linear': loss_linear.item(),  # 线性动力学损失值
        'pred': loss_pred.item(),      # 多步预测损失值
    }

    return total_loss, loss_dict
