from utils import *
import numpy as np
import torch
import torch.nn as nn
from typing import *
import sys
import argparse

# 全局设备配置（确保utils中定义了device，这里兜底定义）
if 'device' not in globals():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='args')
parser.add_argument('--num_episodes', type=int, help='number of episodes')
parser.add_argument('--checkpoint', type=int, help='the interval of saving models')
parser.add_argument('--use_wandb', action='store_true', help='use wandb for experiment tracking (requires wandb installed)')
parser.add_argument('--wandb_project', type=str, default='gobang-rl-AI3002', help='wandb project name')
parser.add_argument('--wandb_name', type=str, default=None, help='wandb run name')
args = parser.parse_args()
num_episodes = args.num_episodes
checkpoint = args.checkpoint


class SEBlock(nn.Module):
    """Squeeze-and-Excitation (SE) block for channel-wise attention."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.reduction = reduction
        mid = max(1, channels // reduction)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.GELU(),
            nn.Linear(mid, channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        s = self.avgpool(x).view(b, c)
        s = self.fc(s).view(b, c, 1, 1)
        return x * s


class Actor(nn.Module):
    """
    Actor（策略网络）负责生成可靠的策略，以尽可能最大化累积回报。
    它接受形状为 (B, 1, N, N) 或 (N, N) 的一批数组作为输入，并输出形状为 (B, N ** 2) 的张量
    作为生成的策略。
    """

    def __init__(self, board_size: int, lr=1e-4, use_se: bool = True, channels: int = 32, reduction: int = 16, hidden: int = 256, dropout: float = 0.2):
        """两层卷积结构 + 可选 SE。

        参数：
        - channels: 卷积输出通道数
        - reduction: SE 中 reduction
        - hidden: 全连接层中间维度
        - dropout: 全连接层中的 dropout 概率
        """
        self.use_se = use_se
        super().__init__()
        self.board_size = board_size
        """
        # 在此处定义你的神经网络结构。Torch 模块必须在初始化过程中注册。
        # 例如，你可以按如下方式定义卷积神经网络（CNN）结构：

        # self.conv_blocks = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=kernel_size, padding=padding),
        #     nn.MaxPool2d(kernel_size=kernel_size, padding=padding, stride=stride),
        #     nn.ReLU(),
        # )

        # 这里的 channels、kernel_size、padding 和 stride 是深度学习中的“超参数”。

        # 卷积之后，你可以使用 nn.Flatten() 将隐藏的二维表示展平为一维表示。
        # 然后可以使用全连接层得到 n**2 维的表示，其中每个数值表示“原始策略分数”，
        # （在下一步中需要对其进行额外的约束和处理）。

        # self.linear_blocks = nn.Sequential(
        #     nn.Linear(in_features=features, out_features=board_size ** 2),
        # )

        # 在得到 n**2 维表示之后，你仍然需要进行额外的数据处理，
        # 包括：
        # i) 确保所有对应非法动作的数值被设为 0（!!!!!最重要!!!!!）;
        # ii) 确保剩余数值满足归一化条件（即它们的和等于 1）。
        # 强烈不建议使用就地(in-place)操作，因为它们可能导致梯度计算失败。
        # 一个更稳妥的替代方案是采用不依赖就地修改的方法来实现目标。

        # 我们也鼓励你尝试更强大的模型和不同的技术，
        # 比如使用注意力模块、不同的激活函数，或者简单地调整超参数设置。
        """

        # BEGIN YOUR CODE
        kernel_size = 3
        padding = 1
        
        conv_layers = [
            # 卷积层1
            nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=kernel_size, padding=padding),
            nn.GELU(),  # 保留ReLU
            
            # 卷积层2
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
        ]

        # 保留SE模块（可选）
        if self.use_se:
            conv_layers.append(SEBlock(channels=channels, reduction=reduction))
        conv_layers.append(nn.Flatten())
        self.conv_blocks = nn.Sequential(*conv_layers)
        # 全连接层：Dropout + SiLU
        conv_output_dim = channels * board_size * board_size
        self.linear_blocks = nn.Sequential(
            nn.Linear(in_features=conv_output_dim, out_features=hidden),
            nn.GELU(),  
            nn.Dropout(dropout),  # 增加Dropout防止过拟合
            nn.Linear(in_features=hidden, out_features=board_size ** 2),
        )
        # END YOUR CODE

        # 在此处定义优化器，用于计算梯度并执行优化步骤。
        # 学习率 (lr) 是另一个需要预先确定的超参数。
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=lr)

    def forward(self, x: np.ndarray):
        if len(x.shape) == 2:
            output = torch.tensor(x).to(device).to(torch.float32).unsqueeze(0).unsqueeze(0)
        else:
            output = torch.tensor(x).to(device).to(torch.float32)
            if len(output.shape) == 3:
                output = output.unsqueeze(1)  # 补全通道维度


        # 在此处进一步处理和转换数据。确保输出形状为 (B, n ** 2)。
        # 我们已经将原始输入的形状统一为 (B, 1, N, N)，
        # 其中 B >= 1 表示批量大小，N = n 即棋盘的尺寸。

        # 你可以使用在初始化时注册的模块继续处理数据。例如：

        # output = self.conv_blocks(output)
        # output = nn.Flatten()(output)
        # output = self.linear_blocks(output)

        # 再次提醒：

        # ****************************************
        # 在得到 n**2 维表示之后，你仍然需要进行额外的数据处理，
        # 包括：
        # i) 确保所有对应非法动作的数值被设为 0（!!!!!最重要!!!!!）;
        # ii) 确保剩余数值满足归一化条件（即它们的和等于 1）。
        # 强烈不建议使用就地(in-place)操作，因为它们可能导致梯度计算失败。
        # ****************************************

        # BEGIN YOUR CODE
        # 前向计算：得到每个位置的 logits，然后屏蔽非法动作并归一化为概率分布
        features = self.conv_blocks(output)
        logits = self.linear_blocks(features)

        # 合法动作掩码（空位置为合法）
        state = output.squeeze(1)  # (B, N, N)
        mask = (state == 0).view(state.shape[0], -1)  # (B, N**2), bool

        # 若某一条样本没有合法动作（极端情况），将其视为全位置合法以避免数值问题
        mask_float = mask.float()
        mask_sum = mask_float.sum(dim=1, keepdim=True)
        mask = torch.where(mask_sum == 0, torch.ones_like(mask), mask)

        # 将非法位置的 logits 设为很小的值，避免 softmax 赋予其概率
        masked_logits = logits.masked_fill(~mask.bool(), float("-1e9"))

        probs = torch.softmax(masked_logits, dim=1)

        # 明确将非法位置概率置为 0 并对合法位置重新归一化（避免就地操作）
        eps_norm = 1e-8
        probs_masked = probs * mask_float  # (B, N**2)
        probs_sum = probs_masked.sum(dim=1, keepdim=True)
        probs = probs_masked / (probs_sum + eps_norm)

        output = probs
        # END YOUR CODE
        return output


class Critic(nn.Module):
    """
    Critic（价值网络）负责生成可靠的 Q 值以拟合 Bellman 方程的解。它接受一批状态数组（形状为 (B, 1, N, N) 或 (N, N)）
    和一批动作（形状为 (B, 2)）作为输入，并输出形状为 (B,) 的张量，表示指定 (s, a) 对应的 Q 值。

    例如，动作可以是：
    [[0, 1],
     [2, 3],
     [5, 6]]
    这表示模型分别在坐标 (0,1)、(2,3) 和 (5,6) 处落子。假设 n=12，这些动作与索引一一对应：0*12+1=1、2*12+3=27、5*12+6=66。
    你可以使用 _position_to_index 将一个位置转换为对应的索引，或使用 _index_to_position 做相反的转换。

    主要思路是：先得到一个形状为 (B, N ** 2) 的张量，表示在给定统一化状态（(B,1,N,N)）下所有可能动作的 Q 值，
    然后从该张量中抽取每个动作 (i,j) 对应的 Q 值（应充分利用 _position_to_index 获取动作索引）。

    最后返回形状为 (B,) 的张量，包含这些 Q 值。
    """

    def __init__(self, board_size: int, lr=1e-4, use_se: bool = True, channels: int = 32, reduction: int = 16, hidden: int = 256, dropout: float = 0.2):  # LR调整为1e-4
        super().__init__()
        self.board_size = board_size
        self.use_se = use_se
        self.channels = channels
        self.reduction = reduction
        self.hidden = hidden
        self.dropout = dropout
        # 同样在此处定义你的神经网络结构。Torch 模块必须在初始化过程中注册。

        # BEGIN YOUR CODE
        # 与 Actor 类似的卷积 + 全连接结构，用于生成每个位置的 Q 值（未筛除非法动作）
         # 两层卷积结构，与Actor保持一致
        kernel_size = 3
        padding = 1
        
        conv_layers = [
            # 卷积层1
            nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            
            # 卷积层2
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
        ]

        # 保留SE模块
        if self.use_se:
            conv_layers.append(SEBlock(channels=channels, reduction=reduction))
        conv_layers.append(nn.Flatten())
        self.conv_blocks = nn.Sequential(*conv_layers)
        # 全连接层：Dropout + SiLU
        conv_output_dim = channels * board_size * board_size
        self.linear_blocks = nn.Sequential(
            nn.Linear(in_features=conv_output_dim, out_features=hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=hidden, out_features=board_size ** 2),
        )
        # END YOUR CODE

        # 在此处定义优化器，用于计算梯度并执行优化步骤。
        # 学习率 (lr) 是另一个需要预先确定的超参数。
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=lr)

    def forward(self, x: np.ndarray, action: np.ndarray):
        # 将输入转为统一张量格式 (B, 1, N, N)
        if len(x.shape) == 2:
            output = torch.tensor(x).to(device).to(torch.float32).unsqueeze(0).unsqueeze(0)
        else:
            output = torch.tensor(x).to(device).to(torch.float32)
            if len(output.shape) == 3:
                output = output.unsqueeze(1)

        # BEGIN YOUR CODE
        # 计算每个位置的 Q 值（B, N**2）
        features = self.conv_blocks(output)
        q_all = self.linear_blocks(features)  # (B, N**2)

        # 处理动作索引（支持 numpy 数组 或 torch 张量）
        if isinstance(action, np.ndarray):
            acts = torch.tensor(action, dtype=torch.long, device=device)
        else:
            acts = action.to(device).long()

        indices = (acts[:, 0] * self.board_size + acts[:, 1]).long()  # (B,)

        q_vals = q_all[torch.arange(indices.size(0), device=device), indices]
        output = q_vals
        # END YOUR CODE

        return output


class GobangModel(nn.Module):
    """
    GobangModel 类将 Actor 和 Critic 两个模块整合用于计算和训练。给定状态张量 `x` 和动作张量 `action`，
    它直接返回 self.actor(x) 和 self.critic(x, action) 分别作为策略和 Q 值。
    """

    def __init__(self, board_size: int, bound: int, use_se: bool = True, channels: int = 32, reduction: int = 16, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.bound = bound
        self.board_size = board_size
        self.use_se = use_se
        self.channels = channels
        self.reduction = reduction
        self.hidden = hidden
        self.dropout = dropout

        """
        在此处注册 actor 和 critic 模块。此步骤无需进一步设计模块结构。
        如有需要，可在 Actor 或 Critic 的 __init__ 方法中添加额外的参数以方便使用。
        """

        # BEGIN YOUR CODE
        self.actor = Actor(board_size=board_size, use_se=self.use_se, channels=channels, reduction=reduction, hidden=hidden, dropout=dropout)
        self.critic = Critic(board_size=board_size, use_se=self.use_se, channels=channels, reduction=reduction, hidden=hidden, dropout=dropout)
        # END YOUR CODE

        self.to(device)

    def forward(self, x, action):
        """
        给定状态 `x` 和动作 `action`，返回策略向量 π(s) 和 Q 值 Q(s, a)。
        """
        return self.actor(x), self.critic(x, action)

    def optimize(self, policy, qs, actions, rewards, next_qs, gamma, eps=1e-6):
        """
        此函数计算 actor 和 critic 的损失。
        使用得到的损失，我们可以通过 actor.optimizer 和 critic.optimizer 应用优化算法，
        分别最大化 actor 的目标或最小化 critic 的损失。

        函数 "optimize" 中存在 3 个错误，导致模型无法正确执行优化。
        找出并修复所有错误。
        """

        # 目标 q 值和 critic 损失
        targets = rewards + gamma * next_qs
        q_clip = 100.0         # 可调
        targets = torch.clamp(targets, -q_clip, q_clip)
        critic_loss = nn.SmoothL1Loss()(qs, targets)

        # 计算动作索引（支持 torch 张量或 numpy）
        if isinstance(actions, torch.Tensor):
            acts = actions.to(device).long()
        else:
            acts = torch.tensor(actions, dtype=torch.long, device=device)
        indices = (acts[:, 0] * self.board_size + acts[:, 1]).long()

        # 选中动作的概率
        aimed_policy = policy[torch.arange(indices.size(0), device=device), indices]

        # actor loss: 原始策略梯度目标（未加入熵正则）
        actor_loss = -torch.mean(torch.log(aimed_policy + eps) * qs.clone().detach())

        # Bug3修复：分开优化Actor和Critic，确保step()调用
        # 优化Actor
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0) # 防止梯度爆炸
        self.actor.optimizer.step()

        # 优化Critic
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic.optimizer.step()

        return actor_loss, critic_loss


if __name__ == "__main__":
    if args.use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_name,
                config={
                    "num_episodes": num_episodes,
                    "checkpoint": checkpoint,
                    "board_size": 12,
                    "bound": 5,
                }
            )
            print("Wandb initialized successfully.")
        except ImportError:
            print("Warning: wandb not installed. Install with 'pip install wandb' to enable experiment tracking.")
            print("Continuing without wandb...")
    
    # 创建模型（默认启用SE模块）
    agent = GobangModel(board_size=12, bound=5, use_se=True).to(device)
    train_model(agent, num_episodes=num_episodes, checkpoint=checkpoint)
    
    if args.use_wandb:
        try:
            import wandb
            wandb.finish()
        except:
            pass
