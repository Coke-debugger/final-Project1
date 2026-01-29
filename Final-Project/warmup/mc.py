import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
matplotlib.rcParams['axes.unicode_minus'] = False    # 解决负号显示异常问题
import numpy as np
import sys
import time
from collections import defaultdict

from envs import BlackjackEnv  # 导入二十一点游戏环境
import plotting  # 导入绘图工具
matplotlib.style.use('ggplot')  # 设置matplotlib绘图风格

# 初始化二十一点游戏环境
env = BlackjackEnv()

def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    基于给定的Q函数和ε值，创建ε-贪心策略。
    
    参数:
        Q: 字典，映射关系为 状态 -> 动作价值数组。
            每个值是长度为nA的numpy数组（见下文）
        epsilon: 选择随机动作的概率，0到1之间的浮点数。
        nA: 环境中的动作数量。
    
    返回:
        一个函数，接收观测（状态）作为参数，返回
        每个动作的概率（长度为nA的numpy数组）。
    """
    def policy_fn(observation):
        # 初始化所有动作的概率为 ε/nA
        A = np.ones(nA, dtype=float) * epsilon / nA
        # 找到当前状态下Q值最大的最优动作
        best_action = np.argmax(Q[observation])
        # 给最优动作增加 (1 - ε) 的概率
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def mc_first_visit(env, num_episodes, discount_factor=1.0, epsilon=0.1):
    """
    基于ε-贪心策略的蒙特卡洛控制（首次访问法）。
    求解最优的ε-贪心策略。
    
    参数:
        env: OpenAI Gym环境（此处为二十一点环境）。
        num_episodes: 要采样的回合数。
        discount_factor: 折扣因子γ。
        epsilon: 随机选择动作的概率，0到1之间的浮点数。
    
    返回:
        一个元组 (Q, policy)。
        Q: 字典，映射关系为 状态 -> 动作价值。
        policy: 函数，接收观测（状态）作为参数，返回动作概率。
    """
    
    # 跟踪每个（状态-动作）对的回报总和与计数，用于计算平均值
    # 相比保存所有回报（如教材中的方式），这种方式更节省内存
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)  # 计数用int类型更合理
    
    # 最终的动作价值函数
    # 嵌套字典：状态 -> (动作 -> 动作价值)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # 我们当前遵循的策略（ε-贪心）
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    
    for i_episode in range(1, num_episodes + 1):
        # 打印当前回合数，方便调试
        if i_episode % 1000 == 0:
            print("\r回合 {}/{}。".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        #########################Implement your code here#########################
        raise NotImplementedError("Not implemented")
        # Step 1: Generate an episode: an array of (state, action, reward) tuples

        # Step 2: Find first-visit index for each (state, action) pair

        # Step 3: Calculate returns backward, update only at first-visit time step
        #########################Implement your code end#########################
    return Q, policy


def mc_every_visit(env, num_episodes, discount_factor=1.0, epsilon=0.1):
    """
    基于ε-贪心策略的蒙特卡洛控制（每次访问法）。
    求解最优的ε-贪心策略。
    """
    
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)  # 计数用int类型更合理
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    
    for i_episode in range(1, num_episodes + 1):
        if i_episode % 1000 == 0:
            print("\r回合 {}/{}。".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        #########################Implement your code here#########################
        raise NotImplementedError("Not implemented")
        # Step 1: Generate an episode
        
        # Step 2: Calculate returns for each (state, action) pair (every-visit)
        
        #########################Implement your code end#########################

    return Q, policy

if __name__ == "__main__":
    # First-Visit Monte Carlo
    Q, policy = mc_first_visit(env, num_episodes=10000, epsilon=0.1)
    V = defaultdict(float)
    for state, actions in Q.items():
        V[state] = np.max(actions)
    # 绘制价值函数并保存图片
    plotting.plot_value_function(V, title="最优价值函数", 
        file_name="First_Visit_Value_Function_Episodes_500000")
    
    # Every-Visit Monte Carlo
    # Q, policy = mc_every_visit(env, num_episodes=10000, epsilon=0.1)
    # V = defaultdict(float)
    # for state, actions in Q.items():
    #     V[state] = np.max(actions)
    # plotting.plot_value_function(V, title="Optimal Value Function", 
    #     file_name="Every_Visit_Value_Function_Episodes_10000")