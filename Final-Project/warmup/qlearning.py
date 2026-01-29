# 导入matplotlib库，用于数据可视化
import matplotlib
# 导入numpy库，用于数值计算和数组操作
import numpy as np
# 导入sys库，用于系统相关的功能，如标准输出刷新
import sys

# 从collections模块导入defaultdict，用于创建默认字典
# defaultdict允许为不存在的键提供默认值
from collections import defaultdict
# 从envs模块导入CliffWalkingEnv，这是悬崖行走环境的实现
from envs import CliffWalkingEnv
# 导入自定义的plotting模块，用于绘制统计图表
import plotting

# 设置matplotlib使用ggplot样式，这是一种美观的绘图风格
matplotlib.style.use('ggplot')

# 创建悬崖行走环境的实例
# CliffWalkingEnv是一个经典的网格世界环境，用于强化学习算法测试
env = CliffWalkingEnv()

def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    创建一个基于给定Q函数和ε值的ε-贪婪策略。
    ε-贪婪策略平衡了探索（随机选择动作）和利用（选择当前最佳动作）。
    
    参数:
        Q: 一个字典，将状态映射到动作价值数组。
            每个值是一个长度为nA的numpy数组
        epsilon: 随机选择动作的概率。0到1之间的浮点数。
        nA: 环境中的动作数量。
    
    返回:
        一个函数，接收观测（状态）作为参数，返回
        每个动作的概率，形式为长度为nA的numpy数组。
    """
    def policy_fn(observation):
        # 初始化动作概率数组，每个动作的初始概率为ε/nA
        # 这确保了总探索概率为ε（ε/nA * nA = ε）
        A = np.ones(nA, dtype=float) * epsilon / nA
        
        # 找到当前状态下具有最高Q值的动作（最佳动作）
        # np.argmax返回最大值的索引
        best_action = np.argmax(Q[observation])
        
        # 将(1-ε)的概率加到最佳动作上
        # 这确保了最佳动作的总概率为ε/nA + (1-ε)
        A[best_action] += (1.0 - epsilon)
        
        # 返回动作概率分布数组
        # 数组元素之和为1：nA*(ε/nA) + (1-ε) = ε + (1-ε) = 1
        return A
    
    # 返回策略函数
    return policy_fn


def make_epsilon_greedy_policy_double(Q1, Q2, epsilon, nA):
    """
    为Double Q-learning创建基于Q1+Q2的ε-贪婪策略。
    Double Q-learning使用两个独立的Q函数来减少最大化偏差。
    
    参数:
        Q1: 第一个Q函数字典。
        Q2: 第二个Q函数字典。
        epsilon: 随机选择动作的概率。0到1之间的浮点数。
        nA: 环境中的动作数量。
    
    返回:
        一个函数，接收观测（状态）作为参数，返回
        每个动作的概率，形式为长度为nA的numpy数组。
    """
    def policy_fn(observation):
        # 使用Q1 + Q2的组合值进行动作选择（更稳定和标准的方法）
        # 将两个Q函数的值相加，得到更稳健的动作价值估计
        combined_Q = Q1[observation] + Q2[observation]
        
        # 初始化动作概率数组，每个动作的初始概率为ε/nA
        A = np.ones(nA, dtype=float) * epsilon / nA
        
        # 在组合Q值中找到最佳动作
        best_action = np.argmax(combined_Q)
        
        # 将(1-ε)的概率加到最佳动作上
        A[best_action] += (1.0 - epsilon)
        
        # 返回动作概率分布数组
        return A
    
    # 返回策略函数
    return policy_fn


def q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1, max_steps=10000):
    """
    Q-Learning算法: 离线策略TD控制。
    在遵循ε-贪婪策略的同时，找到最优的贪婪策略
    
    参数:
        env: OpenAI环境。
        num_episodes: 运行的回合数。
        discount_factor: Gamma折扣因子。
        alpha: TD学习率。
        epsilon: 随机选择动作的概率。0到1之间的浮点数。
        max_steps: 每个回合的最大步数（安全限制）。
    
    返回:
        一个元组 (Q, stats)。
        Q是最优动作价值函数，一个将状态映射到动作值的字典。
        stats是EpisodeStats对象，包含回合长度和回合奖励的两个numpy数组。
    """
    
    # 最终的动作价值函数
    # 一个嵌套字典，映射状态 -> (动作 -> 动作价值)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # 跟踪有用的统计数据
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes)
    )    
    
    # 我们正在遵循的策略
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    
    for i_episode in range(num_episodes):
        # 打印当前回合数，便于调试
        if (i_episode + 1) % 100 == 0:
            print("\r回合 {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()
        
        # 重置环境
        state = env.reset()

        # 在环境中执行一步（有max_steps安全限制）
        for t in range(max_steps):
            #########################请在此处实现代码#########################
            # 步骤1: 执行一步动作
            # 根据当前策略获取动作概率分布
            action_probs = policy(state)
            # 根据概率分布随机选择动作
            action = np.random.choice(np.arange(env.action_space.n), p=action_probs)
            # 执行动作，获取下一个状态、奖励和是否结束
            next_state, reward, done, _ = env.step(action)
            
            # 更新统计数据
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t + 1
            
            # 步骤2: TD更新（处理终止状态）
            # Q-learning更新公式: Q(s,a) ← Q(s,a) + α[r + γ * maxₐ' Q(s',a') - Q(s,a)]
            if not done:
                # 非终止状态的TD目标：即时奖励 + 折扣因子 * 下一个状态的最大Q值
                td_target = reward + discount_factor * np.max(Q[next_state])
            else:
                # 终止状态的TD目标：只有即时奖励（没有下一个状态）
                td_target = reward
            
            # 计算TD误差：目标值与当前值的差异
            td_delta = td_target - Q[state][action]
            # 更新Q值：当前值 + 学习率 * TD误差
            Q[state][action] += alpha * td_delta
            
            # 步骤3: 转移到下一个状态并处理回合结束
            state = next_state
            
            # 如果回合结束，跳出循环
            if done:
                break
            #########################代码实现结束#########################
    return Q, stats


def double_q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1, max_steps=10000):
    """
    Double Q-Learning算法: 减少最大化偏差的Q-learning变体。
    维护两个Q函数来解耦动作选择和值评估。
    
    参数:
        env: OpenAI环境。
        num_episodes: 运行的回合数。
        discount_factor: Gamma折扣因子。
        alpha: TD学习率。
        epsilon: 随机选择动作的概率。0到1之间的浮点数。
        max_steps: 每个回合的最大步数（安全限制）。
    
    返回:
        一个元组 (Q1, Q2, stats)。
        Q1, Q2是两个Q函数。
        stats是EpisodeStats对象，包含回合长度和回合奖励的两个numpy数组。
    """
    
    # 初始化两个独立的Q函数
    Q1 = defaultdict(lambda: np.zeros(env.action_space.n))
    Q2 = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # 跟踪有用的统计数据
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))
    
    # 使用Q1+Q2的组合作为行为策略（标准Double Q-learning方法）
    policy = make_epsilon_greedy_policy_double(Q1, Q2, epsilon, env.action_space.n)

    for i_episode in range(num_episodes):
        if (i_episode + 1) % 100 == 0:
            print("\r回合 {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()

        # 重置环境
        state = env.reset()

        for t in range(max_steps):
            #########################请在此处实现代码#########################       
            # 步骤1: 使用组合的Q1+Q2策略执行一步动作
            # 根据当前策略获取动作概率分布
            action_probs = policy(state)
            # 根据概率分布随机选择动作
            action = np.random.choice(np.arange(env.action_space.n), p=action_probs)
            # 执行动作，获取下一个状态、奖励和是否结束
            next_state, reward, done, _ = env.step(action)
            
            # 更新统计数据
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t + 1
            
            # 步骤2: Double Q-learning更新
            # 随机选择更新哪个Q函数（各50%概率）
            if np.random.rand() < 0.5:
                # 更新Q1，使用Q2评估下一个状态的最佳动作值
                # 根据Q1选择下一个状态的最佳动作
                best_action_q1 = np.argmax(Q1[next_state])
                # TD目标：使用Q2评估Q1选择的最佳动作值
                if not done:
                    td_target = reward + discount_factor * Q2[next_state][best_action_q1]
                else:
                    td_target = reward
                # 计算TD误差并更新Q1
                td_delta = td_target - Q1[state][action]
                Q1[state][action] += alpha * td_delta
            else:
                # 更新Q2，使用Q1评估下一个状态的最佳动作值
                # 根据Q2选择下一个状态的最佳动作
                best_action_q2 = np.argmax(Q2[next_state])
                # TD目标：使用Q1评估Q2选择的最佳动作值
                if not done:
                    td_target = reward + discount_factor * Q1[next_state][best_action_q2]
                else:
                    td_target = reward
                # 计算TD误差并更新Q2
                td_delta = td_target - Q2[state][action]
                Q2[state][action] += alpha * td_delta
            
            # 步骤3: 转移到下一个状态并处理回合结束
            state = next_state
            
            # 如果回合结束，跳出循环
            if done:
                break
            #########################代码实现结束#########################
                
    return Q1, Q2, stats


if __name__ == '__main__':
    # Q-Learning
    Q, stats = q_learning(env, 1000)
    plotting.plot_episode_stats(stats, file_name='episode_stats_q_learning')
    
    # Double Q-Learning
    # Q1, Q2, stats = double_q_learning(env, 1000)
    # plotting.plot_episode_stats(stats, file_name='episode_stats_double_q_learning')
