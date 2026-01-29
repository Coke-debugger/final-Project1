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

        #########################请在此处实现代码#########################
        # raise NotImplementedError("Not implemented")  # 移除这行注释以实现代码
        
        # Step 1: 生成一个完整的回合数据：数组形式，每个元素是 (状态, 动作, 奖励) 元组
        episode = []  # 存储回合的(状态, 动作, 奖励)
        state = env.reset()  # 重置环境，获取初始状态
        while True:
            # 根据当前策略选择动作（按概率采样）
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            # 执行动作，获取下一个状态、奖励和是否结束
            next_state, reward, done, _ = env.step(action)
            # 将(状态, 动作, 奖励)存入回合列表
            episode.append((state, action, reward))
            if done:  # 回合结束则退出循环
                break
            state = next_state  # 更新状态
        
        # Step 2: 找到每个（状态-动作）对在回合中首次出现的索引
        sa_first_visit = {}  # 存储(状态, 动作) -> 首次出现的时间步
        for t, (state, action, reward) in enumerate(episode):
            sa_pair = (state, action)
            if sa_pair not in sa_first_visit:
                sa_first_visit[sa_pair] = t
        
        # Step 3: 反向计算回报，仅在首次访问的时间步更新Q值
        G = 0  # 累计回报
        # 从后往前遍历回合（t从最后一步到0）
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            sa_pair = (state, action)
            # 累计回报 = 即时奖励 + 折扣因子 * 之前的累计回报
            G = reward + discount_factor * G
            
            # 仅当当前时间步是该（状态-动作）对的首次访问时，更新Q值
            if t == sa_first_visit.get(sa_pair):
                returns_sum[sa_pair] += G  # 累加回报总和
                returns_count[sa_pair] += 1  # 累加计数
                # 更新Q值：平均值 = 总回报 / 计数
                Q[state][action] = returns_sum[sa_pair] / returns_count[sa_pair]
        #########################代码实现结束#########################
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

        #########################请在此处实现代码#########################
        # raise NotImplementedError("Not implemented")  # 移除这行注释以实现代码
        
        # Step 1: 生成一个完整的回合数据
        episode = []
        state = env.reset()
        while True:
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state
        
        # Step 2: 计算每个（状态-动作）对的回报（每次访问法）
        G = 0  # 累计回报
        # 从后往前遍历回合
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            sa_pair = (state, action)
            # 累计回报 = 即时奖励 + 折扣因子 * 之前的累计回报
            G = reward + discount_factor * G
            
            # 每次访问到该（状态-动作）对，都更新回报总和和计数
            returns_sum[sa_pair] += G
            returns_count[sa_pair] += 1
            # 更新Q值：平均值 = 总回报 / 计数
            Q[state][action] = returns_sum[sa_pair] / returns_count[sa_pair]
        #########################代码实现结束#########################

    return Q, policy

if __name__ == "__main__":
    # 首次访问蒙特卡洛
    Q, policy = mc_first_visit(env, num_episodes=500000, epsilon=0.1)
    # 计算状态价值函数V（每个状态的最大动作价值）
    V = defaultdict(float)
    for state, actions in Q.items():
        V[state] = np.max(actions)
    # 绘制价值函数并保存图片
    plotting.plot_value_function(V, title="最优价值函数", 
        file_name="First_Visit_Value_Function_Episodes_500000")
    
    # 每次访问蒙特卡洛（取消注释可运行）
    # Q, policy = mc_every_visit(env, num_episodes=10000, epsilon=0.1)
    # V = defaultdict(float)
    # for state, actions in Q.items():
    #     V[state] = np.max(actions)
    # plotting.plot_value_function(V, title="最优价值函数", 
    #     file_name="Every_Visit_Value_Function_Episodes_10000")