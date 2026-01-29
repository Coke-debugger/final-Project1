import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
matplotlib.rcParams['axes.unicode_minus'] = False    # 解决负号显示异常问题
import numpy as np
import sys

from collections import defaultdict
from envs import CliffWalkingEnv  
import plotting  
matplotlib.style.use('ggplot')  # 设置matplotlib绘图风格为ggplot

# agent需从起点走到终点，避开悬崖
env = CliffWalkingEnv()

def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    基于给定的Q函数和ε值，创建ε-贪心策略。
    ε-贪心策略核心：以1-ε的概率选择当前最优动作（Q值最大），以ε的概率随机选动作，平衡探索与利用。
    
    参数:
        Q: 字典，映射关系为「状态 → 动作价值数组」。
            每个值是长度为nA的numpy数组（nA为动作数）
        epsilon: 选择随机动作的概率，0到1之间的浮点数。
        nA: 环境中的动作总数（悬崖行走环境中为4：上/下/左/右）。
    
    返回:
        一个策略函数，接收观测（状态）作为输入，返回
        每个动作的选择概率（长度为nA的numpy数组）。
    """
    def policy_fn(observation):
        # 初始化所有动作的概率为 ε/nA（随机探索部分）
        A = np.ones(nA, dtype=float) * epsilon / nA
        # 找到当前状态下Q值最大的最优动作
        best_action = np.argmax(Q[observation])
        # 给最优动作增加 (1 - ε) 的概率（利用部分）
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def sarsa(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1, max_steps=10000):
    """
    SARSA算法实现：在线（On-policy）时序差分（TD）控制算法。
    核心特点：仅使用当前策略采样的「状态-动作-奖励-下一状态-下一动作」序列更新Q值，
    最终收敛到最优的ε-贪心策略。
    
    参数:
        env: OpenAI Gym环境（此处为悬崖行走环境）。
        num_episodes: 要采样的训练回合数。
        discount_factor: 折扣因子γ，用于计算未来回报的现值。
        alpha: TD学习率，控制Q值更新的步长（0<α≤1）。
        epsilon: 随机选择动作的概率，0到1之间的浮点数（ε-贪心策略参数）。
        max_steps: 每个回合的最大步数（安全限制，避免无限循环）。
    
    返回:
        一个元组 (Q, stats)。
        Q: 最优动作价值函数，字典映射「状态 → 动作价值数组」。
        stats: EpisodeStats对象，包含两个numpy数组：
            episode_lengths: 每个回合的步数
            episode_rewards: 每个回合的总奖励
    """
    
    # 最终的动作价值函数（嵌套字典：状态 → (动作 → 动作价值)）
    # 初始化为默认值：每个状态的所有动作价值都是0
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # 跟踪训练过程中的关键统计信息（用于后续绘图）
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),  # 记录每个回合的步数
        episode_rewards=np.zeros(num_episodes))  # 记录每个回合的总奖励

    # 构建当前遵循的ε-贪心策略（策略会随Q值更新动态变化）
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    
    for i_episode in range(num_episodes):
        # 打印当前训练进度（每100回合更新一次），方便调试
        if (i_episode + 1) % 100 == 0:
            print("\r回合 {}/{}。".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()

        # 重置环境，获取初始状态（每个回合开始时回到起点）
        state = env.reset()
        
        # 选择初始动作（关键：SARSA需先选动作，再进入循环）
        action_probs = policy(state)  # 获取当前状态的动作概率分布
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)  # 按概率采样动作
        
        # 执行当前回合的交互（带max_steps安全限制，防止死循环）
        for t in range(max_steps):
            #########################请在此处实现核心代码#########################
            # raise NotImplementedError("Not implemented")  
            
            # Step 1: 执行当前动作，获取环境反馈
            next_state, reward, done, _ = env.step(action)
            
            # Step 2: 选择下一个动作（SARSA核心：必须基于当前策略选下一动作）
            next_action_probs = policy(next_state)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)
            
            # Step 3: TD误差计算与Q值更新（SARSA核心公式）
            # TD目标：即时奖励 + 折扣因子 * 下一状态-下一动作的Q值
            td_target = reward + discount_factor * Q[next_state][next_action]
            # TD误差：目标值 - 当前Q值
            td_error = td_target - Q[state][action]
            # 更新当前状态-动作的Q值：Q(s,a) ← Q(s,a) + α*TD误差
            Q[state][action] += alpha * td_error
            
            # Step 4: 更新统计信息
            stats.episode_rewards[i_episode] += reward  # 累计当前回合奖励
            stats.episode_lengths[i_episode] = t + 1    # 记录当前回合已走步数
            
            # Step 5: 终止条件判断（回合结束则退出循环）
            if done:
                break
            
            # Step 6: 移动到下一个状态-动作对（为下一轮循环做准备）
            state = next_state
            action = next_action
            #########################核心代码实现结束#########################
    return Q, stats

if __name__ == '__main__':
    # 训练SARSA算法（1000个回合）
    Q, stats = sarsa(env, 1000)
    # 绘制回合统计信息（奖励/步数变化曲线）并保存图片
    plotting.plot_episode_stats(stats, file_name='episode_stats_sarsa')