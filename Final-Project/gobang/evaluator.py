from utils import *
from model_loader import get_model
from opponent_loader import get_opponent

if __name__ == "__main__":
    # 定义游戏设置。
    board_size = 12
    bound = 5
    num_episodes = 1000

    # 载入训练好的模型（黑子）和对手模型（白子）。
    model = get_model()
    opponent = get_opponent()

    model.eval()
    opponent.eval()

    # 开始评估过程。
    chess_board = Gobang(board_size=board_size, bound=bound, training=False)

    # 开始使用随机噪声进行测试（设置 random_response=True），
    # 或者使用另一个训练好的模型进行测试（设置 random_response=False）。
    # 请确认模型（表示黑子）和对手模型（表示白子）在评估前均已加载。
    chess_board.evaluate_agent_performance(random_response=False, model=model, opponent=opponent, episodes=num_episodes)
