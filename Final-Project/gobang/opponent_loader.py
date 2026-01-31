import torch.nn as nn
from typing import *
from utils import *
import numpy as np
import torch

board_size = 12
bound = 5


# Load models using functions 'get_model' without passing any extra
# parameters, so that we can directly call get_model() in player.py and evaluator.py.


def get_opponent():
    # 如果存在训练好的对手模型则载入，否则返回一个随机对手（随机噪声）
    import os
    ckpt_path = os.path.join(os.path.dirname(__file__), 'opponent.pth')
    if os.path.exists(ckpt_path):
        from submission import GobangModel
        opponent = GobangModel(board_size=board_size, bound=bound)
        opponent.load_state_dict(torch.load(ckpt_path, map_location=device))
        return opponent

    # 随机对手实现：actor(state) 返回在空位上的均匀概率分布，兼容现有接口
    class RandomOpponent:
        def __init__(self, board_size):
            self.board_size = board_size
        def eval(self):
            return
        def actor(self, state):
            # state: (N,N) 或 (1,N,N)
            if isinstance(state, np.ndarray):
                s = state
            else:
                s = state.cpu().numpy() if hasattr(state, 'cpu') else np.array(state)
            if s.ndim == 3:
                s = s[0]
            flat = (s == 0).astype(float).ravel()
            if flat.sum() == 0:
                flat = np.ones_like(flat)
            probs = flat / flat.sum()
            return torch.tensor([probs], device=device, dtype=torch.float32)

    return RandomOpponent(board_size)


__all__ = ['get_opponent']
