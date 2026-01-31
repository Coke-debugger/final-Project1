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
    import os
    ckpt_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'model_2999_11.pth')

    if os.path.exists(ckpt_path):
        from submission import GobangModel
        ckpt = torch.load(ckpt_path, map_location=device)
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            state = ckpt['state_dict']
            arch = ckpt.get('arch', None)
        else:
            state = ckpt
            arch = None

        def infer_arch_from_state(sd: dict):
            ch = None
            hidden = None
            use_se = False
            if 'actor.conv_blocks.0.weight' in sd:
                ch = sd['actor.conv_blocks.0.weight'].shape[0]
            if 'actor.linear_blocks.0.weight' in sd:
                hidden = sd['actor.linear_blocks.0.weight'].shape[0]
            use_se = any(k.startswith('actor.conv_blocks') and '.fc.' in k for k in sd.keys())
            return {'channels': ch or 64, 'hidden': hidden or 512, 'use_se': use_se}

        if arch is not None:
            opponent = GobangModel(board_size=board_size, bound=bound, use_se=arch.get('use_se', True), channels=arch.get('channels', 64) or 64, reduction=arch.get('reduction', 16) or 16, hidden=arch.get('hidden', 512) or 512, dropout=arch.get('dropout', 0.2) or 0.2)
            try:
                opponent.load_state_dict(state)
                print(f"Loaded opponent checkpoint with arch metadata (strict load).")
                return opponent
            except RuntimeError as e:
                print(f"Strict load failed with arch metadata: {e}")
        # fallback to inference
        inferred = infer_arch_from_state(state)
        print(f"Inferred arch for opponent: {inferred}")
        opponent = GobangModel(board_size=board_size, bound=bound, use_se=inferred['use_se'], channels=inferred['channels'], hidden=inferred['hidden'])
        try:
            opponent.load_state_dict(state)
            print(f"Loaded opponent checkpoint after inference (strict load).")
            return opponent
        except RuntimeError as e:
            print(f"Strict load failed after inference: {e}")
            res = opponent.load_state_dict(state, strict=False)
            missing = getattr(res, 'missing_keys', None)
            unexpected = getattr(res, 'unexpected_keys', None)
            print(f"Non-strict load after inference. Missing: {missing}; Unexpected: {unexpected}")
            matched = len(opponent.state_dict()) - (len(missing) if missing else 0)
            if matched > 0:
                print(f"Using opponent model after non-strict load (matched {matched} keys).")
                return opponent

    # 若找不到模型文件或加载失败，返回一个随机对手实现：在所有空位上均匀采样
    class RandomOpponent:
        def __init__(self, board_size):
            self.board_size = board_size
        def eval(self):
            return
        def actor(self, state):
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
