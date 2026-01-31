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
    ckpt_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'model_2999.pth')

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

        def _safe_load(model, state_dict):
            try:
                model.load_state_dict(state_dict)
                return True, {'method': 'strict', 'skipped': [], 'converted': []}
            except RuntimeError as e:
                print(f"Strict load failed: {e}")
                model_sd = model.state_dict()
                filtered = {}
                skipped = []
                converted = []
                for k, v in state_dict.items():
                    if k not in model_sd:
                        skipped.append((k, v.shape, None))
                        continue
                    tgt = model_sd[k]
                    if v.shape == tgt.shape:
                        filtered[k] = v
                    else:
                        if v.dim() == 1 and tgt.dim() >= 3 and v.shape[0] == tgt.shape[0]:
                            try:
                                new_v = v.view(-1, 1, 1).expand_as(tgt).clone()
                                filtered[k] = new_v
                                converted.append(k)
                                continue
                            except Exception:
                                skipped.append((k, v.shape, tgt.shape))
                                continue
                        skipped.append((k, v.shape, tgt.shape))
                try:
                    res = model.load_state_dict(filtered, strict=False)
                    missing = getattr(res, 'missing_keys', None)
                    unexpected = getattr(res, 'unexpected_keys', None)
                    info = {'method': 'filtered_non_strict', 'skipped': skipped, 'converted': converted, 'missing': missing, 'unexpected': unexpected}
                    return True, info
                except RuntimeError as e2:
                    print(f"Filtered non-strict load also failed: {e2}")
                    return False, {'error': str(e2)}

        if arch is not None:
            opponent = GobangModel(board_size=board_size, bound=bound, use_se=arch.get('use_se', True), channels=arch.get('channels', 64) or 64, reduction=arch.get('reduction', 16) or 16, hidden=arch.get('hidden', 512) or 512, dropout=arch.get('dropout', 0.2) or 0.2)
            ok, info = _safe_load(opponent, state)
            if ok:
                print(f"Loaded opponent checkpoint with arch metadata (method={info['method']}).")
                if info.get('skipped'):
                    print(f"Skipped keys due to incompatibility: {info['skipped']}")
                if info.get('converted'):
                    print(f"Converted keys by broadcasting: {info['converted'][:10]}{'...' if len(info['converted'])>10 else ''}")
                return opponent
        # fallback to inference
        inferred = infer_arch_from_state(state)
        print(f"Inferred arch for opponent: {inferred}")
        opponent = GobangModel(board_size=board_size, bound=bound, use_se=inferred['use_se'], channels=inferred['channels'], hidden=inferred['hidden'])
        ok, info = _safe_load(opponent, state)
        if ok:
            print(f"Loaded opponent checkpoint after inference (method={info['method']}).")
            if info.get('skipped'):
                print(f"Skipped keys due to incompatibility: {info['skipped']}")
            if info.get('converted'):
                print(f"Converted keys by broadcasting: {info['converted'][:10]}{'...' if len(info['converted'])>10 else ''}")
            return opponent
        else:
            print(f"Failed to load opponent checkpoint after inference: {info.get('error')}")


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
