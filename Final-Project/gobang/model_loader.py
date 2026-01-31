import torch.nn as nn
from typing import *
from utils import *
import numpy as np
import torch
import os

board_size = 12
bound = 5


# Load models using functions 'get_model' without passing any extra
# parameters, so that we can directly call get_model() in player.py and evaluator.py.


def get_model():
    from submission import GobangModel
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'model_2999_7.pth')

    if not os.path.exists(ckpt_path):
        print(f"Warning: checkpoint not found at {ckpt_path}. Returning default model (no weights).")
        return GobangModel(board_size=board_size, bound=bound).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)

    # Parse checkpoint format: may include arch metadata
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state = ckpt['state_dict']
        arch = ckpt.get('arch', None)
    else:
        state = ckpt
        arch = None

    def infer_arch_from_state(sd: dict):
        # Infer channels from first conv weight, hidden from first linear out_features, detect use_se
        ch = None
        hidden = None
        use_se = False
        if 'actor.conv_blocks.0.weight' in sd:
            ch = sd['actor.conv_blocks.0.weight'].shape[0]
        elif any(k.startswith('actor.conv_blocks') and k.endswith('.weight') for k in sd.keys()):
            # fallback: pick first conv weight found
            for k in sd.keys():
                if k.startswith('actor.conv_blocks') and k.endswith('.weight'):
                    ch = sd[k].shape[0]
                    break
        if 'actor.linear_blocks.0.weight' in sd:
            hidden = sd['actor.linear_blocks.0.weight'].shape[0]
        # detect SE by searching for fc layers within conv_blocks
        use_se = any(k.startswith('actor.conv_blocks') and '.fc.' in k for k in sd.keys())
        return {'channels': ch or 64, 'hidden': hidden or 512, 'use_se': use_se}

    # If arch metadata present, construct model accordingly. Otherwise infer from state_dict.
    def _safe_load(model, state_dict):
        """Try loading and fall back to a filtered/converted state_dict when sizes mismatch.
        - If a checkpoint param is 1-D and target is 3-D with matching channel dim, broadcast it.
        - Otherwise skip keys with incompatible shapes and perform a non-strict load.
        Returns a tuple (loaded: bool, info: dict)
        """
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
                    # Allow broadcasting of 1-D (C,) to (C,H,W) for norm params
                    if v.dim() == 1 and tgt.dim() >= 3 and v.shape[0] == tgt.shape[0]:
                        try:
                            new_v = v.view(-1, 1, 1).expand_as(tgt).clone()
                            filtered[k] = new_v
                            converted.append(k)
                            continue
                        except Exception:
                            skipped.append((k, v.shape, tgt.shape))
                            continue
                    # otherwise skip incompatible shapes
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
        arch_use_se = arch.get('use_se', True)
        arch_channels = arch.get('channels', 64) or 64
        arch_hidden = arch.get('hidden', 512) or 512
        arch_reduction = arch.get('reduction', 16) or 16
        arch_dropout = arch.get('dropout', 0.2) or 0.2
        model = GobangModel(board_size=board_size, bound=bound, use_se=arch_use_se, channels=arch_channels, reduction=arch_reduction, hidden=arch_hidden, dropout=arch_dropout).to(device)
        ok, info = _safe_load(model, state)
        if ok:
            print(f"Loaded checkpoint {ckpt_path} with arch metadata (method={info['method']}).")
            if info.get('skipped'):
                print(f"Skipped keys due to incompatibility: {info['skipped']}")
            if info.get('converted'):
                print(f"Converted keys by broadcasting: {info['converted'][:10]}{'...' if len(info['converted'])>10 else ''}")
            return model
        else:
            print(f"Failed to load checkpoint with arch metadata: {info.get('error')}")
            # fall through to try inference/fallback

    # Try constructing model by inferring arch from state_dict
    inferred = infer_arch_from_state(state)
    print(f"Inferred arch from checkpoint: {inferred}")
    model = GobangModel(board_size=board_size, bound=bound, use_se=inferred['use_se'], channels=inferred['channels'], hidden=inferred['hidden']).to(device)
    try:
        model.load_state_dict(state)
        print(f"Loaded checkpoint {ckpt_path} after inferring arch (strict load).")
        return model
    except RuntimeError as e:
        print(f"Strict load failed after inference: {e}")
        # try non-strict load
        try:
            res = model.load_state_dict(state, strict=False)
            missing = getattr(res, 'missing_keys', None)
            unexpected = getattr(res, 'unexpected_keys', None)
            print(f"Non-strict load after inference. Missing: {missing}; Unexpected: {unexpected}")
            matched = len(model.state_dict()) - (len(missing) if missing else 0)
            if matched > 0:
                print(f"Using model after non-strict load (matched {matched} keys).")
                return model
        except RuntimeError as e2:
            print(f"Non-strict load also failed: {e2}")

    # Fallback
    print("Warning: failed to cleanly load checkpoint. Returning uninitialized model.")
    return GobangModel(board_size=board_size, bound=bound).to(device)



__all__ = ['get_model']
