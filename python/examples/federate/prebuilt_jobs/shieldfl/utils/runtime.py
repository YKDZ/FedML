import logging
import os
import random

import numpy as np
import torch


def configure_runtime(args):
    runtime_mode = getattr(args, "runtime_mode", "cpu-deterministic")
    enforce_determinism = bool(getattr(args, "enforce_determinism", True))
    seed = int(getattr(args, "random_seed", 0))

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if runtime_mode in ("cpu-deterministic", "single-gpu-deterministic"):
        if enforce_determinism:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True)
            except Exception as exc:
                logging.warning("Deterministic algorithms not fully enabled: %s", exc)
    elif runtime_mode == "single-gpu-fast":
        # GPU 高吞吐模式：关闭确定性以换取速度
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    setattr(args, "runtime_mode", runtime_mode)
    setattr(args, "sort_client_updates", bool(getattr(args, "sort_client_updates", True)))
    _print_runtime_summary(args, runtime_mode, seed)


def _print_runtime_summary(args, runtime_mode, seed):
    """启动时打印运行时上下文摘要，便于结果追溯。"""
    logging.info(
        "ShieldFL runtime configured: mode=%s seed=%s using_gpu=%s model=%s dataset=%s",
        runtime_mode,
        seed,
        getattr(args, "using_gpu", False),
        getattr(args, "model", "unknown"),
        getattr(args, "dataset", "unknown"),
    )
