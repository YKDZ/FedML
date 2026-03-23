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

    if enforce_determinism:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as exc:
            logging.warning("Deterministic algorithms not fully enabled: %s", exc)

    setattr(args, "runtime_mode", runtime_mode)
    setattr(args, "sort_client_updates", bool(getattr(args, "sort_client_updates", True)))
    logging.info(
        "ShieldFL runtime configured: mode=%s seed=%s using_gpu=%s",
        runtime_mode,
        seed,
        getattr(args, "using_gpu", False),
    )
