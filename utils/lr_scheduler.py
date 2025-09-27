import numpy as np


def linear_warmup_cosine_decay(warmup_steps: int, total_steps: int):
    def fn(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))

        progress = float(step - warmup_steps) / float(max(
            1, total_steps - warmup_steps))

        # cosine decay
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return fn
