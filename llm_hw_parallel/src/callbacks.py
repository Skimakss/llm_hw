import time

import torch.distributed as dist
from transformers import TrainerCallback


class TimeoutCallback(TrainerCallback):
    """Stop training after a specified timeout."""

    def __init__(self, timeout_seconds, check_every_n_steps=1):
        self.timeout_seconds = float(timeout_seconds)
        self.check_every_n_steps = int(check_every_n_steps)
        self.start_time = None
        self.step = 0
        self.is_dist = False
        self.rank = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.monotonic()
        self.is_dist = dist.is_available() and dist.is_initialized()
        if self.is_dist:
            self.rank = dist.get_rank()

    def on_step_end(self, args, state, control, **kwargs):
        self.step += 1
        if self.step % self.check_every_n_steps != 0:
            return control

        local_stop = False
        if self.rank == 0:
            local_stop = (time.monotonic() - self.start_time) > self.timeout_seconds

        if self.is_dist:
            flag = [local_stop]
            dist.broadcast_object_list(flag, src=0)
            should_stop = bool(flag[0])
        else:
            should_stop = local_stop

        if should_stop:
            control.should_training_stop = True
            if (not self.is_dist) or self.rank == 0:
                elapsed = time.monotonic() - self.start_time
                print(f"Training stopped after {elapsed:.2f}s (timeout)")

        return control