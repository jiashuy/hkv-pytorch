import torch


class GPUTimer:
    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def start(self):
        self.start_event.record()

    def stop(self):
        self.end_event.record()

    def elapsed_time(self):
        """
        return in ms
        """
        self.dist_sync()
        torch.cuda.synchronize()
        return self.start_event.elapsed_time(self.end_event)

    def dist_sync(self):
        pass
        # torch.distributed.barrier(device_ids=[torch.cuda.current_device()])