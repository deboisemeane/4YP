import time
import torch


class Timer:
    def __init__(self):
        self._start_time = None

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise Exception("Timer is running. Use .stop() to stop it")

        # Synchronize CUDA operations before starting the timer
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self._start_time = time.perf_counter()

    def stop(self, msg: str):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise Exception("Timer is not running. Use .start() to start it")

        # Synchronize CUDA operations before stopping the timer
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        print(f"{msg}: {elapsed_time:0.4f} seconds")
