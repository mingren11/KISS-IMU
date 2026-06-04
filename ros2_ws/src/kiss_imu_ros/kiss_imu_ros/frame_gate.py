"""Non-blocking gate that drops backlogged work instead of queuing it.

The scan callback calls try_enter() before processing; if a previous step() is
still running (lock held), the new scan is dropped (counter incremented) so
odometry latency does not accumulate under load."""
import threading


class FrameGate:
    def __init__(self):
        self._lock = threading.Lock()
        self.dropped = 0

    def try_enter(self) -> bool:
        if self._lock.acquire(blocking=False):
            return True
        self.dropped += 1
        return False

    def exit(self) -> None:
        self._lock.release()
