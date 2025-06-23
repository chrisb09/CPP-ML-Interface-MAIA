import time

class TimerManager:
    def __init__(self):
        self._start_times = {}
        self._elapsed = {}

    def start(self, name):
        if name in self._start_times:
            raise RuntimeError(f"Timer '{name}' is already running.")
        self._start_times[name] = time.perf_counter()

    def stop(self, name):
        if name not in self._start_times:
            raise RuntimeError(f"Timer '{name}' was not started.")
        elapsed = time.perf_counter() - self._start_times.pop(name)
        self._elapsed.setdefault(name, []).append(elapsed)
        return elapsed

    def peek(self, name):
        if name not in self._start_times:
            raise RuntimeError(f"Timer '{name}' is not running.")
        return time.perf_counter() - self._start_times[name]

    def reset(self, name=None):
        if name:
            self._start_times.pop(name, None)
            self._elapsed.pop(name, None)
        else:
            self._start_times.clear()
            self._elapsed.clear()

    def summary(self):
        print("\n--- Timer Summary ---")
        for name, times in self._elapsed.items():
            total = sum(times)
            count = len(times)
            avg = total / count
            print(f"{name}:")
            print(f"  Total: {total:.6f}s  |  Runs: {count}  |  Avg: {avg:.6f}s")
        print("---------------------\n")
