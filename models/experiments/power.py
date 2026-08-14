"""
Power monitoring utilities for GPU energy measurement (issue #49).

Extracted from models/experiments/run_latency_table3.py so all runners
can use it without duplicating code.

Usage:
    from models.experiments.power import PowerMonitor, measure_idle_power

    idle = measure_idle_power(device=0, duration_s=5)
    pm = PowerMonitor(device=0, interval=0.05)
    pm.start()
    # ... run inference ...
    pm.stop()
    avg_power = pm.avg_power_watts()
    energy_j = (avg_power - idle) * wall_time_s
    j_per_event = energy_j / n_events
"""

import subprocess
import threading
import time


class PowerMonitor:
    """Polls nvidia-smi for GPU power draw in a background thread.

    Usage:
        pm = PowerMonitor(device=0, interval=0.05)
        pm.start()
        # ... run inference ...
        pm.stop()
        avg_power = pm.avg_power_watts()  # board-level, including idle
    """

    def __init__(self, device=0, interval=0.05):
        self.device = device
        self.interval = interval
        self._readings = []
        self._thread = None
        self._stop = False
        self._start_time = None
        self._end_time = None

    def start(self):
        self._stop = False
        self._readings = []
        self._start_time = time.perf_counter()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop = True
        self._end_time = time.perf_counter()
        if self._thread:
            self._thread.join(timeout=2)

    def _poll(self):
        while not self._stop:
            try:
                out = subprocess.check_output(
                    ["nvidia-smi",
                     f"--id={self.device}",
                     "--query-gpu=power.draw",
                     "--format=csv,noheader,nounits"],
                    timeout=2,
                ).decode().strip()
                if out:
                    self._readings.append(float(out.split("\n")[0]))
            except (subprocess.SubprocessError, ValueError, FileNotFoundError):
                pass
            time.sleep(self.interval)

    def avg_power_watts(self):
        if not self._readings:
            return 0.0
        return sum(self._readings) / len(self._readings)

    def peak_power_watts(self):
        if not self._readings:
            return 0.0
        return max(self._readings)

    def has_readings(self):
        return len(self._readings) > 0

    def duration_s(self):
        if self._start_time and self._end_time:
            return self._end_time - self._start_time
        return 0.0

    def energy_j(self, idle_power_w=0.0):
        """Energy in joules = (avg_power - idle) * duration."""
        if not self._readings:
            return 0.0
        return (self.avg_power_watts() - idle_power_w) * self.duration_s()

    def summary(self, idle_power_w=0.0, n_events=None):
        """Return a summary dict for JSON output."""
        s = {
            "avg_power_w": round(self.avg_power_watts(), 2),
            "peak_power_w": round(self.peak_power_watts(), 2),
            "idle_power_w": round(idle_power_w, 2),
            "duration_s": round(self.duration_s(), 3),
            "n_readings": len(self._readings),
        }
        energy = self.energy_j(idle_power_w)
        s["energy_j_total"] = round(energy, 2)
        if n_events and n_events > 0:
            s["j_per_event"] = round(energy / n_events, 4)
        return s


def measure_idle_power(device=0, duration_s=5):
    """Measure idle GPU power for baseline subtraction."""
    pm = PowerMonitor(device=device, interval=0.5)
    pm.start()
    time.sleep(duration_s)
    pm.stop()
    return pm.avg_power_watts()


def measure_peak_mem_gb(device=0):
    """Query max GPU memory usage (best-effort, may fail)."""
    try:
        import torch
        return round(torch.cuda.max_memory_allocated(device) / 1e9, 2)
    except Exception:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", f"--id={device}",
                 "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                timeout=2,
            ).decode().strip()
            return round(float(out) / 1024, 2) if out else None
        except Exception:
            return None


def cost_per_event(j_per_event, gpu_rate_per_hour=2.50,
                   electricity_rate_per_kwh=0.12):
    """Estimate $/event from energy and GPU rental rate."""
    gpu_cost = gpu_rate_per_hour / 3600.0  # $/s
    electricity_cost = j_per_event / 3.6e6 * electricity_rate_per_kwh
    return {"gpu_cost_per_event": round(gpu_cost, 6),
            "electricity_cost_per_event": round(electricity_cost, 8),
            "total_per_event": round(gpu_cost + electricity_cost, 6)}
