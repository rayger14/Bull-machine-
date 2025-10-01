#!/usr/bin/env python3
"""
Bull Machine Resource Guardrails
Prevents system crashes during long optimization runs
"""

import os
import signal
import psutil
import resource
import time
import threading
import logging
from typing import Optional, Callable

class ResourceGuard:
    """System resource monitor and protection"""

    def __init__(self,
                 max_memory_gb: float = 12.0,
                 max_cpu_percent: float = 90.0,
                 max_runtime_minutes: int = 60,
                 check_interval_s: int = 30):

        self.max_memory_bytes = max_memory_gb * 1024**3
        self.max_cpu_percent = max_cpu_percent
        self.max_runtime_s = max_runtime_minutes * 60
        self.check_interval_s = check_interval_s

        self.start_time = time.time()
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.violation_callback: Optional[Callable] = None

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Install signal handlers for graceful shutdown
        self._install_signal_handlers()

        # Set system resource limits
        self._set_system_limits()

    def _install_signal_handlers(self):
        """Install signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            self.logger.warning(f"Received signal {signum}, shutting down gracefully...")
            self.stop_monitoring()
            os._exit(1)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _set_system_limits(self):
        """Set OS-level resource limits"""
        try:
            # Memory limit (virtual memory)
            memory_limit = int(self.max_memory_bytes)
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))

            # File descriptor limit
            resource.setrlimit(resource.RLIMIT_NOFILE, (4096, 8192))

            # Process limit
            resource.setrlimit(resource.RLIMIT_NPROC, (1024, 2048))

            self.logger.info(f"Set resource limits: {self.max_memory_bytes/1024**3:.1f}GB memory, 4096 files, 1024 processes")

        except (OSError, ValueError) as e:
            self.logger.warning(f"Could not set resource limits: {e}")

    def check_resources(self) -> dict:
        """Check current resource usage"""
        try:
            process = psutil.Process()

            # Memory usage
            mem_info = process.memory_info()
            memory_mb = mem_info.rss / 1024**2
            memory_percent = (mem_info.rss / psutil.virtual_memory().total) * 100

            # CPU usage
            cpu_percent = process.cpu_percent()

            # Runtime
            runtime_s = time.time() - self.start_time

            # System-wide metrics
            system_mem = psutil.virtual_memory()
            system_cpu = psutil.cpu_percent()

            return {
                'process_memory_mb': memory_mb,
                'process_memory_percent': memory_percent,
                'process_cpu_percent': cpu_percent,
                'runtime_s': runtime_s,
                'system_memory_percent': system_mem.percent,
                'system_cpu_percent': system_cpu,
                'system_memory_available_gb': system_mem.available / 1024**3
            }

        except Exception as e:
            self.logger.error(f"Error checking resources: {e}")
            return {}

    def _check_violations(self, stats: dict) -> list:
        """Check for resource limit violations"""
        violations = []

        # Memory violations
        if stats.get('process_memory_mb', 0) > (self.max_memory_bytes / 1024**2):
            violations.append(f"Process memory exceeded: {stats['process_memory_mb']:.1f}MB > {self.max_memory_bytes/1024**2:.1f}MB")

        if stats.get('system_memory_percent', 0) > 95:
            violations.append(f"System memory critical: {stats['system_memory_percent']:.1f}% > 95%")

        # CPU violations
        if stats.get('system_cpu_percent', 0) > self.max_cpu_percent:
            violations.append(f"System CPU overload: {stats['system_cpu_percent']:.1f}% > {self.max_cpu_percent}%")

        # Runtime violations
        if stats.get('runtime_s', 0) > self.max_runtime_s:
            violations.append(f"Runtime exceeded: {stats['runtime_s']:.1f}s > {self.max_runtime_s}s")

        return violations

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                stats = self.check_resources()
                violations = self._check_violations(stats)

                if violations:
                    self.logger.error("Resource violations detected:")
                    for violation in violations:
                        self.logger.error(f"  - {violation}")

                    if self.violation_callback:
                        self.violation_callback(violations, stats)
                    else:
                        self.logger.critical("Terminating process due to resource violations")
                        os._exit(1)

                else:
                    # Log periodic status
                    self.logger.info(f"Resources OK: {stats['process_memory_mb']:.1f}MB RAM, "
                                   f"{stats['process_cpu_percent']:.1f}% CPU, "
                                   f"{stats['runtime_s']:.1f}s runtime")

                time.sleep(self.check_interval_s)

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.check_interval_s)

    def start_monitoring(self, violation_callback: Optional[Callable] = None):
        """Start resource monitoring in background thread"""
        if self.monitoring:
            return

        self.violation_callback = violation_callback
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

        self.logger.info("Resource monitoring started")

    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

        self.logger.info("Resource monitoring stopped")

    def __enter__(self):
        """Context manager entry"""
        self.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_monitoring()

class ProcessTimeoutWrapper:
    """Wrapper to add timeout and resource limits to any function"""

    def __init__(self, timeout_s: int = 300, max_memory_gb: float = 8.0):
        self.timeout_s = timeout_s
        self.max_memory_gb = max_memory_gb

    def __call__(self, func):
        """Decorator to wrap function with timeout and resource limits"""
        def wrapper(*args, **kwargs):

            def target_func():
                try:
                    with ResourceGuard(max_memory_gb=self.max_memory_gb,
                                     max_runtime_minutes=self.timeout_s//60):
                        return func(*args, **kwargs)
                except Exception as e:
                    return {"status": "error", "error": str(e)}

            # Run with timeout
            import multiprocessing
            import queue

            result_queue = multiprocessing.Queue()
            process = multiprocessing.Process(target=lambda: result_queue.put(target_func()))
            process.start()
            process.join(timeout=self.timeout_s)

            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
                if process.is_alive():
                    process.kill()
                return {"status": "timeout", "timeout_s": self.timeout_s}

            try:
                return result_queue.get_nowait()
            except queue.Empty:
                return {"status": "no_result"}

        return wrapper

# Example usage and testing
if __name__ == "__main__":
    print("Testing resource guardrails...")

    # Test basic monitoring
    with ResourceGuard(max_memory_gb=1.0, max_runtime_minutes=1, check_interval_s=5) as guard:
        print("Resource guard active...")

        # Check current stats
        stats = guard.check_resources()
        print(f"Current stats: {stats}")

        # Simulate some work
        time.sleep(10)

    print("Resource guard test complete.")