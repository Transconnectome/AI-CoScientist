#!/usr/bin/env python3
"""
Remote Spark Monitor
Unified monitoring for DGX-Spark: tmux sessions, GPU usage, and system resources.
"""

import subprocess
import argparse
import sys
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

class RemoteSparkMonitor:
    def __init__(self, host: str = "juke@192.168.0.79"):
        self.host = host
        self.timeout = 10

    def run_remote_command(self, cmd: str) -> str:
        """Run a command on the remote host via SSH."""
        try:
            result = subprocess.run(
                ["ssh", "-o", f"ConnectTimeout={self.timeout}", self.host, cmd],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            # Silence error if tmux ls finds no sessions (exit code 1)
            if "tmux ls" in cmd and e.returncode == 1:
                return ""
            return f"Error: {e.stderr.strip()}"
        except Exception as e:
            return f"Exception: {str(e)}"

    def get_tmux_sessions(self) -> str:
        return self.run_remote_command("tmux ls 2>/dev/null")

    def get_gpu_status(self) -> str:
        # Get simplified nvidia-smi output
        return self.run_remote_command("nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader")

    def get_system_status(self) -> Dict[str, str]:
        uptime = self.run_remote_command("uptime -p")
        load = self.run_remote_command("uptime | awk -F'load average:' '{ print $2 }'")
        # Handle both English (Mem) and Korean (메모리) headers
        memory = self.run_remote_command("free -h | awk '/(Mem:|메모리:)/ {print $3 \" / \" $2}'")
        return {
            "uptime": uptime,
            "load": load.strip(),
            "memory": memory
        }

    def format_gpu_output(self, raw_gpu: str) -> List[str]:
        if not raw_gpu or "Error" in raw_gpu:
            return ["GPU Info: Not available or nvidia-smi failed"]
        
        lines = []
        for line in raw_gpu.split('\n'):
            if line:
                parts = [p.strip() for p in line.split(',')]
                # parts: index, name, util_gpu, util_mem, used_mem, total_mem, temp
                # Handle [N/A] values for Blackwell
                used = parts[4] if parts[4] != "[N/A]" else "N/A"
                total = parts[5] if parts[5] != "[N/A]" else "N/A"
                gpu_str = f"GPU [{parts[0]}] {parts[1]}: {parts[2]} Util | {used}/{total} | {parts[6]}°C"
                lines.append(gpu_str)
        return lines

    def display(self):
        print(f"\n{'='*80}")
        print(f"🚀 DGX-Spark Remote Monitor | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Host: {self.host}")
        print(f"{'='*80}\n")

        # System Info
        sys_info = self.get_system_status()
        print(f"📊 [System]")
        print(f"   Uptime: {sys_info['uptime']}")
        print(f"   Load:   {sys_info['load']}")
        print(f"   Memory: {sys_info['memory']}")
        print()

        # GPU Info
        gpu_raw = self.get_gpu_status()
        gpu_lines = self.format_gpu_output(gpu_raw)
        print(f"🎮 [GPU]")
        for line in gpu_lines:
            print(f"   {line}")
        print()

        # Tmux Info
        tmux_info = self.get_tmux_sessions()
        print(f"🪟 [Tmux Sessions]")
        if not tmux_info:
            print("   No active sessions")
        else:
            for line in tmux_info.split('\n'):
                print(f"   {line}")
        print()

    def monitor_loop(self, interval: int = 10):
        try:
            while True:
                subprocess.run(["clear"])
                self.display()
                print(f"Refreshing every {interval}s... (Ctrl+C to stop)")
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")

def main():
    parser = argparse.ArgumentParser(description="Remote Spark Monitor")
    parser.add_argument("-m", "--monitor", action="store_true", help="continuous monitoring mode")
    parser.add_argument("-i", "--interval", type=int, default=10, help="refresh interval (seconds)")
    args = parser.parse_args()

    monitor = RemoteSparkMonitor()
    if args.monitor:
        monitor.monitor_loop(args.interval)
    else:
        monitor.display()

if __name__ == "__main__":
    main()
