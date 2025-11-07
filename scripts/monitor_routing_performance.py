#!/usr/bin/env python3
"""
Week 4: Performance Monitoring and Analytics
Real-time monitoring dashboard for model routing performance
"""

import json
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

# Paths
LOG_PATH = Path("/home/juke/git/AI-CoScientist/logs/model_router.log")
ANALYTICS_PATH = Path("/home/juke/git/AI-CoScientist/logs/routing_analytics.json")


class RoutingAnalytics:
    """Analyze routing decisions and performance"""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.analytics = defaultdict(lambda: {
            "count": 0,
            "total_time": 0.0,
            "avg_time": 0.0,
            "by_model": defaultdict(int),
            "by_task_type": defaultdict(int)
        })

    def parse_log_file(self, hours: int = 24) -> List[Dict]:
        """Parse routing decisions from log file"""
        decisions = []
        cutoff_time = datetime.now() - timedelta(hours=hours)

        if not self.log_path.exists():
            print(f"⚠️  Log file not found: {self.log_path}")
            return decisions

        with open(self.log_path, 'r') as f:
            for line in f:
                if "Routing decision:" in line:
                    try:
                        # Extract JSON from log line
                        json_start = line.index("{")
                        json_str = line[json_start:]
                        decision = json.loads(json_str)

                        # Check timestamp
                        timestamp = datetime.fromisoformat(decision["timestamp"])
                        if timestamp >= cutoff_time:
                            decisions.append(decision)
                    except (ValueError, json.JSONDecodeError, KeyError) as e:
                        print(f"⚠️  Error parsing log line: {e}")
                        continue

        return decisions

    def analyze_decisions(self, decisions: List[Dict]) -> Dict:
        """Analyze routing decisions"""
        if not decisions:
            return {}

        total = len(decisions)

        # Model distribution
        model_counts = defaultdict(int)
        for d in decisions:
            model_counts[d["selected_model"]] += 1

        # Task type distribution
        task_counts = defaultdict(int)
        for d in decisions:
            task_counts[d["task_type"]] += 1

        # Complexity distribution
        complexity_bins = {"low": 0, "medium": 0, "high": 0}
        for d in decisions:
            complexity = d["complexity"]
            if complexity < 0.3:
                complexity_bins["low"] += 1
            elif complexity < 0.7:
                complexity_bins["medium"] += 1
            else:
                complexity_bins["high"] += 1

        return {
            "total_decisions": total,
            "model_distribution": dict(model_counts),
            "task_type_distribution": dict(task_counts),
            "complexity_distribution": complexity_bins,
            "time_range_hours": 24
        }

    def print_dashboard(self, analysis: Dict):
        """Print monitoring dashboard"""
        print("\n" + "="*80)
        print("MODEL ROUTING PERFORMANCE DASHBOARD")
        print("="*80)

        print(f"\n📊 Total Routing Decisions: {analysis['total_decisions']}")
        print(f"⏰ Time Range: Last {analysis['time_range_hours']} hours")

        # Model distribution
        print("\n" + "-"*80)
        print("🤖 MODEL USAGE DISTRIBUTION")
        print("-"*80)

        model_dist = analysis["model_distribution"]
        total = analysis["total_decisions"]

        for model, count in sorted(model_dist.items(), key=lambda x: -x[1]):
            percentage = (count / total * 100) if total > 0 else 0
            bar = "█" * int(percentage / 2)
            print(f"{model:30s} {count:5d} ({percentage:5.1f}%) {bar}")

        # Task type distribution
        print("\n" + "-"*80)
        print("📋 TASK TYPE DISTRIBUTION")
        print("-"*80)

        task_dist = analysis["task_type_distribution"]

        for task_type, count in sorted(task_dist.items(), key=lambda x: -x[1]):
            percentage = (count / total * 100) if total > 0 else 0
            bar = "█" * int(percentage / 2)
            print(f"{task_type:30s} {count:5d} ({percentage:5.1f}%) {bar}")

        # Complexity distribution
        print("\n" + "-"*80)
        print("🎚️  COMPLEXITY DISTRIBUTION")
        print("-"*80)

        complexity_dist = analysis["complexity_distribution"]

        for level, count in [("low", complexity_dist["low"]),
                             ("medium", complexity_dist["medium"]),
                             ("high", complexity_dist["high"])]:
            percentage = (count / total * 100) if total > 0 else 0
            bar = "█" * int(percentage / 2)
            print(f"{level:30s} {count:5d} ({percentage:5.1f}%) {bar}")

    def calculate_performance_metrics(self, decisions: List[Dict]) -> Dict:
        """Calculate performance improvement metrics"""
        # Estimated baseline (all DeepSeek-R1)
        baseline_avg_speed = 4.5  # tokens/sec

        # Estimated hybrid performance
        model_speeds = {
            "deepseek-r1:32b": 4.5,
            "nemotron-nano-9b-v2": 17.5
        }

        total_decisions = len(decisions)
        if total_decisions == 0:
            return {}

        # Calculate weighted average speed
        model_counts = defaultdict(int)
        for d in decisions:
            model_counts[d["selected_model"]] += 1

        weighted_speed = sum(
            model_speeds.get(model, baseline_avg_speed) * count
            for model, count in model_counts.items()
        ) / total_decisions

        improvement = ((weighted_speed - baseline_avg_speed) / baseline_avg_speed) * 100

        return {
            "baseline_speed_tps": baseline_avg_speed,
            "current_avg_speed_tps": weighted_speed,
            "improvement_percentage": improvement,
            "total_requests": total_decisions
        }

    def print_performance_metrics(self, metrics: Dict):
        """Print performance improvement metrics"""
        print("\n" + "-"*80)
        print("⚡ PERFORMANCE METRICS")
        print("-"*80)

        print(f"Baseline (DeepSeek-R1 only):  {metrics['baseline_speed_tps']:.1f} tokens/sec")
        print(f"Current (Hybrid routing):     {metrics['current_avg_speed_tps']:.1f} tokens/sec")
        print(f"Performance Improvement:      +{metrics['improvement_percentage']:.1f}%")
        print(f"Total Requests Analyzed:      {metrics['total_requests']}")

    def save_analytics(self, analysis: Dict, metrics: Dict):
        """Save analytics to JSON file"""
        ANALYTICS_PATH.parent.mkdir(parents=True, exist_ok=True)

        analytics_data = {
            "timestamp": datetime.now().isoformat(),
            "analysis": analysis,
            "performance_metrics": metrics
        }

        with open(ANALYTICS_PATH, 'w') as f:
            json.dump(analytics_data, f, indent=2)

        print(f"\n📄 Analytics saved: {ANALYTICS_PATH}")


def gpu_utilization_check():
    """Check GPU utilization with nvidia-smi"""
    import subprocess

    print("\n" + "-"*80)
    print("🖥️  GPU UTILIZATION")
    print("-"*80)

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                parts = line.split(", ")
                if len(parts) >= 5:
                    gpu_id, name, mem_used, mem_total, util = parts
                    mem_pct = (float(mem_used) / float(mem_total)) * 100 if float(mem_total) > 0 else 0
                    print(f"GPU {gpu_id} ({name}): {util}% util, {mem_used}/{mem_total}MB ({mem_pct:.1f}%)")
        else:
            print("❌ nvidia-smi command failed")

    except FileNotFoundError:
        print("⚠️  nvidia-smi not found (not on GPU server?)")
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")


def ollama_model_status():
    """Check Ollama model status"""
    import httpx

    print("\n" + "-"*80)
    print("🔧 OLLAMA MODEL STATUS")
    print("-"*80)

    try:
        response = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"Loaded models: {len(models)}")
            for model in models:
                name = model.get("name")
                size = model.get("size", 0) / (1024**3)  # Convert to GB
                print(f"  - {name} ({size:.1f} GB)")
        else:
            print(f"❌ Ollama API error: {response.status_code}")
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")


def proxy_status():
    """Check model router proxy status"""
    import httpx

    print("\n" + "-"*80)
    print("🚀 MODEL ROUTER PROXY STATUS")
    print("-"*80)

    try:
        response = httpx.get("http://localhost:11435/health", timeout=5.0)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Proxy Status: {data.get('status', 'unknown')}")
            print(f"   Service: {data.get('service', 'unknown')}")
        else:
            print(f"❌ Proxy health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Proxy unreachable: {e}")


if __name__ == "__main__":
    print("\n🚀 Starting Performance Monitoring Dashboard")

    # System status checks
    proxy_status()
    ollama_model_status()
    gpu_utilization_check()

    # Routing analytics
    analyzer = RoutingAnalytics(LOG_PATH)

    # Parse decisions from last 24 hours
    decisions = analyzer.parse_log_file(hours=24)

    if decisions:
        # Analyze decisions
        analysis = analyzer.analyze_decisions(decisions)

        # Print dashboard
        analyzer.print_dashboard(analysis)

        # Calculate performance metrics
        metrics = analyzer.calculate_performance_metrics(decisions)

        # Print metrics
        analyzer.print_performance_metrics(metrics)

        # Save analytics
        analyzer.save_analytics(analysis, metrics)
    else:
        print("\n⚠️  No routing decisions found in the last 24 hours")
        print("   Make sure the model router proxy is running and handling requests")

    print("\n✅ Monitoring complete!")
