"""Track metrics over time across multiple evaluation runs."""
import json
from pathlib import Path
from datetime import datetime
from typing import Any
import csv


class MetricsTracker:
    """Track evaluation metrics across runs for trend analysis."""
    
    def __init__(self, tracking_file: str = "metrics_history.json"):
        self.tracking_file = Path(tracking_file)
        self.history = self._load_history()
    
    def _load_history(self) -> list[dict]:
        """Load historical metrics."""
        if self.tracking_file.exists():
            with open(self.tracking_file) as f:
                return json.load(f)
        return []
    
    def add_run(self, metrics: dict[str, Any], run_name: str = None) -> None:
        """Add a new metrics run to history."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "run_name": run_name or f"run_{len(self.history) + 1}",
            **metrics,
        }
        self.history.append(entry)
        self._save_history()
    
    def _save_history(self) -> None:
        """Save history to file."""
        with open(self.tracking_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def export_csv(self, output_file: str = "metrics_history.csv") -> None:
        """Export history as CSV for spreadsheet analysis."""
        if not self.history:
            print("No history to export")
            return
        
        keys = set()
        for entry in self.history:
            keys.update(entry.keys())
        keys = sorted(list(keys))
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.history)
        
        print(f"Exported to {output_file}")
    
    def get_trend(self, metric_key: str) -> dict:
        """Get trend statistics for a metric."""
        values = [entry.get(metric_key) for entry in self.history 
                 if metric_key in entry and entry[metric_key] is not None]
        
        if not values:
            return {}
        
        return {
            "latest": values[-1],
            "best": min(values) if metric_key in ["stt_wer", "stt_cer", "stt_finalize_latency_ms", "e2e_latency_ms"] else max(values),
            "worst": max(values) if metric_key in ["stt_wer", "stt_cer", "stt_finalize_latency_ms", "e2e_latency_ms"] else min(values),
            "average": sum(values) / len(values),
            "num_runs": len(values),
        }
    
    def print_summary(self) -> None:
        """Print a summary of tracked metrics."""
        if not self.history:
            print("No metrics history")
            return
        
        print(f"\nMetrics History Summary ({len(self.history)} runs)")
        print("=" * 60)
        
        metrics_to_track = [
            "stt_wer", "stt_cer", "stt_finalize_latency_ms", 
            "e2e_latency_ms", "question_detection_accuracy"
        ]
        
        for metric in metrics_to_track:
            trend = self.get_trend(metric)
            if trend:
                print(f"\n{metric}:")
                print(f"  Latest: {trend['latest']:.2f}")
                print(f"  Average: {trend['average']:.2f}")
                print(f"  Best: {trend['best']:.2f}")
                print(f"  Worst: {trend['worst']:.2f}")
    
    def compare_runs(self, run_indices: list[int] = None) -> None:
        """Compare specific runs."""
        if run_indices is None:
            run_indices = [len(self.history) - 2, len(self.history) - 1]  # Last 2 runs
        
        if not all(0 <= i < len(self.history) for i in run_indices):
            print("Invalid run indices")
            return
        
        print("\nRun Comparison")
        print("=" * 60)
        
        runs = [self.history[i] for i in run_indices]
        metrics_to_compare = [
            "stt_wer", "stt_cer", "stt_finalize_latency_ms", 
            "e2e_latency_ms", "question_detection_accuracy"
        ]
        
        for metric in metrics_to_compare:
            print(f"\n{metric}:")
            for idx, run in zip(run_indices, runs):
                value = run.get(metric, "N/A")
                run_name = run.get("run_name", f"Run {idx}")
                print(f"  {run_name}: {value}")


if __name__ == "__main__":
    tracker = MetricsTracker()
    
    # Example: Add a run
    example_metrics = {
        "num_prompts": 20,
        "stt_wer": 5.2,
        "stt_cer": 2.1,
        "stt_finalize_latency_ms": 450,
        "e2e_latency_ms": 1200,
        "question_detection_accuracy": 0.95,
    }
    tracker.add_run(example_metrics, "test_run")
    
    # Print summary
    tracker.print_summary()
    
    # Export as CSV
    tracker.export_csv()
