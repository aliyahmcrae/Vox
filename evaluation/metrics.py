"""Metrics and visualization for Vox voice assistant evaluation results."""
import json
import statistics as st
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime


@dataclass
class MetricsSummary:
    """Summary of evaluation metrics."""
    timestamp: str
    num_prompts: int
    stt_wer: float
    stt_cer: float
    stt_finalize_latency_ms: float
    e2e_latency_ms: float
    question_detection_accuracy: float
    raw_data: dict[str, Any]


class MetricsAnalyzer:
    """Analyze voice test results and compute metrics."""
    
    def __init__(self, results_file: str | Path = "voice_results.json"):
        self.results_file = Path(results_file)
        self.data = None
        self.timestamp = None
        
    def load_results(self) -> dict[str, Any]:
        """Load results from JSON file."""
        if not self.results_file.exists():
            raise FileNotFoundError(f"Results file not found: {self.results_file}")

        with open(self.results_file) as f:
            raw = json.load(f)

        # Convert voice_results.json format into the format expected by the rest
        # of the metrics code.
        self.data = {
            "prompts": [
                {
                    "stt_wer": p["wer"],
                    "stt_cer": p["cer"],
                    "stt_finalize_latency_ms": (
                        p["stt_finalize_s"] * 1000
                        if p["stt_finalize_s"] is not None
                        else None
                    ),
                    "e2e_latency_ms": (
                        p["e2e_first_audio_s"] * 1000
                        if p["e2e_first_audio_s"] is not None
                        else None
                    ),
                    "question_detected": p["qd_fires"] > 0,
                }
                for p in raw
            ]
        }

        self.timestamp = datetime.now().isoformat()
        return self.data

    def compute_metrics(self) -> MetricsSummary:
        """Compute aggregate metrics from results."""
        if self.data is None:
            self.load_results()
        
        prompts = self.data.get("prompts", [])
        num_prompts = len(prompts)
        
        # STT metrics
        wer_values = [p.get("stt_wer", 0) for p in prompts if "stt_wer" in p]
        cer_values = [p.get("stt_cer", 0) for p in prompts if "stt_cer" in p]
        
        # Latency metrics (in ms)
        finalize_latencies = [
            p["stt_finalize_latency_ms"]
            for p in prompts
            if p.get("stt_finalize_latency_ms") is not None
        ]

        e2e_latencies = [
            p["e2e_latency_ms"]
            for p in prompts
            if p.get("e2e_latency_ms") is not None
        ]
        
        # Question detection
        question_detections = [p.get("question_detected", False) for p in prompts]
        question_accuracy = (sum(question_detections) / len(question_detections) 
                            if question_detections else 0)
        
        return MetricsSummary(
            timestamp=self.timestamp,
            num_prompts=num_prompts,
            stt_wer=st.mean(wer_values) if wer_values else 0,
            stt_cer=st.mean(cer_values) if cer_values else 0,
            stt_finalize_latency_ms=st.mean(finalize_latencies) if finalize_latencies else 0,
            e2e_latency_ms=st.mean(e2e_latencies) if e2e_latencies else 0,
            question_detection_accuracy=question_accuracy,
            raw_data=self.data,
        )
    
    def get_percentiles(self, metric_key: str, percentiles: list[int] = [50, 95, 99]) -> dict:
        """Get percentile values for a metric."""
        if self.data is None:
            self.load_results()
        
        values = []
        for prompt in self.data.get("prompts", []):
            if metric_key in prompt:
                values.append(prompt[metric_key])
        
        if not values:
            return {}
        
        return {f"p{p}": np.percentile(values, p) for p in percentiles}


class MetricsVisualizer:
    """Generate visualizations from metrics data."""
    
    def __init__(self, analyzer: MetricsAnalyzer):
        self.analyzer = analyzer
        self.fig_count = 0
    
    def plot_stt_accuracy(self, save_path: str | None = None) -> plt.Figure:
        """Plot STT accuracy metrics (WER/CER)."""
        if self.analyzer.data is None:
            self.analyzer.load_results()
        
        prompts = self.analyzer.data.get("prompts", [])
        wer_values = [p.get("stt_wer", 0) for p in prompts]
        cer_values = [p.get("stt_cer", 0) for p in prompts]
        prompt_ids = list(range(len(wer_values)))
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(prompt_ids, wer_values, marker='o', label='WER', linewidth=2)
        ax.plot(prompt_ids, cer_values, marker='s', label='CER', linewidth=2)
        ax.axhline(y=np.mean(wer_values), color='blue', linestyle='--', alpha=0.5, label=f'Avg WER: {np.mean(wer_values):.2f}%')
        ax.axhline(y=np.mean(cer_values), color='orange', linestyle='--', alpha=0.5, label=f'Avg CER: {np.mean(cer_values):.2f}%')
        
        ax.set_xlabel("Prompt Index")
        ax.set_ylabel("Error Rate (%)")
        ax.set_title("Speech-to-Text Accuracy (WER/CER)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_latencies(self, save_path: str | None = None) -> plt.Figure:
        """Plot latency metrics."""
        if self.analyzer.data is None:
            self.analyzer.load_results()
        
        prompts = self.analyzer.data.get("prompts", [])
        
        finalize_latencies = [
            p.get("stt_finalize_latency_ms", 0) or 0
            for p in prompts
        ]

        e2e_latencies = [
            p.get("e2e_latency_ms", 0) or 0
            for p in prompts
        ]

        prompt_ids = list(range(len(finalize_latencies)))
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # STT finalize latency
        ax1.bar(prompt_ids, finalize_latencies, alpha=0.7, color='steelblue')
        ax1.axhline(y=np.mean(finalize_latencies), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(finalize_latencies):.1f}ms')
        ax1.set_ylabel("Latency (ms)")
        ax1.set_title("STT Finalize Latency")
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # End-to-end latency
        ax2.bar(prompt_ids, e2e_latencies, alpha=0.7, color='darkgreen')
        ax2.axhline(y=np.mean(e2e_latencies), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(e2e_latencies):.1f}ms')
        ax2.set_xlabel("Prompt Index")
        ax2.set_ylabel("Latency (ms)")
        ax2.set_title("End-to-End Latency (STT → LLM → TTS)")
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_metrics_summary(self, metrics: MetricsSummary, save_path: str | None = None) -> plt.Figure:
        """Plot summary of key metrics as a bar chart."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        labels = ['WER (%)', 'CER (%)', 'Finalize Latency (×10ms)', 'E2E Latency (×100ms)', 'Question Detection (%)']
        values = [
            metrics.stt_wer,
            metrics.stt_cer,
            metrics.stt_finalize_latency_ms / 10,  # Scale for visibility
            metrics.e2e_latency_ms / 100,  # Scale for visibility
            metrics.question_detection_accuracy * 100,
        ]
        colors = ['#ff7f0e', '#ff7f0e', '#2ca02c', '#d62728', '#1f77b4']
        
        bars = ax.bar(labels, values, color=colors, alpha=0.7)
        ax.set_ylabel("Value")
        ax.set_title("Voice Assistant Metrics Summary")
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=10)
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_distribution(self, metric_key: str, save_path: str | None = None) -> plt.Figure:
        """Plot distribution of a metric."""
        if self.analyzer.data is None:
            self.analyzer.load_results()
        
        values = [p.get(metric_key, 0) for p in self.analyzer.data.get("prompts", []) if metric_key in p]
        
        if not values:
            print(f"No data found for metric: {metric_key}")
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(values, bins=20, alpha=0.7, edgecolor='black', color='steelblue')
        ax.axvline(x=np.mean(values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(values):.2f}')
        ax.axvline(x=np.median(values), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(values):.2f}')
        
        ax.set_xlabel(metric_key)
        ax.set_ylabel("Frequency")
        ax.set_title(f"Distribution: {metric_key}")
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def generate_report(self, output_dir: str = "metrics_output") -> dict:
        """Generate all visualizations and save to output directory."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        metrics = self.analyzer.compute_metrics()
        
        files_generated = {}
        
        # Generate plots
        figs = {
            "stt_accuracy.png": self.plot_stt_accuracy(output_path / "stt_accuracy.png"),
            "latencies.png": self.plot_latencies(output_path / "latencies.png"),
            "summary.png": self.plot_metrics_summary(metrics, output_path / "summary.png"),
        }
        
        for name, fig in figs.items():
            if fig:
                plt.close(fig)
                files_generated[name] = str(output_path / name)
        
        # Save metrics summary as JSON
        summary_data = {
            "timestamp": metrics.timestamp,
            "num_prompts": metrics.num_prompts,
            "stt_wer": metrics.stt_wer,
            "stt_cer": metrics.stt_cer,
            "stt_finalize_latency_ms": metrics.stt_finalize_latency_ms,
            "e2e_latency_ms": metrics.e2e_latency_ms,
            "question_detection_accuracy": metrics.question_detection_accuracy,
        }
        
        summary_file = output_path / "metrics_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        files_generated["metrics_summary.json"] = str(summary_file)
        
        return files_generated


if __name__ == "__main__":
    # Example usage
    analyzer = MetricsAnalyzer("voice_results.json")
    visualizer = MetricsVisualizer(analyzer)
    
    # Generate all reports
    files = visualizer.generate_report("metrics_output")
    print("Generated files:")
    for name, path in files.items():
        print(f"  {name}: {path}")
    
    # Print summary
    metrics = analyzer.compute_metrics()
    print(f"\nMetrics Summary:")
    print(f"  Prompts tested: {metrics.num_prompts}")
    print(f"  STT WER: {metrics.stt_wer:.2f}%")
    print(f"  STT CER: {metrics.stt_cer:.2f}%")
    print(f"  STT Finalize Latency: {metrics.stt_finalize_latency_ms:.1f}ms")
    print(f"  E2E Latency: {metrics.e2e_latency_ms:.1f}ms")
    print(f"  Question Detection: {metrics.question_detection_accuracy*100:.1f}%")
