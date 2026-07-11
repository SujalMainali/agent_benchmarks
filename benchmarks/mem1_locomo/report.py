"""
MEM1 LoCoMo Reporter - Writes evaluation artifacts.

Generates:
- Per-sample JSON files (trajectory, think history)
- Batch summary (CSV, JSON)
- Markdown report
"""

import json
import os
from dataclasses import asdict
from datetime import datetime
from typing import Optional

from benchmarks.mem1_locomo.config import Mem1LoCoMoSettings, load_mem1_locomo_settings
from benchmarks.mem1_locomo.runner import Mem1RunResult


class Mem1LoCoMoReporter:
    """Writes MEM1 LoCoMo evaluation reports."""
    
    def __init__(self, settings: Optional[Mem1LoCoMoSettings] = None):
        self.settings = settings or load_mem1_locomo_settings()
        self.output_dir = self.settings.output_dir
    
    def _ensure_dir(self, path: str) -> None:
        """Ensure directory exists."""
        os.makedirs(path, exist_ok=True)
    
    def write_sample_result(self, result: Mem1RunResult) -> str:
        """
        Write a single sample result to disk.
        
        Returns path to the output directory.
        """
        sample_dir = os.path.join(self.output_dir, result.sample_id)
        self._ensure_dir(sample_dir)
        
        # Write main result
        result_dict = {
            "sample_id": result.sample_id,
            "question": result.question,
            "expected": result.expected,
            "prediction": result.prediction,
            "final_think": result.final_think,
            "reasoning_steps": result.reasoning_steps,
            "search_count": result.search_count,
            "latency_ms": result.latency_ms,
        }
        
        with open(os.path.join(sample_dir, "result.json"), "w") as f:
            json.dump(result_dict, f, indent=2)
        
        # Write trajectory if available
        if result.trajectory and self.settings.save_trajectories:
            trajectory_dict = {
                "events": [
                    {
                        "event_type": e.event_type,
                        "content": e.content,
                        "timestamp": e.timestamp,
                        "metadata": e.metadata,
                    }
                    for e in result.trajectory.events
                ],
                "duration_ms": result.trajectory.duration_ms,
            }
            with open(os.path.join(sample_dir, "trajectory.json"), "w") as f:
                json.dump(trajectory_dict, f, indent=2)
        
        # Write think history if available
        if result.think_history and self.settings.save_think_history:
            think_history = []
            for step in result.think_history:
                think_history.append({
                    "step": step.step_num,
                    "think": step.parsed.think,
                    "search": step.parsed.search,
                    "answer": step.parsed.answer,
                })
            with open(os.path.join(sample_dir, "think_history.json"), "w") as f:
                json.dump(think_history, f, indent=2)
        
        return sample_dir
    
    def write_batch_summary(
        self,
        results: list[Mem1RunResult],
        evaluation_scores: Optional[list[dict]] = None,
    ) -> str:
        """
        Write batch summary files.
        
        Returns path to summary JSON.
        """
        self._ensure_dir(self.output_dir)
        
        # Build summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_samples": len(results),
            "settings": {
                "model_id": self.settings.mem1.model_id,
                "max_context_chars": self.settings.mem1.max_context_chars,
                "max_reasoning_steps": self.settings.mem1.max_reasoning_steps,
            },
            "aggregate_metrics": {
                "avg_latency_ms": sum(r.latency_ms for r in results) / len(results) if results else 0,
                "avg_reasoning_steps": sum(r.reasoning_steps for r in results) / len(results) if results else 0,
                "avg_search_count": sum(r.search_count for r in results) / len(results) if results else 0,
                "avg_think_length": sum(len(r.final_think) for r in results) / len(results) if results else 0,
            },
            "samples": [
                {
                    "sample_id": r.sample_id,
                    "prediction": r.prediction,
                    "expected": r.expected,
                    "reasoning_steps": r.reasoning_steps,
                    "latency_ms": r.latency_ms,
                }
                for r in results
            ],
        }
        
        # Add evaluation scores if provided
        if evaluation_scores:
            summary["evaluation"] = evaluation_scores
            correct = sum(1 for s in evaluation_scores if s.get("exact_match", 0) > 0)
            summary["aggregate_metrics"]["accuracy"] = correct / len(evaluation_scores) if evaluation_scores else 0
        
        # Write JSON summary
        summary_path = os.path.join(self.output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        # Write CSV
        csv_path = os.path.join(self.output_dir, "results.csv")
        with open(csv_path, "w") as f:
            headers = ["sample_id", "prediction", "expected", "reasoning_steps", "search_count", "latency_ms"]
            f.write(",".join(headers) + "\n")
            for r in results:
                row = [
                    r.sample_id,
                    f'"{r.prediction[:100]}"',
                    f'"{r.expected[:100]}"',
                    str(r.reasoning_steps),
                    str(r.search_count),
                    f"{r.latency_ms:.2f}",
                ]
                f.write(",".join(row) + "\n")
        
        # Write Markdown report
        self._write_markdown_report(results, evaluation_scores)
        
        return summary_path
    
    def _write_markdown_report(
        self,
        results: list[Mem1RunResult],
        evaluation_scores: Optional[list[dict]] = None,
    ) -> None:
        """Write human-readable Markdown report."""
        report_path = os.path.join(self.output_dir, "report.md")
        
        with open(report_path, "w") as f:
            f.write("# MEM1 LoCoMo Evaluation Report\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Model:** `{self.settings.mem1.model_id}`\n\n")
            f.write(f"**Samples:** {len(results)}\n\n")
            
            f.write("## Aggregate Metrics\n\n")
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            
            avg_latency = sum(r.latency_ms for r in results) / len(results) if results else 0
            avg_steps = sum(r.reasoning_steps for r in results) / len(results) if results else 0
            avg_searches = sum(r.search_count for r in results) / len(results) if results else 0
            
            f.write(f"| Avg Latency | {avg_latency:.2f} ms |\n")
            f.write(f"| Avg Reasoning Steps | {avg_steps:.2f} |\n")
            f.write(f"| Avg Searches | {avg_searches:.2f} |\n")
            
            if evaluation_scores:
                correct = sum(1 for s in evaluation_scores if s.get("exact_match", 0) > 0)
                accuracy = correct / len(evaluation_scores) if evaluation_scores else 0
                f.write(f"| Accuracy | {accuracy:.2%} |\n")
            
            f.write("\n## Sample Results\n\n")
            
            for i, r in enumerate(results[:10]):  # First 10 samples
                f.write(f"### Sample: {r.sample_id}\n\n")
                f.write(f"**Question:** {r.question[:200]}...\n\n")
                f.write(f"**Expected:** {r.expected}\n\n")
                f.write(f"**Prediction:** {r.prediction}\n\n")
                f.write(f"**Steps:** {r.reasoning_steps}, **Searches:** {r.search_count}\n\n")
                f.write("---\n\n")
            
            if len(results) > 10:
                f.write(f"*...and {len(results) - 10} more samples (see summary.json)*\n")