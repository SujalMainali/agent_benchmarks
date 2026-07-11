#!/usr/bin/env python3
"""
MEM1 LoCoMo Benchmark Runner - CLI Entry Point.

Usage:
    # Single sample mode
    python -m benchmarks.mem1_locomo.run
    
    # With environment variables
    MEM1_LOCOMO_RUN_MODE=batch python -m benchmarks.mem1_locomo.run

Environment Variables:
    MEM1_LOCOMO_DATA_FILE    - Path to LoCoMo data file
    MEM1_LOCOMO_RUN_MODE     - "single" or "batch"
    MEM1_LOCOMO_SAMPLE_ID    - Sample ID for single mode
    MEM1_LOCOMO_MAX_SAMPLES  - Max samples for batch mode (-1 for all)
    MEM1_LOCOMO_OUTPUT_DIR   - Output directory
"""

import json
import sys
from pathlib import Path

from benchmarks.mem1_locomo.config import load_mem1_locomo_settings
from benchmarks.mem1_locomo.runner import Mem1LoCoMoRunner
from benchmarks.mem1_locomo.report import Mem1LoCoMoReporter


def load_data(data_file: str) -> list[dict]:
    """Load LoCoMo data from JSON or JSONL file."""
    path = Path(data_file)
    
    if not path.exists():
        print(f"Error: Data file not found: {data_file}")
        sys.exit(1)
    
    samples = []
    
    if path.suffix == ".jsonl":
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
    else:
        with open(path, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                samples = data
            else:
                samples = [data]
    
    return samples


def normalize_sample(raw: dict) -> dict:
    """Normalize sample to standard format."""
    # Handle different LoCoMo formats
    return {
        "id": raw.get("id", raw.get("sample_id", "")),
        "question": raw.get("question", raw.get("query", "")),
        "expected": raw.get("answer", raw.get("expected", raw.get("ground_truth", ""))),
        "context": raw.get("conversation", raw.get("context", raw.get("history", []))),
    }


def run_single(settings, runner, reporter):
    """Run evaluation on a single sample."""
    print(f"Loading data from: {settings.data_file}")
    samples = load_data(settings.data_file)
    
    if not samples:
        print("Error: No samples found")
        return
    
    # Find sample by ID or use first
    sample = None
    if settings.sample_id:
        for s in samples:
            if s.get("id", s.get("sample_id", "")) == settings.sample_id:
                sample = s
                break
        if not sample:
            print(f"Error: Sample '{settings.sample_id}' not found")
            return
    else:
        sample = samples[0]
    
    sample = normalize_sample(sample)
    print(f"\nRunning sample: {sample['id']}")
    print(f"Question: {sample['question'][:100]}...")
    
    # Run
    result = runner.run_sample(
        sample_id=sample["id"],
        question=sample["question"],
        expected=sample["expected"],
        context=sample["context"],
    )
    
    # Report
    output_path = reporter.write_sample_result(result)
    
    print(f"\n{'='*60}")
    print(f"Prediction: {result.prediction}")
    print(f"Expected:   {result.expected}")
    print(f"Steps:      {result.reasoning_steps}")
    print(f"Latency:    {result.latency_ms:.2f} ms")
    print(f"Output:     {output_path}")
    print(f"{'='*60}")


def run_batch(settings, runner, reporter):
    """Run evaluation on multiple samples."""
    print(f"Loading data from: {settings.data_file}")
    samples = load_data(settings.data_file)
    
    if not samples:
        print("Error: No samples found")
        return
    
    # Limit samples if specified
    if settings.max_samples > 0:
        samples = samples[:settings.max_samples]
    
    samples = [normalize_sample(s) for s in samples]
    print(f"Running {len(samples)} samples...\n")
    
    def progress(current, total):
        pct = current / total * 100
        print(f"  [{current}/{total}] {pct:.1f}%", end="\r")
    
    # Run batch
    results = runner.run_batch(samples, progress_callback=progress)
    print()
    
    # Write individual results
    for result in results:
        reporter.write_sample_result(result)
    
    # Write summary (without evaluation scores for now)
    summary_path = reporter.write_batch_summary(results)
    
    print(f"\n{'='*60}")
    print(f"Completed: {len(results)} samples")
    print(f"Avg latency: {sum(r.latency_ms for r in results) / len(results):.2f} ms")
    print(f"Avg steps:   {sum(r.reasoning_steps for r in results) / len(results):.2f}")
    print(f"Output:      {settings.output_dir}")
    print(f"Summary:     {summary_path}")
    print(f"{'='*60}")


def main():
    """Main entry point."""
    print("=" * 60)
    print("  MEM1 LoCoMo Benchmark Runner")
    print("=" * 60)
    
    # Load settings
    settings = load_mem1_locomo_settings()
    print(f"\nSettings:")
    print(f"  Model:     {settings.mem1.model_id}")
    print(f"  Data:      {settings.data_file}")
    print(f"  Mode:      {settings.run_mode}")
    print(f"  Output:    {settings.output_dir}")
    
    # Create runner and reporter
    runner = Mem1LoCoMoRunner(settings)
    reporter = Mem1LoCoMoReporter(settings)
    
    # Run based on mode
    if settings.run_mode == "batch":
        run_batch(settings, runner, reporter)
    else:
        run_single(settings, runner, reporter)


if __name__ == "__main__":
    main()