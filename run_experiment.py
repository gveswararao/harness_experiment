"""
Orchestrates the 3x3x10 design: for each task, harness, model, and trial,
resets the buggy file, runs the harness, executes pytest, and logs the
result to results.csv.

Usage:
    python run_experiment.py --trials 3

Before running:
  1. `ollama serve` must be running
  2. All three models must be pulled (see README instructions)
  3. Fill in tasks/ with all 10 tasks (2 are provided as examples)
"""
import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from harness import run_h1_minimal, run_h2_improved, run_h3_full

TASKS_DIR = Path(__file__).parent / "tasks"
MODELS = ["phi4-mini"]  # llama3.2:latest already has a completed results.csv from the earlier run;
                          # running it again here would waste time re-doing 90 trials you already have.
                          # tinyllama and qwen3:4b remain excluded: too slow/unstable on this hardware.
HARNESSES = ["H1", "H2", "H3"]
RESULTS_CSV_DEFAULT = Path(__file__).parent / "results.csv"


def discover_tasks():
    return sorted([d for d in TASKS_DIR.iterdir() if d.is_dir()])


def backup_original(task_dir):
    shutil.copy(task_dir / "solution.py", task_dir / "solution.py.orig")


def restore_original(task_dir):
    shutil.copy(task_dir / "solution.py.orig", task_dir / "solution.py")


def run_tests(task_dir):
    """Runs pytest in the task directory. Returns (passed: bool, output: str)."""
    print(f"    -> running pytest in {task_dir.name} ...", flush=True)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "test_solution.py", "-q"],
        cwd=task_dir, capture_output=True, text=True, timeout=60,
    )
    print(f"    -> pytest {'PASSED' if proc.returncode == 0 else 'FAILED'}", flush=True)
    return proc.returncode == 0, (proc.stdout + proc.stderr)[-2000:]  # last 2000 chars


def run_one_trial(task_dir, task_meta, harness, model):
    restore_original(task_dir)
    prompt = task_meta["prompt"]
    started = time.time()
    if harness == "H1":
        meta = run_h1_minimal(task_dir, prompt, model)
        passed, output = run_tests(task_dir)
    elif harness == "H2":
        meta = run_h2_improved(task_dir, prompt, model)
        passed, output = run_tests(task_dir)
    elif harness == "H3":
        meta = run_h3_full(task_dir, prompt, model, run_tests)
        passed, output = run_tests(task_dir)
    else:
        raise ValueError(harness)
    elapsed = time.time() - started
    return {
        "task_id": task_meta["id"],
        "harness": harness,
        "model": model,
        "passed": int(passed),
        "elapsed_sec": round(elapsed, 1),
        "attempts": meta.get("attempts", 1),
        "repair_rounds": meta.get("repair_rounds", 0),
    }


def main(trials, output_path):
    tasks = discover_tasks()
    for t in tasks:
        backup_original(t)

    rows = []
    for task_dir in tasks:
        task_meta = json.loads((task_dir / "task.json").read_text())
        for model in MODELS:
            for harness in HARNESSES:
                for trial in range(1, trials + 1):
                    print(f"[{task_meta['id']}] {model} / {harness} / trial {trial}")
                    row = run_one_trial(task_dir, task_meta, harness, model)
                    row["trial"] = trial
                    rows.append(row)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} trial results to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--output", type=str, default=str(RESULTS_CSV_DEFAULT),
                         help="Output CSV path. Use a different name (e.g. results_phi4mini.csv) "
                              "to avoid overwriting an existing results.csv from a prior model.")
    args = parser.parse_args()
    main(args.trials, Path(args.output))
