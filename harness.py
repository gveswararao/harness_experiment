"""
Three harness configurations (H1 minimal, H2 improved, H3 full) for the
local-LLM-agent evaluation described in the paper. Each function takes a
task directory and a model name, calls Ollama's REST API, writes the
model's fixed file back to disk, and returns the raw response text.

Requires: `pip install requests`
Requires: `ollama serve` running locally with the target model pulled.
"""
import json
import re
import time
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
CALL_TIMEOUT_SECONDS = 600  # generous: local CPU inference + cold model load can be slow


def _call_model(model, prompt, temperature=0.2, seed=42):
    """Single call to the local Ollama model. Returns the raw text response."""
    print(f"    -> calling {model} ... (first call can take several minutes on CPU while the model loads)", flush=True)
    started = time.time()
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature, "seed": seed},
            },
            timeout=CALL_TIMEOUT_SECONDS,
        )
    except requests.exceptions.ReadTimeout:
        raise RuntimeError(
            f"'{model}' did not respond within {CALL_TIMEOUT_SECONDS}s. This usually means "
            f"the model is still loading into memory (common on the very first call) or your "
            f"machine can't keep up on CPU. Try running `ollama run {model}` by hand once "
            f"first (this pre-loads it and lets you see its raw speed), then re-run this script."
        )
    print(f"    -> {model} responded in {time.time() - started:.1f}s", flush=True)
    if resp.status_code == 404:
        raise RuntimeError(
            f"Ollama returned 404 for model '{model}'. This means Ollama doesn't "
            f"recognize that exact tag. Run `ollama list` and update the MODELS "
            f"list in run_experiment.py to match the exact name shown there. "
            f"Server response: {resp.text}"
        )
    resp.raise_for_status()
    return resp.json()["response"]


def _strip_code_fences(text):
    """Models often wrap code in ```python ... ``` even when told not to."""
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def _write_solution(task_dir, code_text):
    with open(f"{task_dir}/solution.py", "w") as f:
        f.write(code_text)


def _read_solution(task_dir):
    with open(f"{task_dir}/solution.py") as f:
        return f.read()


def run_h1_minimal(task_dir, task_prompt, model):
    """H1: single shot, no compaction, no retry, no verification."""
    buggy_code = _read_solution(task_dir)
    prompt = f"{task_prompt}\n\n--- solution.py ---\n{buggy_code}"
    response = _call_model(model, prompt)
    fixed_code = _strip_code_fences(response)
    _write_solution(task_dir, fixed_code)
    return {"attempts": 1, "final_response": response}


def run_h2_improved(task_dir, task_prompt, model, max_retries=3):
    """H2: same as H1 but with exponential-backoff retry on API/parse failure
    (not on test failure — H2 has no verification hook)."""
    buggy_code = _read_solution(task_dir)
    prompt = f"{task_prompt}\n\n--- solution.py ---\n{buggy_code}"
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            response = _call_model(model, prompt)
            fixed_code = _strip_code_fences(response)
            if fixed_code.strip():
                _write_solution(task_dir, fixed_code)
                return {"attempts": attempt, "final_response": response}
        except requests.RequestException as e:
            last_err = e
        time.sleep(2 ** attempt)  # exponential backoff
    return {"attempts": max_retries, "final_response": None, "error": str(last_err)}


def run_h3_full(task_dir, task_prompt, model, run_tests_fn, max_repair_rounds=3):
    """H3: H2 behavior + a post-edit verification loop. If pytest fails,
    feed the failure output back to the model and ask for a repair, up to
    `max_repair_rounds` times. Success is silent; failure is verbose
    (matches the harness spec in the paper's Table 1)."""
    result = run_h2_improved(task_dir, task_prompt, model)
    for round_n in range(1, max_repair_rounds + 1):
        passed, test_output = run_tests_fn(task_dir)
        if passed:
            return {**result, "repair_rounds": round_n - 1, "final_status": "pass"}
        current_code = _read_solution(task_dir)
        repair_prompt = (
            f"{task_prompt}\n\nYour previous attempt failed these tests:\n"
            f"{test_output}\n\n--- current solution.py ---\n{current_code}\n\n"
            "Return the FULL corrected contents of solution.py and nothing else."
        )
        response = _call_model(model, repair_prompt)
        _write_solution(task_dir, _strip_code_fences(response))
        result["final_response"] = response
    passed, test_output = run_tests_fn(task_dir)
    return {**result, "repair_rounds": max_repair_rounds,
            "final_status": "pass" if passed else "fail", "last_test_output": test_output}
