"""
Run this BEFORE run_experiment.py to confirm Ollama is reachable and every
model name in MODELS actually resolves. Saves you from discovering a typo'd
tag halfway through a long experiment run.

Usage:
    python check_setup.py
"""
import sys
import time
import requests

MODELS = ["phi4-mini", "llama3.2:latest"]  # both listed here so check_setup.py can verify both respond,
                                             # even though run_experiment.py only re-runs phi4-mini


def main():
    # 1. Is Ollama running at all?
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        r.raise_for_status()
    except requests.RequestException as e:
        print("FAILED: Could not reach Ollama at http://localhost:11434")
        print(f"  {e}")
        print("  -> Is `ollama serve` running? On Windows, check the Ollama")
        print("     tray icon, or run `ollama serve` in a separate terminal.")
        sys.exit(1)

    installed = [m["name"] for m in r.json().get("models", [])]
    print("Ollama is running. Installed models:")
    for name in installed:
        print(f"  - {name}")
    print()

    # 2. Does each model we plan to use actually exist and respond?
    all_ok = True
    for model in MODELS:
        if model not in installed:
            print(f"MISSING: '{model}' is not in `ollama list`.")
            print(f"  Closest installed names: "
                  f"{[m for m in installed if model.split(':')[0] in m] or 'none found'}")
            print(f"  -> Run: ollama pull {model}")
            print(f"     Or update MODELS in run_experiment.py to match an installed name.")
            all_ok = False
            continue
        try:
            print(f"  warming up '{model}' (loading into memory, can take a while the first time)...")
            started = time.time()
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model, "prompt": "Say OK.", "stream": False},
                timeout=600,
            )
            resp.raise_for_status()
            text = resp.json()["response"].strip().replace("\n", " ")
            print(f"OK: '{model}' responded in {time.time() - started:.1f}s: {text[:60]}")
        except requests.RequestException as e:
            print(f"FAILED: '{model}' is listed but did not respond correctly: {e}")
            all_ok = False

    print()
    print("All models ready." if all_ok else "Fix the issues above before running run_experiment.py.")


if __name__ == "__main__":
    main()
