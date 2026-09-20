"""
Smoke Test A — no simulation, sanity that the rotator is wired in but
inert in production-like conditions.

Configuration override for this smoke test only:
  - GROQ_API_KEYS: whatever is already in .env (single key is fine).
  - SIMULATE_QUOTA_EXHAUSTION_AFTER_N_CALLS: unset (production behaviour).

Runs `eval/run_eval.py --n_questions 5 --pipeline_mode baseline
--sleep 1.0` as a subprocess so the smoke test exercises the real CLI
path — no bespoke wrapper. Post-run, reads the summary.json and asserts
the invariants Part 16 requires.

Run from the project root:
    python scripts/smoke_test_rotator_a_no_sim.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run() -> Path:
    env = os.environ.copy()
    # Force NO simulation. Reset in case something else in the shell
    # left it set.
    env.pop("SIMULATE_QUOTA_EXHAUSTION_AFTER_N_CALLS", None)
    # Shorten cooldown to a safe value for the smoke test only — the
    # default 3600s would be catastrophic if the harness accidentally
    # tripped it here. Test A shouldn't rotate at all, so this is
    # belt-and-braces defensive.
    env["KEY_ROTATION_COOLDOWN_S"] = "30"
    env["PYTHONIOENCODING"] = "utf-8"

    cmd = [
        str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"),
        str(PROJECT_ROOT / "eval" / "run_eval.py"),
        "--n_questions", "5",
        "--pipeline_mode", "baseline",
        "--sleep", "1.0",
    ]
    print(f"[smoke A] running: {' '.join(cmd)}")
    print("[smoke A] SIMULATE_QUOTA_EXHAUSTION_AFTER_N_CALLS = <unset>")
    log_path = PROJECT_ROOT / "smoke_a_stdout.txt"
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(
            cmd, cwd=PROJECT_ROOT, env=env, stdout=f, stderr=subprocess.STDOUT
        )
    if proc.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise SystemExit(f"[smoke A] run_eval.py exited {proc.returncode}. Tail:\n{tail}")
    return log_path


def _find_latest_run_dir() -> Path:
    results = PROJECT_ROOT / "eval" / "results"
    dirs = sorted(
        [d for d in results.iterdir() if d.is_dir() and (d / "summary.json").exists()],
        key=lambda d: d.stat().st_mtime,
    )
    if not dirs:
        raise SystemExit("No completed run directories under eval/results/")
    return dirs[-1]


def _assert(cond: bool, msg: str) -> None:
    if cond:
        print(f"  [PASS] {msg}")
    else:
        print(f"  [FAIL] {msg}")
        raise SystemExit(1)


def main() -> None:
    log_path = _run()
    run_dir = _find_latest_run_dir()
    print(f"[smoke A] run_dir = {run_dir}")

    with (run_dir / "summary.json").open("r", encoding="utf-8") as f:
        payload = json.load(f)
    summary = payload["summary"]

    print("\n=== Smoke Test A — RESULTS ===")
    print(f"  run_id            : {summary['run_id']}")
    print(f"  n_questions       : {summary['n_questions']}")
    print(f"  n_keys_configured : {summary['n_keys_configured']}")
    print(f"  key_rotations     : {summary['key_rotations']}")
    print(f"  cooldown_events   : {summary['cooldown_events']}")
    print(f"  rate_limit_errors : {summary['rate_limit_errors']}")
    print(f"  per_key_calls     : {summary['per_key_call_counts']}")
    print(f"  wall_time_s       : {summary['wall_time_s']}")
    print(f"  overall           : {summary['overall']}")

    # Assertions per Part 16
    _assert(summary["n_questions"] == 5, "5 questions completed")
    _assert(summary["key_rotations"] == 0, "no rotations when simulation is off")

    # Cooldowns are allowed ONLY when n_keys==1 AND rate_limit_errors>0.
    # That path is real 429 load exhausting the openai SDK's own retries,
    # bubbling to our rotator, which — having nowhere to rotate to on a
    # single-key config — calls enter_cooldown() and resumes. With
    # multiple keys the same 429 would rotate silently and never
    # trigger cooldown.
    if summary["n_keys_configured"] == 1 and summary["rate_limit_errors"] > 0:
        print(f"  [NOTE] cooldown_events={summary['cooldown_events']} with 1-key "
              f"config + {summary['rate_limit_errors']} real 429s — this is the "
              "designed recovery path, not a regression.")
    else:
        _assert(summary["cooldown_events"] == 0, "no cooldowns")

    # All 4 RAGAS metrics returned real numbers on all 5 questions
    results_lines = (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    _assert(len(results_lines) == 5, "results.jsonl has exactly 5 rows")

    all_metrics_real = True
    for line in results_lines:
        row = json.loads(line)
        for m in ("faithfulness", "answer_relevancy",
                  "context_precision", "context_recall"):
            if row[m] is None:
                print(f"  [FAIL] {row['question_id']} has None for {m}")
                all_metrics_real = False
    _assert(all_metrics_real, "every question got all 4 RAGAS metrics")

    # Check generator + judge config were the locked values
    snap = json.load((run_dir / "progress.json").open(encoding="utf-8"))["config_snapshot"]
    _assert(snap["generation_model"] == "groq/qwen/qwen3.6-27b",
            "generator locked to qwen/qwen3.6-27b")
    _assert(snap["generation_reasoning_effort"] == "none",
            "generator reasoning_effort=none")
    _assert(snap["generation_max_tokens"] == 1200,
            "generator max_tokens=1200")
    _assert(snap["judge_model"] == "openai/gpt-oss-120b",
            "judge locked to openai/gpt-oss-120b")
    _assert(snap["judge_reasoning_effort"] == "low",
            "judge reasoning_effort=low")

    # Check no raw key strings appeared in log or summary
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    _assert("gsk_" not in log_text, "no `gsk_` prefix found in stdout log")
    summary_text = json.dumps(payload, ensure_ascii=False)
    _assert("gsk_" not in summary_text, "no `gsk_` prefix in summary.json payload")

    print("\n[smoke A] all assertions passed. run_dir kept for report.")


if __name__ == "__main__":
    main()
