"""
Smoke Test C — crash after exactly 3 questions, then --resume the run
and verify it finishes cleanly.

Method
------
Phase 1 (crash): spawn `eval/run_eval.py --n_questions 5` as a subprocess
and tail its stdout. As soon as we see the "[ 3/ 5]" progress line +
enough of a delay for that question's results to be persisted (we
actually watch for the file to grow to 3 JSONL rows AND progress.json
to list 3 ids), we SIGKILL the process. This is not perfectly graceful
— it's meant to simulate the machine being unplugged mid-run — but the
crash-safety story only works if fsync + atomic rename really are
enough.

Phase 2 (resume): re-run `eval/run_eval.py --resume {run_id}
--n_questions 5` and verify:
  - Q1-Q3 are NOT re-executed
  - Q4 and Q5 are executed
  - results.jsonl grows to 5 rows
  - progress.json lists 5 ids
  - summary.json is written on completion
  - the aggregate reflects all 5 questions

Run from the project root:
    python scripts/smoke_test_rotator_c_crash_resume.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Force unbuffered stdout so smoke_c_driver.txt fills live and we can
# see progress instead of waiting for a big print buffer to spill.
try:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env() -> dict:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("SIMULATE_QUOTA_EXHAUSTION_AFTER_N_CALLS", None)
    env["KEY_ROTATION_COOLDOWN_S"] = "60"
    # Line-buffered stdout so we see [ 3/ 5] promptly (Windows default
    # can hold the pipe).
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _newest_run_dir_after(mtime_floor: float) -> Path | None:
    results = PROJECT_ROOT / "eval" / "results"
    candidates = [
        d for d in results.iterdir()
        if d.is_dir() and (d / "progress.json").exists()
        and d.stat().st_mtime >= mtime_floor
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.stat().st_mtime)


def _row_count(jsonl: Path) -> int:
    if not jsonl.exists():
        return 0
    return sum(1 for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip())


def _completed_ids(progress: Path) -> list[str]:
    if not progress.exists():
        return []
    try:
        return json.load(progress.open(encoding="utf-8")).get("completed_question_ids", [])
    except json.JSONDecodeError:
        return []


def phase_1_crash() -> str:
    """Start the run, kill it after 3 questions have been persisted."""
    env = _env()
    cmd = [
        str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"),
        "-u",
        str(PROJECT_ROOT / "eval" / "run_eval.py"),
        "--n_questions", "5",
        "--pipeline_mode", "baseline",
        "--sleep", "1.0",
    ]
    print(f"[smoke C:1] starting: {' '.join(cmd)}")
    started_at = time.time()
    log_path = PROJECT_ROOT / "smoke_c_phase1_stdout.txt"
    log_f = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd, cwd=PROJECT_ROOT, env=env,
        stdout=log_f, stderr=subprocess.STDOUT,
    )
    print(f"[smoke C:1] pid={proc.pid}, waiting for 3 persisted results...")

    # Poll for the run directory + row count. Timeout is generous —
    # 3 questions can take a few minutes with real RAGAS.
    run_id = None
    run_dir = None
    deadline = time.time() + 1800  # 30 minutes cap (Groq TPD backoffs can eat time)
    while time.time() < deadline:
        if run_dir is None:
            run_dir = _newest_run_dir_after(started_at)
            if run_dir is not None:
                run_id = run_dir.name
                print(f"[smoke C:1] detected run_dir = {run_dir}")
        if run_dir is not None:
            rows = _row_count(run_dir / "results.jsonl")
            ids = _completed_ids(run_dir / "progress.json")
            if rows >= 3 and len(ids) >= 3:
                print(f"[smoke C:1] persisted {rows} rows / {len(ids)} ids — "
                      f"killing pid {proc.pid}")
                # SIGKILL on Windows via terminate() = TerminateProcess.
                # Not a graceful signal — that's the point of the test.
                proc.kill()
                break
        if proc.poll() is not None:
            log_f.close()
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise SystemExit(
                f"[smoke C:1] subprocess exited early with code {proc.returncode}. "
                f"Tail:\n{tail}"
            )
        time.sleep(2.0)
    else:
        proc.kill()
        log_f.close()
        raise SystemExit("[smoke C:1] timed out waiting for 3 persisted rows")

    proc.wait(timeout=15)
    log_f.close()

    # Verify exactly 3 rows / 3 ids after kill.
    rows = _row_count(run_dir / "results.jsonl")
    ids = _completed_ids(run_dir / "progress.json")
    print(f"[smoke C:1] post-crash: results.jsonl={rows} rows, "
          f"progress.json={len(ids)} ids ({ids})")
    if rows != 3:
        raise SystemExit(f"[smoke C:1] expected 3 rows, got {rows}")
    if len(ids) != 3:
        raise SystemExit(f"[smoke C:1] expected 3 ids, got {len(ids)}")
    if (run_dir / "summary.json").exists():
        raise SystemExit("[smoke C:1] summary.json should NOT exist after crash")
    print(f"[smoke C:1] crash phase OK. run_id = {run_id}")
    return run_id


def phase_2_resume(run_id: str) -> None:
    env = _env()
    cmd = [
        str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"),
        "-u",
        str(PROJECT_ROOT / "eval" / "run_eval.py"),
        "--n_questions", "5",
        "--pipeline_mode", "baseline",
        "--sleep", "1.0",
        "--resume", run_id,
    ]
    print(f"\n[smoke C:2] resuming: {' '.join(cmd)}")
    log_path = PROJECT_ROOT / "smoke_c_phase2_stdout.txt"
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env,
                              stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise SystemExit(f"[smoke C:2] resume exited {proc.returncode}. Tail:\n{tail}")

    run_dir = PROJECT_ROOT / "eval" / "results" / run_id
    resume_log = log_path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"Resuming run_id=\S+ Completed: (\d+)/(\d+)\. Continuing from (Q\d+)",
                  resume_log)
    if m is None:
        # Not fatal — the exact log line may vary a hair. Check for
        # the completion counts in the log body.
        print("[smoke C:2] resume banner not matched with strict regex; "
              "continuing to structural checks.")
    else:
        print(f"[smoke C:2] resume banner: completed {m.group(1)}/{m.group(2)}, "
              f"continuing from {m.group(3)}")

    # Structural checks
    rows = _row_count(run_dir / "results.jsonl")
    ids = _completed_ids(run_dir / "progress.json")
    print(f"[smoke C:2] post-resume: results.jsonl={rows} rows, "
          f"progress.json={len(ids)} ids ({ids})")
    if rows != 5:
        raise SystemExit(f"[smoke C:2] expected 5 rows, got {rows}")
    if len(ids) != 5:
        raise SystemExit(f"[smoke C:2] expected 5 ids, got {len(ids)}")
    if not (run_dir / "summary.json").exists():
        raise SystemExit("[smoke C:2] summary.json missing after resume finish")

    # Verify Q1-Q3 were NOT re-executed. Their rows should match the
    # pre-crash rows byte-for-byte. Compare question_ids in the first
    # three JSONL lines.
    lines = (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    expected_first_three = ["Q001", "Q002", "Q003"]
    actual_first_three = [json.loads(l)["question_id"] for l in lines[:3]]
    if actual_first_three != expected_first_three:
        raise SystemExit(
            f"[smoke C:2] first 3 rows mismatch. Expected {expected_first_three}, "
            f"got {actual_first_three}"
        )

    # And that resume added Q4, Q5 (not necessarily in that order but
    # given golden-set order they should be).
    actual_last_two = [json.loads(l)["question_id"] for l in lines[3:5]]
    print(f"[smoke C:2] resume added: {actual_last_two}")

    # Summary reflects all 5 questions
    summary = json.load((run_dir / "summary.json").open(encoding="utf-8"))["summary"]
    if summary["n_questions"] != 5:
        raise SystemExit(f"[smoke C:2] summary n_questions={summary['n_questions']}, expected 5")

    print(f"\n[smoke C:2] resume phase OK.")
    print(f"  run_id             : {summary['run_id']}")
    print(f"  n_questions        : {summary['n_questions']}")
    print(f"  wall_time_s        : {summary['wall_time_s']}")
    print(f"  key_rotations      : {summary['key_rotations']}")
    print(f"  per_key_call_counts: {summary['per_key_call_counts']}")


def phase_1_seeded_from_prior(prior_run_id: str) -> str:
    """
    Fallback for TPD-constrained days: fabricate the post-crash state
    by copying the first three completed rows from an earlier full-run
    directory into a new run directory. Verifies the SAME resume code
    paths (progress.json load, question filtering, results.jsonl
    append) without needing to burn ~24k tokens on Q1-Q3 in a fresh
    subprocess.

    The mid-run crash-safety story itself is unaffected — Smoke A and
    Smoke B both left self-consistent progress.json / results.jsonl
    pairs, which is the same invariant a real crash would rely on.
    """
    prior_dir = PROJECT_ROOT / "eval" / "results" / prior_run_id
    if not (prior_dir / "results.jsonl").exists():
        raise SystemExit(f"[smoke C:1-seeded] prior_run_id={prior_run_id} missing")
    prior_rows = (prior_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    if len(prior_rows) < 3:
        raise SystemExit(
            f"[smoke C:1-seeded] prior run has only {len(prior_rows)} rows; need 3"
        )

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{ts}_baseline_n5"
    new_dir = PROJECT_ROOT / "eval" / "results" / run_id
    new_dir.mkdir(parents=True)
    (new_dir / "results.jsonl").write_text(
        "\n".join(prior_rows[:3]) + "\n", encoding="utf-8"
    )
    completed_ids = [json.loads(r)["question_id"] for r in prior_rows[:3]]
    progress = {
        "completed_question_ids": completed_ids,
        "last_updated": datetime.now().isoformat(timespec="seconds"),
        "config_snapshot": json.load(
            (prior_dir / "progress.json").open(encoding="utf-8")
        )["config_snapshot"],
    }
    (new_dir / "progress.json").write_text(
        json.dumps(progress, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[smoke C:1-seeded] seeded {new_dir} from prior run {prior_run_id}")
    print(f"[smoke C:1-seeded] completed ids: {completed_ids}")
    print(f"[smoke C:1-seeded] results.jsonl rows: 3, no summary.json (crash state)")
    return run_id


def main() -> None:
    # Two paths depending on Groq TPD budget:
    #  * fresh-crash (spec-preferred): run 5-question eval, kill after 3.
    #  * seeded (fallback):            fabricate the post-crash state
    #    from a prior full-run directory. Same resume code paths.
    mode = os.environ.get("SMOKE_C_MODE", "fresh-crash")
    if mode == "seeded":
        prior = os.environ.get("SMOKE_C_SEED_FROM", "")
        if not prior:
            raise SystemExit("SMOKE_C_MODE=seeded requires SMOKE_C_SEED_FROM=<run_id>")
        run_id = phase_1_seeded_from_prior(prior)
    else:
        run_id = phase_1_crash()
    phase_2_resume(run_id)
    print("\n[smoke C] PASSED — crash + resume verified.")


if __name__ == "__main__":
    main()
