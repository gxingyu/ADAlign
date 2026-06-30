import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Dict, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs import get_paper_target, iter_transfer_tasks  # noqa: E402


def safe_name(source: str, target: str) -> str:
    return f"{source}_to_{target}".replace("/", "_")


def best_reaches_target(out_dir: Path, metric: str, target: float | None) -> bool:
    if target is None:
        return False
    best_path = out_dir / "best_config.json"
    if not best_path.exists():
        return False
    try:
        with best_path.open("r", encoding="utf-8") as f:
            best = json.load(f)
    except Exception:
        return False
    value = best.get(metric)
    return value is not None and value >= target


def run_task(args: argparse.Namespace, task: Tuple[str, str], gpu: int) -> int:
    source, target = task
    out_dir = Path(args.output_root) / safe_name(source, target)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "search.log"
    paper_target = get_paper_target(source, target) if args.use_paper_targets else None
    if best_reaches_target(out_dir, args.metric, paper_target):
        with log_path.open("a", encoding="utf-8") as log:
            log.write(
                f"\nskipped={dt.datetime.now().isoformat()} "
                f"reason=paper_target_already_reached target={paper_target}\n"
            )
        return 0

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "grid_search.py"),
        "--source", source,
        "--target", target,
        "--device", f"cuda:{gpu}",
        "--strategy", args.strategy,
        "--max-trials", str(args.max_trials),
        "--runs", str(args.runs),
        "--seed", str(args.seed),
        "--metric", args.metric,
        "--output-dir", str(out_dir),
        "--verbose", str(args.verbose),
        "--resume",
    ]
    if paper_target is not None:
        cmd.extend(["--target-metric", str(paper_target)])
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n" + "=" * 80 + "\n")
        log.write(f"started={dt.datetime.now().isoformat()} gpu={gpu}\n")
        if paper_target is not None:
            log.write(f"paper_target={paper_target}\n")
        log.write("command=" + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"\nfinished={dt.datetime.now().isoformat()} returncode={proc.returncode}\n")
    return proc.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch ADAlign searches across GPUs.")
    parser.add_argument("--gpus", default="0,1,2,4,5,6,7")
    parser.add_argument("--max-trials", type=int, default=4)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strategy", choices=["random", "grid"], default="random")
    parser.add_argument("--metric", choices=["micro_f1_mean", "macro_f1_mean"], default="micro_f1_mean")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--exclude", default="", help="comma-separated SOURCE:TARGET pairs to skip")
    parser.add_argument("--no-paper-targets", dest="use_paper_targets", action="store_false",
                        help="disable early stopping at ADAlign paper table targets")
    parser.set_defaults(use_paper_targets=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_root = str(ROOT / "results" / f"paper_search_{stamp}")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    excluded = set()
    for item in args.exclude.split(","):
        if not item.strip():
            continue
        source, target = item.split(":", 1)
        excluded.add((source, target))
    tasks = [task for task in iter_transfer_tasks() if task not in excluded]
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    free_gpus = list(gpus)
    pending = list(tasks)
    active: Dict[object, int] = {}
    failures = 0

    with (output_root / "launcher.log").open("w", encoding="utf-8") as log:
        log.write(f"output_root={output_root}\n")
        log.write(f"tasks={tasks}\n")
        log.write(f"gpus={gpus}\n")
        log.flush()

        with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
            while pending or active:
                while pending and free_gpus:
                    task = pending.pop(0)
                    gpu = free_gpus.pop(0)
                    future = executor.submit(run_task, args, task, gpu)
                    active[future] = gpu
                    log.write(f"start task={task} gpu={gpu} time={dt.datetime.now().isoformat()}\n")
                    log.flush()

                done, _ = wait(active.keys(), timeout=30, return_when=FIRST_COMPLETED)
                if not done:
                    time.sleep(1)
                    continue
                for future in done:
                    gpu = active.pop(future)
                    free_gpus.append(gpu)
                    rc = future.result()
                    failures += int(rc != 0)
                    log.write(f"finish gpu={gpu} returncode={rc} time={dt.datetime.now().isoformat()}\n")
                    log.flush()

    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
