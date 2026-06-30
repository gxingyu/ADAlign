import argparse
import json
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Monitor ADAlign JSON trial outputs.")
    parser.add_argument("output_root")
    parser.add_argument("--metric", default="micro_f1_mean")
    parser.add_argument("--expected-trials", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.output_root)
    trials = sorted(root.glob("**/trial_*.json"))
    best = None
    mtimes = []
    for path in trials:
        try:
            with path.open("r", encoding="utf-8") as f:
                row = json.load(f)
        except Exception:
            continue
        mtimes.append(path.stat().st_mtime)
        value = row.get(args.metric)
        if value is not None and (best is None or value > best[0]):
            best = (value, path, row)

    print(f"root={root}")
    print(f"completed_trials={len(trials)}")
    if args.expected_trials:
        print(f"expected_trials={args.expected_trials}")
        print(f"progress={len(trials) / args.expected_trials:.4%}")
    if len(mtimes) >= 2:
        elapsed = max(mtimes) - min(mtimes)
        rate = (len(mtimes) - 1) / elapsed if elapsed > 0 else 0
        print(f"rate_trials_per_min={rate * 60:.2f}")
        if args.expected_trials and rate > 0:
            remaining = max(args.expected_trials - len(trials), 0) / rate
            print(f"eta_hours={remaining / 3600:.2f}")
    if best:
        value, path, row = best
        print(f"best_metric={args.metric}")
        print(f"best_value={value}")
        print(f"best_path={path}")
        print(json.dumps(row.get("params", row), indent=2))
    print(f"checked_at={time.strftime('%Y-%m-%dT%H:%M:%S%z')}")


if __name__ == "__main__":
    main()
