import argparse
import csv
import itertools
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs import get_search_space, resolve_domain_name  # noqa: E402


def product_dict(space: Dict[str, List[float]]) -> Iterable[Dict[str, float]]:
    keys = list(space.keys())
    for values in itertools.product(*(space[key] for key in keys)):
        yield dict(zip(keys, values))


def params_to_args(params: Dict[str, float]) -> List[str]:
    pnums = int(params["pnums"])
    return [
        "--lr", str(params["lr"]),
        "--weight_decay", str(params["weight_decay"]),
        "--t_batchsize", str(int(params["t_batchsize"])),
        "--s_pnums", str(pnums),
        "--t_pnums", str(pnums),
        "--weight", str(params["weight"]),
        "--dropout", str(params["dropout"]),
        "--nhid", str(int(params["nhid"])),
        "--alpha", str(params["alpha"]),
        "--epochs", str(int(params["epochs"])),
    ]


def run_trial(args: argparse.Namespace, trial_id: int, params: Dict[str, float]) -> Dict[str, float]:
    trial_json = Path(args.output_dir) / f"trial_{trial_id:05d}.json"
    if args.resume and trial_json.exists():
        with trial_json.open("r", encoding="utf-8") as f:
            result = json.load(f)
        return {"trial": trial_id, **params, **result}

    cmd = [
        sys.executable,
        str(ROOT / "main.py"),
        "--source", args.source,
        "--target", args.target,
        "--device", args.device,
        "--runs", str(args.runs),
        "--seed", str(args.seed + trial_id * args.runs),
        "--results-json", str(trial_json),
    ] + params_to_args(params)
    if args.synthetic:
        cmd.append("--synthetic")
    if args.verbose is not None:
        cmd.extend(["--verbose", str(args.verbose)])

    if args.dry_run:
        return {"trial": trial_id, "command": " ".join(cmd), **params}

    subprocess.run(cmd, check=True, cwd=ROOT)
    with trial_json.open("r", encoding="utf-8") as f:
        result = json.load(f)
    return {"trial": trial_id, **params, **result}


def load_existing_best(output_dir: Path, metric: str) -> Optional[Dict[str, float]]:
    best_path = output_dir / "best_config.json"
    if not best_path.exists():
        return None
    with best_path.open("r", encoding="utf-8") as f:
        best = json.load(f)
    if metric not in best:
        return None
    return best


def write_early_stop(output_dir: Path, best: Dict[str, float], target: float, metric: str) -> None:
    payload = {
        "reason": "target_metric_reached",
        "metric": metric,
        "target": target,
        "best_value": best[metric],
        "best_trial": best,
    }
    with (output_dir / "early_stop.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid/random search for ADAlign hyperparameters.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metric", choices=["micro_f1_mean", "macro_f1_mean"], default="micro_f1_mean")
    parser.add_argument("--strategy", choices=["grid", "random"], default="grid")
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--output-dir", default="results/search")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--target-metric", type=float, default=None,
                        help="stop once the selected metric reaches this value")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.source = resolve_domain_name(args.source)
    args.target = resolve_domain_name(args.target)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trials = list(product_dict(get_search_space(args.source, args.target)))
    if args.strategy == "random":
        rng = random.Random(args.seed)
        rng.shuffle(trials)
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")
    indexed_trials = [(i, params) for i, params in enumerate(trials) if i % args.num_shards == args.shard_index]
    if args.max_trials is not None:
        indexed_trials = indexed_trials[:args.max_trials]

    rows = []
    best = load_existing_best(output_dir, args.metric)
    if best is not None and args.target_metric is not None and best[args.metric] >= args.target_metric:
        write_early_stop(output_dir, best, args.target_metric, args.metric)
        print(json.dumps(best, indent=2))
        return

    for trial_id, params in indexed_trials:
        row = run_trial(args, trial_id, params)
        rows.append(row)
        if not args.dry_run and (best is None or row[args.metric] > best[args.metric]):
            best = row
            with (output_dir / "best_config.json").open("w", encoding="utf-8") as f:
                json.dump(best, f, indent=2)
        if (
            not args.dry_run
            and args.target_metric is not None
            and best is not None
            and best[args.metric] >= args.target_metric
        ):
            write_early_stop(output_dir, best, args.target_metric, args.metric)
            break

    csv_path = output_dir / "trials.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if best is not None:
        print(json.dumps(best, indent=2))
    else:
        print(f"Wrote {len(rows)} dry-run commands to {csv_path}")


if __name__ == "__main__":
    main()
