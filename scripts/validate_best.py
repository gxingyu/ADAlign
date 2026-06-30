import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Validate the best search trial with repeated runs.")
    parser.add_argument("output_root")
    parser.add_argument("--metric", default="micro_f1_mean")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.output_root)
    best = None
    best_path = None
    for path in sorted(root.glob("**/trial_*.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                row = json.load(f)
        except Exception:
            continue
        value = row.get(args.metric)
        if value is not None and (best is None or value > best.get(args.metric, float("-inf"))):
            best = row
            best_path = path
    if best is None:
        raise RuntimeError(f"No trial JSON with metric {args.metric} under {root}")

    params = best["params"]
    output_json = args.output_json or str(root / "best_5run_validation.json")
    cmd = [
        sys.executable,
        str(ROOT / "main.py"),
        "--source", best["source"],
        "--target", best["target"],
        "--device", args.device,
        "--runs", str(args.runs),
        "--seed", str(args.seed),
        "--results-json", output_json,
        "--nhid", str(params["nhid"]),
        "--num_layers", str(params["num_layers"]),
        "--dropout", str(params["dropout"]),
        "--s_pnums", str(params["s_pnums"]),
        "--t_pnums", str(params["t_pnums"]),
        "--weight", str(params["weight"]),
        "--weight_decay", str(params["weight_decay"]),
        "--lr", str(params["lr"]),
        "--epochs", str(params["epochs"]),
        "--t_batchsize", str(params["t_batchsize"]),
        "--alpha", str(params["alpha"]),
        "--verbose", "0",
    ]
    meta = {
        "selected_from": str(best_path),
        "selection_metric": args.metric,
        "selection_value": best[args.metric],
        "validation_command": cmd,
        "selected_trial": best,
    }
    with (root / "best_selection_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
