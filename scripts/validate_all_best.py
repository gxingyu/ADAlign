import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Validate every per-transfer best_config.json.")
    parser.add_argument("output_root")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--gpus", default="0,1,2,4,5,6,7")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.output_root)
    gpus = [x for x in args.gpus.split(",") if x.strip()]
    best_files = sorted(root.glob("*/best_config.json"))
    for idx, best_file in enumerate(best_files):
        with best_file.open("r", encoding="utf-8") as f:
            best = json.load(f)
        params = best["params"]
        gpu = gpus[idx % len(gpus)]
        output_json = best_file.parent / "best_5run_validation.json"
        log_path = best_file.parent / "best_5run_validation.log"
        cmd = [
            sys.executable,
            str(ROOT / "main.py"),
            "--source", best["source"],
            "--target", best["target"],
            "--device", f"cuda:{gpu}",
            "--runs", str(args.runs),
            "--seed", str(args.seed),
            "--results-json", str(output_json),
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
        with log_path.open("w", encoding="utf-8") as log:
            log.write("command=" + " ".join(cmd) + "\n\n")
            subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    main()
