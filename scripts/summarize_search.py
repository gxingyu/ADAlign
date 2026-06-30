import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize ADAlign search best configs.")
    parser.add_argument("output_root")
    parser.add_argument("--metric", default="micro_f1_mean")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.output_root)
    rows = []
    for path in sorted(root.glob("**/best_config.json")):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        params = data.get("params", {})
        rows.append({
            "scenario": path.parent.name,
            "source": data.get("source"),
            "target": data.get("target"),
            "metric": args.metric,
            "metric_value": data.get(args.metric),
            "micro_f1_mean": data.get("micro_f1_mean"),
            "macro_f1_mean": data.get("macro_f1_mean"),
            "lr": params.get("lr", data.get("lr")),
            "weight_decay": params.get("weight_decay", data.get("weight_decay")),
            "t_batchsize": params.get("t_batchsize", data.get("t_batchsize")),
            "s_pnums": params.get("s_pnums", data.get("pnums")),
            "t_pnums": params.get("t_pnums", data.get("pnums")),
            "weight": params.get("weight", data.get("weight")),
            "dropout": params.get("dropout", data.get("dropout")),
            "nhid": params.get("nhid", data.get("nhid")),
            "alpha": params.get("alpha", data.get("alpha")),
            "epochs": params.get("epochs", data.get("epochs")),
        })

    csv_path = root / "best_summary.csv"
    md_path = root / "best_summary.md"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        with md_path.open("w", encoding="utf-8") as f:
            f.write("| scenario | micro_f1 | macro_f1 | lr | wd | M | pnums | lambda | dropout | kappa |\n")
            f.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
            for row in rows:
                f.write(
                    f"| {row['scenario']} | {row['micro_f1_mean']} | {row['macro_f1_mean']} | "
                    f"{row['lr']} | {row['weight_decay']} | {row['t_batchsize']} | "
                    f"{row['t_pnums']} | {row['weight']} | {row['dropout']} | {row['alpha']} |\n"
                )
    print(f"wrote {len(rows)} rows to {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
