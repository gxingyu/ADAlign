import argparse
import json
import os
import os.path as osp
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from pygda.datasets import AirportDataset, BlogDataset, CitationDataset, MAGDataset, TwitchDataset
from torch_geometric.data import Data
from torch_geometric.utils import degree

from configs import PAPER_DEFAULTS, resolve_domain_name
from model.ncfm import NCFM


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataset(domain: str, root: str):
    domain = resolve_domain_name(domain)
    if domain in {"DBLPv7", "ACMv9", "Citationv1"}:
        path = osp.join(root, "data", "Citation", domain)
        return CitationDataset(path, domain)
    if domain in {"BRAZIL", "EUROPE", "USA"}:
        path = osp.join(root, "data", "Airport", domain)
        return AirportDataset(path, domain)
    if domain in {"Blog1", "Blog2"}:
        path = osp.join(root, "data", "Blog", domain)
        return BlogDataset(path, domain)
    if domain in {"MAG_CN", "MAG_DE", "MAG_FR", "MAG_JP", "MAG_RU", "MAG_US"}:
        path = osp.join(root, "data", "MAG", domain)
        return MAGDataset(path, domain)
    if domain in {"DE", "EN", "ES", "FR", "PT", "RU"}:
        path = osp.join(root, "data", "Twitch", domain)
        return TwitchDataset(path, domain)
    raise ValueError(f"Unsupported domain: {domain}")


def ensure_features(data: Data, device: str, default_num_features: int = 241) -> Data:
    if not hasattr(data, "x") or data.x is None:
        node_degrees = degree(data.edge_index[0], num_nodes=data.num_nodes).long()
        num_classes = max(default_num_features, int(node_degrees.max().item()) + 1)
        data.x = torch.nn.functional.one_hot(node_degrees, num_classes=num_classes).float()
    return data.to(device)


def load_transfer(source: str, target: str, device: str, synthetic: bool = False) -> Tuple[Data, Data]:
    if synthetic:
        edge_index = torch.tensor(
            [[0, 1, 2, 3, 4, 5, 0, 2, 4, 6], [1, 2, 3, 4, 5, 6, 2, 4, 6, 0]],
            dtype=torch.long,
        )
        source_data = Data(
            x=torch.randn(8, 5),
            edge_index=edge_index,
            y=torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long),
        )
        target_data = Data(
            x=torch.randn(8, 5),
            edge_index=edge_index,
            y=torch.tensor([0, 1, 1, 0, 0, 1, 1, 0], dtype=torch.long),
        )
        return source_data.to(device), target_data.to(device)

    root = osp.dirname(osp.realpath(__file__))
    source_dataset = build_dataset(source, root)
    target_dataset = build_dataset(target, root)
    source_data = ensure_features(source_dataset[0], device)
    target_data = ensure_features(target_dataset[0], device)
    return source_data, target_data


def build_model(args: argparse.Namespace, source_data: Data) -> NCFM:
    num_features = source_data.x.size(1)
    num_classes = len(np.unique(source_data.y.detach().cpu().numpy()))
    return NCFM(
        in_dim=num_features,
        hid_dim=args.nhid,
        num_classes=num_classes,
        device=args.device,
        num_layers=args.num_layers,
        dropout=args.dropout,
        s_pnums=args.s_pnums,
        t_pnums=args.t_pnums,
        weight=args.weight,
        weight_decay=args.weight_decay,
        lr=args.lr,
        epoch=args.epochs,
        t_batchsize=args.t_batchsize,
        alpha=args.alpha,
        verbose=args.verbose,
    )


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    micro_f1_scores: List[float] = []
    macro_f1_scores: List[float] = []

    for run_idx in range(args.runs):
        seed = args.seed + run_idx
        set_seed(seed)
        source_data, target_data = load_transfer(args.source, args.target, args.device, args.synthetic)
        model = build_model(args, source_data)
        max_mi, max_ma = model.fit(source_data, target_data)
        micro_f1_scores.append(float(max_mi))
        macro_f1_scores.append(float(max_ma))
        print(f"run={run_idx} seed={seed} max_micro_f1={max_mi:.6f} max_macro_f1={max_ma:.6f}")

    result = {
        "source": resolve_domain_name(args.source),
        "target": resolve_domain_name(args.target),
        "runs": args.runs,
        "params": {
            "nhid": args.nhid,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "s_pnums": args.s_pnums,
            "t_pnums": args.t_pnums,
            "weight": args.weight,
            "weight_decay": args.weight_decay,
            "lr": args.lr,
            "epochs": args.epochs,
            "t_batchsize": args.t_batchsize,
            "alpha": args.alpha,
        },
        "micro_f1_scores": micro_f1_scores,
        "macro_f1_scores": macro_f1_scores,
        "micro_f1_mean": float(np.mean(micro_f1_scores)),
        "micro_f1_std": float(np.std(micro_f1_scores)),
        "macro_f1_mean": float(np.mean(macro_f1_scores)),
        "macro_f1_std": float(np.std(macro_f1_scores)),
    }

    print(json.dumps(result, indent=2))
    if args.results_json:
        os.makedirs(osp.dirname(osp.abspath(args.results_json)), exist_ok=True)
        with open(args.results_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ADAlign / NCFM graph domain adaptation.")
    parser.add_argument("--source", type=str, default="ACMv9", help="source domain")
    parser.add_argument("--target", type=str, default="Citationv1", help="target domain")
    parser.add_argument("--device", type=str, default="cuda:0", help="cpu or cuda device")
    parser.add_argument("--runs", type=int, default=5, help="number of repeated runs")
    parser.add_argument("--seed", type=int, default=42, help="base random seed")
    parser.add_argument("--nhid", type=int, default=PAPER_DEFAULTS["nhid"], help="hidden feature dimension")
    parser.add_argument("--num_layers", type=int, default=PAPER_DEFAULTS["num_layers"])
    parser.add_argument("--dropout", type=float, default=PAPER_DEFAULTS["dropout"])
    parser.add_argument("--s_pnums", type=int, default=PAPER_DEFAULTS["s_pnums"])
    parser.add_argument("--t_pnums", type=int, default=PAPER_DEFAULTS["t_pnums"])
    parser.add_argument("--weight", type=float, default=PAPER_DEFAULTS["weight"], help="alignment loss weight lambda")
    parser.add_argument("--weight_decay", type=float, default=PAPER_DEFAULTS["weight_decay"])
    parser.add_argument("--lr", type=float, default=PAPER_DEFAULTS["lr"])
    parser.add_argument("--epochs", type=int, default=PAPER_DEFAULTS["epochs"])
    parser.add_argument("--t_batchsize", type=int, default=PAPER_DEFAULTS["t_batchsize"], help="frequency number M")
    parser.add_argument("--alpha", type=float, default=PAPER_DEFAULTS["alpha"], help="amplitude weight kappa")
    parser.add_argument("--verbose", type=int, default=2)
    parser.add_argument("--results-json", type=str, default=None)
    parser.add_argument("--synthetic", action="store_true", help="run a tiny synthetic smoke test instead of loading datasets")
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
