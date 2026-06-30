from __future__ import annotations

from copy import deepcopy
from typing import Dict, Iterable, List, Tuple


DOMAIN_ALIASES = {
    "A": "ACMv9",
    "C": "Citationv1",
    "D": "DBLPv7",
    "U": "USA",
    "B": "BRAZIL",
    "E": "EUROPE",
    "B1": "Blog1",
    "B2": "Blog2",
}


PAPER_DEFAULTS = {
    "lr": 0.001,
    "weight_decay": 0.001,
    "t_batchsize": 2048,
    "s_pnums": 0,
    "t_pnums": 10,
    "weight": 0.4,
    "dropout": 0.25,
    "nhid": 128,
    "alpha": 0.5,
    "epochs": 150,
    "num_layers": 2,
}


BASE_SEARCH_SPACE = {
    "lr": [0.001, 0.005, 0.01],
    "weight_decay": [0.001, 0.005, 0.01],
    "t_batchsize": [128, 1024, 2048, 4096],
    "pnums": [0, 1, 10, 15],
    "weight": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "dropout": [0.1, 0.25, 0.5],
    "nhid": [128],
    "alpha": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "epochs": [150],
}


SCENARIO_OVERRIDES = {
    ("USA", "BRAZIL"): {"pnums": [0, 1, 5, 10, 15]},
    ("USA", "EUROPE"): {
        "t_batchsize": [32, 64, 128, 1024, 2048, 4096],
        "pnums": [0, 1, 5, 10, 15],
        "dropout": [0.1, 0.2, 0.25, 0.5],
    },
    ("BRAZIL", "USA"): {
        "t_batchsize": [32, 64, 128, 1024, 2048, 4096],
        "pnums": [0, 1, 5, 10, 15],
    },
    ("BRAZIL", "EUROPE"): {
        "lr": [0.001, 0.003, 0.005, 0.01],
        "t_batchsize": [32, 64, 128, 1024, 2048, 4096],
        "pnums": [0, 1, 5, 10, 15],
    },
    ("EUROPE", "USA"): {
        "t_batchsize": [32, 64, 128, 1024, 2048, 4096],
    },
    ("EUROPE", "BRAZIL"): {
        "lr": [0.001, 0.003, 0.005, 0.01],
        "t_batchsize": [32, 64, 128, 1024, 2048, 4096],
    },
    ("Blog1", "Blog2"): {
        "t_batchsize": [32, 64, 128, 1024, 2048, 4096],
    },
    ("Blog2", "Blog1"): {
        "t_batchsize": [32, 64, 128, 1024, 2048, 4096],
    },
    ("DE", "EN"): {
        "t_batchsize": [32, 64, 128, 1024, 2048, 4096],
    },
    ("EN", "DE"): {
        "t_batchsize": [32, 64, 128, 1024, 2048, 4096],
    },
}


PAPER_TARGETS = {
    ("ACMv9", "Citationv1"): 0.8221,
    ("ACMv9", "DBLPv7"): 0.7806,
    ("Citationv1", "ACMv9"): 0.7624,
    ("Citationv1", "DBLPv7"): 0.7951,
    ("DBLPv7", "ACMv9"): 0.7438,
    ("DBLPv7", "Citationv1"): 0.8184,
    ("Blog1", "Blog2"): 0.5508,
    ("Blog2", "Blog1"): 0.5242,
    ("USA", "BRAZIL"): 0.7252,
    ("USA", "EUROPE"): 0.5323,
    ("BRAZIL", "USA"): 0.5571,
    ("BRAZIL", "EUROPE"): 0.5849,
    ("EUROPE", "USA"): 0.5541,
    ("EUROPE", "BRAZIL"): 0.7582,
    ("DE", "EN"): 0.6192,
    ("EN", "DE"): 0.6569,
}


def resolve_domain_name(name: str) -> str:
    return DOMAIN_ALIASES.get(name, name)


def scenario_key(source: str, target: str) -> Tuple[str, str]:
    return resolve_domain_name(source), resolve_domain_name(target)


def get_search_space(source: str, target: str) -> Dict[str, List[float]]:
    space = deepcopy(BASE_SEARCH_SPACE)
    space.update(SCENARIO_OVERRIDES.get(scenario_key(source, target), {}))
    return space


def get_paper_target(source: str, target: str) -> float | None:
    return PAPER_TARGETS.get(scenario_key(source, target))


def iter_transfer_tasks() -> Iterable[Tuple[str, str]]:
    citation = [("ACMv9", "Citationv1"), ("ACMv9", "DBLPv7"), ("Citationv1", "ACMv9"),
                ("Citationv1", "DBLPv7"), ("DBLPv7", "ACMv9"), ("DBLPv7", "Citationv1")]
    airport = [("USA", "BRAZIL"), ("USA", "EUROPE"), ("BRAZIL", "USA"),
               ("BRAZIL", "EUROPE"), ("EUROPE", "USA"), ("EUROPE", "BRAZIL")]
    blog = [("Blog1", "Blog2"), ("Blog2", "Blog1")]
    twitch = [("DE", "EN"), ("EN", "DE")]
    yield from citation + airport + blog + twitch
