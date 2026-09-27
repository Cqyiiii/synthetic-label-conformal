from __future__ import annotations

MAIN_METHODS: tuple[str, ...] = (
    "Split",
    "SPPI",
    "NNM",
    "CQR",
    "CQR-PPI",
    "RCP",
    "RCP-PTFT",
    "RCP-PPI",
    "RCP-PPI-CV",
    "RCP-Aug",
)

ABLATION_METHODS: tuple[str, ...] = (
    "RCP",
    "RCP-PTFT",
    "RCP-PPI",
    "RCP-PPI-CV",
    "RCP-Aug",
    "RCP-Aug-Permuted",
    "RCP-Extra",
    "RCP-Pool",
)

COUNTERPARTS: dict[str, str | None] = {
    "Split": None,
    "SPPI": "Split",
    "NNM": "Split",
    "CQR": None,
    "CQR-PPI": "CQR",
    "RCP": None,
    "RCP-PTFT": "RCP",
    "RCP-PPI": "RCP",
    "RCP-PPI-CV": "RCP",
    "RCP-Aug": "RCP",
    "RCP-Aug-Permuted": "RCP",
    "RCP-Extra": "RCP",
    "RCP-Pool": "RCP",
}

FORBIDDEN_MAIN_PREFIXES: tuple[str, ...] = ("Linear-",)
FORBIDDEN_MAIN_NAMES: frozenset[str] = frozenset(
    {
        "RCP-PPI-unit-one-step",
        "RCP-PPI-deterministic-gate",
        "RCP-synthetic-weighted-ERM",
        "RCP-synthetic-pretrain-one-step",
        "Budget-matched-supervised-RCP",
    }
)
