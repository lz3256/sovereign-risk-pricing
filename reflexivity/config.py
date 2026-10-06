from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHOCK_ROLES = ("inflation_shock", "growth_shock")
PRIMARY_CONTROLS = ("growth_z", "credit_z", "real_rate_z", "carry_z", "momentum_z", "fragility")
VINTAGE_CUTOFFS = {
    "ALLIEDGDP_NSA": "2024-01-01",
    "GGFCDSGDP_NSA": "2024-01-01",
    "CGFCDSGDP_NSA": "2024-01-01",
}
SOURCES = {
    "API": "https://docs.macrosynergy.com/latest/macrosynergy.download.jpmaqs.html",
    "debt": "https://macrosynergy.com/academy/notebooks/external-and-foreign-currency-debt/",
    "surprises": "https://macrosynergy.com/academy/notebooks/economic-surprises/",
    "fx": "https://macrosynergy.com/academy/notebooks/fx-forward-returns/",
    "metrics": "https://docs.macrosynergy.com/latest/common_definitions.html",
}


def load_config(path: Path | str = ROOT / "config.json") -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"cids", "categories", "start_date", "end_date", "delays", "hold_days"}
    if required - cfg.keys():
        raise ValueError(f"Config missing keys: {sorted(required - cfg.keys())}")
    from datetime import date
    if date.fromisoformat(cfg["start_date"]) >= date.fromisoformat(cfg["end_date"]):
        raise ValueError("start_date must precede end_date")
    roles = {"inflation_shock", "growth_shock", "fx_return", "carry", "growth", "credit", "real_rate", "cds", "debt", "reserves", "fx_untradable", "fx_targeted"}
    if roles - cfg["categories"].keys():
        raise ValueError(f"Missing category roles: {sorted(roles - cfg['categories'].keys())}")
    if len(set(cfg["categories"].values())) != len(cfg["categories"]):
        raise ValueError("Each category must have a unique role")
    if not cfg["cids"] or len(cfg["cids"]) != len(set(cfg["cids"])):
        raise ValueError("cids must be nonempty and unique")
    if cfg["categories"]["fx_return"] != "FXXRUSD_NSA":
        raise ValueError("This implementation requires unscaled USD-based FXXRUSD_NSA returns")
    if not set(cfg["fragility_roles"]) <= {"debt", "reserves"}:
        raise ValueError("Fragility roles must be debt and/or reserves; directions are explicitly defined")
    for key in ("delays", "daily_response_days", "cumulative_horizons", "feedback_horizons"):
        if not cfg[key] or any(not isinstance(x, int) or x < 1 for x in cfg[key]):
            raise ValueError(f"{key} must contain positive integers")
        cfg[key] = sorted(set(cfg[key]))
    if cfg["daily_response_days"][0] < 2:
        raise ValueError("Daily responses start at t+2, after next-close execution")
    for key in ("hold_days", "min_scale_days", "min_shock_events", "min_regression_rows", "bootstrap_block_days", "signal_window_days", "oos_test_years"):
        if not isinstance(cfg[key], int) or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["bootstrap_reps"] < 50:
        raise ValueError("Use at least 50 bootstrap replications; 499+ recommended for final research")
    if not 1 <= cfg["min_fragility_components"] <= len(cfg["fragility_roles"]):
        raise ValueError("Invalid minimum fragility component count")
    if cfg["max_grading"] not in (1, 2, 3):
        raise ValueError("max_grading must be 1, 2 or 3")
    if cfg["cost_bps_one_way"] < 0 or min(cfg["cost_scenarios_bps"]) < 0:
        raise ValueError("Transaction costs must be nonnegative")
    cfg["cost_scenarios_bps"] = sorted(set([*cfg["cost_scenarios_bps"], cfg["cost_bps_one_way"]]))
    if cfg["ridge_alpha"] <= 0 or cfg["max_forward_fill_days"] < 0:
        raise ValueError("ridge_alpha must be positive; forward fill must be nonnegative")
    if not 0 < cfg["signal_min_observed_fraction"] <= 1:
        raise ValueError("Invalid signal observation fraction")
    return cfg
