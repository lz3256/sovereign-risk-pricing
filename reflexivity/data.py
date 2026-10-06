from __future__ import annotations

import hashlib
import json
import os
import shlex
from pathlib import Path

import numpy as np
import pandas as pd

from .config import VINTAGE_CUTOFFS

METRICS = ["value", "grading", "eop_lag", "mop_lag"]
ENV_KEYS = {"DQ_CLIENT_ID", "DQ_CLIENT_SECRET", "DQ_PROXY_URL"}


def read_env(path: Path, allowed_keys=None) -> dict:
    """A deliberately small .env reader: literal values only, no shell expansion."""
    if not path.exists():
        return {}
    allowed_keys = ENV_KEYS if allowed_keys is None else set(allowed_keys)
    result = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:]
        if "=" not in line:
            raise ValueError(f"Invalid .env syntax at line {number}")
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in allowed_keys or key in result:
            raise ValueError(f"Unsupported .env key at line {number}; see .env.example")
        try:
            parts = shlex.split(value, comments=True, posix=True)
        except ValueError:
            raise ValueError(f"Invalid quoting in .env at line {number}") from None
        if len(parts) > 1:
            raise ValueError(f"Quote values containing spaces in .env line {number}")
        result[key] = parts[0] if parts else ""
    return result


def load_env(path: Path, allowed_keys=None) -> None:
    for key, value in read_env(path, allowed_keys).items():
        if not os.getenv(key):
            os.environ[key] = value


def normalize_qdf(df: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(df).copy()
    df = df.rename(columns={"date": "real_date", "grade": "grading"})
    required = {"real_date", "cid", "xcat", "value"}
    if required - set(df.columns):
        raise ValueError(f"Input must be long/QDF format. Missing: {sorted(required - set(df.columns))}")
    if df.empty:
        raise ValueError("Input contains no observations")
    if df[["real_date", "cid", "xcat"]].isna().any().any():
        raise ValueError("Date/cid/xcat cannot be missing")
    df["real_date"] = pd.to_datetime(df["real_date"], errors="raise", utc=True).dt.tz_convert(None).dt.normalize()
    df["cid"] = df["cid"].astype(str)
    df["xcat"] = df["xcat"].astype(str)
    for col in METRICS:
        if col not in df:
            df[col] = np.nan
        else:
            before = df[col].notna()
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if (before & df[col].isna()).any():
                raise ValueError(f"Non-numeric values in {col}")
            if np.isinf(df[col].to_numpy(dtype=float)).any():
                raise ValueError(f"Infinite values in {col}")
    if df.duplicated(["real_date", "cid", "xcat"]).any():
        raise ValueError("Duplicate date/cid/xcat keys; resolve duplicates before analysis")
    if ((df["grading"].dropna() < 1) | (df["grading"].dropna() > 3)).any():
        raise ValueError("grading must be between 1 and 3")
    # Lags describe the reference observation period, not data availability.
    # A period can end after the real-time information date (e.g. surveys or
    # estimates). Preserve signed metadata; never move values to that period.
    return df.sort_values(["cid", "xcat", "real_date"]).reset_index(drop=True)


def read_qdf(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return normalize_qdf(pd.read_parquet(path))
    if path.suffix.lower() in (".csv", ".gz"):
        return normalize_qdf(pd.read_csv(path))
    raise ValueError("Input must be CSV, CSV.gz or Parquet in long/QDF format")


def download_live(cfg: dict, cache: Path, refresh: bool = False, client_factory=None, credentials=None):
    identity = {k: cfg[k] for k in ("cids", "categories", "start_date", "end_date")}
    identity["client_version"] = "1.8.0"
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    folder = cache / fingerprint
    raw_path = folder / "raw.parquet"
    meta_path = folder / "download_audit.json"
    if not refresh and raw_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        meta["used_cache"] = True
        return read_qdf(raw_path), meta
    credential_source = os.environ if credentials is None else credentials
    client_id, client_secret = credential_source.get("DQ_CLIENT_ID"), credential_source.get("DQ_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise ValueError("Missing DQ_CLIENT_ID / DQ_CLIENT_SECRET. Fill local .env, then rerun --mode live.")
    if client_factory is None:
        from macrosynergy.download import JPMaQSDownload
        client_factory = JPMaQSDownload
    kwargs = dict(client_id=client_id, client_secret=client_secret, oauth=True,
                  check_connection=True, debug=False, print_debug_data=False)
    proxy = credential_source.get("DQ_PROXY_URL")
    if proxy:
        kwargs["proxy"] = {"http": proxy, "https": proxy}
    pieces, missing, failures = [], [], []
    try:
        connection = client_factory(**kwargs)
    except Exception as exc:
        raise RuntimeError(f"DataQuery connection failed ({type(exc).__name__}). Check local OAuth credentials, entitlement and proxy.") from None
    with connection as client:
        try:
            catalogue = set(client.get_catalogue(verbose=False))
        except Exception as exc:
            raise RuntimeError(f"DataQuery catalogue failed ({type(exc).__name__}). Check credentials/API entitlement/proxy. No demo fallback.") from None
        for role, xcat in cfg["categories"].items():
            wanted = [f"{cid}_{xcat}" for cid in cfg["cids"]]
            tickers = [ticker for ticker in wanted if ticker in catalogue]
            missing.extend(ticker for ticker in wanted if ticker not in catalogue)
            if not tickers:
                continue
            print(f"Downloading {role}: {len(tickers)} tickers", flush=True)
            try:
                piece = client.download(tickers=tickers, metrics=METRICS,
                                        start_date=cfg["start_date"], end_date=cfg["end_date"],
                                        get_catalogue=False, show_progress=False,
                                        debug=False, suppress_warning=False,
                                        as_dataframe=True, dataframe_format="qdf")
                if isinstance(piece, pd.DataFrame) and not piece.empty:
                    pieces.append(piece)
                else:
                    failures.append({"role": role, "error_type": "EmptyResponse"})
            except Exception as exc:
                # Do not serialize arbitrary API exception messages: they can contain tokens/URLs.
                failures.append({"role": role, "error_type": type(exc).__name__})
    if not pieces:
        raise RuntimeError("DataQuery returned no usable data. Check API entitlement; live failures never switch to simulation.")
    raw = normalize_qdf(pd.concat(pieces, ignore_index=True))
    meta = {"mode": "live", "used_cache": False, "request": identity,
            "jpmaqs_full_access": getattr(connection, "jpmaqs_access", None),
            "missing_catalogue_tickers": missing, "download_failures": failures,
            "retrieved_rows": len(raw), "retrieved_at_utc": pd.Timestamp.now(tz="UTC").isoformat()}
    folder.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(raw_path, index=False)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return raw, meta


def audit_coverage(raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = []
    groups = {key: part for key, part in raw.groupby(["cid", "xcat"], observed=True)}
    for cid in cfg["cids"]:
        for role, xcat in cfg["categories"].items():
            part = groups.get((cid, xcat))
            valid = part.dropna(subset=["value"]) if part is not None else pd.DataFrame()
            row = {"cid": cid, "role": role, "xcat": xcat, "observations": len(valid),
                   "start": None, "end": None, "coverage_fraction": 0.0,
                   "nonzero_observations": 0, "value_changes": 0, "missing_grade_fraction": np.nan,
                   "negative_eop_lag_observations": 0, "negative_mop_lag_observations": 0,
                   "minimum_eop_lag": np.nan, "minimum_mop_lag": np.nan,
                   "missing_eop_lag_fraction": np.nan, "missing_mop_lag_fraction": np.nan,
                   "acceptable_grade_fraction": np.nan, "pseudo_vintage_cutoff": VINTAGE_CUTOFFS.get(xcat),
                   "status": "missing"}
            if len(valid):
                calendar = pd.bdate_range(cfg["start_date"], cfg["end_date"])
                row.update(start=str(valid.real_date.min().date()), end=str(valid.real_date.max().date()),
                           coverage_fraction=float(valid.real_date.isin(calendar).sum() / max(1, len(calendar))),
                           nonzero_observations=int(valid.value.ne(0).sum()),
                           value_changes=int(valid.value.diff().dropna().ne(0).sum()),
                           negative_eop_lag_observations=int(valid.eop_lag.lt(0).sum()),
                           negative_mop_lag_observations=int(valid.mop_lag.lt(0).sum()),
                           minimum_eop_lag=float(valid.eop_lag.min()),
                           minimum_mop_lag=float(valid.mop_lag.min()),
                           missing_grade_fraction=float(valid.grading.isna().mean()),
                           missing_eop_lag_fraction=float(valid.eop_lag.isna().mean()),
                           missing_mop_lag_fraction=float(valid.mop_lag.isna().mean()),
                           acceptable_grade_fraction=float(valid.grading.le(cfg["max_grading"]).mean()),
                           status="available")
            rows.append(row)
    return pd.DataFrame(rows)


def read_spot(path: Path, convention: str = "local_per_usd") -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"real_date", "cid", "spot"}
    if required - set(df):
        raise ValueError("Spot CSV requires real_date,cid,spot; quote convention must be specified")
    df["real_date"] = pd.to_datetime(df.real_date, errors="raise", utc=True).dt.tz_convert(None).dt.normalize()
    if df[["real_date", "cid"]].isna().any().any() or df.duplicated(["real_date", "cid"]).any():
        raise ValueError("Spot dates/cids must be present with unique date/cid keys")
    df["spot"] = pd.to_numeric(df.spot, errors="raise")
    if not np.isfinite(df.spot).all() or (df.spot <= 0).any():
        raise ValueError("Spot must be finite and strictly positive")
    if convention == "usd_per_local":
        df["spot"] = 1.0 / df.spot
    elif convention != "local_per_usd":
        raise ValueError("Unknown spot convention")
    df["cid"] = df.cid.astype(str)
    return df[["real_date", "cid", "spot"]]
