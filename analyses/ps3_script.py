# %% Imports
"""Run the PS3 claims modelling exercises and PS4 model evaluation."""

import argparse
import hashlib
import json
import shutil
import socket
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from analyses.claims_evaluation import (
    prepare_model_data,
    run_baseline,
    run_initial_lightgbm,
    run_tuned_lightgbm,
    sample_frames,
    write_model_metrics,
)
from analyses.model_comparison import read_model_metrics, save_comparison, write_metrics_csv


# %% PS3 Exercise 1 — Load claims and apply the fixed policy split

def load_model_data(frame=None):
    """Prepare pure premium and apply the fixed train/test split by policy ID."""
    return prepare_model_data(frame)


def load_input_data(data_path, network_timeout):
    """Load a local cache or use the instructor's data loader."""
    previous_timeout = socket.getdefaulttimeout()
    try:
        socket.setdefaulttimeout(network_timeout)
        return load_model_data(pd.read_parquet(data_path) if data_path else None)
    finally:
        socket.setdefaulttimeout(previous_timeout)


# %% PS3 Exercise 3 — Train the Tweedie GLM benchmark
# Dividing ClaimAmountCut by Exposure puts the target on a per-unit basis.
# Exposure weights reflect the different observation periods in fitting and scoring.

def save_tweedie_benchmark(frame, results_dir, verification):
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        metrics, coefficients, _, predictions = run_baseline(frame)
    print(coefficients.to_string())
    coefficients.rename_axis("feature").to_csv(
        results_dir / "tweedie_glm_coefficients.csv"
    )
    save_verified_run(metrics, predictions, results_dir, verification)


# %% PS3 Exercise 3 — Fit a more flexible parametric pipeline
# The quadratic specification uses polynomial terms for BonusMalus and VehPower.

def save_parametric_glm(frame, results_dir, verification):
    from analyses.parametric_glm import run_parametric_glm

    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        metrics, _, predictions = run_parametric_glm(frame)
    metrics["evaluation_caveat"] = (
        "The quadratic specification was chosen after test-tail diagnostics, "
        "so this test score is exploratory."
    )
    save_verified_run(metrics, predictions, results_dir, verification)


# %% PS3 Exercise 3 — Fit the initial LightGBM with the Tweedie objective

def save_initial_lightgbm(frame, results_dir, verification):
    metrics, predictions = run_initial_lightgbm(frame)
    save_verified_run(metrics, predictions, results_dir, verification)


# %% PS3 Exercise 3 — Tune LightGBM on the training sample

def save_tuned_lightgbm(frame, results_dir, verification):
    audit_dir = Path("data/tuning_runs") / datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    metrics, predictions = run_tuned_lightgbm(frame, audit_dir)
    save_verified_run(metrics, predictions, results_dir, verification)


# %% PS3 Exercise 3 — Compare pure premium rankings and plot Lorenz curves

def save_ranking_comparison(frame, results_dir, verification):
    train, test = sample_frames(frame)
    existing = read_model_metrics(results_dir)
    if "data_fingerprint" in existing and any(
        existing["data_fingerprint"].dropna() != verification["data_fingerprint"]
    ):
        raise ValueError("Saved metrics do not match this prepared dataset")
    save_comparison(train, test, results_dir)


# %% PS4 begins — Exercise 1: LightGBM monotonicity

def run_ps4_ex1(frame, results_dir):
    from analyses.lgbm_monotonicity import run_monotonicity_analysis

    _, _, metrics, _ = run_monotonicity_analysis(frame, results_dir)
    print(json.dumps(metrics, indent=2, allow_nan=False))
    return metrics


# %% Shared report and command-line support

def save_run(metrics, test_predictions, results_dir):
    """Save the audit metrics and a gitignored test-predictions file."""
    path = write_model_metrics(metrics, results_dir / "evaluated")
    test_predictions.rename("PurePremiumPredicted").rename_axis("IDpol").to_csv(
        results_dir / f"{metrics['model']}_predictions.csv"
    )
    print(json.dumps(metrics, indent=2, allow_nan=False))
    print(f"Saved {path}")


def save_verified_run(metrics, predictions, results_dir, verification):
    metrics.update(verification, verification_status="locally_trained")
    save_run(metrics, predictions, results_dir)


def run_provenance(frame):
    """Fingerprint the exact prepared values and executed modelling sources."""
    ordered = frame.sort_values("IDpol")
    digest = hashlib.sha256()
    digest.update(str(list(zip(ordered.columns, ordered.dtypes.astype(str)))).encode())
    digest.update(pd.util.hash_pandas_object(ordered, index=False).values.tobytes())
    sources = [*Path("analyses").glob("*.py"), *Path("ps3").rglob("*.py")]
    return digest.hexdigest(), {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_head_at_run": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(sources)
        },
        "versions": {
            name: version(name)
            for name in ["glum", "dask-ml", "lightgbm", "numpy", "pandas", "scikit-learn"]
        },
    }


def archive_results(results_dir):
    """Keep prior review evidence locally before a command replaces outputs."""
    if not results_dir.exists():
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = Path("data/run_backups") / stamp
    shutil.copytree(results_dir, destination)
    print(f"Archived prior outputs to {destination}")


def main(argv=None):
    ps3_sections = {
        "benchmark": save_tweedie_benchmark,
        "parametric": save_parametric_glm,
        "lightgbm": save_initial_lightgbm,
        "tuned-lightgbm": save_tuned_lightgbm,
        "compare": save_ranking_comparison,
    }
    ps4_sections = {"ps4-ex1": run_ps4_ex1}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        nargs="?",
        default="benchmark",
        choices=["check-data", *ps3_sections, "metrics", "all", *ps4_sections],
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--data", type=Path,
        help="Explicit local prepared Parquet cache; otherwise use the instructor loader",
    )
    parser.add_argument(
        "--save-data", type=Path,
        help="Save the shared prepared frame locally for independent rescoring",
    )
    parser.add_argument(
        "--network-timeout", type=float, default=300.0,
        help="Remote socket read timeout in seconds",
    )
    args = parser.parse_args(argv)

    if args.action == "metrics":
        archive_results(args.results_dir)
        print(f"Saved {write_metrics_csv(args.results_dir)}")
        return

    if args.network_timeout <= 0:
        parser.error("--network-timeout must be positive")
    frame = load_input_data(args.data, args.network_timeout)
    if args.save_data:
        args.save_data.parent.mkdir(parents=True, exist_ok=True)
        if args.save_data.exists():
            raise FileExistsError("Choose a new cache path to preserve the existing data cache")
        frame.to_parquet(args.save_data, index=False)
    if args.action == "check-data":
        train, test = sample_frames(frame)
        print(f"Validated {len(frame):,} policies ({len(train):,} train; {len(test):,} test)")
        return
    if args.action in ps4_sections:
        ps4_sections[args.action](frame, args.results_dir)
        return

    archive_results(args.results_dir)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    data_fingerprint, provenance = run_provenance(frame)
    verification = {"data_fingerprint": data_fingerprint, "run_provenance": provenance}
    sections = ps3_sections.values() if args.action == "all" else (ps3_sections[args.action],)
    for run_section in sections:
        run_section(frame, args.results_dir, verification)


# %% Command-line entry point
# In an interactive window, call main(["benchmark"]) or select another section.
if (
    __name__ == "__main__"
    and "__file__" in globals()
    and Path(sys.argv[0]).resolve() == Path(__file__).resolve()
):
    main()
