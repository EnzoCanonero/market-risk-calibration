from __future__ import annotations

import hashlib
import json
import platform
from importlib.metadata import version
from math import sqrt
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from scipy.stats import norm

from experiments.simulation.scenarios import (
    generate_scale_scenario,
    generate_shape_scenario,
    shape_cdf,
    student_t_cdf,
)
from market_risk_calibration.metrics import pinball_loss, summarize_forecasts
from market_risk_calibration.sequential import sequential_forecasts
from market_risk_calibration.volatility import expanding_volatility, ewma_volatility


ALPHA = 0.05
WINDOW = 500
GAMMA = 0.001
RAW_QUANTILE = float(norm.ppf(ALPHA))
REPLICATIONS = 200
WARMUP = 1000
CALIBRATION_SIZE = 1000
EVALUATION_SIZE = 4000
COVERAGE_WINDOW = 250
CHANGE_OFFSETS = (750, 1500, 2250, 3000, 3750)
CHANGE_JITTER = 50
# Separate streams, both disjoint from the pilot's seeds 19457–19520.
INNOVATION_SEED_START = 100000
TIMING_SEED_START = 200000

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "results" / "simulation"
METRICS = (
    "mean_probability_error",
    "mean_pinball_loss",
    "mean_excess_pinball_loss",
    "mean_local_coverage_error",
    "breach_rate",
    "mean_downside_buffer",
    "non_finite_count",
)


def run_replication(replication: int) -> pd.DataFrame:
    innovation_seed = INNOVATION_SEED_START + replication
    timing_seed = TIMING_SEED_START + replication
    evaluation_start = WARMUP + CALIBRATION_SIZE
    length = evaluation_start + EVALUATION_SIZE

    timing_rng = np.random.default_rng(timing_seed)
    jitter = timing_rng.integers(-CHANGE_JITTER, CHANGE_JITTER + 1, len(CHANGE_OFFSETS))
    change_points = (evaluation_start + np.array(CHANGE_OFFSETS) + jitter).tolist()

    # Reuse innovations across scale magnitudes and timing across changing scenarios.
    scenarios = {
        "stationary": generate_scale_scenario(length, seed=innovation_seed),
        "scale_1.5": generate_scale_scenario(
            length, seed=innovation_seed, change_points=change_points, high_scale=1.5,
        ),
        "scale_2.5": generate_scale_scenario(
            length, seed=innovation_seed, change_points=change_points, high_scale=2.5,
        ),
        "shape": generate_shape_scenario(
            length, seed=innovation_seed, change_points=change_points,
        ),
    }

    records: list[dict[str, Union[str, int, float]]] = []
    for scenario_name, scenario in scenarios.items():
        returns = scenario["return"]
        evaluation = scenario.iloc[evaluation_start:]
        oracle_loss = pinball_loss(
            evaluation["return"], evaluation["oracle_quantile"], alpha=ALPHA,
        )

        if scenario_name in ("scale_1.5", "scale_2.5"):
            models = {
                "expanding": expanding_volatility(returns, warmup=WARMUP),
                "ewma": ewma_volatility(returns, warmup=WARMUP),
                "oracle": scenario["true_volatility"],
            }
        else:
            models = {"oracle": scenario["true_volatility"]}

        # Each method receives the same returns and archived scale forecasts.
        for model_name, volatility in models.items():
            for method, gamma in (("raw", float("nan")), ("rolling", 0.0), ("aci", GAMMA)):
                if method == "raw":
                    bounds = volatility.iloc[evaluation_start:] * RAW_QUANTILE
                    forecasts = pd.DataFrame({
                        "return": evaluation["return"],
                        "bound": bounds,
                        "breach": evaluation["return"] < bounds,
                    })
                else:
                    forecasts = sequential_forecasts(
                        returns, volatility, warmup=WARMUP,
                        calibration_size=CALIBRATION_SIZE, window=WINDOW,
                        alpha=ALPHA, gamma=gamma,
                    )
                metrics = summarize_forecasts(
                    forecasts, alpha=ALPHA, coverage_window=COVERAGE_WINDOW,
                )
                if scenario_name == "shape":
                    probabilities = shape_cdf(
                        forecasts["bound"], evaluation["jump_probability"],
                    )
                else:
                    probabilities = student_t_cdf(
                        forecasts["bound"], evaluation["true_volatility"],
                    )

                metrics["mean_probability_error"] = float((probabilities - ALPHA).abs().mean())
                metrics["mean_excess_pinball_loss"] = (
                    metrics["mean_pinball_loss"] - float(oracle_loss.mean())
                )
                metrics["mean_downside_buffer"] = float((-forecasts["bound"]).mean())

                record: dict[str, Union[str, int, float]] = {
                    "replication": replication,
                    "innovation_seed": innovation_seed,
                    "timing_seed": timing_seed,
                    "scenario": scenario_name,
                    "model": model_name,
                    "method": method,
                    "window": float("nan") if method == "raw" else WINDOW,
                    "gamma": gamma,
                }
                record.update(metrics)
                records.append(record)

    return pd.DataFrame(records)


def summarize_replications(replications: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Union[str, int, float]]] = []
    for (scenario, model), group in replications.groupby(["scenario", "model"], sort=False):
        raw = group.loc[group["method"] == "raw"]
        raw = raw.set_index("replication", verify_integrity=True).sort_index()
        rolling = group.loc[group["method"] == "rolling"]
        rolling = rolling.set_index("replication", verify_integrity=True).sort_index()
        aci = group.loc[group["method"] == "aci"]
        aci = aci.set_index("replication", verify_integrity=True).sort_index()
        if not raw.index.equals(rolling.index) or not rolling.index.equals(aci.index):
            raise ValueError("Raw, rolling and ACI must use the same replications.")

        for metric in METRICS:
            # Subtract within each trajectory before estimating uncertainty.
            comparisons = {
                "raw": raw[metric],
                "rolling": rolling[metric],
                "aci": aci[metric],
                "raw_minus_rolling": raw[metric] - rolling[metric],
                "raw_minus_aci": raw[metric] - aci[metric],
                "rolling_minus_aci": rolling[metric] - aci[metric],
            }
            for comparison, values in comparisons.items():
                samples = values.to_numpy(dtype=float)
                count = len(samples)
                finite = np.isfinite(samples)
                # NumPy keeps NaN/inf in the mean; never summarize a filtered subset.
                estimate = float(np.mean(samples))
                mc_se = float("nan")
                if finite.all() and count > 1:
                    mc_se = float(np.std(samples, ddof=1) / sqrt(count))

                records.append({
                    "scenario": str(scenario),
                    "model": str(model),
                    "metric": metric,
                    "comparison": comparison,
                    "n_replications": count,
                    "estimate": estimate,
                    "mc_se": mc_se,
                    "ci_lower": estimate - 1.96 * mc_se,
                    "ci_upper": estimate + 1.96 * mc_se,
                    "non_finite_replications": int((~finite).sum()),
                })

    return pd.DataFrame(records)


def save_results(replications: pd.DataFrame, summary: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    replications.to_csv(OUTPUT_DIR / "replications.csv", index=False, na_rep="NaN")
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False, na_rep="NaN")

    settings = {
        "alpha": ALPHA,
        "window": WINDOW,
        "gamma": GAMMA,
        "methods": ["raw", "rolling", "aci"],
        "raw_forecast": {
            "distribution": "standard_normal",
            "mean": 0.0,
            "standardized_quantile": RAW_QUANTILE,
        },
        "replications": REPLICATIONS,
        "warmup": WARMUP,
        "calibration_size": CALIBRATION_SIZE,
        "evaluation_size": EVALUATION_SIZE,
        "coverage_window": COVERAGE_WINDOW,
        "change_offsets": CHANGE_OFFSETS,
        "change_jitter": CHANGE_JITTER,
        "innovation_seed_start": INNOVATION_SEED_START,
        "timing_seed_start": TIMING_SEED_START,
        "scenario_models": {
            "stationary": ["oracle"],
            "scale_1.5": ["expanding", "ewma", "oracle"],
            "scale_2.5": ["expanding", "ewma", "oracle"],
            "shape": ["oracle"],
        },
    }
    settings_bytes = json.dumps(settings, sort_keys=True).encode()
    source_paths = [Path(__file__), Path(__file__).with_name("scenarios.py"), ROOT / "pyproject.toml"]
    source_paths.extend(sorted((ROOT / "src" / "market_risk_calibration").glob("*.py")))
    source_hashes = {}
    for path in source_paths:
        source_hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()

    output_hashes = {}
    for name in ("replications.csv", "summary.csv"):
        output_hashes[name] = hashlib.sha256((OUTPUT_DIR / name).read_bytes()).hexdigest()

    metadata = {
        "settings": settings,
        "settings_sha256": hashlib.sha256(settings_bytes).hexdigest(),
        "source_sha256": source_hashes,
        "output_sha256": output_hashes,
        "versions": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "scipy": version("scipy"),
        },
        "evaluation": "Whole evaluation period; primary settings and regime schedule only.",
        "differences": (
            "Raw minus rolling, raw minus ACI, and rolling minus ACI, paired by replication; "
            "positive favors the second method for errors/losses."
        ),
        "intervals": "Pointwise 95% normal intervals across trajectories: mean +/- 1.96 MC SE.",
        "non_finite": "Retained in estimates; MC SE and intervals unavailable for affected groups.",
    }
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    frames = []
    for replication in range(REPLICATIONS):
        frames.append(run_replication(replication))
    replications = pd.concat(frames, ignore_index=True)
    summary = summarize_replications(replications)
    save_results(replications, summary)
    print(f"Saved {len(replications)} replication rows and {len(summary)} summaries to {OUTPUT_DIR}.")


if __name__ == "__main__":
    main()
