# Market Risk Calibration

**When does adaptive calibration improve risk forecasts, and when does it mainly regulate recent breach counts?**

A focused study of the lower 5% quantile of next-day US market returns: a threshold that returns should fall below about 5% of the time. Writing $r_t$ for the return and $L_t$ for the forecast lower bound, the target and breach indicator are:

$$
\Pr(r_t < L_t \mid \text{past information}) \approx \alpha = 0.05,
\qquad e_t = \mathbf{1}\{r_t < L_t\}.
$$

A breach has $e_t=1$; otherwise $e_t=0$. A very conservative threshold can reduce breaches while giving poor quantile estimates, so breach frequency alone cannot establish forecast quality.

Rolling conformal calibration adjusts the risk threshold using recent deviations from a base forecast, with a fixed calibration level $\alpha_t=\alpha$. Adaptive Conformal Inference (ACI) starts at the same level and updates it after each observed return:

$$
\alpha_{t+1} = \alpha_t + \gamma(\alpha - e_t).
$$

The step size $\gamma>0$ controls responsiveness. A breach lowers the calibration level, pushing toward a more conservative bound. The study compares quantile accuracy and coverage, both overall and over recent observations.

Historical returns use a frozen Fama/French daily-factor snapshot covering July 1926 to August 2026, with market returns constructed as `(Mkt-RF + RF) / 100`. Controlled simulations will provide known risk probabilities; the historical study will examine practical trade-offs in an exploratory comparison on one market series.

**Implemented so far**

The shared Python core includes data loading, expanding and EWMA volatility forecasts, rolling/ACI calibration, and sequential forecasting using only past information. Quantile accuracy is measured by pinball loss:

$$
\ell_t = (r_t - L_t)(\alpha - e_t).
$$

This penalizes errors on both sides of the bound, with greater weight on downside misses. Evaluation also covers breach rate, local coverage error over 250 observations, and non-finite forecasts. Focused tests check the shared core; CI runs pytest, Ruff and mypy.

The primary simulation compares rolling calibration and ACI on stationary returns, volatility changes and changes in distribution shape at constant variance. It uses 200 paired trajectories per scenario, W=500 and gamma=0.001. Known distributions let us measure true breach-probability error alongside loss and observed coverage. Summaries include trajectory-level Monte Carlo uncertainty; the full sensitivity study and historical comparison remain to be implemented.

**Next milestones**

1. Add the simulation notebook to explain the primary comparison and interpret its results.
2. Complete the simulation grid and analysis after regime changes.
3. Add GARCH-t volatility forecasting, monthly refits and an EWMA fallback.
4. Add one quantile boosting model, lagged features and conformal calibration.
5. Compare rolling/ACI for the three volatility models and raw/rolling/ACI for ML, with common dates, bootstrap uncertainty and fixed sensitivity checks.
6. Produce the final report: two tables, three figures and reproduction commands.

**Repository structure**

```text
src/market_risk_calibration/  Data, forecasting, calibration and evaluation
tests/                        Correctness and chronology checks
configs/                      Unused placeholder
experiments/                  Simulation, historical study and report commands
docs/                         Methodology notes
data/                         Local snapshots (ignored by Git)
results/                      Generated outputs (ignored by Git)
.github/workflows/            Tests, lint and type checks in CI
```

`experiments/simulation/` contains stationary, scale-change and constant-variance
shape-change generators and the primary comparison runner.
`configs/` and `docs/` currently contain placeholders.

**Local development**

Requires Python 3.9 or later. Data snapshots and generated results remain outside Git; tests use small offline examples.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check src experiments tests
python -m mypy
```

Run the primary simulation from the repository root:

```bash
python -m experiments.simulation.run
```

Fixed settings are at the top of `experiments/simulation/run.py`. The command writes
`replications.csv`, `summary.csv` and `metadata.json` under the gitignored
`results/simulation/`, replacing the previous outputs there. Differences are rolling
minus ACI; positive error/loss differences favor ACI. Non-finite outcomes remain in
the results and affected summaries have no confidence interval.
