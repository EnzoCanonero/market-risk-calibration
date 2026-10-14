# Market Risk Calibration

**When does adaptive calibration improve risk forecasts, and when does it mainly regulate recent breach counts?**

A focused study of the lower 5% quantile of next-day US market returns: a threshold that returns should fall below about 5% of the time. Writing $r_t$ for the return and $L_t$ for the forecast lower bound, the target and breach indicator are:

$$
\Pr(r_t < L_t \mid \text{past information}) \approx \alpha = 0.05,
\qquad e_t = \mathbf{1}\{r_t < L_t\}.
$$

A breach has $e_t=1$; otherwise $e_t=0$. A very conservative threshold can reduce breaches while giving poor quantile estimates, so breach frequency alone cannot establish forecast quality.

The comparison follows **raw → rolling → ACI**: what does calibration add to the original forecast, and what does adaptive feedback add beyond rolling? In simulation, the raw benchmark assumes Gaussian innovations and uses $L_t^{\mathrm{raw}}\approx-1.645\,\hat\sigma_t$, where $\hat\sigma_t$ is the volatility forecast.

Rolling conformal calibration estimates the tail from recent returns standardized by their forecast volatility, at a fixed calibration level $\alpha_t=\alpha$. Adaptive Conformal Inference (ACI) starts at the same level and updates it after each observed return:

$$
\alpha_{t+1} = \alpha_t + \gamma(\alpha - e_t).
$$

The step size $\gamma>0$ controls responsiveness. A breach lowers the calibration level, pushing toward a more conservative bound. The study compares quantile accuracy and coverage, both overall and over recent observations.

Historical returns use a frozen Fama/French daily-factor snapshot covering July 1926 to August 2026, with market returns constructed as `(Mkt-RF + RF) / 100`. Controlled simulations provide known risk probabilities; the historical study will examine practical trade-offs in an exploratory comparison on one market series.

**Implemented so far**

The shared Python core includes data loading, expanding and EWMA volatility forecasts, rolling/ACI calibration, and sequential forecasting using only past information. Quantile accuracy is measured by pinball loss:

$$
\ell_t = (r_t - L_t)(\alpha - e_t).
$$

This penalizes errors on both sides of the bound, with greater weight on downside misses. Evaluation also covers breach rate, local coverage error over 250 observations, and non-finite forecasts. Focused tests check the shared core; CI runs pytest, Ruff and mypy.

The primary simulation compares raw Gaussian forecasts, rolling calibration and ACI on stationary Student-t returns, changes in their volatility, and shape changes at constant variance. It uses 200 paired trajectories per scenario, a calibration window W=500 and ACI step size gamma=0.001. Known distributions let us measure true breach-probability error alongside loss and observed coverage.

At these settings, rolling improves risk accuracy over raw with expanding volatility under scale changes and with changing tail shape. Raw is more accurate with EWMA or known volatility in the Student-t scenarios. ACI improves local observed coverage over rolling in every case, but improves both risk metrics only under large scale changes with expanding volatility. The [simulation notebook](experiments/simulation/analysis.ipynb) presents the evidence, paired Monte Carlo uncertainty and exploratory limitations. Sensitivity analysis and the historical comparison remain to be implemented.

**Next milestones**

1. Complete the simulation grid and analysis after regime changes.
2. Add GARCH-t volatility forecasting, monthly refits and an EWMA fallback.
3. Add one quantile boosting model, lagged features and conformal calibration.
4. Compare raw/rolling/ACI for all four forecasters on common dates, with bootstrap uncertainty and fixed sensitivity checks. Raw uses Gaussian quantiles for expanding/EWMA, fitted Student-t quantiles for GARCH-t, and the directly predicted quantile for ML.
5. Complete a compact report across the simulation and historical notebooks, with conclusions, tables, figures and reproduction commands.

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
shape-change generators, the primary comparison runner and its analysis notebook.
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
`results/simulation/`, replacing the previous outputs there. Paired differences are
raw minus rolling, raw minus ACI, and rolling minus ACI; positive error/loss differences
favor the second method. Non-finite outcomes remain in the results and affected
summaries have no confidence interval.

The notebook reads these saved outputs and contains four scenario tables and two figures for browsing
on GitHub. To execute it locally, install the notebook dependencies and select the
`.venv` kernel in your editor:

```bash
python -m pip install -e ".[notebooks]"
```

After generating the simulation outputs, it can also be executed from the repository root:

```bash
python -m jupyter nbconvert --to notebook --execute --inplace experiments/simulation/analysis.ipynb
```
