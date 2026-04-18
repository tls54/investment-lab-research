# Heston Historical Calibration

This directory implements the **backward-looking (historical) calibration** arm
of the investment lab's Heston stochastic volatility framework. It uses only
price/return data and realized variance — no options prices, no FFT pricing.

The companion options-calibration directory handles the forward-looking
(Q-measure) arm. 

---

## The Heston model

$$dS_t = \mu S_t\,dt + \sqrt{v_t}\,S_t\,dW_t^S$$
$$dv_t = \kappa(\theta - v_t)\,dt + \sigma\sqrt{v_t}\,dW_t^v, \qquad dW^S dW^v = \rho\,dt$$

| Parameter | Meaning | Identifiability from returns + RV |
|---|---|---|
| κ | Mean-reversion speed | Good — via RV autocorrelation |
| θ | Long-run variance | Good — via mean RV |
| ρ | Leverage (return–vol correlation) | Good — via return/RV covariance |
| σ | Vol-of-vol | Weak — needs high-frequency RV or options |
| v₀ | Initial variance | Very weak — process forgets it in ~1/κ days |

**Key insight:** κ and θ are individually non-identifiable from daily returns
alone (the κ–θ banana). Adding realized variance (RV) as a second observation
channel breaks most of this degeneracy. σ remains hard to pin down without
either high-frequency intraday data or options prices.

---

## Notebooks

The notebooks form a progression. Read them in order to understand the
reasoning behind each design decision.

### 1. [markov.ipynb](markov.ipynb) — Foundations
Discrete Markov chains and basic Metropolis-Hastings. Starting point before
any Heston-specific work.

### 2. [damped-spring-mcmc.ipynb](damped-spring-mcmc.ipynb) — Adaptive MH on a physics model
Parameter recovery for a damped oscillator. Introduces adaptive
Metropolis-Hastings with empirical covariance tuning and corner plots.

### 3. [Heston-back-fit.ipynb](Heston-back-fit.ipynb) — Particle MCMC
First Heston calibration using a bootstrap particle filter for likelihood
estimation and adaptive Metropolis-Hastings for sampling.

**Key findings:**
- κ–θ banana clearly visible — data constrains κ·θ but not κ and θ individually
- Particle filter is stochastic → noisy likelihood → slow MH mixing
- Motivates the move to a deterministic likelihood (UKF) and gradient sampler (NUTS)

### 4. [heston-ukf-nuts.ipynb](heston-ukf-nuts.ipynb) — UKF + NUTS, four experiments
Replaces the particle filter with a 2D Unscented Kalman Filter (deterministic,
differentiable via JAX `lax.scan`) and replaces MH with NUTS.

Four experiments establish what data actually identifies the parameters:

| Experiment | Data | κ in CI? | θ in CI? | σ in CI? | ρ in CI? |
|---|---|---|---|---|---|
| Baseline | 252 daily returns | NO | NO | NO | NO |
| A1 — hourly | 2016 hourly returns | NO | NO | NO | NO |
| **A2 — returns + RV** | **252 returns + 252 RV** | **yes** | **yes** | NO | **yes** |
| B — 5yr daily | 1260 daily returns | NO | NO | yes | yes |

**The RV channel (A2) is the key breakthrough.** More daily returns (A1, B)
sharpens the banana without resolving it. Adding RV provides a near-direct
observation of v_t each day, which breaks the degeneracy. σ remains outside
CI because daily RV with H=8 bars is too noisy to resolve vol-of-vol.

### 5. [heston-ukf-nuts-warmstart.ipynb](heston-ukf-nuts-warmstart.ipynb) — MAP warm-start
Adds an MLE/MAP optimisation stage (L-BFGS-B with JAX analytical gradients)
before NUTS to warm-start chains at the posterior mode.

**Key findings:**
- MAP optimisation takes ~0.1s and puts chains close to the posterior mode
- On the A2 posterior (already well-conditioned by RV), warm-start makes no
  visible difference — cold start at κ=1.5 is only ~0.7 from posterior mean
- `dense_mass=True` needs ≥800 warmup to converge; with 500 it adds overhead
  without benefit
- Warm-start + dense mass would matter on harder posteriors (returns-only,
  5yr data) where chains can lock into different modes

### 6. [heston-calibration-production.ipynb](heston-calibration-production.ipynb) — Production pipeline
The notebook to use on real data. Consolidates all lessons from the research
notebooks into a clean, configurable workflow.

**Design:**
- Single config cell at the top — the only thing to edit per dataset
- v₀ fixed to early realized variance by default (`"auto"` mode); the process
  forgets v₀ in ~1/κ days so the data contains almost no information about it
- σ is configurable: calibrate (wider posterior) or fix to an external value
  (e.g. from the options-calibration sibling)
- NumPyro model built dynamically — only free parameters are sampled
- MAP warm-start on free parameters only
- Full diagnostic suite that works without ground truth:
  - UKF innovation whiteness (Ljung-Box on standardised innovations)
  - Feller condition check across posterior draws
  - κ·θ plausibility bounds
  - Posterior predictive: return distribution, QQ plot, ACF(r²) with envelope
  - Filtered variance path

**To run on real data:** set `USE_SYNTHETIC = False`, load `r_daily` and
`RV_daily` into the data cell, and set `TRUE_PARAMS` values to `None`.

---

## Why σ is hard

σ controls how much v_t diffuses around its mean. To measure it you need to
observe that diffusion above the noise floor of the RV estimator.

With H intraday bars per day:

```
Signal (true σ variance):  Var(Δv_t) ≈ σ²·θ·dt  ~  2.5×10⁻⁶
Noise (RV sampling error):  Var(RV_t) ≈ 2v_t²·dt²/H  ~  4×10⁻⁸ × (8/H)
Signal-to-noise ratio:  ≈ σ²·H·dt / (4θ)
```

| H | Source | SNR | Practical σ recovery |
|---|---|---|---|
| 8 | Hourly | 0.03 | No — noise 30× larger than signal |
| 13 | 30-min | 0.05 | No |
| 78 | 5-min | 0.31 | Reasonable |
| 390 | 1-min | 1.5 | Good |

The rolling-RV-variance approach (moment estimator) is conceptually correct but
gives σ ≈ 0.15 vs true 0.4 with H=8 because measurement noise dominates. More
intraday bars, not more trading days, is the fix.

---

## Recurring conventions

All Heston notebooks use identical synthetic parameters for comparability:

```python
TRUE = {"kappa": 3.0, "theta": 0.04, "sigma": 0.4, "rho": -0.7, "v0": 0.04}
seed = 0
T    = 252   # daily steps (1 year)
mu   = 0.05  # drift — known, not calibrated
```

**Unconstrained parameterisation** (used by NUTS and L-BFGS-B):
```
ψ = (log κ,  log θ,  log σ,  arctanh ρ,  log v₀)
```

**JAX requirements:**
- `jax.config.update("jax_enable_x64", True)` before any JAX operations
- `os.environ["JAX_PLATFORM_NAME"] = "cpu"` — Metal GPU lacks the `popcnt`
  operation used in NUTS tree traversal

---

## Dependencies

```
numpy scipy matplotlib
jax
numpyro
arviz
corner
```