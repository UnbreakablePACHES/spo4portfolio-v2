# SPO4Portfolio

A minimal research sandbox for stochastic programming on ETF portfolios. The codebase wires up data fetching, model training, and backtesting so you can compare prediction-driven and allocation-driven workflows without extra scaffolding.

## Project layout
- **DataPipeline/** builds training samples from OHLCV features and wraps them in PyTorch dataloaders (`monthly_window`).
- **models/** exposes a linear forecaster and a softmax allocator selected through `model.type` in the config.
- **portfolio/** defines the optimization layer used by SPO+ losses.
- **losses/** includes SPO+, a softmax-friendly SPO surrogate, plus return- and Sharpe-focused objectives.
- **optimizers/** and **utils/** handle training loops, logging, plotting, and Optuna tuning for rolling experiments.

## Getting started
1. Install dependencies (Gurobi is required for SPO+):
   ```bash
   pip install -r requirements.txt
   ```
2. Inspect a config like `configs/spo_plus_linear.yaml` to set your data path, ETF tickers, and training window.
3. Run a rolling backtest (defaults to the linear + SPO+ config):
   ```bash
   python - <<'PY'
   from Backtest import rolling_backtest
   rolling_backtest("configs/spo_plus_linear.yaml")
   PY
   ```

## Configuration highlights
- `data`: root folder for features, tickers list, and feature names used for both models.
- `model`: choose `linear` (predict returns, optimized via SPO+) or `softmax` (direct weight allocation with hidden layers/dropout options).
- `loss`: select `spo_plus`, `softmax_spo`, `max_return`, or `max_sharpe` depending on whether you train a predictor or allocator.
- `backtest`: control the rolling window size, rebalance frequency (e.g., month start), and initial capital for the performance trace.

## Parameter meanings
- **model**
  - `type`: toggles the training target. `linear` predicts returns that are fed into the portfolio optimizer, while `softmax` outputs allocation weights directly.
  - `params`: high-level hyperparameters for the chosen model (e.g., feature dimension for `linear`, hidden layers/dropout for `softmax`).
- **portfolio**
  - `type`: selects the layer that converts predicted returns into weights; the default `basic` layer enforces a budgeted allocation.
  - `params`: `budget` controls total capital to allocate (1.0 means fully invested), `lb`/`ub` set lower/upper bounds per asset to cap exposure.
- **loss**
  - `type`: `spo_plus` and `softmax_spo` are SPO-style surrogates that align the model with the downstream optimizer; `max_return` pushes weights toward higher average return; `max_sharpe` encourages higher risk-adjusted performance.
  - `params`: top-level knobs for each loss (e.g., `temperature` for `max_sharpe` smoothing); keeps you from editing model internals when tuning objectives.
