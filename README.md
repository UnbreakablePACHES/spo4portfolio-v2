# SPO4Portfolio

A minimal research sandbox for stochastic programming on ETF portfolios. The codebase wires up data fetching, model training, and backtesting so you can compare prediction-driven and allocation-driven workflows without extra scaffolding.

## Project layout
- **DataPipeline/** builds training samples from OHLCV features and wraps them in PyTorch dataloaders (`monthly_window`).
- **models/** exposes a linear forecaster and a softmax allocator selected through `model.type` in the config.
- **portfolio/** defines the optimization layer used by SPO+ losses.
- **losses/** includes SPO+, a softmax-friendly SPO surrogate, plus return- and Sharpe-focused objectives.
- **optimizers/** and **utils/** handle training loops, logging, plotting, and Optuna tuning for rolling experiments.

## Getting started
1. Install dependencies :
   ```bash
   pip install -r requirements.txt
   ```
2. Inspect a config like `configs/spo_plus_linear.yaml` to set your data path, ETF tickers, and training window.
3. Run a rolling backtest :
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
