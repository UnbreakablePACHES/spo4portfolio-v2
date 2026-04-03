# SPO4Portfolio

Research sandbox for ETF portfolio backtesting with rolling retraining.

## Structure
- `Backtest.py`: main rolling backtest entry.
- `configs/`: experiment settings.
- `DataPipeline/`: dataset and dataloader building.
- `models/`: prediction/allocation models.
- `losses/`: training objectives.
- `portfolio/`: portfolio optimization layer.
- `optimizers/`: optimizer factory.
- `utils/`: logging, metrics, plotting, tuning.

## Quick start
1. Install:
   ```bash
   pip install -r requirements.txt
   ```
2. Check config (`configs/spo_plus_linear.yaml` or `configs/softmax_linear.yaml`).
3. Run:
   ```python
   from Backtest import rolling_backtest
   rolling_backtest("configs/spo_plus_linear.yaml")
   ```

## Outputs
Saved under `outputs/<exp_name_timestamp>/`:
- `rolling_performance.csv`
- `rolling_weights.csv`
- plots and logs
