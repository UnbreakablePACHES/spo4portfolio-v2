# DataPipeline

Builds aligned features/labels and wraps them into a PyTorch `DataLoader`.

## Files
- `DataBuilder.py`: loads per-ticker CSVs, selects features, builds labels.
- `Dataloader.py`: `PortfolioDataset` reshapes data to `(N, F)` per sample.
- `factory.py`: builds dataloader (`monthly_window` currently).

## Required CSV columns
- `Date`
- `log_return`
- all feature names listed in config

## Common config keys
- `data.root`
- `data.etfs`
- `data.features`
- `data.label_window`
- `dataloader.params.batch_size`
