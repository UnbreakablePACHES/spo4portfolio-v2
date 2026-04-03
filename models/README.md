# models

Two model paths:
- `linear`: predict per-asset scores, then optimize weights.
- `softmax`: output weights directly.

## Files
- `LinearInferencer.py`: linear head on flattened `(N, F)` features.
- `SoftmaxAllocator.py`: optional MLP + softmax for weights.
- `factory.py`: model builder from `model.type`.
