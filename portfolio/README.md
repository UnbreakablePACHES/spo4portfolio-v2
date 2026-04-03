# portfolio

Decision layer used by SPO-style training.

- `factory.py` builds the portfolio model.
- Training can call `solve(...)` on predicted costs to get weights.
- Extend this module for extra constraints (sector, caps, turnover, etc.).
