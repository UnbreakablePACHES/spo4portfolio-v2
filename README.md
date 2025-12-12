## File Structure
```
spo4portfolio/
├── configs/                     # Configuration files
│   └── spo_plus_linear.yaml     # Config for Linear SPO+ experiment
├── data/                        # Data directory
│   ├── DailyOracle/             # Oracle weights/objectives
│   ├── DailyReturn/             # Daily asset returns
│   ├── FeatureData/             # Preprocessed feature CSVs per ticker
│   └── RawData/                 # Raw data sourced from yfinance
├── DataPipeline/                # Data processing
│   ├── DataBuilder.py           # Feature engineering
│   └── Dataloader.py            # PyTorch Dataset implementation
├── models/                      # Model definitions
│   ├── LinearInferencer.py      # Linear predictor model
│   └── PortfolioModel.py        # Optimization model (Gurobi interface)
├── utils/                       # Utility scripts
├── LI+SPO_plus.py               # [Main Entry] Linear Model + SPO+ Loss training & backtest
└── requirements.txt             # Project dependencies         
```
