# Smart Predict-then-Optimize for Portfolio Optimization

## Introduction  
This project implements an end-to-end portfolio optimization framework based on the Smart Predict-then-Optimize(SPO) methodology.  
(For more details: [Smart Predict-then-Optimize](https://arxiv.org/abs/1710.08005))

We aims to improve portfolio allocation by jointly optimizing prediction accuracy and investment decisions by integrating deep learning time series forecasting models with SPO+ loss functions.

## Usage
- Deep Learning Framework: PyTorch (version >=1.8)
- Required Python Packages:  
  - gurobipy==12.0.2
  - pandas==2.3.0
  - pyepo==1.0.2
  - scikit_learn==1.3.2
  - torch==2.7.1
  - yfinance==0.2.61
- Open the project in VSCode using:

```bash
SPO4Portfolio.code-workspace
```
## Dataset  
Daily data in MSCI ishares ETF from 2023-01-01 to 2024-12-31, sourced from yfinance.

## File Structure
```
spo4portfolio/
├── DataPipeline/                
├── data/                        
│   ├── FeatureData/             # Featured data
│   ├── RawData/                 # Raw data from yfinance
│   └── TradingDay_info.csv      # Trading calendar info
├── loss/                        
├── models/                     
│   ├── LR.py                    # Linear regression model
│   ├── LinearInferencer.py      # SPO+ linear predictor
│   └── PortfolioModel.py        # Portfolio optimization model
├── tools/                       
├── 8tickers_test.ipynb          # Test notebook for 8 stock tickers
├── DataAccess.ipynb             # Notebook for accessing data
├── README.md                    
├── SPO4Portfolio.code-workspace # VSCode workspace settings
└── requirements.txt             
```
