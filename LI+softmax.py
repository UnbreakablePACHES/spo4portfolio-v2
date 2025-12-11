import pandas as pd
import numpy as np
import torch
import random

from dateutil.relativedelta import relativedelta

from models.LinearInferencer import LinearPredictorTorch
from DataPipeline.Dataloader import PortfolioDataset
from torch.utils.data import DataLoader
from DataPipeline.DataBuilder import build_dataset
from torch import nn
from torch.optim import Adam
pd.options.display.float_format = '{:.6f}'.format
np.set_printoptions(precision=6, suppress=True)

tickers = ["EEM","EFA","JPXN","SPY","XLK",'VTI','AGG','DBC']

# Random Seed
seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

class FNNSoftmaxAllocator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_assets):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_assets)
        )

    def forward(self, preds):  # preds: (B, N)
        scores = self.net(preds)     # (B, N)
        weights = torch.softmax(scores, dim=1)
        return weights
    
def spo_plus_loss(pred_y, true_y, allocator):
    pred_weights = allocator(pred_y)        # ŵ = g(ŷ)
    oracle_weights = allocator(true_y)      # w* ≈ g(y)

    regret = torch.sum((oracle_weights - pred_weights) * true_y, dim=1)
    return regret.mean()

# 模型超参数
input_dim = 7         # 每个资产的特征数
num_assets = 8        # ETF 数量
hidden_dim = 32       # allocator 隐层宽度
epochs = 30           # 训练轮数
device = 'cuda' if torch.cuda.is_available() else 'cpu'


results = []
monthly_returns = []

# 预读取所有 ETF 的 log_return
return_df = pd.DataFrame()
for ticker in tickers:
    file_path = f"data/FeatureData/{ticker}.csv"  # 不用 os.path
    df = pd.read_csv(file_path, parse_dates=["Date"])
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df = df.set_index("Date")["log_return"].rename(ticker)
    return_df = pd.concat([return_df, df], axis=1)

# 设置初始月份
base_month = pd.to_datetime("2016-01-01")

for i in range(108):
    # 当前月份范围
    infer_start = base_month + relativedelta(months=i)
    infer_end = (infer_start + relativedelta(months=1)) - pd.Timedelta(days=1)
    train_start = infer_start - relativedelta(years=1)
    train_end = infer_start - pd.Timedelta(days=1)

    print(f"\n📅 第 {i+1} 次迭代：训练 {train_start.date()} ~ {train_end.date()}，推断 {infer_start.date()} ~ {infer_end.date()}")

    # 1. 训练数据
    features_df, labels_df = build_dataset(
        tickers=tickers,
        start_date=str(train_start.date()),
        end_date=str(train_end.date())
    )
    oracle_df = pd.read_csv("data/DailyOracle/oracle_weights_with_fee.csv", index_col=0)
    oracle_df.index = pd.to_datetime(oracle_df.index).normalize()
    features_df.index = pd.to_datetime(features_df.index).normalize()
    oracle_df = oracle_df.loc[features_df.index]
    labels_df = oracle_df.copy()

    dataset = PortfolioDataset(features_df, labels_df, num_assets=8)
    train_loader = DataLoader(dataset, batch_size=63, shuffle=True)

    # 2. 初始化模型
    predictor = LinearPredictorTorch(input_dim * num_assets, num_assets).to(device)
    allocator = FNNSoftmaxAllocator(num_assets, hidden_dim, num_assets).to(device)
    optimizer = Adam(list(predictor.parameters()) + list(allocator.parameters()), lr=1e-3)

    # 3. 训练模型
    for epoch in range(epochs):
        total_loss = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            pred_y = predictor(x)
            loss = spo_plus_loss(pred_y, y, allocator)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1:02d} - Loss: {avg_loss:.6f}")

    # 4. 构建目标月份的特征
    features_future, labels_future = build_dataset(
        tickers=tickers,
        start_date=str(infer_start.date()),
        end_date=str(infer_end.date())
    )
    features_future.index = pd.to_datetime(features_future.index).normalize()
    labels_future.index = pd.to_datetime(labels_future.index).normalize()
    inference_dataset = PortfolioDataset(features_future, labels_future, num_assets=8)
    x_tensor = torch.stack([inference_dataset[i][0] for i in range(len(inference_dataset))]).to(device)

    # 5. 推断
    predictor.eval()
    allocator.eval()
    with torch.no_grad():
        pred_y = predictor(x_tensor)
        pred_weights = allocator(pred_y)
    w_month = pred_weights.mean(dim=0).cpu().numpy()

    # 6. 计算月度组合收益
    try:
        arith_return_month = np.expm1(return_df.loc[infer_start:infer_end, tickers].values)  # 将log return转为算术收益率
        daily_return = arith_return_month @ w_month                                           # 每日组合算术收益率
        monthly_return = np.prod(1 + daily_return) - 1   
    except Exception as e:
        print(f"⚠️ 无法计算 {infer_start.strftime('%Y-%m')} 的组合收益：{e}")
        monthly_return = np.nan

    # 7. 打印与记录
    print('组合比率:')
    for ticker, weight in zip(tickers, w_month):
        print(f"{ticker}: {weight:.4f}")
    print(f"📈 {infer_start.strftime('%Y-%m')} 月组合收益：{monthly_return:.4%}")

    results.append((infer_start.strftime('%Y-%m'), w_month))
    monthly_returns.append((infer_start.strftime('%Y-%m'), monthly_return))

# 保存所有月度收益结果
monthly_returns_df = pd.DataFrame(results, columns=["Month", "PortfolioWeights"])
monthly_returns_df["MonthlyReturn"] = [r for _, r in monthly_returns]
monthly_returns_df["CumulativeReturn"] = (1 + monthly_returns_df["MonthlyReturn"]).cumprod() - 1
# 保存到 CSV 文件
monthly_returns_df.to_csv("result\8_ticker_1ytrain1yinfer\LP+softmax.csv", index=False)

print("已保存所有月度收益到 'result'")






