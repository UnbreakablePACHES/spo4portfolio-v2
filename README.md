# SPO4Portfolio 配置选项说明 (OPTIONS.md)

## 1. 模型架构 (model: type)

### A. linear (预测型)
- 原理: 预测资产成本/收益，不直接决策。
- 适用: SPO+ Loss (spo_plus), Soft-SPO (softmax_spo)。
- 配置:
  model:
    type: "linear"
    params:
      input_dim: 7

### B. softmax_allocator (决策型)
- 原理: 神经网络直接输出投资权重 (MLP + Softmax)。
- 适用: Max Return (max_return), Max Sharpe (max_sharpe)。
- 配置:
  model:
    type: "softmax_allocator"
    params:
      input_dim: 7
      hidden_layers: [64, 32]
      dropout_rate: 0.2

---

## 2. 损失函数 (loss: type)

### A. 预测-优化类 (配合 linear 模型)
1. spo_plus
   - 描述: 标准 SPO+，需调用 Gurobi，训练慢但严谨。
   - 配置: { type: "spo_plus" }

2. softmax_spo
   - 描述: 软化版 SPO，用 Softmax 代替求解器，训练快。
   - 配置: { type: "softmax_spo", params: { temperature: 1.0 } }

### B. 直接策略类 (配合 softmax_allocator 模型)
3. max_return
   - 描述: 最大化组合绝对收益，风险偏好高。
   - 配置: { type: "max_return" }

4. max_sharpe
   - 描述: 最大化夏普比率，兼顾收益与波动，稳健推荐。
   - 配置: { type: "max_sharpe", params: { risk_free_rate: 0.02 } }
