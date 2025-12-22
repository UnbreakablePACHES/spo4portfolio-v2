# SPO4Portfolio 配置选项说明文档（OPTIONS_README.md）

本文档详细说明了项目中可用的 **模型架构（Models）** 和 **损失函数（Losses）** 选项，以及它们在 `config.yaml` 中的对应写法。

---

## 1. 模型架构（Models）

在配置文件中通过 `model.type` 指定。

---

### A. `linear`（预测型模型）

- **对应类**：`LinearInferencer`
- **范式**：Predict-then-Optimize
- **原理**：标准线性预测模型  
  - **输入**：资产特征向量 \( x \)  
  - **输出**：预测的 **成本向量（Cost Vector）** \( \hat{c} \)（或收益率 \( \hat{r} \)）
- **角色定位**：
  - 只负责预测市场
  - **不直接做决策**
  - 决策由后续优化器（如 Gurobi）或 Loss 内部逻辑完成
- **适用 Loss**：
  - `spo_plus`
  - `softmax_spo`

#### YAML 配置示例

```yaml
model:
  type: "linear"
  params:
    input_dim: 7

