# SPL-PU：基于自步学习的正例-无标签分类框架

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-orange.svg)](https://pytorch.org/)
[![UCL Dissertation](https://img.shields.io/badge/UCL-MSc%20Dissertation%202022-purple.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> UCL 数据科学与机器学习 硕士论文项目 · 2022年
> 作者：付子桢 · 导师：Alastair Moore

---

## 背景与问题

在信贷审批等真实业务场景中，数据往往面临两个核心挑战：

- **标签稀缺**：只有少量确认的正例（如已确认违约的客户），大量样本无法获得可靠标签
- **样本不均衡**：正负样本比例悬殊（本项目数据集为 81% vs 19%），朴素模型倾向于忽略少数类

直接将无标签数据视为负例训练，会引入大量标签噪声，导致模型对真实风险客户的召回能力极差——本项目基准模型（逻辑回归）对负例的召回率为 **0**，即完全无法识别违约客户。

**核心判断**：问题的根源不在于模型结构，而在于训练数据的质量——噪声标签和不均衡分布使模型在训练初期就偏离了正确方向。

---

## 解决方案

本项目将 **自步学习（Self-Paced Learning, SPL）** 与 **正例-无标签学习（PU Learning）** 结合，从数据层面解决上述问题，核心思路是让模型像人一样学习：**先从简单、置信度高的样本入手，再逐步纳入困难和模糊的样本**。

### 实验结果

在 Fair For You（英国社区金融机构）真实信贷数据上（1000条样本，149个特征）：

| 模型 | Precision | Recall | F1-Score | AUC |
|------|-----------|--------|----------|-----|
| 逻辑回归（基准） | 0.845 | 0.845 | 0.845 | — |
| **SPL + 神经网络（本项目）** | **0.89** | **0.88** | **0.885** | **0.942** |

负例（违约客户）的召回率从 **0 → 29%**，在不增加误判的前提下，模型获得了真正识别高风险客户的能力。

---

## 算法设计

### 1. 正例-无标签学习（PU Learning）

标准监督学习要求正负标签都已知，而 PU Learning 只需要少量已标注正例和大量无标签数据。核心思路是用无偏风险估计替代朴素的负例假设：

$$R_{pu}(f) = \pi \cdot \mathbb{E}_{p(x|y=1)}[\ell(f(x), +1)] + \mathbb{E}_{p(x)}[\ell(f(x), -1)] - \pi \cdot \mathbb{E}_{p(x|y=1)}[\ell(f(x), -1)]$$

本项目支持两种 PU 损失：
- **uPU**：无偏 PU 损失，理论上无偏但可能出现负值
- **nnPU**：非负 PU 损失，在 uPU 基础上加入非负约束，训练更稳定

### 2. 自步学习（SPL）与样本权重

SPL 在训练损失中引入可学习的样本权重 $v_i \in [0,1]$ 和年龄参数 $\lambda$：

$$\min_{w,\, v \in [0,1]^N}\ \sum_{i=1}^{N} v_i \cdot \ell_i(w) + g(v;\,\lambda)$$

其中 $g(v;\lambda)$ 为 SP 正则化器，控制权重的分配方式。以线性正则化器为例，最优权重的解析解为：

$$v_i^* = \max\!\left(0,\ 1 - \frac{\ell_i}{\lambda}\right)$$

**效果**：损失大（模型不确定）的样本权重自动降低，损失小（容易学习）的样本权重保持；随 $\lambda$ 增大，更多难样本被逐渐纳入训练，实现从易到难的课程式学习。

本项目实现了 **10 种 SP 正则化器**，适配不同的噪声分布：

| 类型 | 权重函数 $v^*(l;\lambda)$ | 特点 |
|------|--------------------------|------|
| `hard` | $\mathbf{1}[l < \lambda]$ | 二值化，非此即彼 |
| `linear` | $\max(0,\ 1 - l/\lambda)$ | 平滑线性衰减 |
| `log` | 对数衰减 | 对噪声样本惩罚更陡 |
| `welsch` | $\exp(-l/\lambda^2)$ | 对大损失极度鲁棒 |
| `cauchy` | $1/(1 + l/\lambda^2)$ | 重尾分布，鲁棒性强 |
| `huber` | $\min(1,\ \lambda/\sqrt{l})$ | 兼顾稳定性和连续性 |
| `poly` | $(1 - l/\lambda)^{1/(t-1)}$ | 多项式衰减，可调陡峭度 |
| `logistic` | Sigmoid 型 | S 形平滑过渡 |

### 3. 训练调度器

$\lambda$ 的增长策略直接决定了"从易到难"的节奏，本项目支持 5 种调度函数：

| 调度器 | 公式 | 效果 |
|--------|------|------|
| `exp` | $\lambda_0 \cdot \eta^t$ | 指数增长，前期保守后期激进 |
| `linear` | $\lambda_0 + \frac{\lambda_{max}-\lambda_0}{T} \cdot t$ | 匀速增加难度 |
| `geom` | 对数空间线性插值 | 几何增长，平滑过渡 |
| `rootp` | $p$ 次根函数增长 | 前期快速后期趋缓 |
| `const` | $\lambda_0$ | 固定阈值，无课程 |

### 4. 双层超参数自动优化

SPL 的关键超参数（初始阈值 $\alpha$、增长率 $\eta$）对最终效果影响很大，手动网格搜索代价极高。本项目实现了基于**隐式微分 + 共轭梯度法（CG）** 的双层优化，将超参数视为外层变量，自动优化验证集损失：

$$\min_{\alpha,\,\eta}\ \mathcal{L}_{\text{val}}\!\left(w^*(\alpha, \eta)\right) \quad \text{s.t.}\ w^* = \arg\min_w \mathcal{L}_{\text{train}}(w;\,\alpha,\eta)$$

外层梯度通过 CG 近似求解线性方程组（避免显式计算 Hessian），内层通过 Adam 优化模型参数，两层交替更新，实现端到端的超参数自动搜索。

---

## 模型架构

### MLP（用于结构化数据）
```
输入层(x_dim) → Linear(512) → Sigmoid → Linear(100) → Sigmoid → Linear(1)
```
Xavier 初始化，适用于信贷、UCI 等结构化数据集。

### CNN（用于图像数据）
9 层卷积网络（含 BatchNorm + ReLU），Kaiming 初始化，用于 CIFAR-10。

---

## 项目结构

```
SPL-PU-Learning/
├── main.py / main_2.py      # 主训练入口（标准 SPL-PU 训练）
├── train_hyper.py            # 双层超参数优化训练逻辑
├── models.py                 # 网络结构
├── lossFunc.py               # PU 损失函数（BCE、nnPU、uPU、熵损失）
├── spl_utills.py             # SPL 核心：调度器 + 10 种权重函数
├── utils.py                  # 数据加载与 PU 数据集构建
├── Metrics.py                # 评估指标
├── helpers.py                # 日志工具与 AverageMeter
├── hypergrad/                # 双层优化模块
│   ├── hypergradients.py     # CG、不动点、Neumann、反向传播等
│   ├── diff_optimizers.py    # 可微优化器（GD、HeavyBall、Momentum）
│   └── CG_torch.py           # 共轭梯度求解器
├── data/sample/              # ✅ 已内置真实信贷数据集
│   ├── all_data.csv          # 完整数据（1000条，149个特征）
│   ├── train_data.csv        # 训练集（800条）
│   └── test_data.csv         # 测试集（200条）
├── configs/                  # 各场景配置文件
│   ├── mnist.yaml
│   └── risk.yaml
└── scripts/                  # 一键运行脚本
    ├── run_exp.sh            # 运行内置信贷数据集
    ├── run_mnist.sh
    ├── run_risk.sh
    └── run_bilevel.sh
```

---

## 快速开始

### 安装

```bash
git clone https://github.com/YOUR_USERNAME/SPL-PU-Learning.git
cd SPL-PU-Learning
pip install -r requirements.txt
```

### 在 MNIST 上运行（开箱即用）

```bash
bash scripts/run_mnist.sh
```

### 在自有数据集上运行

参考 `utils.py` 中 `exp` 数据集的预处理逻辑，替换为格式相似的 CSV 文件（包含特征列和标签列），然后执行：

```bash
python main_2.py \
    --dataset exp \
    --data_dir YOUR_DATA_DIR/ \
    --prior 0.2 \
    --spl_type linear \
    --scheduler_type exp \
    --alpha 0.1 \
    --eta 1.1 \
    --pre_loss nnpu \
    --epochs 100 \
    --no_cuda
```

### 开启双层超参数自动优化

```bash
bash scripts/run_bilevel.sh
```

---

## 支持的数据集

| 数据集 | 类型 | 规模 | 获取方式 |
|--------|------|------|----------|
| `exp` | 结构化 | 1,000条 | 来自真实金融机构，涉及保密协议，不可公开 |
| `mnist` | 图像 | 70,000条 | ✅ torchvision 自动下载，开箱即用 |
| `cifar10` | 图像 | 60,000条 | ✅ torchvision 自动下载，开箱即用 |
| `risk` | 结构化 | 307,511条 | [Home Credit Default Risk · Kaggle](https://www.kaggle.com/c/home-credit-default-risk) |
| `mushroom` | 结构化 | 8,124条 | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/mushroom) |

> **关于 `exp` 数据集**：实验数据来自 Fair For You（英国社区金融机构）的真实信贷记录，包含 1000 条样本、149 个特征，正负样本比例约为 4:1。由于签署了数据保密协议，原始数据无法公开。如需在类似结构化数据上复现实验，可参考 `utils.py` 中的预处理逻辑，或使用上方表格中的公开替代数据集。

---

## 技术栈

`PyTorch 1.10+` · `scikit-learn` · `pandas` · `higher`（隐式微分）· `transformers`（学习率调度）· `MLflow`（实验追踪）

---

## 引用

```bibtex
@mastersthesis{fu2022spl,
  title  = {Credit Risk Engine using Self-paced Learning and Neural Network},
  author = {Zizhen Fu},
  school = {University College London},
  year   = {2022}
}
```

**主要参考文献**：
- Kumar et al., "Self-Paced Learning for Latent Variable Models," NeurIPS 2010
- Kiryo et al., "Positive-Unlabeled Learning with Non-Negative Risk Estimator," NeurIPS 2017
- Wang et al., "A Survey on Curriculum Learning," IEEE TPAMI 2022

---

## 开源协议

[MIT License](LICENSE)
