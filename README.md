<div align="center">

# Informer: A Time Series Forecasting Framework
### High-Efficiency Long-Sequence Forecasting for Financial Markets

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](./LICENSE)
[![Paper](https://img.shields.io/badge/AAAI-21-b31b1b?style=for-the-badge)](https://arxiv.org/abs/2012.07436)

<br />

<img src="https://github.com/zhouhaoyi/Informer2020/raw/main/img/informer.png" alt="Informer Architecture" width="80%">

<br />
<br />

**[Abstract](#abstract) • [Methodology](#methodology) • [Installation](#installation) • [Experiments](#experiments) • [Citation](#citation)**

</div>

---

## 📑 Abstract

This repository implements the **Informer** architecture for long-sequence time-series forecasting (LSTF), specifically optimized for **Stock Market Data Analysis**. While traditional Transformer models suffer from high memory consumption and quadratic time complexity, the Informer model leverages a ProbSparse self-attention mechanism to achieve $\mathcal{O}(L \log L)$ complexity.

This project demonstrates the model's capability to capture long-range dependencies in volatile financial datasets, providing accurate predictions for open/close prices and market trends.

---

## 🧠 Methodology

### The Informer Architecture
The core innovation of this project lies in addressing the limitations of the vanilla Transformer when applied to LSTF. We utilize three distinct mechanisms to enhance prediction efficiency:

#### 1. ProbSparse Self-Attention
To handle the quadratic complexity of canonical self-attention, we employ ProbSparse attention. This mechanism selects the "active" queries based on a measurement of Kullback-Leibler divergence, allowing the model to focus only on dominant features.

The standard attention is defined as:
$$
\mathcal{A}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

In contrast, our **ProbSparse** attention restricts the query set to the top-$u$ dominant queries:
$$
\mathcal{A}(Q, K, V) = \text{Softmax}\left(\frac{\bar{Q}K^T}{\sqrt{d}}\right)V
$$
Where $\bar{Q}$ contains only the active queries derived from the sparsity measurement.

#### 2. Self-Attention Distilling
To prevent feature redundancy in deep networks, we use a distilling operation that halves the input length in each layer, drastically reducing memory usage:
$$
X_{j+1}^t = \text{MaxPool}(\text{ELU}(\text{Conv1d}([X_j^t]_{\text{AB}})))
$$

#### 3. Generative Style Decoder
Unlike standard encoder-decoder structures that generate outputs step-by-step (dynamic decoding), the Informer uses a generative decoder to predict the entire long sequence in a single forward pass, mitigating error accumulation during inference.

---

## ⚡ Installation & Usage

### Prerequisites
Ensure you have Python 3.8+ and PyTorch installed.

```bash
git clone [https://github.com/yourusername/informer-stock-forecasting.git](https://github.com/yourusername/informer-stock-forecasting.git)
cd informer-stock-forecasting
pip install -r requirements.txt
