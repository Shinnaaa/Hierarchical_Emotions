<p align="center">
  <a href="../README.md"><img alt="English" src="https://img.shields.io/badge/English-eaeef2?style=for-the-badge"></a>
  <a href="README.zh-CN.md"><img alt="简体中文" src="https://img.shields.io/badge/%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-1f2328?style=for-the-badge"></a>
  <a href="README.ja.md"><img alt="日本語" src="https://img.shields.io/badge/%E6%97%A5%E6%9C%AC%E8%AA%9E-eaeef2?style=for-the-badge"></a>
</p>

# Emotional Earth Mover’s Distance for Fine-Grained Hierarchical Emotion Analysis

论文 **"Emotional Earth Mover’s Distance for Fine-Grained Hierarchical Emotion Analysis"** 的官方代码。作者：Hai-Tao Yu、Dawei Li、Xin Kang。*Advanced Data Mining and Applications (ADMA 2025)*，Springer，第 296–310 页。[[论文]](https://link.springer.com/chapter/10.1007/978-981-95-3453-1_20) · DOI [10.1007/978-981-95-3453-1_20](https://doi.org/10.1007/978-981-95-3453-1_20)

细粒度的情感标签之间并不独立：把"喜悦"预测成"愉悦"，比预测成"悲伤"错得轻。常规的多标签训练（二元交叉熵）却把这两种错误一视同仁。本文提出 **Emotional Earth Mover's Distance（EEMD）**：把情感标签的层级结构编码成运输代价，既作为与 BCE 并用的训练损失，也作为评估指标，让"接近的错误"比"离得远的错误"受到更小的惩罚。

完整的方法说明、配置和推理用法见[英文 README](../README.md)。

## 仓库内容

| 路径 | 内容 | 在论文中的角色 |
|---|---|---|
| 根目录 | BERT + 可学习权重的 BCE/EEMD 组合损失 | 本文方法 |
| [`baselines/bert/`](../baselines/bert) | 只用 BCE 的 BERT（[GoEmotions-pytorch](https://github.com/monologg/GoEmotions-pytorch)），加上了 EEMD 评估指标 | 微调基线 |
| [`baselines/llm/`](../baselines/llm) | 用思维链提示大模型（gpt-3.5-turbo），按同样的指标评估 | 大模型基线 |
| [`data/original/`](../data/original) | GoEmotions 的 train/dev/test 划分，28 个标签，三组实验共用 | 数据集 |

三组实验都在同一数据划分上报告 micro/macro/weighted F1、准确率、Hamming loss 和 EMD，数值可以直接对比。具体结果见论文。

## 复现

```bash
pip install -r requirements.txt

# 本文方法（在仓库根目录运行）
python run_goemotions.py --taxonomy original

# 微调 BERT 基线（在它的目录下运行，读取 ../../data/original）
cd baselines/bert && python run_goemotions.py --taxonomy original && cd ../..

# 大模型基线（评估 dev 集，会消耗 API 额度）
export OPENAI_API_KEY=sk-...
python baselines/llm/Fine-grained-emotions-analysis-by-LLM.py
```

BERT 相关实验固定使用论文时的 `torch==1.4.0` 和 `transformers==2.11.0`，用 Python 3.7 环境安装最省事。大模型基线需要 `openai`、`pandas`、`POT` 和 `scikit-learn`，可以通过 `OPENAI_MODEL` 换用其他模型。

## 引用

```bibtex
@inproceedings{yu2025emotional,
  title={Emotional Earth Mover’s Distance for Fine-Grained Hierarchical Emotion Analysis},
  author={Yu, Hai-Tao and Li, Dawei and Kang, Xin},
  booktitle={International Conference on Advanced Data Mining and Applications},
  pages={296--310},
  year={2025},
  organization={Springer}
}
```

## 许可证

[Apache-2.0](../LICENSE)。训练代码基于 [monologg/GoEmotions-pytorch](https://github.com/monologg/GoEmotions-pytorch)（Apache-2.0）。
