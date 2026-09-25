<p align="center">
  <a href="../README.md"><img alt="English" src="https://img.shields.io/badge/English-eaeef2?style=for-the-badge"></a>
  <a href="README.zh-CN.md"><img alt="简体中文" src="https://img.shields.io/badge/%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-eaeef2?style=for-the-badge"></a>
  <a href="README.ja.md"><img alt="日本語" src="https://img.shields.io/badge/%E6%97%A5%E6%9C%AC%E8%AA%9E-1f2328?style=for-the-badge"></a>
</p>

# Emotional Earth Mover’s Distance for Fine-Grained Hierarchical Emotion Analysis

論文 **"Emotional Earth Mover’s Distance for Fine-Grained Hierarchical Emotion Analysis"** の公式実装です。著者：Hai-Tao Yu、Dawei Li、Xin Kang。*Advanced Data Mining and Applications (ADMA 2025)*、Springer、pp. 296–310。[[論文]](https://link.springer.com/chapter/10.1007/978-981-95-3453-1_20) · DOI [10.1007/978-981-95-3453-1_20](https://doi.org/10.1007/978-981-95-3453-1_20)

細粒度の感情ラベルは互いに独立ではありません。「喜び」を「楽しさ」と予測するのは、「悲しみ」と予測するより軽い誤りです。しかし通常のマルチラベル学習（二値交差エントロピー）は両者を同じように扱います。本研究では **Emotional Earth Mover's Distance（EEMD）** を提案します。感情ラベルの階層構造を輸送コストとして符号化し、BCE と併用する学習損失として、また「近い誤り」を「遠い誤り」より軽く評価する指標として用います。

手法の詳細、設定、推論の使い方は[英語版 README](../README.md) を参照してください。

## リポジトリの構成

| パス | 内容 | 論文での位置づけ |
|---|---|---|
| ルート | BERT ＋ 学習可能な重みによる BCE/EEMD 複合損失 | 提案手法 |
| [`baselines/bert/`](../baselines/bert) | BCE のみの BERT（[GoEmotions-pytorch](https://github.com/monologg/GoEmotions-pytorch)）に EEMD 評価指標を追加 | ファインチューニングのベースライン |
| [`baselines/llm/`](../baselines/llm) | 大規模言語モデル（gpt-3.5-turbo）への Chain-of-Thought プロンプト。同じ指標で評価 | LLM ベースライン |
| [`data/original/`](../data/original) | GoEmotions の train/dev/test 分割（28 ラベル）。3 つの実験で共通 | データセット |

3 つの実験はすべて同じ分割で micro/macro/weighted F1・正解率・Hamming loss・EMD を報告するため、数値をそのまま比較できます。結果は論文をご覧ください。

## 再現方法

```bash
pip install -r requirements.txt

# 提案手法（リポジトリのルートで実行）
python run_goemotions.py --taxonomy original

# BERT ベースライン（そのフォルダで実行。../../data/original を読み込みます）
cd baselines/bert && python run_goemotions.py --taxonomy original && cd ../..

# LLM ベースライン（dev 分割を評価。API 利用料がかかります）
export OPENAI_API_KEY=sk-...
python baselines/llm/Fine-grained-emotions-analysis-by-LLM.py
```

BERT の実験は論文当時の `torch==1.4.0` と `transformers==2.11.0` に固定しています。Python 3.7 の環境で入れるのが簡単です。LLM ベースラインには `openai`・`pandas`・`POT`・`scikit-learn` が必要で、`OPENAI_MODEL` で別のモデルも試せます。

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

## ライセンス

[Apache-2.0](../LICENSE)。学習コードは [monologg/GoEmotions-pytorch](https://github.com/monologg/GoEmotions-pytorch)（Apache-2.0）をベースにしています。
