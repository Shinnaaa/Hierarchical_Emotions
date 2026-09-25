# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

Code for the ADMA 2025 paper "Emotional Earth Mover's Distance for
Fine-Grained Hierarchical Emotion Analysis" (Springer, DOI
10.1007/978-981-95-3453-1_20). GoEmotions, 28 labels, `data/original/`.

- Repository root: the proposed method — BERT whose loss mixes BCE and an EMD
  over the emotion hierarchy with a learnable weight (`model.py`,
  `cost_matric.py`).
- `baselines/bert/`: BCE-only BERT (monologg/GoEmotions-pytorch) with the EMD
  evaluation metric added. Merged via `git subtree` from the former
  `Shinnaaa/GoEmotions-baseline` fork, keeps its Apache-2.0 license. Runs from
  its own folder (`config/` is read relative to the working directory); its
  `data_dir` is `../../data/original`.
- `baselines/llm/`: chain-of-thought prompting of gpt-3.5-turbo, same metrics.
  Merged via `git subtree` from `Shinnaaa/Fine-grained-emotions-analysis-by-LLM`.
  Reads `OPENAI_API_KEY` / `OPENAI_MODEL` / `OPENAI_BASE_URL`; paths are
  relative to the script.

## Ground rules

- This is the code behind published numbers. Don't change training,
  evaluation or prompt behavior; fixes to paths, configuration or docs are
  fine. If behavior must change, keep the paper setting as the default.
- The BERT code pins `torch==1.4.0` / `transformers==2.11.0` (Python 3.7 era);
  don't upgrade them in place.
- The two BERT folders share `data/original/` and its `cached_*` feature files;
  their data loaders are identical, so the cache is safe to share.
- Known quirk kept on purpose: in `baselines/llm`, `dataset_to_predict` is
  unused — the loop evaluates `dev_df` (documented in its README).
