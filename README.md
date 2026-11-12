# Agentic Memory 🧠

A novel agentic memory system for LLM agents that can dynamically organize memories in an agentic way.

> **Note:** This repository is a reproduction fork of the paper implementation, based on
> [WujiangXu/AgenticMemory](https://github.com/WujiangXu/AgenticMemory) at commit `f9a173d`.
> If you want to use A-Mem in your own agents, please use the library version:
> [A-mem-sys](https://github.com/WujiangXu/A-mem-sys).

For more details, please refer to the paper:
[A-Mem: Agentic Memory for LLM Agents](https://arxiv.org/abs/2502.12110).

## About this fork

The authors' code is used essentially unchanged. This fork adds only what was needed to run
the LoCoMo experiments and to audit the evaluation:

| Addition | What it does |
|---|---|
| `--query_mode {keywords,question}` | `keywords` (default) is A-Mem's LLM query expansion; `question` uses the raw question as the retrieval query and skips one LLM call per question. This is the only functional change to the system. |
| `score_c5_legacy.py` + `c5_legacy_scores.json` | Recomputes category-5 (adversarial) scores under both the corrected reference and the paper-era trap reference, from the stored predictions, without re-running any model. |
| `ast.Str` import removed | The original code does not import on Python ≥ 3.12. |
| `LOGS.md` | Index of the per-run log archives: command, result file, timestamps and notes. |

## Introduction 🌟

Large Language Model (LLM) agents have demonstrated remarkable capabilities in handling complex real-world tasks through external tool usage. However, to effectively leverage historical experiences, they require sophisticated memory systems. Traditional memory systems, while providing basic storage and retrieval functionality, often lack advanced memory organization capabilities.

The project introduces an innovative **Agentic Memory** system that revolutionizes how LLM agents manage and utilize their memories:

<div align="center">
  <img src="Figure/intro-a.jpg" alt="Traditional Memory System" width="600"/>
  <img src="Figure/intro-b.jpg" alt="Proposed Agentic Memory" width="600"/>
  <br/>
  <em>Comparison between traditional memory system (top) and the proposed agentic memory (bottom). The system enables dynamic memory operations and flexible agent-memory interactions.</em>
</div>

## Key Features ✨

- 🔄 Dynamic memory organization based on Zettelkasten principles
- 🔍 Intelligent indexing and linking of memories
- 📝 Comprehensive note generation with structured attributes
- 🌐 Interconnected knowledge networks
- 🔄 Continuous memory evolution and refinement
- 🤖 Agent-driven decision making for adaptive memory management

## Framework 🏗️

<div align="center">
  <img src="Figure/framework.jpg" alt="Agentic Memory Framework" width="800"/>
  <br/>
  <em>The framework of the Agentic Memory system showing the dynamic interaction between LLM agents and memory components.</em>
</div>

## How It Works 🛠️

When a new memory is added to the system:

1. Generates comprehensive notes with structured attributes
2. Creates contextual descriptions and tags
3. Analyzes historical memories for relevant connections
4. Establishes meaningful links based on similarities
5. Enables dynamic memory evolution and updates

## Requirements

| Component | Requirement |
|---|---|
| Python | 3.10 or newer (tested on 3.14.4) |
| OS | Linux, macOS, or Windows via WSL2 (tested on Ubuntu 26.04 / WSL2) |
| Disk | ~2 GB for model downloads, plus ~1 GB per full run for caches and logs |
| GPU | Optional. Everything runs on CPU; a full run took 8.5 h on two cores |
| Access | An OpenAI-compatible API key (DeepSeek is used here) or a local Ollama server |

## Installation

```bash
git clone https://github.com/okra5455/A-mem.git
cd A-mem

python3 -m venv .venv
source .venv/bin/activate          # Windows (PowerShell): .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

The first run downloads the models used by the pipeline:

- `all-MiniLM-L6-v2` (~90 MB) for note and query embeddings;
- NLTK `punkt` and `wordnet`, downloaded automatically on first import;
- BERTScore's `roberta-large` (~1.4 GB) the first time an answer is scored.

## Configuration

The harness reads credentials from environment variables; it does **not** parse `.env`
itself. Copy the example file and fill in your key:

```bash
cp .env.example .env
# edit .env, then export the variables into the current shell:
set -a; source .env; set +a
```

On Windows PowerShell, set the same variables directly:

```powershell
$env:OPENAI_API_KEY      = "sk-..."
$env:OPENAI_BASE_URL     = "https://api.deepseek.com"
$env:OLLAMA_HOST         = "http://127.0.0.1:11434"
# $env:CUDA_VISIBLE_DEVICES = ""   # uncomment to force CPU-only execution
```

Any OpenAI-compatible endpoint works: set `OPENAI_BASE_URL` and pass the matching model
name with `--model`. The runs in this repository used **DeepSeek-V4.1-Flash-1017 in
non-thinking mode**, served under the legacy alias `deepseek-chat`.

## Quick start

Run conversation 0 (199 questions) end to end:

```bash
python test_advanced_robust.py \
  --backend openai \
  --model deepseek-chat \
  --dataset data/locomo10.json \
  --ratio 0.1 \
  --seed 42 \
  --retrieve_k 10 \
  --output results_quick.json
```

Notes:

- `--ratio 0.1` takes the first 10% of the dataset (conversation 0), not a random sample.
- `--seed 42` fixes the order of the two answer options shown for category-5 questions;
  without a fixed seed, repeated runs are not comparable.
- `--retrieve_k 10` is the retrieval depth used in every run in this repository. The paper
  tuned it per model and category; we fixed it at 10.
- `--output` is required. Without it the results are only written to the log.

## Full runs

The five runs behind the committed result files (`--dataset data/locomo10.json --seed 42
--retrieve_k 10` in all of them):

```bash
# 1. Baseline, conversation 0
python test_advanced_robust.py --backend openai --model deepseek-chat \
  --dataset data/locomo10.json --ratio 0.1 --seed 42 --retrieve_k 10 \
  --output results_robust_deepseek-chat_ratio01.json

# 2. Baseline, all ten conversations
python test_advanced_robust.py --backend openai --model deepseek-chat \
  --dataset data/locomo10.json --seed 42 --retrieve_k 10 \
  --output results_robust_deepseek-chat.json

# 3. Ablation (raw-question retrieval), conversation 0
python test_advanced_robust.py --backend openai --model deepseek-chat \
  --dataset data/locomo10.json --ratio 0.1 --seed 42 --retrieve_k 10 \
  --query_mode question \
  --output results_deepseek-chat-ablation_question_ratio01.json

# 4. Ablation (raw-question retrieval), all ten conversations
python test_advanced_robust.py --backend openai --model deepseek-chat \
  --dataset data/locomo10.json --seed 42 --retrieve_k 10 \
  --query_mode question \
  --output results_deepseek-chat-ablation_question.json

# 5. Qwen2.5-1.5B, conversation 0 (see the Ollama section first)
python test_advanced_robust.py --backend ollama --model qwen2.5-1.5b \
  --dataset data/locomo10.json --ratio 0.1 --seed 42 --retrieve_k 10 \
  --output results_robust_ollama_qwen15b_ratio01.json
```

Wall clock on the reference machine (Intel Pentium G4400, 2 cores / 2 threads, CPU only,
7.7 GiB WSL2):

| Run | Questions | Wall clock |
|---|---|---|
| Baseline, conversation 0 | 199 | 3.16 h |
| Baseline, all conversations | 1,986 | 8.45 h |
| Ablation, conversation 0 | 199 | 0.23 h |
| Ablation, all conversations | 1,986 | 2.32 h |
| Qwen2.5-1.5B, conversation 0 | 199 | 6.57 h |

Memory construction, not answering, dominates a cold run: roughly 39 minutes per
conversation on this hardware. The two ablation timings are lower because they reused the
memory cache built by the baseline runs; they are not evidence of a speed-up.

## Local model with Ollama (Qwen2.5-1.5B)

```bash
ollama serve
ollama pull qwen2.5:1.5b

# Create a colon-free alias: the harness embeds the model name in the cache directory
# name, and ":" is not a valid character in Windows paths.
ollama cp qwen2.5:1.5b qwen2.5-1.5b
```

Then run with `--backend ollama --model qwen2.5-1.5b` (command 5 above). The `ollama`
Python client reads `OLLAMA_HOST`; the default is `http://127.0.0.1:11434`.

## Command-line reference

`test_advanced_robust.py` is the recommended harness: it has no JSON-schema dependency and
works with any OpenAI-compatible API, vLLM/SGLang or Ollama. (`test_advanced.py` is the
original harness and requires an OpenAI endpoint with structured-output support.)

| Flag | Default | Description |
|---|---|---|
| `--dataset` | `data/locomo10.json` | Dataset path. |
| `--backend` | `openai` | `openai`, `ollama`, `vllm` or `sglang`. |
| `--model` | `gpt-4o-mini` | Model name passed to the backend. |
| `--output` | none | Result JSON path. Required to persist results. |
| `--ratio` | `1.0` | Fraction of the dataset; takes a prefix, not a random sample. |
| `--retrieve_k` | `10` | Number of memories retrieved per question. |
| `--seed` | `42` | Seed for the order of category-5 answer options. |
| `--temperature_c5` | `0.5` | Temperature for category-5 questions (others use 0.7). |
| `--query_mode` | `keywords` | `keywords`: LLM query expansion (original). `question`: raw question (ablation). |
| `--legacy_c5_reference` | off | Score category 5 against the trap answer (paper-era protocol) instead of `"Not mentioned in the conversation"`. |
| `--refresh_index` | off | Re-embed neighbours immediately after memory evolution; uses a separate cache directory. |
| `--memory_scope` | `conversation` | `conversation`: each question uses its own conversation's memory (LoCoMo protocol). `pooled`: one memory over all evaluated conversations. |
| `--sglang_host`, `--sglang_port` | `http://localhost`, `30000` | Server address for the `vllm`/`sglang` backends. |

`run_all_experiments.sh` is the authors' cluster script (hard-coded paths, vLLM on
multiple GPUs); it is kept for reference and is not directly runnable elsewhere.

## Outputs and logs

Each `--output` file contains:

- the run parameters (`model`, `ratio`, `seed`, `query_mode`, `memory_scope`,
  `legacy_c5_reference`);
- `aggregate_metrics`: means for `overall` and `category_1` … `category_5`;
- `individual_results`: one entry per question, with the prediction, the reference and all
  metrics;
- `llm_failures`, `cached_samples` and `stale_cache_samples`.

**Check `llm_failures` before trusting any number.** A non-empty counter invalidates the
run, regardless of the score. In one pilot, a thinking-mode model returned empty
completions 400 times and still produced a complete, plausible-looking metric summary that
was higher than the valid run's:

```json
"llm_failures": {"empty_completion": 400}
```

Logs are written to `logs/eval_robust_{model}_{backend}_ratio{ratio}_{scope}_{query_mode}_{timestamp}.log`.
They record every prompt and prediction and can reach ~180 MB for a full run; `logs/` is
gitignored. `LOGS.md` indexes the six archived runs kept at the repository root.

## Category-5 scoring

Category 5 (adversarial) questions are unanswerable by design. The repository scores them
against `"Not mentioned in the conversation"` and separately reports
`adversarial_correct` (declines and does not state the trap). The paper-era protocol scored
them against the trap answer itself, which rewards reproducing the trap.

Both can be recomputed from the stored predictions without re-running a model:

```bash
python score_c5_legacy.py            # every results_*.json in the repository
python score_c5_legacy.py --full     # slower: all metrics, including BERTScore/ROUGE
```

This writes `c5_legacy_scores.json` with the fixed-reference and legacy-reference scores,
the `adversarial_correct` rate, and a breakdown of predictions. `--legacy_c5_reference`
selects the same protocol when running the harness directly.

## Caching and re-running

Memories are built once and cached under `cached_memories_robust_{backend}_{model}/`
(`…_refresh/` when `--refresh_index` is used). The cache is keyed on backend and model
only and is guarded by a `.version` marker, so changing `--ratio`, `--seed`, `--retrieve_k`
or `--query_mode` reuses the existing memories — which is what makes the `--query_mode`
ablation a controlled comparison.

To rebuild from scratch, delete the directory:

```bash
rm -rf cached_memories_robust_*
```

`cached_memories_*/` is gitignored.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `CUDA error: no kernel image is available for execution on the device` | The installed PyTorch build has no kernels for the GPU's compute capability (e.g. a GTX 960, sm_52) | Force CPU with `CUDA_VISIBLE_DEVICES=` (empty), or install a PyTorch build that supports the GPU |
| Predictions are empty; `llm_failures` is `{"empty_completion": N}` | A thinking-mode model consumed the 1,000-token budget before answering | Use a non-thinking model/alias |
| `invalid_link` is non-zero | The model's link output was rejected by the parser (index out of range or wrong format) | Not fatal; check that the rate is low. On conversation 0, Qwen2.5-1.5B produced 31 of 419 notes (7.4%) |
| `ImportError: cannot import name 'Str' from 'ast'` | The original code imports a module removed in Python 3.12 | This fork removed the import; pull the latest commit |
| First run is very slow and downloads ~1.5 GB | MiniLM + NLTK + BERTScore `roberta-large` are being fetched | Expected; the models are cached in `~/.cache/huggingface` and `~/nltk_data` |
| `bert_f1` is always 0 | BERTScore failed, usually because the model could not be downloaded | Check network/proxy; the other metrics are unaffected |
| Zero-byte log files | The run was restarted or killed | Use the log that ends with a metric summary; `LOGS.md` flags these cases |
| A model name with `:` breaks the cache path on Windows | `:` is invalid in Windows paths and the name is used in a directory name | Create a colon-free alias: `ollama cp qwen2.5:1.5b qwen2.5-1.5b` |

## Repository layout

```text
.
├── test_advanced_robust.py   # evaluation harness (recommended)
├── test_advanced.py          # original harness (needs OpenAI JSON schema support)
├── memory_layer_robust.py    # memory system + LLM backends
├── memory_layer.py           # original memory layer
├── utils.py                  # metrics (F1, BLEU, ROUGE, BERTScore, METEOR, SBERT)
├── score_c5_legacy.py        # category-5 rescoring from stored predictions
├── c5_legacy_scores.json     # output of score_c5_legacy.py
├── data/locomo10.json        # 1,986 QA pairs, 5,882 turns, 10 conversations
├── results_*.json            # committed run outputs
├── LOGS.md                   # index of the archived execution logs
├── Figure/                   # figures used above
├── requirements.txt
└── cached_memories_*/        # generated memory caches (gitignored)
```

## Results

All numbers below use the corrected category-5 reference and are **not directly
comparable with the paper**: the model is different, retrieval `k` is fixed at 10, and the
public dataset has 1,986 questions rather than the 7,512 reported in the paper.

| Run | Questions | F1 | BLEU-1 | Exact match |
|---|---|---|---|---|
| DeepSeek-V4.1-Flash-1017, baseline, all conversations | 1,986 | 47.43 | 42.37 | 28.95 |
| DeepSeek-V4.1-Flash-1017, ablation (`--query_mode question`), all conversations | 1,986 | 46.78 | 41.85 | 28.80 |
| Qwen2.5-1.5B, baseline, conversation 0 | 199 | 22.62 | 17.77 | 3.52 |

Per-category F1 for the full baseline: 32.44 (multi-hop), 30.02 (temporal), 11.56
(open-domain), 47.86 (single-hop) and 76.35 (adversarial). Under the paper-era trap
reference, the adversarial category scores 26.59 instead of 76.35; `score_c5_legacy.py`
reports both.

## Citation 📚

If you use this code in your research, please cite the original work:

```bibtex
@inproceedings{xu2025amem,
  title={A-Mem: Agentic memory for llm agents},
  author={Xu, Wujiang and Liang, Zujie and Mei, Kai and Gao, Hang and Tan, Juntao and Zhang, Yongfeng},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```

## License 📄

This project is licensed under the MIT License. See LICENSE for details.
