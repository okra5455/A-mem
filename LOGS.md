# Run log index

This file is the key to the six archives at the repository root. Each archive
contains the raw evaluation log(s) for one run of `test_advanced_robust.py`.
The JSON result files are the primary artifacts; the archives are the execution
evidence (commands, per-question prompts and predictions, warnings, timings).

Each archive was produced with `zip <archive> <log>`, so the log can be read
with any unzip tool. The three 0-byte logs are aborted restarts of the same
run; they are kept rather than deleted, and the completed log is identified
below.

| Archive | Run | Result file | Log(s) inside | Window |
|---|---|---|---|---|
| `eval_robust_deepseek-chat_openai_ratio0.1_conversation_2026-11-07-14-05.zip` | DeepSeek-V4.1-Flash Baseline, conversation 0 | `results_robust_deepseek-chat_ratio01.json` | 20.0 MB | 2026-11-07 14:05-17:14 |
| `eval_robust_deepseek-chat_openai_ratio0.1_conversation_question_2026-11-10-02-37.zip` | DeepSeek-V4.1-Flash Ablation, conversation 0 | `results_deepseek-chat-ablation_question_ratio01.json` | 19.9 MB | 2026-11-10 02:37-02:50 |
| `eval_robust_deepseek-chat_openai_ratio1.0_conversation_2026-11-08-04-14.zip` | DeepSeek-V4.1-Flash Baseline, all conversations | `results_robust_deepseek-chat.json` | three 0-byte restarts (`04-14`, `04-58`, `05-16`) plus the completed 178.6 MB `05-31` log | 2026-11-08 05:31-13:58 |
| `eval_robust_deepseek-chat_openai_ratio1.0_conversation_question_2026-11-10-03-13.zip` | DeepSeek-V4.1-Flash Ablation, all conversations | `results_deepseek-chat-ablation_question.json` | 177.8 MB | 2026-11-10 03:13-05:33 |
| `eval_robust_deepseek-flash_openai_ratio0.1_conversation_2026-11-07-02-16.zip` | Pilot (excluded) | none - the pilot wrote to a generic name later overwritten by the baseline | 4.1 MB | 2026-11-07 02:16-06:11 |
| `eval_robust_qwen2.5-1.5b_ollama_ratio0.1_conversation_2026-11-09-21-18.zip` | Qwen2.5-1.5B, conversation 0 | `results_robust_ollama_qwen15b_ratio01.json` | two 0-byte restarts (`21-18`, `21-19`) plus the completed 6.2 MB `21-45` log | 2026-11-09 21:45-2026-11-10 04:19 |

## Commands

All commands were run from the repository root after sourcing `.env`
(`OPENAI_API_KEY`, `OPENAI_BASE_URL=https://api.deepseek.com`,
`CUDA_VISIBLE_DEVICES=`).

```bash
# baseline, conversation 0
python test_advanced_robust.py --backend openai --model deepseek-chat \
  --dataset data/locomo10.json --ratio 0.1 --seed 42 --retrieve_k 10 \
  --output results_robust_deepseek-chat_ratio01.json

# baseline, all conversations
python test_advanced_robust.py --backend openai --model deepseek-chat \
  --dataset data/locomo10.json --seed 42 --retrieve_k 10 \
  --output results_robust_deepseek-chat.json

# Qwen2.5-1.5B, conversation 0
python test_advanced_robust.py --backend ollama --model qwen2.5-1.5b \
  --dataset data/locomo10.json --ratio 0.1 --seed 42 --retrieve_k 10 \
  --output results_robust_ollama_qwen15b_ratio01.json

# ablation, conversation 0
python test_advanced_robust.py --backend openai --model deepseek-chat \
  --dataset data/locomo10.json --ratio 0.1 --seed 42 --retrieve_k 10 \
  --query_mode question --output results_deepseek-chat-ablation_question_ratio01.json

# ablation, all conversations
python test_advanced_robust.py --backend openai --model deepseek-chat \
  --dataset data/locomo10.json --seed 42 --retrieve_k 10 \
  --query_mode question --output results_deepseek-chat-ablation_question.json

# pilot (excluded): thinking-mode alias, 400 empty completions
python test_advanced_robust.py --backend openai --model deepseek-flash \
  --dataset data/locomo10.json --ratio 0.1 --output results_robust_deepseek.json
```

## Reading a run

Every result JSON contains the per-question `prediction`, `reference`,
`category` and `metrics`, plus the run parameters (`model`, `ratio`, `seed`,
`query_mode`, `memory_scope`, `legacy_c5_reference`) and `llm_failures`.
A non-empty `llm_failures` invalidates the run as evidence, regardless of the
score: the pilot above reports `{"empty_completion": 400}` and its apparent F1
was higher than the valid run's.
