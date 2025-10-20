# Titan 80M Synthetic Dataset

Target: ≥1,000,000 JSONL chat traces covering conversational, instructional, agentic, coding, math, safety, and creative behaviors for the Titan 80M model.

## Slice Allocation
- S1 Conversational (multi-turn): 320,000
- S2 Instruction following: 120,000
- S3 Agentic Copilot (tools): 260,000
- S4 Coding reasoning: 80,000
- S5 Math & formal (incl. set theory): 120,000
- S6 Safety / refusal: 50,000
- S7 Creative / style: 50,000

Curriculum passes: Pass A (200k seed, short/direct), Pass B (+400k, medium, adversarial prompts), Pass C (+400k, long, richer tool loops).

## Layout
```
Titan80M_Dataset/
  shards/          # slice → split → part-*.jsonl shards (~10k rows each)
  meta/            # schemas, sampling config, tokenizer contract, license
  stats/           # populated by validation pipeline (distinct-n, histograms, etc.)
```

Train/val/test split: 96 % / 2 % / 2 %, stratified per slice and length bucket.

## Generation Workflow
1. Create a virtual environment (`python -m venv .venv`) and install `orjson`, `xxhash`, `numpy`, `pyyaml`.
2. Run `python tools/make_titan80m_dataset.py --cfg Titan80M_Dataset/meta/sampling_config_v1.yaml --out Titan80M_Dataset/shards --workers 8`.
3. The generator writes slice shards plus consolidated `Titan80M_Dataset/{train,val,test}.jsonl` for training.
4. Use the validator pipeline to ensure schema, masking, diversity, and entropy checks pass before promotion.

See `meta/schema_chat.md`, `meta/schema_agentic.md`, and `meta/tokenizer_special_tokens.json` for format specifics.
