Setup (PowerShell):
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt

Generate Titan80M dataset:
python tools\make_titan80m_dataset.py --cfg Titan80M_Dataset\meta\sampling_config_v1.yaml --out Titan80M_Dataset\shards --workers 8

Train:
python -m src.titan.train_titan --config configs\titan_100m.yaml

./start_training.ps1

Split dataset:
python tools\split_train_val.py --in_path data\repetition_basic.jsonl --val_frac 0.1 --dedup none `
  --train_out data\train.jsonl --val_out data\val.jsonl

Sample Greedy:
$ckpt = "runs\checkpoints\titan_v0\titan_step25000.pt"

python tools\sample_greedy.py --ckpt $ckpt --tokenizer tokenizer_local_200k --prompt harbor --max_new 64 --d 512 --layers 36 --device cpu --dtype fp32

Sample Structured:
python tools\sample_structured.py --ckpt $ckpt --tokenizer tokenizer_local_200k --prompts harbor "how many r are in strawberry?" --d 512 --layers 36 --device cpu --dtype fp32 --max_new 64

Check Memorization:
python tools\eval_memorization.py --ckpt runs\checkpoints\titan_v0\titan_final.pt `
  --tokenizer tokenizer_local --train_jsonl data\train.jsonl --val_jsonl data\val.jsonl --topk 20

Check echo:
# (Optional) convert plain-text val set into prompts.jsonl
Get-Content data\val.jsonl | % { ($_ | ConvertFrom-Json).text } |
  % { @{prompt=$_} | ConvertTo-Json -Compress } > data\echo_val.jsonl

$ckpt = "runs\checkpoints\titan_v0\titan_step25000.pt"
python tools\eval_echo.py --ckpt $ckpt `
  --tokenizer tokenizer_local_200k --prompts_jsonl data\val.jsonl `
  --limit 100 --max_new 64 --d 512 --layers 36 `
  --device cpu --dtype fp32

Generate Echo Dataset:
# generate 200k echo lines
python tools\gen_echo_dataset.py

# (optional) split to train/val exactly as before
python tools\split_train_val.py --in_path data\repetition_basic.jsonl `
  --val_frac 0.1 --dedup none `
  --train_out data\train.jsonl --val_out data\val.jsonl

Generate Termination booster dataset:
# regen the booster (gentle)
python tools\gen_termination_booster.py --n 30000 --ratio_immediate 0.05 --ratio_onechar 0.15

# merge with your 200k echo set
Get-Content data\repetition_basic.jsonl, data\termination_booster.jsonl `
  | Set-Content -Encoding utf8 data\repetition_with_termination.jsonl

# re-split
python tools\split_train_val.py --in_path data\dictionary_echo.jsonl `
  --val_frac 0.1 --dedup none `
  --train_out data\train.jsonl --val_out data\val.jsonl

Generate Context Dataset:
python tools/gen_context_dataset.py 

Train Context Core:
python -m src.titan.train_context --config configs\context_core.yaml

# Evaluate Context Core:
python tools\eval_context.py --ckpt runs\checkpoints\context_core_v0\context_step8000.pt `
  --tokenizer tokenizer_local --jsonl data\context_val.jsonl `
  --d 384 --layers 12 --heads 6 --topics 16 --intents 8

# Finetune Titan (LoRA/QLoRA):
python -m src.titan.titan_finetune `
  --config configs\titan_finetune.yaml `
  --base-ckpt runs\checkpoints\titan_v0\titan_step25000.pt

# Generate Alphabet_digits:
python -m py_compile tools\gen_alphabet_digits_dataset.py
python tools\gen_alphabet_digits_dataset.py data\alphabet_digits.jsonl --count 1_000_000

python tools\split_train_val.py --in_path data\alphabet_digits.jsonl `
  --val_frac 0.1 --dedup none `
  --train_out data\train.jsonl --val_out data\val.jsonl

# Generate Alphabet_Digits_Definitions:
python tools\gen_alpha_digit_definitions.py data\letter_digit_defs.jsonl --count 1000000

python tools\split_train_val.py --in_path data\letter_digit_defs.jsonl `
  --val_frac 0.1 --dedup none `
  --train_out data\train.jsonl --val_out data\val.jsonl

# Generate Symbols Dataset:
python tools\gen_symbol_dataset.py data\symbols.jsonl --count 1000000

python tools\split_train_val.py --in_path data\symbols.jsonl `
  --val_frac 0.1 --dedup none `
  --train_out data\train.jsonl --val_out data\val.jsonl

# Generate is verb use Dataset:
python tools\gen_is_verb_dataset.py data\is_verb.jsonl --count 1000000

python tools\split_train_val.py --in_path data\is_verb.jsonl `
  --val_frac 0.1 --dedup none `
  --train_out data\train.jsonl --val_out data\val.jsonl

# Mix dataset is verb, letter, numbers defs :
python tools\mix_curriculum.py data\mixed_train.jsonl --count 1000000

python tools\split_train_val.py --in_path data\mixed_train.jsonl `
  --val_frac 0.1 --dedup none `
  --train_out data\train.jsonl --val_out data\val.jsonl
