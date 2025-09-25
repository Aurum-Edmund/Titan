.\.venv\Scripts\activate
Set-Item Env:PYTHONPATH src

# Requires venv already activated; run from repo root
#Write-Host "Generating echo dataset..." -ForegroundColor Cyan
#python scripts/gen_echo_dataset.py --total 50000

Write-Host "Starting Titan Finetuning (100M preset)..." -ForegroundColor Cyan
python -m src.titan.titan_finetune `
  --config configs\titan_finetune.yaml `
  --base-ckpt runs\checkpoints\titan_v0\titan_step25000.pt