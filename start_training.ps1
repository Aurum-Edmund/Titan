.\.venv\Scripts\activate
Set-Item Env:PYTHONPATH src

# Requires venv already activated; run from repo root
#Write-Host "Generating echo dataset..." -ForegroundColor Cyan
#python scripts/gen_echo_dataset.py --total 50000

Write-Host "Starting Titan training (100M preset)..." -ForegroundColor Cyan
python -m src.titan.train_titan --config configs\titan_100m.yaml

Write-Host "Probing final model..." -ForegroundColor Cyan
python tools\titan_probe.py --tokenizer tokenizer_local --d 512 --layers 36 --seq 128 --batch 2 --device cuda --dtype fp16
