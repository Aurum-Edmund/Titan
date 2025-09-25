import torch
from transformers import AutoTokenizer
from src.titan.core import TitanModel
from src.titan.sampler import generate_with_math

def test_echo_minimal():
    tok = AutoTokenizer.from_pretrained("tokenizer_local")
    m = TitanModel(len(tok), d=256, n_blocks=4)
    out = generate_with_math(m, tok, "Thirteen\n<think>", max_new_tokens=8, temperature=0.0, device="cpu")
    assert "<think>" in out
