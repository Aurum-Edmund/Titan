import torch
from transformers import AutoTokenizer
from src.titan.core import TitanModel

def test_overfit_tiny_batch():
    tok = AutoTokenizer.from_pretrained("tokenizer_local")
    if tok.pad_token_id is None: tok.add_special_tokens({"pad_token":"<|pad|>"})
    text = "Cat\n<think>Cat</think>\n<final>Cat</final>"
    ids = tok.encode(text, add_special_tokens=False)
    x = torch.tensor([ids[:-1]]); y = torch.tensor([ids[1:]])
    model = TitanModel(len(tok), d=256, n_blocks=4)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    lossf = torch.nn.CrossEntropyLoss()
    for _ in range(300):
        logits = model(x)["logits"]
        loss = lossf(logits.view(-1, logits.size(-1)), y.view(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.05, f"did not overfit tiny batch, loss={loss.item():.3f}"
