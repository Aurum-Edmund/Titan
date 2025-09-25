import torch, json
from transformers import AutoTokenizer
from src.titan.train_titan import JsonlLM, make_collate

def test_ignore_index_padding(tmp_path):
    p = tmp_path/"d.jsonl"
    with open(p,"w",encoding="utf-8") as f:
        f.write(json.dumps({"text":"A\n<think>A</think>\n<final>A</final>"})+"\n")
    tok = AutoTokenizer.from_pretrained("tokenizer_local")
    if tok.pad_token_id is None: tok.add_special_tokens({"pad_token":"<|pad|>"})
    ds = JsonlLM(str(p), tok, max_len=8)
    X,Y = make_collate(tok.pad_token_id)([ds[0], ds[0]])
    assert (Y== -100).sum().item() > 0, "labels must pad to -100"
    assert X.shape==Y.shape
