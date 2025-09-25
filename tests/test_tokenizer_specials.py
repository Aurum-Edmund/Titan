from transformers import AutoTokenizer
def test_specials():
    tok = AutoTokenizer.from_pretrained("tokenizer_local")
    specials = ["<think>","</think>","<final>","</final>"]
    ids = [tok.convert_tokens_to_ids(s) for s in specials]
    assert all(i is not None for i in ids)
    for s in specials:
        e = tok.encode(s, add_special_tokens=False)
        assert len(e)==1, f"{s} must be single token"
