import torch

def _ids_to_str(tok, ids):
    return tok.decode(ids, clean_up_tokenization_spaces=False)

def _mask_to_digit(logits_row, target_id):
    mask = torch.full_like(logits_row, -1e9)
    mask[0, target_id] = 0.0
    return logits_row + mask

def generate_with_math(model, tok, prompt, max_new_tokens=64, temperature=0.0, device="cpu",
                       math_guard=None):
    model.eval()
    ids = tok.encode(prompt, add_special_tokens=False)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    specials = {s: tok.convert_tokens_to_ids(s) for s in ["<think>","</think>","<final>","</final>"]}
    in_final = False
    target_ids = None
    with torch.no_grad(), torch.cuda.amp.autocast('cuda',enabled=False):
        for _ in range(max_new_tokens):
            out = model(x); logits = out["logits"][:, -1:, :]  # [1,1,V]
            if in_final and target_ids is not None:
                # hard-mask the next token to follow the known target sequence
                idx = x.size(1) - start_pos
                if idx < len(target_ids):
                    logits = _mask_to_digit(logits, target_ids[idx:idx+1])
            nxt = logits.argmax(-1)      # greedy
            x = torch.cat([x, nxt], dim=1)
            text = _ids_to_str(tok, x[0].tolist())
            if not in_final and text.endswith("<final>") and math_guard is not None:
                # ask math core what digits to emit; provide position
                start_pos = x.size(1)
                target_str = math_guard()  # returns string like "518"
                target_ids = torch.tensor(tok.encode(target_str, add_special_tokens=False), device=device)
                in_final = True
            if text.endswith("</final>"): break
    return _ids_to_str(tok, x[0].tolist())
