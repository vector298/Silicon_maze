#!/usr/bin/env python3
"""Logit-lens the F.R.I.D.A.Y. core on the 32 logged lines.

Usage:
    pip install torch transformers safetensors
    python friday_vault.py --model DIR --prompts prompts.csv [--target SHA256]

DIR is the folder holding config.json, the tokenizer files and the weights.
Only a small summary is printed, so you can paste it back.

Design (from the story):
  * N transformer blocks, indexed from 0 (lobby = floor 0).
  * Top two blocks are corrupt -> highest intact block is index N-3.
    In HF `hidden_states`, the output of block i is hidden_states[i+1].
  * The "voice box" shared by every floor is the output head (lm_head).
  * Raw log, nothing prepended -> add_special_tokens=False (BOS also tried).
  * Code = predicted next-token ids, one per line, in log order, joined by commas.
Because details are ambiguous, several variants are tried; --target (the
vault's SHA-256) tells you which one is right.
"""
import argparse, csv, hashlib, itertools, sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def find_parts(model):
    """Return (blocks, final_norm, lm_head) for common decoder layouts."""
    head = model.get_output_embeddings()
    base = getattr(model, model.base_model_prefix, model)
    norm = None
    for name in ("norm", "ln_f", "final_layernorm", "final_layer_norm", "norm_f"):
        if hasattr(base, name):
            norm = getattr(base, name)
            break
    return head, norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--target", default=None, help="vault SHA-256 (lowercase hex)")
    ap.add_argument("--trust", action="store_true", help="trust_remote_code")
    a = ap.parse_args()
    target = a.target.strip().lower() if a.target else None

    with open(a.prompts, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    prompts = [r["prompt"] for r in rows]
    print(f"{len(prompts)} prompts loaded")

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=a.trust)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.float32, trust_remote_code=a.trust
    ).eval()
    cfg = model.config
    N = cfg.num_hidden_layers
    head, norm = find_parts(model)
    print(f"layers N={N}; norm={type(norm).__name__ if norm else None}; "
          f"head={type(head).__name__}; tied={getattr(cfg,'tie_word_embeddings',None)}")

    results = {}
    # layer_idx = block index whose output we read (story: N-3); try neighbours too
    for add_bos in (False, True):
        hs_cache = []
        for p in prompts:
            ids = tok(p, add_special_tokens=False)["input_ids"]
            if add_bos and tok.bos_token_id is not None:
                ids = [tok.bos_token_id] + ids
            with torch.no_grad():
                out = model(torch.tensor([ids]), output_hidden_states=True)
            hs_cache.append((out.hidden_states, out.logits[0, -1]))
        for layer_idx in (N - 3, N - 2, N - 4):
            for use_norm in (True, False):
                pred = []
                for hs, _ in hs_cache:
                    h = hs[layer_idx + 1][0, -1]
                    with torch.no_grad():
                        if use_norm and norm is not None:
                            h = norm(h)
                        pred.append(int(head(h).argmax()))
                code = ",".join(map(str, pred))
                digest = hashlib.sha256(code.encode()).hexdigest()
                key = (add_bos, layer_idx, use_norm)
                results[key] = (pred, code, digest)

    hit = None
    for (bos, li, un), (pred, code, digest) in results.items():
        tag = f"bos={bos} layer={li} final_norm={un}"
        mark = ""
        if target and digest == target:
            mark = "   <== MATCH"
            hit = (tag, pred)
        print(f"\n[{tag}]{mark}\n sha256={digest}\n ids={code}")
        if mark or not target:
            print(" words=" + " | ".join(repr(tok.decode([t])) for t in pred))

    if target:
        if hit:
            print("\nVAULT CODE FOUND:", hit[0])
            print("flag/words:", "".join(tok.decode([t]) for t in hit[1]))
        else:
            print("\nNo variant matched the target. Paste the output back to me.")


if __name__ == "__main__":
    sys.exit(main())
