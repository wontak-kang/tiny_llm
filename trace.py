# -*- coding: utf-8 -*-
"""문장 하나를 모델에 통과시키며 각 단계의 텐서 모양을 찍는다. d 와 L 이 뭔지 보려는 것."""
import torch, sentencepiece as spm
from model import GPT, Config

ck = torch.load("runs/20260913_010147_A/ckpt.pt", map_location="cpu", weights_only=False)
cfg = Config(**ck["cfg"])
model = GPT(cfg); model.load_state_dict(ck["model"]); model.eval()
sp = spm.SentencePieceProcessor(model_file="tok/sp4000.model")

text = "Once upon a time"
ids = sp.encode(text)
print(f"입력 문장   : {text!r}")
print(f"토큰 조각   : {sp.encode(text, out_type=str)}")
print(f"토큰 ID     : {ids}   ({len(ids)}개)")
print(f"\n설정        : d_model={cfg.d_model}  n_layer={cfg.n_layer}  vocab={cfg.vocab_size}\n")

x = torch.tensor([ids])
print(f"{'단계':<34}{'모양':<20}{'숫자 개수':>10}")
print("-" * 66)
print(f"{'입력 (토큰 ID)':<34}{str(tuple(x.shape)):<20}{x.numel():>10}")

h = model.emb(x)
print(f"{'임베딩 통과 -> 벡터가 됨':<34}{str(tuple(h.shape)):<20}{h.numel():>10}")

cos, sin = model._rope(h.shape[1], h.device)
with torch.no_grad():
    for i, b in enumerate(model.blocks):
        h, _ = b(h, cos, sin)
        print(f"{f'블록 {i} 통과':<34}{str(tuple(h.shape)):<20}{h.numel():>10}")
    h = model.norm(h)
    logits = model.head(h)
print(f"{'출력층 통과 -> 어휘 점수':<34}{str(tuple(logits.shape)):<20}{logits.numel():>10}")

print(f"\n마지막 위치({sp.id_to_piece(ids[-1])!r}) 다음에 올 토큰 상위 5개:")
for p, i in zip(*logits[0, -1].softmax(-1).topk(5)):
    print(f"   {sp.id_to_piece(int(i)):<12} {float(p)*100:5.1f}%")

print(f"\n블록 하나의 부품 (전부 d x d = {cfg.d_model} x {cfg.d_model}):")
for name, p in model.blocks[0].named_parameters():
    print(f"   {name:<16}{str(tuple(p.shape)):<14}{p.numel():>8,}개")
tot = sum(p.numel() for p in model.blocks[0].parameters())
print(f"   {'블록 1개 합계':<16}{'':<14}{tot:>8,}개  x {cfg.n_layer}층 = {tot*cfg.n_layer:,}개")
