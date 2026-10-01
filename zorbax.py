# -*- coding: utf-8 -*-
"""처음 보는 이름을 모델이 어떻게 이어 쓰는지, 어텐션 행 하나를 실제로 꺼내 본다."""
import torch, numpy as np, sentencepiece as spm
from model import GPT, Config
from gen import find

TEXT = "Zorbax was hungry. Zorbax went to find food. Zor"
HEAD = (3, 5)          # inspect_attn.py 가 찾아낸 C 모델의 induction head

sp = spm.SentencePieceProcessor(model_file="tok/sp4000.model")
ck = torch.load(find("C"), map_location="cpu", weights_only=False)
model = GPT(Config(**ck["cfg"])); model.load_state_dict(ck["model"]); model.eval()

ids = sp.encode(TEXT)
pcs = sp.encode(TEXT, out_type=str)
T = len(ids)
with torch.no_grad():
    logits, _, attns = model(torch.tensor([ids]), return_attn=True)

print(f"입력: {TEXT!r}   (토큰 {T}개, 마지막은 {pcs[-1]!r})\n")

print("모델이 예측한 다음 토큰 상위 5개:")
for p, i in zip(*logits[0, -1].softmax(-1).topk(5)):
    print(f"   {sp.id_to_piece(int(i)):<8} {float(p)*100:5.1f}%")

l, h = HEAD
row = attns[l][0, h, T - 1].numpy()      # 마지막 위치가 각 위치를 보는 정도
print(f"\n[층{l} head{h}] 이 마지막 위치 {T-1}({pcs[-1]!r}) 에서 어디를 보는가")
print(f"  {'위치':>4} {'토큰':<9} {'보는 정도':>9}")
for i in np.argsort(-row)[:6]:
    mark = "  <- 첫 Zorbax 의 'b'" if i == 2 else ("  <- 둘째 Zorbax 의 'b'" if i == 9 else "")
    print(f"  {i:>4} {pcs[i]!r:<9} {row[i]*100:>8.1f}%{mark}")
print(f"  (합계 {row.sum()*100:.0f}%, 볼 수 있는 자리 {T}개라 아무데나 보면 자리당 {100/T:.1f}%)")

# 대조군: induction 이 아닌 head 하나
l2, h2 = 0, 0
row2 = attns[l2][0, h2, T - 1].numpy()
print(f"\n[층{l2} head{h2}] 대조군 — 같은 자리에서 어디를 보는가")
for i in np.argsort(-row2)[:4]:
    print(f"  {i:>4} {pcs[i]!r:<9} {row2[i]*100:>8.1f}%")

# 'b' 가 나오는 자리(2, 9)에 얼마나 주는지 head 별로
print(f"\n64개 head 중 'b'(위치 2,9)를 가장 많이 보는 head 상위 5개")
L, H = len(attns), attns[0].shape[1]
sc = sorted(((float(attns[l][0, h, T - 1, 2] + attns[l][0, h, T - 1, 9]), l, h)
             for l in range(L) for h in range(H)), reverse=True)
for v, l, h in sc[:5]:
    tag = "  <- 우리가 찾아낸 head" if (l, h) == HEAD else ""
    print(f"  층{l} head{h}   {v*100:5.1f}%{tag}")
