# -*- coding: utf-8 -*-
"""어텐션 계산의 결과물인 행렬 하나를 통째로 찍는다."""
import torch, sentencepiece as spm
from model import GPT, Config
from gen import find

TEXT = "Zorbax was hungry. Zorbax went to find food. Zor"
LAYER, HEAD = 3, 5

sp = spm.SentencePieceProcessor(model_file="tok/sp4000.model")
ck = torch.load(find("C"), map_location="cpu", weights_only=False)
model = GPT(Config(**ck["cfg"])); model.load_state_dict(ck["model"]); model.eval()

ids = sp.encode(TEXT)
pcs = [p.replace("▁", "_") for p in sp.encode(TEXT, out_type=str)]
T = len(ids)
with torch.no_grad():
    _, _, attns = model(torch.tensor([ids]), return_attn=True)
M = attns[LAYER][0, HEAD].numpy()

print(f"문장: {TEXT!r}\n")
print("위치별 토큰:")
for i, p in enumerate(pcs):
    print(f"  {i:>2}={p:<8}", end="\n" if (i + 1) % 6 == 0 else "")
print()

print(f"어텐션 행렬 ({T} x {T}).  값은 %, 반올림해서 0이면 빈칸")
print(f"{'':>11}" + "".join(f"{c:>4}" for c in range(T)) + "    <- 보이는 위치")
print(f"{'':>11}" + "".join(f"{p[:3]:>4}" for p in pcs))
print(f"{'':>11}" + "----" * T)
for r in range(T):
    row = "".join(f"{round(M[r, c] * 100):>4}" if M[r, c] >= 0.005 else "   ."
                  for c in range(T))
    mark = "   <- 마지막 위치" if r == T - 1 else ""
    print(f"{r:>3} {pcs[r][:6]:>6} |{row}{mark}")
print(f"\n각 행의 합: {[round(float(M[r].sum()), 2) for r in range(0, T, 6)]} ... 전부 1.0")
