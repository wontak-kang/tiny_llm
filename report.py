"""A/B/C 모델에 같은 프롬프트를 넣어 결과를 REPORT.md 로 정리한다."""
import glob, json, os
import torch, sentencepiece as spm
from model import GPT, Config

PROMPTS = [
    ("표준 시작 — 데이터 분포 한복판", "Once upon a time"),
    ("등장인물과 인과",              "The dragon was very hungry."),
    ("문맥 이어받기",                "Tom and Lily found a box. Inside the box was"),
    ("감정과 이유",                  "Sara was sad because"),
    ("학습 데이터 밖의 문장",        "The quarterly revenue forecast indicates that"),
]
N, TEMP, TOPK, SEED = 130, 0.8, 40, 7

sp = spm.SentencePieceProcessor(model_file="tok/sp4000.model")
dev = "cuda" if torch.cuda.is_available() else "cpu"

models = []
for ck_path in sorted(glob.glob("runs/*/ckpt.pt")):
    meta = json.load(open(os.path.join(os.path.dirname(ck_path), "config.json")))
    if "best_val_loss" not in meta:
        continue
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    m = GPT(Config(**ck["cfg"])).to(dev).eval()
    m.load_state_dict(ck["model"])
    models.append((meta, m))
models.sort(key=lambda x: x[0]["params"])

out = ["# A / B / C 모델 비교", "",
       f"공통 조건: temperature {TEMP}, top_k {TOPK}, {N}토큰 생성, **시드 {SEED} 고정** "
       "(모델 차이만 보이도록 난수를 맞춤)", "",
       "| | 파라미터 | 구조 | 학습 토큰 | 학습 시간 | val loss |",
       "|---|---|---|---|---|---|"]
for meta, _ in models:
    toks = meta["total_steps"] * meta["tokens_per_step"]
    out.append(f"| **{meta['preset']}** | {meta['params']:,} | "
               f"d={meta['d_model']}, {meta['n_layer']}층, head {meta['n_head']} | "
               f"{toks/1e6:.0f}M | {meta['minutes']:.1f}분 | **{meta['best_val_loss']:.4f}** |")
out.append("")

for title, prompt in PROMPTS:
    out += [f"## {title}", "", f"> 입력: `{prompt}`", ""]
    for meta, m in models:
        torch.manual_seed(SEED)
        ids = torch.tensor([sp.encode(prompt)], dtype=torch.long, device=dev)
        with torch.no_grad():
            gen = m.generate(ids, N, temperature=TEMP, top_k=TOPK)
        text = sp.decode(gen[0].tolist())[len(prompt):].strip()
        out += [f"**{meta['preset']}** ({meta['params']:,} params, val {meta['best_val_loss']:.3f})", "",
                "```", text, "```", ""]

open("REPORT.md", "w", encoding="utf-8").write("\n".join(out))
print("\n".join(out))
