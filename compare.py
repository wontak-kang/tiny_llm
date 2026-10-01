"""같은 프롬프트를 여러 체크포인트에 넣어 나란히 비교한다.

  python compare.py "Once upon a time"           # 가진 것 전부
  python compare.py "..." -r A B                 # 지정한 것만
  python compare.py "..." -t 0.5 -n 100
"""
import argparse, glob, json, os
import torch
import sentencepiece as spm
from model import GPT, Config
from gen import find

p = argparse.ArgumentParser()
p.add_argument("prompt", nargs="+")
p.add_argument("-r", "--runs", nargs="*", default=None, help="A B C 또는 경로들")
p.add_argument("-n", type=int, default=120)
p.add_argument("-t", "--temperature", type=float, default=0.8)
p.add_argument("-k", "--top_k", type=int, default=40)
p.add_argument("--seed", type=int, default=0, help="같은 난수로 비교하려면 고정")
a = p.parse_args()

targets = a.runs or sorted(glob.glob("runs/*/ckpt.pt"))
prompt = " ".join(a.prompt)
sp = spm.SentencePieceProcessor(model_file="tok/sp4000.model")
dev = "cuda" if torch.cuda.is_available() else "cpu"

print(f"프롬프트: {prompt!r}   (t={a.temperature}, top_k={a.top_k}, n={a.n})")
for t in targets:
    path = find(t)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    meta = json.load(open(os.path.join(os.path.dirname(path), "config.json")))
    model = GPT(Config(**ck["cfg"]))
    model.load_state_dict(ck["model"])
    model.to(dev).eval()

    torch.manual_seed(a.seed)   # 모델 차이만 보이게 난수를 맞춘다
    ids = torch.tensor([sp.encode(prompt)], dtype=torch.long, device=dev)
    out = model.generate(ids, a.n, temperature=a.temperature, top_k=a.top_k)
    text = sp.decode(out[0].tolist())

    print(f"\n{'='*78}")
    print(f"[{meta['preset']}] {ck['cfg']['d_model']}d x {ck['cfg']['n_layer']}L  "
          f"{meta['params']:,} params  val {meta.get('best_val_loss', float('nan')):.4f}  "
          f"({meta.get('minutes', 0):.1f}분 학습)")
    print(f"{'='*78}")
    print(text[len(prompt):].strip())
    del model
    torch.cuda.empty_cache()
