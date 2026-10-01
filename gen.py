"""학습한 체크포인트에 글을 넣고 뒤를 이어 쓰게 한다.

  python gen.py "Once upon a time"
  python gen.py "The dragon was angry." -n 200 -t 0.5
  python gen.py                          # 인자 없으면 대화형
"""
import argparse, glob, os
import torch
import sentencepiece as spm
from model import GPT, Config


def find(run):
    """run 이 None 이면 가장 최근, 'A'/'B'/'C' 면 그 preset 의 가장 최근, 아니면 경로 그대로."""
    if run and os.path.exists(run):
        return run
    sub = f"*_{run}" if run else "*"
    # github 100MB 한도 때문에 큰 체크포인트는 fp16 으로만 올라가 있다. 없으면 그쪽을 쓴다.
    for name in ("ckpt.pt", "ckpt_fp16.pt"):
        hits = sorted(glob.glob(f"runs/{sub}/{name}"))
        if hits:
            return hits[-1]
    raise SystemExit(f"체크포인트를 못 찾음: runs/{sub}/")


def load(run=None):
    run = find(run)
    ck = torch.load(run, map_location="cpu", weights_only=False)
    model = GPT(Config(**ck["cfg"]))
    model.load_state_dict(ck["model"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev).eval()
    print(f"[{run}]  step {ck['step']}  val_loss {ck['val_loss']:.4f}  "
          f"params {model.n_params():,}  device {dev}")
    return model, dev


def main():
    p = argparse.ArgumentParser()
    p.add_argument("prompt", nargs="*", help="이어 쓰게 할 앞글")
    p.add_argument("-n", type=int, default=150, help="생성할 토큰 수")
    p.add_argument("-t", "--temperature", type=float, default=0.8,
                   help="낮으면 뻔하게, 높으면 엉뚱하게")
    p.add_argument("-k", "--top_k", type=int, default=40, help="후보 상위 k개만 고름")
    p.add_argument("-r", "--run", default=None,
                   help="A/B/C (해당 preset 최신) 또는 ckpt.pt 경로. 생략하면 가장 최근")
    p.add_argument("--list", action="store_true", help="가진 체크포인트 목록만 보기")
    args = p.parse_args()

    if args.list:
        import json
        for c in sorted(glob.glob("runs/*/ckpt.pt")):
            m = json.load(open(os.path.join(os.path.dirname(c), "config.json")))
            print(f"  {os.path.dirname(c):<34} preset {m['preset']}  "
                  f"{m['params']:>10,} params  val {m.get('best_val_loss', float('nan')):.4f}")
        return

    model, dev = load(args.run)
    sp = spm.SentencePieceProcessor(model_file="tok/sp4000.model")

    def run_one(text):
        ids = torch.tensor([sp.encode(text)], dtype=torch.long, device=dev)
        n_unk = sum(1 for i in ids[0].tolist() if i == sp.unk_id())
        if n_unk:
            print(f"  (주의: {n_unk}개 토큰이 <unk> — vocab 4000 에 없는 글자)")
        out = model.generate(ids, args.n, temperature=args.temperature, top_k=args.top_k)
        full = sp.decode(out[0].tolist())
        print(f"\n\033[2m{text}\033[0m{full[len(text):]}\n")

    if args.prompt:
        run_one(" ".join(args.prompt))
    else:
        print("빈 줄이나 Ctrl-C 로 종료\n")
        while True:
            try:
                t = input("prompt> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not t:
                break
            run_one(t)


if __name__ == "__main__":
    main()
