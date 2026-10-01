"""학습 루프. 모든 Phase 가 이 파일을 인자만 바꿔 쓴다.

  python train.py --preset A          # 1.3M,  약 2분
  python train.py --preset B          # 7.3M,  약 17분
  python train.py --preset C          # 27.2M, 약 4시간
"""
import argparse, json, math, os, time, csv
import torch
import sentencepiece as spm
from model import GPT, Config
from data import Batcher

PRESETS = {   # d_model, n_head, n_layer, batch, 학습토큰(Chinchilla 20N 기준)
    "A": dict(d_model=128, n_head=4, n_layer=4,  batch_size=32, train_tokens=26_000_000),
    "B": dict(d_model=256, n_head=8, n_layer=8,  batch_size=32, train_tokens=147_000_000),
    "C": dict(d_model=512, n_head=8, n_layer=8,  batch_size=24, train_tokens=545_000_000),
}


def lr_at(step, total, base_lr, warmup):
    """워밍업 후 코사인 감쇠. 워밍업이 없으면 초반에 폭발한다 (Phase B 에서 직접 확인)."""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    r = (step - warmup) / max(1, total - warmup)
    return 0.1 * base_lr + 0.45 * base_lr * (1 + math.cos(math.pi * r)) * 2 / 2


@torch.no_grad()
def evaluate(model, batcher, iters, ctx):
    model.eval()
    total = 0.0
    for _ in range(iters):
        x, y = batcher.get()
        with ctx:
            _, loss = model(x, y)
        total += loss.item()
    model.train()
    return total / iters


@torch.no_grad()
def sample(model, sp, device, prompt="Once upon a time", n=120, temperature=0.8, top_k=40):
    model.eval()
    ids = torch.tensor([sp.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(ids, n, temperature=temperature, top_k=top_k)
    model.train()
    return sp.decode(out[0].tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="A", choices=list(PRESETS))
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--tag", default="")
    p.add_argument("--no_ffn", action="store_true", help="Phase B 해부용: FFN 제거")
    args = p.parse_args()

    cfg_d = PRESETS[args.preset]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    sp = spm.SentencePieceProcessor(model_file="tok/sp4000.model")
    cfg = Config(vocab_size=sp.get_piece_size(), seq_len=args.seq_len,
                 d_model=cfg_d["d_model"], n_head=cfg_d["n_head"], n_layer=cfg_d["n_layer"])
    model = GPT(cfg).to(device)

    if args.no_ffn:   # 해부: FFN 을 항등함수로 바꿔버린다
        import torch.nn as nn
        for b in model.blocks:
            b.ffn = nn.Identity()

    batch_size = cfg_d["batch_size"]
    tokens_per_step = batch_size * args.seq_len
    total_steps = cfg_d["train_tokens"] // tokens_per_step
    base_lr = args.lr or (3e-3 if cfg.d_model <= 256 else 1.5e-3)
    warmup = max(20, total_steps // 50)

    run = os.path.join("runs", time.strftime("%Y%m%d_%H%M%S") + f"_{args.preset}" +
                       (f"_{args.tag}" if args.tag else ""))
    os.makedirs(run, exist_ok=True)
    meta = dict(preset=args.preset, params=model.n_params(),
                params_no_emb=model.n_params(embedding=False),
                total_steps=total_steps, tokens_per_step=tokens_per_step,
                base_lr=base_lr, warmup=warmup, device=device,
                **{k: getattr(cfg, k) for k in ("vocab_size", "d_model", "n_head",
                                                "n_layer", "seq_len", "tie_embedding")},
                seed=args.seed, no_ffn=args.no_ffn)
    json.dump(meta, open(f"{run}/config.json", "w"), indent=2)

    log = open(f"{run}/log.txt", "w", encoding="utf-8")
    def say(s):
        print(s, flush=True); log.write(s + "\n"); log.flush()

    say(f"run     : {run}")
    say(f"params  : {meta['params']:,}  (임베딩 제외 {meta['params_no_emb']:,})")
    say(f"steps   : {total_steps:,} x {tokens_per_step:,} 토큰 = {cfg_d['train_tokens']/1e6:.0f}M")
    say(f"lr      : {base_lr}  warmup {warmup}")
    say(f"초기loss 기대값: ln({cfg.vocab_size}) = {math.log(cfg.vocab_size):.3f}")

    train = Batcher("data/train.bin", args.seq_len, batch_size, device, seed=args.seed)
    val = Batcher("data/val.bin", args.seq_len, batch_size, device, seed=args.seed + 1)
    say(f"data    : train {len(train)/1e6:.1f}M / val {len(val)/1e6:.1f}M 토큰")

    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.95),
                            weight_decay=0.1)
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" \
        else torch.autocast("cpu", enabled=False)

    mf = open(f"{run}/metrics.csv", "w", newline="", encoding="utf-8")
    mw = csv.writer(mf); mw.writerow(["step", "train_loss", "val_loss", "lr", "tokens", "sec", "tok_s"])
    sf = open(f"{run}/samples.txt", "w", encoding="utf-8")

    t0 = time.time()
    best = float("inf")
    for step in range(total_steps):
        lr = lr_at(step, total_steps, base_lr, warmup)
        for g in opt.param_groups:
            g["lr"] = lr

        x, y = train.get()
        with ctx:
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if not math.isfinite(loss.item()):      # 발산 즉시 중단 (밤새 헛돌지 않게)
            say(f"[중단] step {step} 에서 loss={loss.item()} — 발산")
            break

        if step % args.eval_every == 0 or step == total_steps - 1:
            vl = evaluate(model, val, 20, ctx)
            el = time.time() - t0
            toks = (step + 1) * tokens_per_step
            mw.writerow([step, f"{loss.item():.4f}", f"{vl:.4f}", f"{lr:.2e}",
                         toks, f"{el:.1f}", f"{toks/el:.0f}"]); mf.flush()
            say(f"step {step:>6}/{total_steps}  train {loss.item():.3f}  val {vl:.3f}  "
                f"lr {lr:.2e}  {toks/el:,.0f} tok/s  {el:.0f}s")
            txt = sample(model, sp, device)
            sf.write(f"\n===== step {step} (val {vl:.3f}) =====\n{txt}\n"); sf.flush()
            if vl < best:
                best = vl
                torch.save(dict(model=model.state_dict(), cfg=cfg.__dict__, step=step,
                                val_loss=vl), f"{run}/ckpt.pt")

    el = time.time() - t0
    say(f"\n완료: {el/60:.1f}분, best val loss {best:.4f}")
    say(f"최종 샘플:\n{sample(model, sp, device)}")
    json.dump({**meta, "best_val_loss": best, "minutes": el / 60},
              open(f"{run}/config.json", "w"), indent=2)
    for f in (log, mf, sf):
        f.close()


if __name__ == "__main__":
    main()
