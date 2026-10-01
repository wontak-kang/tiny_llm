# -*- coding: utf-8 -*-
"""에폭 실험 — 데이터를 1%로 줄이고 같은 것을 1/2/4/8/16번 반복해 보여준다.

에폭이 의미를 가지려면 '같은 데이터를 다시 본다'가 성립해야 한다.
그래서 여기서는 data.py 의 무작위 복원추출 대신 순차 에폭 로더를 쓴다.
"""
import json, math, os, time, csv
import numpy as np
import torch
import sentencepiece as spm
from model import GPT, Config

SUBSET_TOKENS = 5_000_000          # 전체 545M 의 약 1%
EPOCHS = [1, 2, 4, 8, 16]
SEQ, BATCH, SEED = 512, 32, 1234
D_MODEL, N_HEAD, N_LAYER = 256, 8, 8    # Phase B 와 같은 크기
BASE_LR = 3e-3
OUT = "runs_epoch"


class EpochLoader:
    """subset 을 seq 단위로 잘라 매 에폭마다 섞어서 한 번씩 전부 돈다."""

    def __init__(self, data, seq, batch, device, seed=0):
        self.data, self.seq, self.batch, self.device = data, seq, batch, device
        self.starts = np.arange(0, len(data) - seq - 1, seq)
        self.rng = np.random.default_rng(seed)
        self.pos, self.order = 0, self.rng.permutation(self.starts)

    @property
    def steps_per_epoch(self):
        return len(self.starts) // self.batch

    def get(self):
        if self.pos + self.batch > len(self.order):       # 에폭 끝 -> 다시 섞는다
            self.order, self.pos = self.rng.permutation(self.starts), 0
        ix = self.order[self.pos:self.pos + self.batch]
        self.pos += self.batch
        x = np.stack([self.data[i:i + self.seq] for i in ix]).astype(np.int64)
        y = np.stack([self.data[i + 1:i + 1 + self.seq] for i in ix]).astype(np.int64)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        if self.device == "cuda":
            x = x.pin_memory().to("cuda", non_blocking=True)
            y = y.pin_memory().to("cuda", non_blocking=True)
        return x.to(self.device), y.to(self.device)


@torch.no_grad()
def eval_loss(model, data, ctx, device, iters=25, seed=99):
    rng = np.random.default_rng(seed)
    model.eval()
    tot = 0.0
    for _ in range(iters):
        ix = rng.integers(0, len(data) - SEQ - 1, size=BATCH)
        x = torch.from_numpy(np.stack([data[i:i + SEQ] for i in ix]).astype(np.int64)).to(device)
        y = torch.from_numpy(np.stack([data[i + 1:i + 1 + SEQ] for i in ix]).astype(np.int64)).to(device)
        with ctx:
            _, loss = model(x, y)
        tot += loss.item()
    model.train()
    return tot / iters


@torch.no_grad()
def memorization(model, data, device, n=24, prefix=48, cont=64, seed=7):
    """학습 데이터의 앞 48토큰을 주고 탐욕적으로 이어 쓰게 해,
    실제 뒤 64토큰과 몇 개나 정확히 일치하는지 센다. 외웠으면 높게 나온다."""
    rng = np.random.default_rng(seed)
    model.eval()
    hits = 0
    for _ in range(n):
        i = int(rng.integers(0, len(data) - prefix - cont - 1))
        ctx_ids = torch.tensor([data[i:i + prefix].astype(np.int64)], device=device)
        true = data[i + prefix:i + prefix + cont].astype(np.int64)
        cur = ctx_ids
        got = []
        for _ in range(cont):
            logits, _ = model(cur[:, -SEQ:])
            nxt = logits[:, -1, :].argmax(-1, keepdim=True)   # 탐욕적 = 운 요소 제거
            got.append(int(nxt))
            cur = torch.cat([cur, nxt], 1)
        hits += sum(1 for a, b in zip(got, true) if a == b)
    model.train()
    return hits / (n * cont)


def main():
    os.makedirs(OUT, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sp = spm.SentencePieceProcessor(model_file="tok/sp4000.model")

    full = np.memmap("data/train.bin", dtype=np.uint16, mode="r")
    train = np.asarray(full[:SUBSET_TOKENS])          # 1% 만 떼서 RAM 에 올린다 (10MB)
    val = np.memmap("data/val.bin", dtype=np.uint16, mode="r")
    print(f"학습 데이터 {len(train):,} 토큰 (전체의 {len(train)/len(full)*100:.1f}%)")
    print(f"검증 데이터 {len(val):,} 토큰 (학습에 미사용)\n")

    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" \
        else torch.autocast("cpu", enabled=False)
    rows = []
    curves = {}

    for n_ep in EPOCHS:
        torch.manual_seed(SEED)
        cfg = Config(vocab_size=sp.get_piece_size(), seq_len=SEQ,
                     d_model=D_MODEL, n_head=N_HEAD, n_layer=N_LAYER)
        model = GPT(cfg).to(device)
        loader = EpochLoader(train, SEQ, BATCH, device, seed=SEED)
        spe = loader.steps_per_epoch
        total = spe * n_ep
        warmup = max(20, total // 50)
        opt = torch.optim.AdamW(model.parameters(), lr=BASE_LR, betas=(0.9, 0.95),
                                weight_decay=0.1)
        every = max(10, spe // 4)
        hist = []
        t0 = time.time()
        print(f"=== {n_ep:>2} 에폭  ({spe} step/에폭 x {n_ep} = {total} step) ===")

        for step in range(total):
            lr = BASE_LR * (step + 1) / warmup if step < warmup else \
                0.1 * BASE_LR + 0.9 * BASE_LR * 0.5 * (
                    1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
            for g in opt.param_groups:
                g["lr"] = lr
            x, y = loader.get()
            with ctx:
                _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % every == 0 or step == total - 1:
                tl = eval_loss(model, train, ctx, device)
                vl = eval_loss(model, val, ctx, device)
                hist.append((step / spe, tl, vl))
                print(f"  ep {step/spe:5.2f}  train {tl:.4f}  val {vl:.4f}  gap {vl-tl:+.4f}")

        mem_tr = memorization(model, train, device)
        mem_va = memorization(model, np.asarray(val), device)
        tl, vl = hist[-1][1], hist[-1][2]
        best_vl = min(h[2] for h in hist)
        mins = (time.time() - t0) / 60
        rows.append(dict(epochs=n_ep, steps=total, tokens=total * BATCH * SEQ,
                         train_loss=tl, val_loss=vl, best_val=best_vl, gap=vl - tl,
                         memo_train=mem_tr, memo_val=mem_va, minutes=mins))
        curves[n_ep] = hist
        print(f"  -> train {tl:.4f} / val {vl:.4f} (best {best_vl:.4f}) / "
              f"암기율 학습 {mem_tr*100:.1f}% vs 검증 {mem_va*100:.1f}% / {mins:.1f}분\n")
        del model, opt
        torch.cuda.empty_cache()

    with open(f"{OUT}/summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    json.dump({str(k): v for k, v in curves.items()},
              open(f"{OUT}/curves.json", "w"), indent=1)

    print(f"{'에폭':>4}{'train':>9}{'val':>9}{'best val':>10}{'gap':>9}"
          f"{'암기(학습)':>12}{'암기(검증)':>12}")
    print("-" * 65)
    for r in rows:
        print(f"{r['epochs']:>4}{r['train_loss']:>9.4f}{r['val_loss']:>9.4f}"
              f"{r['best_val']:>10.4f}{r['gap']:>+9.4f}"
              f"{r['memo_train']*100:>11.1f}%{r['memo_val']*100:>11.1f}%")

    # 그래프
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for n_ep, h in curves.items():
        e = [p[0] for p in h]
        ax[0].plot(e, [p[1] for p in h], "--", alpha=.5)
        ax[0].plot(e, [p[2] for p in h], "-", label=f"{n_ep} ep")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss")
    ax[0].set_title("solid = val,  dashed = train"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].plot([r["epochs"] for r in rows], [r["val_loss"] for r in rows], "o-", label="final val")
    ax[1].plot([r["epochs"] for r in rows], [r["train_loss"] for r in rows], "o--", label="final train")
    ax[1].set_xscale("log", base=2); ax[1].set_xlabel("number of epochs")
    ax[1].set_title("5M tokens, 7.4M params"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{OUT}/curve.png", dpi=130)
    print(f"\n저장: {OUT}/summary.csv, curves.json, curve.png")


if __name__ == "__main__":
    main()
