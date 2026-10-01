# -*- coding: utf-8 -*-
"""Phase B-2 — 어텐션을 직접 들여다본다.

1) 자연스러운 문장에서 각 head 가 어디를 보는지 히트맵으로 그린다
2) induction head 를 정량적으로 찾는다
   induction head = "앞에 나온 패턴이 반복되면, 그때 뒤따라온 토큰을 보러 가는" head.
   모델이 문맥을 복사해 쓰는 능력의 핵심 부품으로 알려져 있다.

  python inspect_attn.py              # 세 모델 점수 비교 + A 히트맵
  python inspect_attn.py --grid C     # 히트맵을 C 모델로
"""
import argparse, glob, json, os
import numpy as np
import torch
import sentencepiece as spm
from model import GPT, Config
from gen import find

OUT = "runs_inspect"
SENT = "Lily had a red ball. Tom wanted the ball, so Lily gave it to Tom."


def load(tag):
    path = find(tag)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    meta = json.load(open(os.path.join(os.path.dirname(path), "config.json")))
    m = GPT(Config(**ck["cfg"]))
    m.load_state_dict(ck["model"])
    return m.eval(), meta


@torch.no_grad()
def attn_of(model, ids, device):
    x = torch.tensor([ids], device=device)
    _, _, attns = model(x, return_attn=True)
    return [a[0].float().cpu().numpy() for a in attns]      # [L] x (H, T, T)


@torch.no_grad()
def scores(model, device, vocab, n=24, trials=8, seed=0):
    """무작위 토큰 n개를 두 번 반복한 수열을 넣고 head 별 점수를 낸다.

      induction : 두 번째 구간의 위치 n+i 에서 -> 위치 i+1 을 보는 정도
                  (i+1 은 '지난번에 r_i 다음에 왔던 토큰'이 있는 자리)
      prev      : 바로 앞 토큰을 보는 정도 (induction 의 선행 부품으로 알려져 있음)
    """
    rng = np.random.default_rng(seed)
    ind = prev = None
    for _ in range(trials):
        r = rng.integers(10, vocab, size=n)
        ids = np.concatenate([r, r]).tolist()
        A = attn_of(model, ids, device)                      # [L] x (H, 2n, 2n)
        L, H = len(A), A[0].shape[0]
        i_s = np.zeros((L, H))
        p_s = np.zeros((L, H))
        for l in range(L):
            for h in range(H):
                M = A[l][h]
                i_s[l, h] = np.mean([M[n + i, i + 1] for i in range(n - 1)])
                p_s[l, h] = np.mean([M[t, t - 1] for t in range(1, 2 * n)])
        ind = i_s if ind is None else ind + i_s
        prev = p_s if prev is None else prev + p_s
    return ind / trials, prev / trials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="A", help="히트맵을 그릴 모델 (A/B/C)")
    ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sp = spm.SentencePieceProcessor(model_file="tok/sp4000.model")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- 1. 세 모델의 induction head 점수 ----
    print("무작위 토큰 24개를 두 번 반복한 수열로 head 별 점수를 잰다.")
    print("(우연히 한 곳을 보게 될 확률 = 1/위치수 이므로 그보다 크게 나와야 의미가 있다)\n")
    table = []
    for tag in ("A", "B", "C"):
        try:
            model, meta = load(tag)
        except SystemExit:
            continue
        model.to(device)
        ind, prev = scores(model, device, sp.get_piece_size())
        L, H = ind.shape
        # 인과 마스크상 행 n+i 는 (n+i+1)칸만 볼 수 있으므로 그 평균을 우연 기준선으로 쓴다
        chance = float(np.mean([1.0 / (24 + i + 1) for i in range(23)]))
        flat = [(ind[l, h], prev[l, h], l, h) for l in range(L) for h in range(H)]
        flat.sort(reverse=True)
        table.append((tag, meta, ind, prev, flat, chance))
        print(f"[{tag}] {meta['params']:,} params, {L}층 x {H}head = {L*H}개 head")
        print(f"  {'순위':<4}{'위치':<12}{'induction':>11}{'우연대비':>9}{'prev-token':>12}")
        for r, (i_v, p_v, l, h) in enumerate(flat[:a.top], 1):
            print(f"  {r:<4}{'층%d head%d' % (l, h):<12}{i_v:>11.4f}"
                  f"{i_v/chance:>8.1f}x{p_v:>12.4f}")
        print(f"  head 평균 induction {ind.mean():.4f} / 최대 {ind.max():.4f}\n")
        del model
        torch.cuda.empty_cache()

    # ---- 2. 모델별 최고 induction head 비교 그림 ----
    fig, ax = plt.subplots(1, len(table), figsize=(4.2 * len(table), 3.6))
    if len(table) == 1:
        ax = [ax]
    vmax = max(t[2].max() for t in table)          # 세 패널을 같은 스케일로 (비교 가능하게)
    for k, (tag, meta, ind, prev, flat, chance) in enumerate(table):
        im = ax[k].imshow(ind, cmap="magma", aspect="auto", vmin=0, vmax=vmax)
        ax[k].set_title(f"{tag}  ({meta['params']/1e6:.1f}M)\nmax {ind.max():.3f} "
                        f"= {ind.max()/chance:.0f}x chance", fontsize=9)
        ax[k].set_xlabel("head"); ax[k].set_ylabel("layer")
        fig.colorbar(im, ax=ax[k], fraction=.046)
    fig.suptitle("induction score per head", fontsize=11)
    plt.tight_layout(); plt.savefig(f"{OUT}/induction_scores.png", dpi=130); plt.close()

    # ---- 3. 자연 문장 히트맵 ----
    model, meta = load(a.grid)
    model.to(device)
    ids = sp.encode(SENT)
    toks = [p.replace("▁", " ").strip() or "_" for p in sp.encode(SENT, out_type=str)]
    A = attn_of(model, ids, device)
    L, H, T = len(A), A[0].shape[0], len(ids)
    print(f"자연 문장 히트맵: [{a.grid}] {L}층 x {H}head, 토큰 {T}개")
    print(f"  {SENT}")
    fig, axes = plt.subplots(L, H, figsize=(2.05 * H, 2.05 * L))
    axes = np.atleast_2d(axes)
    for l in range(L):
        for h in range(H):
            axx = axes[l, h]
            axx.imshow(A[l][h], cmap="magma", vmin=0, vmax=1)
            axx.set_title(f"L{l} H{h}", fontsize=7, pad=2)
            axx.set_xticks(range(T)); axx.set_yticks(range(T))
            axx.set_xticklabels(toks, fontsize=4.5, rotation=90)
            axx.set_yticklabels(toks, fontsize=4.5)
            axx.tick_params(length=0, pad=1)
    fig.suptitle(f"[{a.grid}]  row = 보는 위치,  column = 보이는 위치  (밝을수록 강하게 봄)",
                 fontsize=11)
    plt.tight_layout(); plt.savefig(f"{OUT}/attn_{a.grid}.png", dpi=150); plt.close()

    # ---- 4. 그 모델의 최고 induction head 하나를 크게 ----
    tag, meta, ind, prev, flat, chance = [t for t in table if t[0] == a.grid][0]
    _, _, bl, bh = flat[0]
    rng = np.random.default_rng(0)
    r = rng.integers(10, sp.get_piece_size(), size=24)
    ids2 = np.concatenate([r, r]).tolist()
    M = attn_of(model, ids2, device)[bl][bh]
    fig, axx = plt.subplots(figsize=(6.4, 5.6))
    im = axx.imshow(M, cmap="magma")
    axx.axvline(23.5, color="#4EC5C5", lw=.8); axx.axhline(23.5, color="#4EC5C5", lw=.8)
    axx.set_title(f"[{a.grid}] layer {bl} head {bh} — repeated random tokens\n"
                  f"induction {ind[bl,bh]:.3f} ({ind[bl,bh]/chance:.0f}x chance)", fontsize=10)
    axx.set_xlabel("position being looked at"); axx.set_ylabel("position looking")
    fig.colorbar(im, ax=axx, fraction=.046)
    plt.tight_layout(); plt.savefig(f"{OUT}/induction_{a.grid}_L{bl}H{bh}.png", dpi=140)
    print(f"\n저장: {OUT}/induction_scores.png, attn_{a.grid}.png, "
          f"induction_{a.grid}_L{bl}H{bh}.png")


if __name__ == "__main__":
    main()
