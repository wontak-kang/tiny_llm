"""Llama 형 소형 트랜스포머. RMSNorm / RoPE / SwiGLU, bias 없음.

용어:
  d_model  각 토큰을 나타내는 벡터의 길이
  n_head   어텐션을 몇 갈래로 나눠서 볼 것인가
  n_layer  블록을 몇 층 쌓는가
"""
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 4000
    d_model: int = 128
    n_head: int = 4
    n_layer: int = 4
    seq_len: int = 512
    tie_embedding: bool = True   # 입력 임베딩과 출력 행렬을 공유 (작은 모델에서 이득이 큼)


class RMSNorm(nn.Module):
    """LayerNorm 과 달리 평균을 빼지 않는다. 제곱평균으로만 크기를 맞춘다."""
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def rope_tables(seq_len, head_dim, device, base=10000.0):
    """RoPE: 위치를 '회전각'으로 준다. 학습되는 위치 임베딩이 없다."""
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv)                      # (T, head_dim/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    # x: (B, H, T, head_dim)  — 짝수/홀수 차원을 짝지어 2차원 회전을 건다
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[None, None, :, :], sin[None, None, :, :]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).flatten(-2)


class Attention(nn.Module):
    """위치 간에 정보가 섞이는 유일한 장치."""
    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, return_attn=False):
        B, T, C = x.shape
        split = lambda t: t.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        if return_attn:   # Phase B 해부용. 느리지만 어텐션 맵을 그대로 꺼낸다
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            att = att.masked_fill(~mask, float("-inf")).softmax(-1)
            y = att @ v
        else:
            att = None
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o(y), att


class SwiGLU(nn.Module):
    """FFN. 위치별로 따로 처리하며 옆 토큰을 보지 않는다. 파라미터의 약 2/3가 여기 있다."""
    def __init__(self, cfg):
        super().__init__()
        hidden = int(8 / 3 * cfg.d_model / 64 + 0.5) * 64   # 64의 배수로 맞춤
        self.gate = nn.Linear(cfg.d_model, hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n1, self.attn = RMSNorm(cfg.d_model), Attention(cfg)
        self.n2, self.ffn = RMSNorm(cfg.d_model), SwiGLU(cfg)

    def forward(self, x, cos, sin, return_attn=False):
        a, att = self.attn(self.n1(x), cos, sin, return_attn)
        x = x + a                      # 잔차 연결: 원본을 더해 학습이 안정된다
        return x + self.ffn(self.n2(x)), att


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embedding:
            self.head.weight = self.emb.weight
        self.apply(self._init)
        self._cache = {}

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def _rope(self, T, device):
        if self._cache.get("T") != T or self._cache.get("dev") != device:
            cos, sin = rope_tables(T, self.cfg.d_model // self.cfg.n_head, device)
            self._cache = {"T": T, "dev": device, "cos": cos, "sin": sin}
        return self._cache["cos"], self._cache["sin"]

    def forward(self, idx, targets=None, return_attn=False):
        cos, sin = self._rope(idx.shape[1], idx.device)
        x, attns = self.emb(idx), []
        for b in self.blocks:
            x, att = b(x, cos, sin, return_attn)
            if return_attn:
                attns.append(att)
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return (logits, loss, attns) if return_attn else (logits, loss)

    def n_params(self, embedding=True):
        n = sum(p.numel() for p in self.parameters())
        if not embedding:
            n -= self.emb.weight.numel()
        return n

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.cfg.seq_len:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k:
                v, _ = logits.topk(min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            nxt = torch.multinomial(logits.softmax(-1), 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx
