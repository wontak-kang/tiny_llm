"""vocab 4k / 8k / 16k 를 실제로 학습시켜 압축률과 임베딩 파라미터를 비교한다."""
import os, time, io
import sentencepiece as spm

D_MODEL, N_LAYER = 256, 8
BODY = 12 * D_MODEL**2 * N_LAYER          # 트랜스포머 본체 파라미터(임베딩 제외)
SAMPLE = "data/sample_100mb.txt"
VALID  = "data/TinyStoriesV2-GPT4-valid.txt"
os.makedirs("tok", exist_ok=True)

# 평가용: valid 앞부분 2MB (학습에 안 쓴 데이터)
with io.open(VALID, encoding="utf-8", errors="ignore") as f:
    eval_text = f.read(2_000_000)
eval_chars = len(eval_text)

SENT = "Once upon a time there was a little girl named Lily."
KOR  = "옛날 옛적에 릴리라는 어린 소녀가 있었습니다."

rows = []
for vocab in (4000, 8000, 16000):
    prefix = f"tok/sp{vocab}"
    if not os.path.exists(prefix + ".model"):
        t0 = time.time()
        spm.SentencePieceTrainer.train(
            input=SAMPLE, model_prefix=prefix, vocab_size=vocab,
            model_type="bpe", character_coverage=1.0,
            input_sentence_size=1_000_000, shuffle_input_sentence=True,
            user_defined_symbols=["<|endoftext|>"],
            train_extremely_large_corpus=False,
        )
        print(f"  [{vocab}] 학습 {time.time()-t0:.0f}초")
    sp = spm.SentencePieceProcessor(model_file=prefix + ".model")
    n = len(sp.encode(eval_text))
    rows.append(dict(
        name=f"직접학습 {vocab//1000}k", vocab=vocab, ntok=n,
        cpt=eval_chars / n,
        emb=vocab * D_MODEL,
        sent=sp.encode(SENT), kor=sp.encode(KOR), sp=sp,
    ))

# 비교군: GPT-2 (byte-level BPE, 50257)
import tiktoken
g = tiktoken.get_encoding("gpt2")
n = len(g.encode(eval_text, allowed_special="all"))
rows.append(dict(name="GPT-2 (기성)", vocab=50257, ntok=n, cpt=eval_chars / n,
                 emb=50257 * D_MODEL, sent=g.encode(SENT), kor=g.encode(KOR), sp=None))

TOTAL_CHARS = 2_227_753_162   # train.txt 전체 바이트(≈문자)
print(f"\n평가 텍스트: valid 앞 {eval_chars:,}자 (학습에 미사용)\n")
hdr = f"{'토크나이저':<14}{'vocab':>7}{'문자/토큰':>10}{'전체토큰(추정)':>15}{'임베딩param':>13}{'모델합계':>11}{'임베딩비중':>9}"
print(hdr); print("-" * len(hdr))
for r in rows:
    total = BODY + r["emb"]
    print(f"{r['name']:<14}{r['vocab']:>7,}{r['cpt']:>10.2f}"
          f"{TOTAL_CHARS/r['cpt']/1e6:>13.0f}M{r['emb']/1e6:>12.2f}M"
          f"{total/1e6:>10.2f}M{r['emb']/total*100:>8.0f}%")

print(f"\n(본체 고정: d_model={D_MODEL}, n_layer={N_LAYER} → {BODY/1e6:.2f}M)")

print(f"\n=== 같은 영어 문장 ===\n{SENT!r}")
for r in rows:
    print(f"  {r['name']:<14} {len(r['sent']):>3}토큰")
print(f"\n=== 같은 한국어 문장 ===\n{KOR!r}")
for r in rows:
    ids = r["kor"]
    unk = sum(1 for i in ids if i == r["sp"].unk_id()) if r["sp"] else 0
    note = f"   <- {unk}개가 <unk>. 한글이 vocab에 없어 정보가 소실됨" if unk else ""
    print(f"  {r['name']:<14} {len(ids):>3}토큰{note}")

print("\n=== 8k 가 문장을 실제로 어떻게 쪼개는가 ===")
sp8 = [r for r in rows if r["vocab"] == 8000][0]["sp"]
print(" | ".join(sp8.encode(SENT, out_type=str)))
print("\n=== 4k vs 16k 차이 (드문 단어) ===")
for w in ["Once upon a time", "The shopkeeper smiled", "a reliable otter", "extraordinary"]:
    a = [r for r in rows if r["vocab"] == 4000][0]["sp"]
    b = [r for r in rows if r["vocab"] == 16000][0]["sp"]
    print(f"  {w!r}")
    print(f"     4k : {' | '.join(a.encode(w, out_type=str))}")
    print(f"    16k : {' | '.join(b.encode(w, out_type=str))}")
