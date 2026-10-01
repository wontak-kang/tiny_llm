"""텍스트를 토큰 ID 배열(uint16)로 변환해 .bin 으로 저장한다.
모든 Phase 가 이 파일을 memmap 으로 읽는다 (RAM 16GB 라 통째로 못 올림)."""
import sys, os, time
import numpy as np
import sentencepiece as spm
from multiprocessing import Pool

MODEL = "tok/sp4000.model"
SEP = "<|endoftext|>"
_sp = None

def init():
    global _sp
    _sp = spm.SentencePieceProcessor(model_file=MODEL)

def enc(stories):
    eos = _sp.piece_to_id(SEP)
    out = []
    for ids in _sp.encode(stories):
        out.extend(ids)
        out.append(eos)          # 이야기 끝 표시 = 모델이 '멈춤'을 배우는 유일한 신호
    return np.array(out, dtype=np.uint16)

def pack(src, dst, workers=6, chunk=20000):
    sp = spm.SentencePieceProcessor(model_file=MODEL)
    assert sp.get_piece_size() < 65536, "uint16 범위 초과"
    t0 = time.time()
    total = 0
    with open(dst, "wb") as out, open(src, encoding="utf-8", errors="ignore") as f:
        buf, batch, pool = "", [], Pool(workers, initializer=init)
        def flush(batch):
            nonlocal total
            for arr in pool.imap(enc, [batch[i:i+chunk//10] for i in range(0, len(batch), chunk//10)]):
                arr.tofile(out); total += len(arr)
        while True:
            data = f.read(1 << 24)
            if not data:
                break
            buf += data
            parts = buf.split(SEP)
            buf = parts.pop()
            batch.extend(p.strip() for p in parts if p.strip())
            if len(batch) >= chunk:
                flush(batch); batch = []
                print(f"  {total/1e6:7.1f}M 토큰  {time.time()-t0:5.0f}초", flush=True)
        if buf.strip():
            batch.append(buf.strip())
        if batch:
            flush(batch)
        pool.close(); pool.join()
    print(f"{dst}: {total:,} 토큰 ({total/1e6:.1f}M), {time.time()-t0:.0f}초, "
          f"{os.path.getsize(dst)/1e6:.0f}MB")
    return total

if __name__ == "__main__":
    pack("data/TinyStoriesV2-GPT4-valid.txt", "data/val.bin")
    pack("data/TinyStoriesV2-GPT4-train.txt", "data/train.bin")
