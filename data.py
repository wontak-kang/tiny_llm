"""토큰 .bin 을 memmap 으로 읽어 학습 배치를 만든다.

RAM 16GB 에 1.1GB 짜리 배열을 통째로 올리지 않기 위해 memmap 을 쓴다.
운영체제가 필요한 부분만 디스크에서 읽어온다.
"""
import numpy as np
import torch


class Batcher:
    def __init__(self, path, seq_len, batch_size, device, seed=1234):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = device
        self.rng = np.random.default_rng(seed)
        if len(self.data) < seq_len + 1:
            raise ValueError(f"{path} 가 너무 짧다: {len(self.data)} 토큰")

    def __len__(self):
        return len(self.data)

    def get(self):
        # 무작위 위치에서 seq_len+1 개를 떠서, 입력 x 와 정답 y 를 한 칸 밀어 만든다.
        # y 가 x 보다 한 칸 뒤라는 것이 '다음 토큰 예측'의 전부다.
        ix = self.rng.integers(0, len(self.data) - self.seq_len - 1, size=self.batch_size)
        x = np.stack([self.data[i:i + self.seq_len] for i in ix]).astype(np.int64)
        y = np.stack([self.data[i + 1:i + 1 + self.seq_len] for i in ix]).astype(np.int64)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        if self.device == "cuda":
            # pin_memory + non_blocking 으로 CPU->GPU 전송을 연산과 겹친다
            x = x.pin_memory().to("cuda", non_blocking=True)
            y = y.pin_memory().to("cuda", non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y
