# -*- coding: utf-8 -*-
"""REPORT.md 와 같은 결과를 읽어 공유용 HTML 한 장을 만든다."""
import glob, json, os, html
import torch, sentencepiece as spm
from model import GPT, Config

PROMPTS = [
    ("표준 시작", "Once upon a time",
     "데이터 분포의 한복판. 셋 다 무난해야 정상이다.",
     "A는 한 문장마다 화제가 튄다 — 드레스 이야기가 갑자기 &ldquo;the ice was very happy&rdquo;로 끝난다. "
     "B는 새 한 마리를 끝까지 끌고 간다. C는 상자 → 장식품 → 주머니 → 빛남으로 사건이 <b>쌓인다</b>."),
    ("등장인물과 인과", "The dragon was very hungry.",
     "배고픔이라는 조건을 끝까지 기억하는가.",
     "A는 &ldquo;The dragon and the dragon became friends&rdquo; — 같은 인물을 둘로 센다. "
     "B는 인과는 맞지만 주어가 새와 곰 사이에서 흔들린다. "
     "C는 배고픔 → 계획 → 납치 → 화해로 이어간 뒤 <b>&ldquo;The dragon was not hungry anymore&rdquo;</b>로 "
     "130토큰 앞의 첫 문장을 회수한다. 이 표에서 가장 큰 차이다."),
    ("문맥 이어받기", "Tom and Lily found a box. Inside the box was",
     "주어진 상황을 이어받는가, 아니면 자기 이야기로 도망가는가.",
     "A는 &ldquo;shiny shiny&rdquo;를 겹쳐 쓰고, 상자를 열었다면서 나무 밑 쪽지를 찾는다. "
     "B는 목걸이로 일관되지만 &ldquo;found a new friend in the box&rdquo;로 미끄러진다. "
     "C는 동전 하나를 놓고 둘이 다투는 장면을 만들고, <b>대사를 인물별로 구분해</b> 주고받는다."),
    ("감정과 이유", "Sara was sad because",
     "이름을 끝까지 유지하는가. 크기와 가장 깨끗하게 비례하는 항목.",
     "A는 Sara가 <b>통째로 사라지고</b> 물고기 이야기가 된다. "
     "B는 Sara로 시작했다가 마지막에 Lucy로 <b>이름이 바뀐다</b>. "
     "C만 Sara와 엄마를 끝까지 유지한다."),
    ("학습 데이터 밖의 문장", "The quarterly revenue forecast indicates that",
     "동화만 본 모델에 사업 문장을 넣으면.",
     "셋 다 아랑곳하지 않고 동화로 끌고 간다 — 본 적 없는 것은 못 한다. "
     "다만 무너지는 방식이 다르다. A는 <b>&ldquo;puisated&rdquo;, &ldquo;bad oxygen&rdquo;</b> 같은 깨진 말을 만들고, "
     "B는 <b>&ldquo;microwly&rdquo;</b>를 만든다. C는 깨진 단어 없이 폭풍 이야기로 착지한다."),
]
N, TEMP, TOPK, SEED = 130, 0.8, 40, 7
RAMP = {"A": "var(--m-a)", "B": "var(--m-b)", "C": "var(--m-c)"}

sp = spm.SentencePieceProcessor(model_file="tok/sp4000.model")
dev = "cuda" if torch.cuda.is_available() else "cpu"
models = []
for p in sorted(glob.glob("runs/*/ckpt.pt")):
    meta = json.load(open(os.path.join(os.path.dirname(p), "config.json")))
    if "best_val_loss" not in meta:
        continue
    ck = torch.load(p, map_location="cpu", weights_only=False)
    m = GPT(Config(**ck["cfg"])).to(dev).eval()
    m.load_state_dict(ck["model"])
    models.append((meta, m))
models.sort(key=lambda x: x[0]["params"])

EOT = ('<span class="eot" title="모델이 스스로 낸 이야기 종료 토큰">'
       '&lt;|endoftext|&gt;</span>')


def prose(t):
    """<|endoftext|> 를 눈에 보이는 표식으로 바꾼다 — 모델이 배운 '이야기의 끝'."""
    return EOT.join(html.escape(x.strip()) for x in t.split("<|endoftext|>"))


cards, sections = [], []
for meta, _ in models:
    toks = meta["total_steps"] * meta["tokens_per_step"]
    cards.append(
        '<article class="card" style="--mc:{c}">'
        '<div class="card-top"><span class="badge">{p}</span>'
        '<span class="params">{n:,}<i>파라미터</i></span></div>'
        '<dl><div><dt>구조</dt><dd>d={d} · {L}층 · head {h}</dd></div>'
        '<div><dt>학습 토큰</dt><dd>{t:.0f}M</dd></div>'
        '<div><dt>학습 시간</dt><dd>{m:.1f}분</dd></div></dl>'
        '<p class="loss"><span>{v:.4f}</span><i>val loss</i></p></article>'.format(
            c=RAMP[meta["preset"]], p=meta["preset"], n=meta["params"],
            d=meta["d_model"], L=meta["n_layer"], h=meta["n_head"],
            t=toks / 1e6, m=meta["minutes"], v=meta["best_val_loss"]))

for title, prompt, why, note in PROMPTS:
    outs = []
    for meta, m in models:
        torch.manual_seed(SEED)
        ids = torch.tensor([sp.encode(prompt)], dtype=torch.long, device=dev)
        with torch.no_grad():
            g = m.generate(ids, N, temperature=TEMP, top_k=TOPK)
        body = sp.decode(g[0].tolist())[len(prompt):].strip()
        outs.append(
            '<div class="out" style="--mc:{c}"><div class="out-rail">'
            '<span class="badge sm">{p}</span>'
            '<span class="out-meta">{n:.1f}M<br>{v:.3f}</span></div>'
            '<p class="story">{s}</p></div>'.format(
                c=RAMP[meta["preset"]], p=meta["preset"], n=meta["params"] / 1e6,
                v=meta["best_val_loss"], s=prose(body)))
    sections.append(
        '<section class="probe"><header class="probe-head"><h2>{t}</h2>'
        '<p class="why">{w}</p></header>'
        '<p class="prompt"><span>입력</span>{p}</p>'
        '<p class="note"><span>여기서 볼 것</span>{n}</p>{o}</section>'.format(
            t=title, w=why, p=html.escape(prompt), n=note, o="".join(outs)))

CSS = """
:root{
  --bg:#F4F6F6; --surface:#FFFFFF; --ink:#12191B; --muted:#5D6E72; --faint:#8A9A9E;
  --line:#DBE3E4; --line-soft:#E9EEEE; --accent:#0E6B6B;
  --m-a:#8FB6B6; --m-b:#3B8B8B; --m-c:#0C4C4C;
  --sans:"IBM Plex Sans",system-ui,-apple-system,"Malgun Gothic",sans-serif;
  --serif:"Newsreader",Georgia,"Batang",serif;
  --mono:"IBM Plex Mono",ui-monospace,Consolas,monospace;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#0C1214; --surface:#131B1D; --ink:#E6EDED; --muted:#93A5A8; --faint:#6E8185;
  --line:#243033; --line-soft:#1B2426; --accent:#4EC5C5;
  --m-a:#2E5555; --m-b:#3F9D9D; --m-c:#7FD9D9;
}}
:root[data-theme="dark"]{
  --bg:#0C1214; --surface:#131B1D; --ink:#E6EDED; --muted:#93A5A8; --faint:#6E8185;
  --line:#243033; --line-soft:#1B2426; --accent:#4EC5C5;
  --m-a:#2E5555; --m-b:#3F9D9D; --m-c:#7FD9D9;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);
  font-size:16px;line-height:1.65;-webkit-font-smoothing:antialiased}
.wrap{max-width:940px;margin:0 auto;padding:0 20px;padding-block:56px 72px}
h1{font-family:var(--serif);font-weight:600;font-size:clamp(2rem,5vw,2.9rem);
  line-height:1.15;margin:.2em 0 .3em;text-wrap:balance;letter-spacing:-.01em}
h2{font-family:var(--serif);font-weight:600;font-size:1.5rem;margin:0;text-wrap:balance}
.eyebrow{font-family:var(--mono);font-size:.74rem;letter-spacing:.16em;
  text-transform:uppercase;color:var(--accent);margin:0}
.lede{color:var(--muted);font-size:1.05rem;max-width:62ch;margin:0 0 1.6em}
.cond{font-family:var(--mono);font-size:.8rem;color:var(--faint);
  border-top:1px solid var(--line);padding-top:14px;margin:0}
.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:34px 0 12px}
@media(max-width:660px){.cards{grid-template-columns:1fr}}
.card{background:var(--surface);border:1px solid var(--line);border-top:3px solid var(--mc);
  border-radius:3px;padding:18px 18px 14px;display:flex;flex-direction:column;gap:14px}
.card-top{display:flex;align-items:baseline;justify-content:space-between;gap:10px}
.badge{font-family:var(--mono);font-weight:500;font-size:.95rem;color:#fff;background:var(--mc);
  width:1.9em;height:1.9em;display:grid;place-items:center;border-radius:2px;flex:none}
:root[data-theme="dark"] .badge{color:#0C1214}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]) .badge{color:#0C1214}}
.badge.sm{font-size:.8rem}
.params{font-family:var(--mono);font-size:1.05rem;font-variant-numeric:tabular-nums;text-align:right}
.params i{display:block;font-style:normal;font-family:var(--sans);font-size:.72rem;color:var(--faint)}
dl{margin:0;display:flex;flex-direction:column;gap:6px;font-size:.86rem}
dl div{display:flex;justify-content:space-between;gap:10px;
  border-bottom:1px dotted var(--line-soft);padding-bottom:5px}
dt{color:var(--faint)} dd{margin:0;font-family:var(--mono);font-variant-numeric:tabular-nums}
.loss{margin:auto 0 0;display:flex;align-items:baseline;gap:8px}
.loss span{font-family:var(--mono);font-size:1.75rem;font-weight:500;color:var(--mc);
  font-variant-numeric:tabular-nums;letter-spacing:-.02em}
.loss i{font-style:normal;font-size:.72rem;color:var(--faint);letter-spacing:.06em}
.readnote{font-size:.9rem;color:var(--muted);max-width:64ch;margin:0 0 46px}
.probe{border-top:1px solid var(--line);padding-top:26px;margin-bottom:44px}
.probe-head{display:flex;flex-wrap:wrap;align-items:baseline;gap:6px 14px;margin-bottom:16px}
.why{margin:0;font-size:.88rem;color:var(--faint)}
.prompt,.note{margin:0 0 14px;padding:11px 14px;border-radius:3px;font-size:.92rem}
.prompt{font-family:var(--mono);background:var(--surface);border:1px solid var(--line);
  border-left:3px solid var(--accent);word-break:break-word}
.note{background:transparent;border:1px dashed var(--line);color:var(--muted);line-height:1.6}
.prompt span,.note span{display:block;font-family:var(--mono);font-size:.68rem;letter-spacing:.14em;
  text-transform:uppercase;color:var(--faint);margin-bottom:5px}
.note b{color:var(--ink);font-weight:600}
.out{display:grid;grid-template-columns:64px 1fr;gap:16px;padding:16px 0;
  border-top:1px solid var(--line-soft)}
.out-rail{display:flex;flex-direction:column;gap:7px;align-items:flex-start}
.out-meta{font-family:var(--mono);font-size:.7rem;color:var(--faint);line-height:1.5;
  font-variant-numeric:tabular-nums}
@media(max-width:560px){.out{grid-template-columns:1fr;gap:8px}
  .out-rail{flex-direction:row;align-items:center;gap:10px}
  .out-meta{text-align:left}}
.story{font-family:var(--serif);font-size:1.06rem;line-height:1.72;margin:0;
  max-width:64ch;color:var(--ink)}
.eot{font-family:var(--mono);font-size:.7em;color:var(--accent);background:var(--line-soft);
  border:1px solid var(--line);border-radius:2px;padding:1px 5px;margin:0 4px;white-space:nowrap}
footer{border-top:1px solid var(--line);padding-top:22px;color:var(--muted);font-size:.88rem}
footer pre{font-family:var(--mono);font-size:.82rem;background:var(--surface);
  border:1px solid var(--line);border-radius:3px;padding:13px 15px;overflow-x:auto;color:var(--ink)}
"""

PAGE = """<title>세 크기의 이야기꾼</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{css}</style>
<div class="wrap">
<p class="eyebrow">Phase A · B · C</p>
<h1>세 크기의 이야기꾼</h1>
<p class="lede">같은 동화 데이터, 같은 토크나이저, 같은 구조. 크기만 21배 차이 나는 세 모델에
같은 문장 다섯 개를 넣었다. RTX 3060 한 장으로 하룻밤에 전부 학습했다.</p>
<p class="cond">temperature 0.8 · top_k 40 · 130토큰 · seed 7 고정 — 난수를 맞춰 모델 차이만 남겼다</p>
<div class="cards">{cards}</div>
<p class="readnote">val loss는 낮을수록 좋다. 다만 그 숫자가 0.9 내려간 것이 글에서 무엇으로
나타나는지가 이 페이지의 내용이다.</p>
{sections}
<footer>
<p>직접 다시 뽑아보려면 &mdash; <code>C:\\ai\\tiny_llm</code></p>
<pre>python gen.py --list
python gen.py -r C "The dragon was very hungry."
python compare.py "Once upon a time" -r A B C</pre>
</footer>
</div>
"""

with open("artifact.html", "w", encoding="utf-8") as f:
    f.write(PAGE.format(css=CSS, cards="".join(cards), sections="".join(sections)))
print("artifact.html 작성 완료")
