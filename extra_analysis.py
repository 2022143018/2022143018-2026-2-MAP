"""보고서 보완 분석
1) KoELECTRA epoch별 손실·정확도 그래프 (학습 로그 값)
2) 학습 데이터(라벨링 2,399건) EDA
3) 부정 리뷰 키워드 네트워크 (앱별)

필요 패키지: pandas, matplotlib, networkx, kiwipiepy
실행: python extra_analysis.py   (과제 폴더에서)
"""
import itertools
import os
from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from matplotlib import font_manager

# ---------- 한글 글꼴 ----------
for name in ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"]:
    if any(name == f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False

OUT = "koelectra_results"
os.makedirs(OUT, exist_ok=True)
NAVER, KAKAO = "#03C75A", "#F2B705"   # 앱 구분 색
LABELS = ["부정", "긍정", "중립", "관계없음"]
LCOLOR = {"부정": "#D64545", "긍정": "#2E7D32", "중립": "#8C8C8C", "관계없음": "#5B7DB1"}

# ---------- 1) epoch별 손실·정확도 ----------
# KOELECTRA_Finetune_map.py 실행 로그에서 옮겨 적은 값
epoch = [1, 2, 3, 4]
loss = [0.6962, 0.3744, 0.1998, 0.0990]
train_acc = [85.9, 93.5, 97.2, 99.1]
val_acc = [80.6, 86.5, 86.3, 88.1]

fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
ax[0].plot(epoch, loss, "o-", color="#333333", lw=2.5)
for x, y in zip(epoch, loss):
    ax[0].annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center")
ax[0].set_title("epoch별 학습 손실 (loss)")
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("평균 손실"); ax[0].set_xticks(epoch); ax[0].set_ylim(0, 0.8)
ax[1].plot(epoch, train_acc, "o-", color="#1F5FBF", lw=2.5, label="학습 데이터 정확도")
ax[1].plot(epoch, val_acc, "s-", color="#D64545", lw=2.5, label="검증 데이터 정확도")
for x, a, b in zip(epoch, train_acc, val_acc):
    ax[1].annotate(f"{a:.1f}", (x, a), textcoords="offset points", xytext=(0, 8), ha="center", color="#1F5FBF")
    ax[1].annotate(f"{b:.1f}", (x, b), textcoords="offset points", xytext=(0, -16), ha="center", color="#D64545")
ax[1].set_title("epoch별 정확도 (%)")
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("정확도 (%)"); ax[1].set_xticks(epoch); ax[1].set_ylim(75, 102)
ax[1].legend(loc="lower right")
for a in ax:
    a.grid(alpha=0.3); a.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(f"{OUT}/K6_epoch별_손실_정확도.png", dpi=150); plt.close()

# ---------- 2) 학습 데이터 EDA ----------
train = pd.read_csv("sentiment/map_reviews_train.txt", sep="\t")
ai = pd.read_csv("sentiment/ai_labels_2400.csv", encoding="utf-8-sig")
train = train.merge(ai[["no", "app_name", "rating"]], left_on="id", right_on="no", how="left")
train["감성"] = train["label"].map(dict(enumerate(LABELS)))
train["길이"] = train["document"].str.len()

ct = pd.crosstab(train["app_name"], train["감성"], normalize="index")[LABELS] * 100
cnt = pd.crosstab(train["app_name"], train["감성"])[LABELS]
fig, ax = plt.subplots(figsize=(10, 3.6))
left = [0, 0]
for lab in LABELS:
    v = ct[lab].values
    ax.barh(ct.index, v, left=left, color=LCOLOR[lab], label=lab)
    for i, (l, w) in enumerate(zip(left, v)):
        if w > 4:
            ax.text(l + w / 2, i, f"{w:.1f}%", ha="center", va="center", color="white", fontweight="bold")
    left = [a + b for a, b in zip(left, v)]
ax.set_xlim(0, 100); ax.set_xlabel("비율 (%)")
ax.set_title("학습 데이터(2,399건)의 앱별 감성 라벨 비율")
ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(f"{OUT}/D1_학습데이터_앱별_감성비율.png", dpi=150); plt.close()

lens = train.groupby("감성")["길이"].agg(["mean", "median"]).loc[LABELS]
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.bar(LABELS, lens["mean"], color=[LCOLOR[l] for l in LABELS])
for i, (m, md) in enumerate(zip(lens["mean"], lens["median"])):
    ax.text(i, m + 2, f"평균 {m:.0f}자\n(중간값 {md:.0f}자)", ha="center")
ax.set_ylabel("리뷰 글자 수"); ax.set_ylim(0, lens["mean"].max() * 1.35)
ax.set_title("학습 데이터의 감성 라벨별 리뷰 길이")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(f"{OUT}/D2_학습데이터_감성별_리뷰길이.png", dpi=150); plt.close()

# ---------- 3) 부정 리뷰 키워드 네트워크 ----------
from kiwipiepy import Kiwi

kiwi = Kiwi()
STOP = set("지도 앱 어플 네이버 카카오 카카오맵 네이버지도 맵 사용 이용 진짜 정말 너무 때 것 수 거 좀 등 점 왜 더 안 못 번 중 제 저 나 내 우리 뭐 이거 그거 이번 요즘 계속 자꾸 항상 기능 문제 생각 경우 정도 부분 개선 해결 확인 부탁 감사 이후".split())
pred = pd.read_csv("dataset/sentiment_predictions.csv", encoding="utf-8-sig")
neg = pred[pred["sentiment_final"] == "부정"]


def nouns(text):
    return sorted({t.form for t in kiwi.tokenize(str(text))
                   if t.tag in ("NNG", "NNP") and len(t.form) > 1 and t.form not in STOP})


summary = []
for app, color in [("네이버지도", NAVER), ("카카오맵", KAKAO)]:
    docs = [nouns(t) for t in neg[neg["app_name"] == app]["review_text"]]
    freq = Counter(w for d in docs for w in d)
    top = [w for w, _ in freq.most_common(35)]
    co = Counter()
    for d in docs:
        ws = [w for w in d if w in top]
        co.update(itertools.combinations(sorted(ws), 2))
    edges = co.most_common(60)
    G = nx.Graph()
    for (a, b), w in edges:
        G.add_edge(a, b, weight=w)
    pos = nx.spring_layout(G, k=1.1, seed=7, weight="weight")
    mx = max(w for _, w in edges)
    fig, ax = plt.subplots(figsize=(11, 9))
    nx.draw_networkx_edges(G, pos, ax=ax, width=[0.5 + 6 * G[u][v]["weight"] / mx for u, v in G.edges()],
                           edge_color="#9A9A9A", alpha=0.6)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=[freq[n] / freq[top[0]] * 3200 + 300 for n in G.nodes()],
                           node_color=color, alpha=0.85, edgecolors="#333333")
    nx.draw_networkx_labels(G, pos, ax=ax, font_family=plt.rcParams["font.family"], font_size=13, font_weight="bold")
    ax.set_title(f"{app} 부정 리뷰 키워드 네트워크 (부정 {len(docs):,}건, 자주 함께 나온 단어 쌍 상위 60개)", fontsize=15)
    ax.axis("off")
    plt.tight_layout(); plt.savefig(f"{OUT}/N{1 if app == '네이버지도' else 2}_{app}_부정_키워드네트워크.png", dpi=150); plt.close()
    summary.append((app, len(docs), freq.most_common(15), edges[:10]))

# ---------- 숫자 요약 ----------
with open(f"{OUT}/extra_analysis.md", "w", encoding="utf-8") as f:
    f.write("## 학습 데이터 EDA\n\n")
    f.write(cnt.assign(합계=cnt.sum(axis=1)).to_markdown() + "\n\n")
    f.write(ct.round(1).to_markdown() + "\n\n")
    f.write(lens.round(1).to_markdown() + "\n\n")
    f.write("## 부정 리뷰 키워드\n\n")
    for app, n, fr, ed in summary:
        f.write(f"### {app} (부정 {n:,}건)\n\n상위 단어: " + ", ".join(f"{w}({c:,})" for w, c in fr) + "\n\n")
        f.write("강한 연결: " + ", ".join(f"{a}–{b}({w:,})" for (a, b), w in ed) + "\n\n")
print(open(f"{OUT}/extra_analysis.md", encoding="utf-8").read())
