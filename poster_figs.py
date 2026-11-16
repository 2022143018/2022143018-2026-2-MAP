"""A0 포스터용 그래프 (글씨를 크게 다시 그림) → poster_figs/
필요 패키지: pandas, matplotlib, networkx, kiwipiepy
"""
import itertools, os
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from matplotlib import font_manager

for name in ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"]:
    if any(name == f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams.update({"axes.unicode_minus": False, "font.size": 22, "axes.titlesize": 24,
                     "axes.labelsize": 22, "xtick.labelsize": 20, "ytick.labelsize": 20, "legend.fontsize": 20})
OUT = "poster_figs"; os.makedirs(OUT, exist_ok=True)
NAVER, KAKAO = "#03C75A", "#F2B705"
W = 9.6  # 포스터 한 칸 너비(인치) → 실제 인쇄 크기와 1:1

def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)

# 1) 앱별 별점 비율 (README 3.2절 값)
star = {"네이버지도": [35.6, 7.2, 10.6, 11.7, 34.9], "카카오맵": [19.4, 4.2, 6.6, 11.4, 58.5]}
scol = ["#C0392B", "#E67E22", "#BDC3C7", "#7FB3D5", "#1F5FBF"]
fig, ax = plt.subplots(figsize=(W, 4.2))
for i, (app, v) in enumerate(star.items()):
    left = 0
    for k, w in enumerate(v):
        ax.barh(i, w, left=left, color=scol[k], label=f"{k+1}점" if i == 0 else None)
        if w >= 9:
            ax.text(left + w / 2, i, f"{w:.0f}%", ha="center", va="center", color="white", fontweight="bold")
        left += w
ax.set_yticks([0, 1], list(star)); ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_xlabel("비율 (%)")
ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.32), frameon=False, handlelength=1.2, columnspacing=1)
clean(ax); plt.tight_layout(); plt.savefig(f"{OUT}/P1_앱별_별점비율.png", dpi=200); plt.close()

# 2) 학습 데이터 앱별 감성 (4.4절 값)
LAB = ["부정", "긍정", "중립", "관계없음"]; LC = ["#D64545", "#2E7D32", "#8C8C8C", "#5B7DB1"]
tr = {"네이버지도": [64.7, 17.8, 14.7, 2.8], "카카오맵": [42.7, 43.5, 10.7, 3.1]}
fig, ax = plt.subplots(figsize=(W, 4.0))
for i, (app, v) in enumerate(tr.items()):
    left = 0
    for k, w in enumerate(v):
        ax.barh(i, w, left=left, color=LC[k], label=LAB[k] if i == 0 else None)
        if w >= 9:
            ax.text(left + w / 2, i, f"{w:.1f}%", ha="center", va="center", color="white", fontweight="bold")
        left += w
ax.set_yticks([0, 1], list(tr)); ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_xlabel("비율 (%)")
ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.34), frameon=False)
clean(ax); plt.tight_layout(); plt.savefig(f"{OUT}/P2_학습데이터_감성비율.png", dpi=200); plt.close()

# 3) epoch별 손실·정확도 (5.2절 값)
ep = [1, 2, 3, 4]; loss = [0.6962, 0.3744, 0.1998, 0.0990]; tra = [85.9, 93.5, 97.2, 99.1]; va = [80.6, 86.5, 86.3, 88.1]
fig, ax = plt.subplots(1, 2, figsize=(W, 4.6))
ax[0].plot(ep, loss, "o-", color="#333", lw=3, ms=9); ax[0].set_title("손실 (loss)")
for x, y in zip(ep, loss):
    ax[0].annotate(f"{y:.2f}", (x, y), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=19)
ax[0].set_ylim(0, 0.85); ax[0].set_xticks(ep); ax[0].set_xlabel("epoch")
ax[1].plot(ep, tra, "o-", color="#1F5FBF", lw=3, ms=9, label="학습")
ax[1].plot(ep, va, "s-", color="#D64545", lw=3, ms=9, label="검증")
for x, a, b in zip(ep, tra, va):
    ax[1].annotate(f"{b:.1f}", (x, b), xytext=(0, -26), textcoords="offset points", ha="center", color="#D64545", fontsize=19)
ax[1].set_title("정확도 (%)"); ax[1].set_ylim(76, 102); ax[1].set_xticks(ep); ax[1].set_xlabel("epoch"); ax[1].legend(loc="lower right")
for a in ax: clean(a); a.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT}/P3_epoch별_손실_정확도.png", dpi=200); plt.close()

# 4) 연도별 부정 비율 (6.3절 값)
yr = list(range(2018, 2026))
nv = [34.7, 53.6, 73.7, 67.8, 64.5, 67.5, 70.1, 74.5]; kk = [None, 26.9, 32.5, 41.0, 45.8, 47.8, 52.1, 61.8]
fig, ax = plt.subplots(figsize=(W, 4.8))
ax.plot(yr, nv, "o-", color=NAVER, lw=4, ms=10, label="네이버지도")
ax.plot(yr[1:], kk[1:], "s-", color=KAKAO, lw=4, ms=10, label="카카오맵")
ax.annotate("74.5%", (2025, 74.5), xytext=(0, 12), textcoords="offset points", ha="center", color="#02913F", fontweight="bold")
ax.annotate("61.8%", (2025, 61.8), xytext=(0, -32), textcoords="offset points", ha="center", color="#B98A00", fontweight="bold")
ax.annotate("26.9%", (2019, 26.9), xytext=(0, -32), textcoords="offset points", ha="center", color="#B98A00", fontweight="bold")
ax.set_ylim(15, 85); ax.set_ylabel("부정 비율 (%)"); ax.set_xticks(yr, [str(y)[2:] + "년" for y in yr])
ax.legend(loc="upper left", frameon=False); ax.grid(alpha=0.3); clean(ax)
plt.tight_layout(); plt.savefig(f"{OUT}/P4_연도별_부정비율.png", dpi=200); plt.close()

# 5) 키워드 네트워크의 가장 강한 연결 (extra_analysis.py와 같은 방법) → 포스터에서 읽기 쉽게 막대로
from kiwipiepy import Kiwi
kiwi = Kiwi()
STOP = set("지도 앱 어플 네이버 카카오 카카오맵 네이버지도 맵 사용 이용 진짜 정말 너무 때 것 수 거 좀 등 점 왜 더 안 못 번 중 제 저 나 내 우리 뭐 이거 그거 이번 요즘 계속 자꾸 항상 기능 문제 생각 경우 정도 부분 개선 해결 확인 부탁 감사 이후".split())
pred = pd.read_csv("dataset/sentiment_predictions.csv", encoding="utf-8-sig")
neg = pred[pred["sentiment_final"] == "부정"]
fig, axes = plt.subplots(1, 2, figsize=(W, 5.4))
for ax, (app, color) in zip(axes, [("네이버지도", NAVER), ("카카오맵", KAKAO)]):
    docs = [sorted({t.form for t in kiwi.tokenize(str(x)) if t.tag in ("NNG", "NNP") and len(t.form) > 1 and t.form not in STOP})
            for x in neg[neg["app_name"] == app]["review_text"]]
    freq = Counter(w for d in docs for w in d)
    top = [w for w, _ in freq.most_common(35)]
    co = Counter()
    for d in docs:
        co.update(itertools.combinations([w for w in d if w in top], 2))
    pairs = co.most_common(8)[::-1]
    ax.barh([f"{a}–{b}" for (a, b), _ in pairs], [w for _, w in pairs], color=color, edgecolor="#333")
    for k, (_, w) in enumerate(pairs):
        ax.text(w, k, f" {w}", va="center", fontsize=18)
    ax.set_title(f"{app}\n(부정 {len(docs):,}개)")
    ax.set_xlim(0, max(w for _, w in pairs) * 1.3); ax.set_xticks([]); clean(ax); ax.spines["bottom"].set_visible(False)
plt.tight_layout(); plt.savefig(f"{OUT}/P5_키워드_강한연결.png", dpi=200); plt.close()
print("ok")
