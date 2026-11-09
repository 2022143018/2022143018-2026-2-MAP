"""
지도 앱 리뷰 전처리 + 라벨링 + 학습 데이터셋 구축

입력 : map_app_reviews_2024_2025_50000.csv  (test1.py로 수집한 원본 50,000건)
출력 : dataset/labeled_reviews.csv          (라벨링된 학습 데이터셋)
       dataset/labeling_summary.csv         (앱별 라벨 분포 요약)
       labeling_results/*.png               (라벨 분포 그래프)

라벨 1. sentiment (감성)  : 평점 기반  4~5점 긍정 / 3점 중립 / 1~2점 부정
라벨 2. topic (주제)      : 키워드 사전 기반, 리뷰가 무엇에 대해 말하는지
"""
import os
import re
import platform

import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 0. 설정
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "map_app_reviews_2024_2025_50000.csv")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
FIG_DIR = os.path.join(BASE_DIR, "labeling_results")
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

MIN_LENGTH = 10          # 10글자 미만 리뷰는 주제를 판단할 정보가 부족하여 제외
MIN_KOREAN_RATIO = 0.5   # 글자 중 한글 비율 50% 이상만 사용



if platform.system() == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
else:
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

# ------------------------------------------------------------------
# 1. 라벨 정의
# ------------------------------------------------------------------
SENTIMENT_ORDER = ["긍정", "중립", "부정"]


def label_sentiment(rating):
    if rating >= 4:
        return "긍정"
    if rating == 3:
        return "중립"
    return "부정"


# 주제 키워드 사전과 규칙은 topic_keywords.py 참고
from topic_keywords import TOPIC_KEYWORDS, label_topic

TOPIC_ORDER = list(TOPIC_KEYWORDS.keys()) + ["일반평가"]


# ------------------------------------------------------------------
# 2. 전처리
# ------------------------------------------------------------------
df = pd.read_csv(INPUT_FILE)
n_raw = len(df)

df["review_text"] = df["review_text"].fillna("").astype(str).str.strip()
df["review_text"] = df["review_text"].str.replace(r"\s+", " ", regex=True)

korean = df["review_text"].str.count(r"[가-힣ㄱ-ㅎㅏ-ㅣ]")
letters = df["review_text"].str.count(r"[^\W\d_]")
korean_ratio = korean / letters.where(letters > 0, 1)

steps = [("원본", n_raw)]

df = df.drop_duplicates(subset=["review_id"])
steps.append(("review_id 중복 제거", len(df)))

mask_korean = (letters.loc[df.index] > 0) & (korean_ratio.loc[df.index] >= MIN_KOREAN_RATIO)
df = df[mask_korean]
steps.append(("한글 위주 리뷰만 (한글 50% 이상)", len(df)))

df = df[df["review_text"].str.len() >= MIN_LENGTH].copy()
steps.append((f"{MIN_LENGTH}글자 이상", len(df)))

print("=" * 60)
print("전처리 단계별 데이터 수")
print("=" * 60)
for name, n in steps:
    print(f"{name:<28} {n:>8,}건")

# ------------------------------------------------------------------
# 3. 라벨링
# ------------------------------------------------------------------
df["sentiment"] = df["rating"].apply(label_sentiment)
topics = df["review_text"].apply(label_topic)
df["topic"] = [t[0] for t in topics]
df["topic_all"] = [t[1] for t in topics]
df["review_length"] = df["review_text"].str.len()

cols = ["app_name", "review_id", "review_date", "rating", "review_text",
        "review_length", "thumbs_up_count", "app_version",
        "sentiment", "topic", "topic_all"]
df = df[cols].sort_values(["app_name", "review_date"], ascending=[True, False])
df.to_csv(os.path.join(DATASET_DIR, "labeled_reviews.csv"),
          index=False, encoding="utf-8-sig")

# 검수용 샘플은 label_check.py 에서 만들고 평가한다.

# ------------------------------------------------------------------
# 4. 라벨 분포 요약
# ------------------------------------------------------------------
sent_tab = pd.crosstab(df["app_name"], df["sentiment"])[SENTIMENT_ORDER]
topic_tab = pd.crosstab(df["topic"], df["app_name"]).reindex(TOPIC_ORDER)
neg = df[df["sentiment"] == "부정"]
neg_topic_ratio = (pd.crosstab(neg["topic"], neg["app_name"], normalize="columns") * 100).reindex(TOPIC_ORDER)

summary = pd.concat({"감성(건)": sent_tab.T, "주제(건)": topic_tab}, axis=0)
summary.to_csv(os.path.join(DATASET_DIR, "labeling_summary.csv"), encoding="utf-8-sig")

print("\n최종 학습 데이터셋:", f"{len(df):,}건")
print(df["app_name"].value_counts().to_string())
print("\n[감성 라벨]")
print(sent_tab)
print("\n[주제 라벨]")
print(topic_tab)
print("\n[부정 리뷰의 주제 비율(%)]")
print(neg_topic_ratio.round(1))

# ------------------------------------------------------------------
# 5. 그래프
# ------------------------------------------------------------------
ax = (pd.crosstab(df["app_name"], df["sentiment"], normalize="index")[SENTIMENT_ORDER] * 100).plot(
    kind="barh", stacked=True, figsize=(9, 3.5), color=["#4C9F70", "#B0B0B0", "#D9534F"])
ax.invert_yaxis()
ax.set_title("앱별 감성 라벨 비율")
ax.set_xlabel("비율 (%)")
ax.set_ylabel("앱", rotation=0, labelpad=15)
plt.legend(title="감성", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "L1_앱별_감성_비율.png"), dpi=200)
plt.close()

ax = topic_tab.plot(kind="barh", figsize=(9, 6))
ax.set_title("앱별 주제 라벨 개수")
ax.set_xlabel("리뷰 개수")
ax.set_ylabel("주제")
ax.invert_yaxis()
ax.legend(title="앱")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "L2_앱별_주제_개수.png"), dpi=200)
plt.close()

ax = neg_topic_ratio.drop("일반평가").plot(kind="barh", figsize=(9, 6))
ax.set_title("부정 리뷰(1~2점)의 주제 비율 (일반평가 제외)")
ax.set_xlabel("앱별 부정 리뷰 중 비율 (%)")
ax.set_ylabel("주제")
ax.invert_yaxis()
ax.legend(title="앱")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "L3_부정리뷰_주제_비율.png"), dpi=200)
plt.close()

topic_rating = df.pivot_table(index="topic", columns="app_name", values="rating", aggfunc="mean").reindex(TOPIC_ORDER)
ax = topic_rating.plot(kind="barh", figsize=(9, 6))
ax.set_title("주제별 평균 평점")
ax.set_xlabel("평균 평점")
ax.set_ylabel("주제")
ax.set_xlim(1, 5)
ax.invert_yaxis()
ax.legend(title="앱")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "L4_주제별_평균_평점.png"), dpi=200)
plt.close()

print("\n저장 완료:", DATASET_DIR, FIG_DIR)
