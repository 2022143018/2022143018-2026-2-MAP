"""
리뷰를 하나씩 읽고 수작업으로 라벨링한 2,000건으로 주제 분류 모델을 학습하고, 전체 36,750건에 주제 라벨을 다시 부여

입력 : dataset/labeled_reviews.csv      (labeling.py 결과, 키워드 규칙 라벨 포함)
       dataset/label_dev_gold.csv       (수작업 라벨 300건)
       dataset/label_test_gold.csv      (수작업 라벨 200건)
       dataset/label_extra_gold.csv     (수작업 라벨 1,500건)
출력 : dataset/gold_2000.csv            (수작업 라벨 2,000건 통합본)
       dataset/labeled_reviews_final.csv(최종 학습 데이터셋: 감성 + 최종 주제 라벨)
       dataset/model_results.csv        (모델별 교차검증 결과)
       labeling_results/M1_모델별_정확도.png, M2_혼동행렬.png
"""
import os
import platform

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer

from topic_keywords import TOPIC_KEYWORDS, topic_scores, label_topic
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
FIG_DIR = os.path.join(BASE_DIR, "labeling_results")
RANDOM_STATE = 42

if platform.system() == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
else:
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

TOPICS = ["대중교통", "길안내", "위치GPS", "장소정보", "오류성능",
          "업데이트UI", "광고정책", "기업이슈", "일반평가"]

# ------------------------------------------------------------------
# 1. 수작업 라벨 2,000건 통합
# ------------------------------------------------------------------
labeled = pd.read_csv(os.path.join(DATASET_DIR, "labeled_reviews.csv"))
gold_parts = []
for file, name in [("label_dev_gold.csv", "개발용"),
                   ("label_test_gold.csv", "평가용"),
                   ("label_extra_gold.csv", "추가")]:
    g = pd.read_csv(os.path.join(DATASET_DIR, file))[["review_id", "gold_topic"]]
    g["gold_set"] = name
    gold_parts.append(g)
gold = pd.concat(gold_parts, ignore_index=True)
assert gold["review_id"].is_unique
gold = gold.merge(labeled[["review_id", "app_name", "rating", "review_text", "topic"]],
                  on="review_id").rename(columns={"topic": "rule_topic"})
gold.to_csv(os.path.join(DATASET_DIR, "gold_2000.csv"), index=False, encoding="utf-8-sig")

print("=" * 60)
print(f"수작업 라벨 데이터: {len(gold):,}건")
print("=" * 60)
print(gold["gold_topic"].value_counts().reindex(TOPICS).to_string())

X = gold["review_text"].values
y = gold["gold_topic"].values

# ------------------------------------------------------------------
# 2. 모델 정의
# ------------------------------------------------------------------
# 한국어는 조사·어미가 붙어 단어 형태가 다양하므로, 형태소 분석기 없이도 잘 동작하는
# 문자 n-gram(2~4글자)과 띄어쓰기 단위 단어 n-gram을 함께 사용한다.
def features():
    return make_union(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True),
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True),
    )


# 키워드 규칙의 지식을 모델에 넣기 위한 피처: 주제별 키워드 점수(9개) + 규칙이 고른 대표 주제(one-hot 9개)
KW_TOPICS = list(TOPIC_KEYWORDS.keys())


def keyword_features(texts):
    rows = []
    for t in texts:
        sc = topic_scores(t)
        primary = label_topic(t)[0]
        rows.append([np.log1p(sc[k]) for k in KW_TOPICS] +
                    [1.0 if primary == k else 0.0 for k in TOPICS])
    return np.array(rows)


def hybrid_features():
    return make_union(features(), FunctionTransformer(keyword_features))


MODELS = {
    "나이브 베이즈 (NB)": make_pipeline(features(), ComplementNB(alpha=0.3)),
    "로지스틱 회귀 (LR)": make_pipeline(features(), LogisticRegression(C=10, max_iter=3000, class_weight="balanced")),
    "선형 SVM (SVM)": make_pipeline(features(), LinearSVC(C=0.5, class_weight="balanced")),
    "키워드+로지스틱 회귀 (하이브리드 LR)": make_pipeline(hybrid_features(), LogisticRegression(C=10, max_iter=3000, class_weight="balanced")),
    "키워드+선형 SVM (하이브리드 SVM)": make_pipeline(hybrid_features(), LinearSVC(C=0.5, class_weight="balanced")),
}

# ------------------------------------------------------------------
# 3. 5-fold 교차검증 비교 (키워드 규칙 포함)
# ------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
rows = [{"모델": "키워드 규칙 (기준)",
         "정확도": accuracy_score(y, gold["rule_topic"]),
         "Macro F1": f1_score(y, gold["rule_topic"], average="macro")}]
preds = {"키워드 규칙 (기준)": gold["rule_topic"].values}
for name, model in MODELS.items():
    p = cross_val_predict(model, X, y, cv=cv)
    preds[name] = p
    rows.append({"모델": name, "정확도": accuracy_score(y, p), "Macro F1": f1_score(y, p, average="macro")})

result = pd.DataFrame(rows)
# 키워드 규칙은 개발용 300건을 보고 다듬었으므로, 그 300건을 뺀 1,700건 기준 성능도 함께 기록
fresh = gold["gold_set"] != "개발용"
result["정확도 (개발용 제외 1,700건)"] = [accuracy_score(y[fresh], preds[m][fresh]) for m in result["모델"]]
result.to_csv(os.path.join(DATASET_DIR, "model_results.csv"), index=False, encoding="utf-8-sig")
print("\n[5-fold 교차검증 결과]")
print(result.round(3).to_string(index=False))

best_name = result.iloc[1:].sort_values("Macro F1", ascending=False).iloc[0]["모델"]
print("\n최종 선정 모델:", best_name)

best_pred = preds[best_name]
print("\n[주제별 F1 비교: 키워드 규칙 vs 최종 모델]")
per_topic = pd.DataFrame({
    "건수": pd.Series(y).value_counts().reindex(TOPICS),
    "키워드 규칙": f1_score(y, gold["rule_topic"], labels=TOPICS, average=None),
    best_name: f1_score(y, best_pred, labels=TOPICS, average=None),
}, index=TOPICS)
print(per_topic.round(3))

# ------------------------------------------------------------------
# 4. 그래프
# ------------------------------------------------------------------
short = result["모델"].str.extract(r"\((.*)\)")[0].replace({"기준": "키워드 규칙"})
ax = result.assign(모델=short).set_index("모델")[["정확도", "Macro F1"]].plot(kind="bar", figsize=(9, 5))
ax.set_title("주제 분류 모델별 성능 (수작업 라벨 2,000건, 5-fold 교차검증)")
ax.set_ylabel("점수")
ax.set_xlabel("")
ax.set_ylim(0, 1)
for c in ax.containers:
    ax.bar_label(c, fmt="%.2f", fontsize=9)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "M1_모델별_정확도.png"), dpi=200)
plt.close()

cm = confusion_matrix(y, best_pred, labels=TOPICS)
cm_ratio = cm / cm.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(8, 7))
ax.imshow(cm_ratio, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(TOPICS)), TOPICS, rotation=45, ha="right")
ax.set_yticks(range(len(TOPICS)), TOPICS)
for i in range(len(TOPICS)):
    for j in range(len(TOPICS)):
        if cm[i, j]:
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm_ratio[i, j] > 0.5 else "black", fontsize=9)
ax.set_xlabel("모델 예측")
ax.set_ylabel("수작업 라벨 (정답)")
ax.set_title(f"혼동행렬: {best_name}")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "M2_혼동행렬.png"), dpi=200)
plt.close()

# ------------------------------------------------------------------
# 5. 최종 모델로 전체 데이터 라벨링
# ------------------------------------------------------------------
final_model = MODELS[best_name].fit(X, y)
final = labeled.copy()
final["topic_rule"] = final["topic"]
final["topic"] = final_model.predict(final["review_text"].values)
final["topic_source"] = "모델"
human = gold.set_index("review_id")["gold_topic"]
is_human = final["review_id"].isin(human.index)
final.loc[is_human, "topic"] = final.loc[is_human, "review_id"].map(human)
final.loc[is_human, "topic_source"] = "수작업"

cols = ["app_name", "review_id", "review_date", "rating", "review_text", "review_length",
        "thumbs_up_count", "app_version", "sentiment", "topic", "topic_source", "topic_rule", "topic_all"]
final[cols].to_csv(os.path.join(DATASET_DIR, "labeled_reviews_final.csv"), index=False, encoding="utf-8-sig")

print("\n최종 학습 데이터셋:", f"{len(final):,}건",
      f"(수작업 라벨 {is_human.sum():,}건 + 모델 라벨 {(~is_human).sum():,}건)")
print("키워드 규칙과 최종 라벨 일치율:", f"{(final['topic'] == final['topic_rule']).mean():.1%}")

# ------------------------------------------------------------------
# 6. 최종 라벨 분포와 가설 확인용 그래프
# ------------------------------------------------------------------
final = final[cols]
topic_tab = pd.crosstab(final["topic"], final["app_name"]).reindex(TOPICS)
neg = final[final["sentiment"] == "부정"]
neg_ratio = (pd.crosstab(neg["topic"], neg["app_name"], normalize="columns") * 100).reindex(TOPICS)
topic_rating = final.pivot_table(index="topic", columns="app_name", values="rating", aggfunc="mean").reindex(TOPICS)

print("\n[최종 주제 라벨 분포]")
print(topic_tab)
print("\n[부정 리뷰의 주제 비율(%)]")
print(neg_ratio.round(1))
print("\n[주제별 평균 평점]")
print(topic_rating.round(2))

for data, title, xlabel, fname, xlim in [
    (topic_tab, "앱별 주제 개수 (최종 라벨)", "리뷰 개수", "F1_앱별_주제_개수.png", None),
    (neg_ratio.drop("일반평가"), "부정 리뷰(1~2점)의 주제 비율 (최종 라벨, 일반평가 제외)", "앱별 부정 리뷰 중 비율 (%)", "F2_부정리뷰_주제_비율.png", None),
    (topic_rating, "주제별 평균 평점 (최종 라벨)", "평균 평점", "F3_주제별_평균_평점.png", (1, 5)),
]:
    ax = data.plot(kind="barh", figsize=(9, 6))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("주제")
    if xlim:
        ax.set_xlim(*xlim)
    ax.invert_yaxis()
    ax.legend(title="앱")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, fname), dpi=200)
    plt.close()

print("\n저장 완료:", DATASET_DIR, FIG_DIR)
