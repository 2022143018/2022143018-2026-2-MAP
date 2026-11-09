"""
라벨 검수: 규칙 기반 라벨(labeling.py 결과)과 리뷰를 직접 읽고 붙인 정답(수작업) 라벨 비교

dataset/label_dev_gold.csv  : 개발용 300건 (앱별 150건). 1차 규칙의 오분류를 보고 키워드 사전을 보완하는 데 사용
dataset/label_test_gold.csv : 평가용 200건 (앱별 100건, 개발용과 겹치지 않음). 규칙 보완이 끝난 뒤 처음 열어 최종 정확도 측정
"""
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

labeled = pd.read_csv(os.path.join(DATASET_DIR, "labeled_reviews.csv"))[["review_id", "sentiment", "topic"]]


def evaluate(name, file, has_v1=False, has_sentiment=False):
    gold = pd.read_csv(os.path.join(DATASET_DIR, file))
    df = gold.merge(labeled, on="review_id")
    print("=" * 60)
    print(f"{name}: {len(df)}건")
    print("=" * 60)
    if has_v1:
        print(f"주제 정확도 (1차 규칙)  : {(df['rule_v1'] == df['gold_topic']).mean():.1%}")
    print(f"주제 정확도 (최종 규칙)  : {(df['topic'] == df['gold_topic']).mean():.1%}")
    for app, g in df.groupby("app_name"):
        print(f"  - {app}: {(g['topic'] == g['gold_topic']).mean():.1%}")
    if has_sentiment:
        print(f"감성 정확도 (평점 기반)  : {(df['sentiment'] == df['gold_sentiment']).mean():.1%}")
        polar = df[df["gold_sentiment"] != "중립"]
        print(f"  - 중립 제외 (긍정/부정만): {(polar['sentiment'] == polar['gold_sentiment']).mean():.1%}")
        print("\n[감성 혼동행렬] 행=수작업 정답, 열=평점 라벨")
        print(pd.crosstab(df["gold_sentiment"], df["sentiment"]))
    print("\n[주제별 정답 수 / 맞힌 수]")
    df["correct"] = df["topic"] == df["gold_topic"]
    rep = df.groupby("gold_topic")["correct"].agg(["count", "sum"])
    rep.columns = ["정답 수", "맞힌 수"]
    rep["재현율"] = (rep["맞힌 수"] / rep["정답 수"]).map("{:.0%}".format)
    print(rep)
    print()
    return df


evaluate("개발용 검수 샘플", "label_dev_gold.csv", has_v1=True)
test = evaluate("평가용 검수 샘플", "label_test_gold.csv", has_sentiment=True)

wrong = test[test["topic"] != test["gold_topic"]]
wrong[["app_name", "review_text", "gold_topic", "topic"]].to_csv(
    os.path.join(DATASET_DIR, "label_test_errors.csv"), index=False, encoding="utf-8-sig")
print("평가용 오분류 사례 저장: dataset/label_test_errors.csv")

# ------------------------------------------------------------------
# 라벨러 간 일치도: 평가용 200건을 두 라벨러가 서로의 결과를 모르는 상태에서 각각 라벨링
# ------------------------------------------------------------------
from sklearn.metrics import cohen_kappa_score

agree = pd.read_csv(os.path.join(DATASET_DIR, "label_agreement.csv"))
print("\n" + "=" * 60)
print(f"라벨러 간 일치도 ({len(agree)}건)")
print("=" * 60)
print(f"일치율        : {(agree['labeler_A'] == agree['labeler_B']).mean():.1%}")
print(f"Cohen's kappa : {cohen_kappa_score(agree['labeler_A'], agree['labeler_B']):.3f}")
