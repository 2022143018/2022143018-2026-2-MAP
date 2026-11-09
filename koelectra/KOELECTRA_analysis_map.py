"""
학습된 KoELECTRA 모델(koelectra-base-map) 평가 및 지도 앱 리뷰 전체 분석
(수업 코드 KOELECTRA_Finetune_map.py 는 수정하지 않고, 저장된 모델을 불러와서 사용)

1. 라벨별 성능: 수업 코드와 같은 방법(random_state=2026, 20%)으로 나눈 검증 데이터로
   라벨별 정밀도·재현율·F1, 혼동행렬 계산
2. 전체 리뷰 분류: 전처리한 리뷰 36,750건의 감성(긍정/부정/중립/관계없음) 예측
3. 스토리라인 분석: 앱별 감성 비율, 부정 리뷰의 주제, 연도별 부정 비율, 평점과 감성 비교

실행 (koelectra 프로젝트, torch251_gpu 환경):
    python KOELECTRA_analysis_map.py

출력 (과제/koelectra_results 폴더):
    valid_report.md, K1_혼동행렬.png ~ K5_*.png, koelectra_results.md
    과제/dataset/sentiment_predictions.csv (전체 리뷰 예측 결과)
"""
import os
import platform

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from transformers import ElectraTokenizer, ElectraForSequenceClassification, logging
from tqdm import tqdm

logging.set_verbosity_error()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "과제"))
TRAIN_FILE = os.path.join(BASE_DIR, "map_reviews_train.txt")
MODEL_DIR = os.path.join(BASE_DIR, "koelectra-base-map")
TOKENIZER_DIR = os.path.join(BASE_DIR, "koelectra_base_v3-discriminator")
ALL_REVIEWS = os.path.join(PROJECT_DIR, "dataset", "labeled_reviews_final.csv")
ID_MAP = os.path.join(PROJECT_DIR, "sentiment", "ai_labels_2400.csv")
OUT_DIR = os.path.join(PROJECT_DIR, "koelectra_results")
PRED_CSV = os.path.join(PROJECT_DIR, "dataset", "sentiment_predictions.csv")
os.makedirs(OUT_DIR, exist_ok=True)

LABELS = ["부정", "긍정", "중립", "관계없음"]          # 0, 1, 2, 3 (make_koelectra_data.py 와 같은 순서)
COLORS = {"긍정": "#4C9F70", "중립": "#B0B0B0", "부정": "#D9534F", "관계없음": "#8C8C8C"}
MAX_LEN = 256                                       # 수업 코드와 같은 최대 길이
BATCH = 64

if platform.system() == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
else:
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용하는 장치:", device)

tokenizer = ElectraTokenizer.from_pretrained(TOKENIZER_DIR)
model = ElectraForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()


def predict(texts):
    """문장 목록의 라벨 번호와 라벨별 확률 반환 (길이가 비슷한 문장끼리 묶어 빠르게 계산)"""
    texts = [str(t) for t in texts]
    order = np.argsort([len(t) for t in texts])
    probs = np.zeros((len(texts), len(LABELS)), dtype=np.float32)
    for s in tqdm(range(0, len(texts), BATCH), desc="예측", leave=False):
        idx = order[s:s + BATCH]
        enc = tokenizer([texts[i] for i in idx], truncation=True, max_length=MAX_LEN,
                        add_special_tokens=True, padding=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(enc["input_ids"].to(device),
                           attention_mask=enc["attention_mask"].to(device)).logits
        probs[idx] = torch.softmax(logits, dim=-1).cpu().numpy()
    return probs.argmax(axis=1), probs


def save_fig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, name), dpi=200)
    plt.close()


md = ["## KoELECTRA 결과\n"]

# ------------------------------------------------------------------
# 1. 검증 데이터로 라벨별 성능 확인
# ------------------------------------------------------------------
dataset = pd.read_csv(TRAIN_FILE, sep="\t").dropna(axis=0)
text = list(dataset["document"].values)
label = dataset["label"].values
# 수업 코드와 같은 random_state, test_size 로 나누면 같은 검증 데이터가 나온다
_, valid_idx, _, valid_y = train_test_split(np.arange(len(text)), label, test_size=0.2, random_state=2026)
valid_pred, _ = predict([text[i] for i in valid_idx])

acc = accuracy_score(valid_y, valid_pred)
report = classification_report(valid_y, valid_pred, labels=range(len(LABELS)), target_names=LABELS,
                               digits=3, zero_division=0, output_dict=True)
print(f"\n검증 데이터 {len(valid_y)}건 정확도: {acc:.4f}")
print(classification_report(valid_y, valid_pred, labels=range(len(LABELS)), target_names=LABELS,
                            digits=3, zero_division=0))

md.append(f"### 1. 검증 데이터 라벨별 성능 ({len(valid_y)}건, 정확도 {acc:.1%})\n")
md.append("| 라벨 | 정밀도 | 재현율 | F1 | 건수 |")
md.append("|---|---|---|---|---|")
for name in LABELS + ["macro avg", "weighted avg"]:
    r = report[name]
    md.append(f"| {name} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1-score']:.3f} | {int(r['support'])} |")
md.append("")

cm = confusion_matrix(valid_y, valid_pred, labels=range(len(LABELS)))
cm_ratio = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
fig, ax = plt.subplots(figsize=(6, 5))
ax.imshow(cm_ratio, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(LABELS)))
ax.set_xticklabels(LABELS)
ax.set_yticks(range(len(LABELS)))
ax.set_yticklabels(LABELS)
for i in range(len(LABELS)):
    for j in range(len(LABELS)):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm_ratio[i, j] > 0.5 else "black")
ax.set_xlabel("모델 예측")
ax.set_ylabel("검수한 라벨 (정답)")
ax.set_title(f"KoELECTRA 혼동행렬 (검증 {len(valid_y)}건)")
save_fig("K1_혼동행렬.png")

# ------------------------------------------------------------------
# 2. 전체 리뷰 감성 예측
# ------------------------------------------------------------------
reviews = pd.read_csv(ALL_REVIEWS)
pred, probs = predict(reviews["review_text"].values)
reviews["sentiment_pred"] = [LABELS[p] for p in pred]
for k, name in enumerate(LABELS):
    reviews[f"prob_{name}"] = probs[:, k].round(4)

# 직접 검수한 리뷰는 검수 라벨을 함께 기록
id_map = pd.read_csv(ID_MAP)[["no", "review_id"]]
checked = dataset.merge(id_map, left_on="id", right_on="no")[["review_id", "label"]]
checked["sentiment_checked"] = checked["label"].map(dict(enumerate(LABELS)))
reviews = reviews.merge(checked[["review_id", "sentiment_checked"]], on="review_id", how="left")
reviews["sentiment_final"] = reviews["sentiment_checked"].fillna(reviews["sentiment_pred"])
reviews["year"] = pd.to_datetime(reviews["review_date"]).dt.year

cols = ["app_name", "review_id", "review_date", "rating", "review_text", "topic",
        "sentiment_final", "sentiment_checked", "sentiment_pred"] + [f"prob_{n}" for n in LABELS]
reviews[cols].to_csv(PRED_CSV, index=False, encoding="utf-8-sig")
print(f"\n전체 {len(reviews):,}건 예측 저장: {PRED_CSV}")

# ------------------------------------------------------------------
# 3. 스토리라인 분석 (검수 라벨이 있으면 검수 라벨, 없으면 모델 예측 사용)
# ------------------------------------------------------------------
S = "sentiment_final"
order4 = ["긍정", "중립", "부정", "관계없음"]

# 3-1 앱별 감성 비율
app_sent = (pd.crosstab(reviews["app_name"], reviews[S], normalize="index") * 100).reindex(columns=order4).fillna(0)
md.append("### 2. 앱별 감성 비율 (전체 리뷰)\n")
md.append("| 앱 | " + " | ".join(order4) + " | 리뷰 수 |")
md.append("|---|" + "---|" * (len(order4) + 1))
counts = reviews["app_name"].value_counts()
for app, row in app_sent.iterrows():
    md.append(f"| {app} | " + " | ".join(f"{v:.1f}%" for v in row) + f" | {counts[app]:,} |")
md.append("")
ax = app_sent.plot(kind="barh", stacked=True, figsize=(9, 3.5), color=[COLORS[c] for c in order4])
ax.invert_yaxis()
ax.set_title("앱별 감성 비율 (KoELECTRA)")
ax.set_xlabel("비율 (%)")
ax.set_ylabel("앱", rotation=0, labelpad=15)
ax.legend(title="감성", bbox_to_anchor=(1.02, 1), loc="upper left")
save_fig("K2_앱별_감성_비율.png")

# 3-2 부정 리뷰의 주제
neg = reviews[reviews[S] == "부정"]
topics = ["길안내", "대중교통", "장소정보", "업데이트UI", "오류성능", "위치GPS", "광고정책", "기업이슈", "일반평가"]
neg_topic = (pd.crosstab(neg["topic"], neg["app_name"], normalize="columns") * 100).reindex(topics).fillna(0)
md.append("### 3. 부정 리뷰의 주제 비율\n")
md.append("| 주제 | " + " | ".join(neg_topic.columns) + " |")
md.append("|---|" + "---|" * len(neg_topic.columns))
for t, row in neg_topic.iterrows():
    md.append(f"| {t} | " + " | ".join(f"{v:.1f}%" for v in row) + " |")
md.append("")
ax = neg_topic.drop("일반평가").plot(kind="barh", figsize=(9, 6))
ax.invert_yaxis()
ax.set_title("부정 리뷰의 주제 비율 (KoELECTRA 부정 + 주제 라벨, 일반평가 제외)")
ax.set_xlabel("앱별 부정 리뷰 중 비율 (%)")
ax.set_ylabel("주제")
ax.legend(title="앱")
save_fig("K3_부정리뷰_주제_비율.png")

# 3-3 주제별 부정 비율 (그 주제를 말한 리뷰 중 부정이 몇 %인가)
topic_neg = reviews.assign(neg=reviews[S] == "부정").pivot_table(
    index="topic", columns="app_name", values="neg", aggfunc="mean").reindex(topics) * 100
md.append("### 4. 주제별 부정 리뷰 비율 (그 주제 리뷰 중 부정의 비율)\n")
md.append("| 주제 | " + " | ".join(topic_neg.columns) + " |")
md.append("|---|" + "---|" * len(topic_neg.columns))
for t, row in topic_neg.iterrows():
    md.append(f"| {t} | " + " | ".join(f"{v:.1f}%" for v in row) + " |")
md.append("")
ax = topic_neg.plot(kind="barh", figsize=(9, 6))
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_title("주제별 부정 리뷰 비율 (KoELECTRA)")
ax.set_xlabel("그 주제 리뷰 중 부정 비율 (%)")
ax.set_ylabel("주제")
ax.legend(title="앱")
save_fig("K4_주제별_부정_비율.png")

# 3-4 연도별 부정 비율
yearly = reviews.assign(neg=reviews[S] == "부정").pivot_table(
    index="year", columns="app_name", values="neg", aggfunc="mean") * 100
md.append("### 5. 연도별 부정 리뷰 비율\n")
md.append("| 연도 | " + " | ".join(yearly.columns) + " |")
md.append("|---|" + "---|" * len(yearly.columns))
for y, row in yearly.iterrows():
    md.append(f"| {y} | " + " | ".join(f"{v:.1f}%" if pd.notna(v) else "-" for v in row) + " |")
md.append("")
ax = yearly.plot(marker="o", figsize=(9, 4.5))
ax.set_ylim(0, 100)
ax.set_title("연도별 부정 리뷰 비율 (KoELECTRA)")
ax.set_xlabel("연도")
ax.set_ylabel("부정 비율 (%)")
ax.legend(title="앱")
save_fig("K5_연도별_부정_비율.png")

# 3-5 평점과 감성 비교: 평점만 보면 놓치는 리뷰
mix = pd.crosstab(reviews["rating"], reviews[S]).reindex(columns=order4).fillna(0).astype(int)
md.append("### 6. 평점별 감성 분포 (건수)\n")
md.append("| 평점 | " + " | ".join(order4) + " |")
md.append("|---|" + "---|" * len(order4))
for r, row in mix.iterrows():
    md.append(f"| {r}점 | " + " | ".join(f"{v:,}" for v in row) + " |")
high_neg = ((reviews["rating"] >= 4) & (reviews[S] == "부정")).sum()
low_pos = ((reviews["rating"] <= 2) & (reviews[S] == "긍정")).sum()
md.append(f"\n- 4~5점인데 내용은 부정: {high_neg:,}건 / 1~2점인데 내용은 긍정: {low_pos:,}건\n")

with open(os.path.join(OUT_DIR, "koelectra_results.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md))
print("\n".join(md))
print(f"\n결과 저장 폴더: {OUT_DIR}")
