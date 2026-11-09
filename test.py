import pandas as pd
import matplotlib.pyplot as plt
import os

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

INPUT_FILE = "map_app_reviews_2024_2025_50000.csv"
OUTPUT_DIR = "eda_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("데이터 분석 시작")
print("=" * 60)

print("데이터 크기:", df.shape)
print("\n앱별 리뷰 수:")
print(df["app_name"].value_counts())

print("\n결측치:")
print(df.isnull().sum())

df["review_date"] = pd.to_datetime(
    df["review_date"],
    errors="coerce"
)

df["review_month"] = df["review_date"].dt.to_period("M").astype(str)

df["review_length"] = (
    df["review_text"]
    .fillna("")
    .astype(str)
    .str.len()
)

df["thumbs_up_count"] = pd.to_numeric(
    df["thumbs_up_count"],
    errors="coerce"
).fillna(0)

plt.figure(figsize=(8, 5))
df["app_name"].value_counts().plot(kind="bar")
plt.title("앱별 리뷰 개수")
plt.xlabel("앱")
plt.ylabel("리뷰 개수")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_앱별_리뷰_개수.png", dpi=300)
plt.show()

plt.figure(figsize=(8, 5))
df["rating"].value_counts().sort_index().plot(kind="bar")
plt.title("전체 평점 분포")
plt.xlabel("평점")
plt.ylabel("리뷰 개수")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_전체_평점_분포.png", dpi=300)
plt.show()

rating_app = pd.crosstab(
    df["rating"],
    df["app_name"]
)

rating_app.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("앱별 평점 분포")
plt.xlabel("평점")
plt.ylabel("리뷰 개수")
plt.xticks(rotation=0)
plt.legend(title="앱")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_앱별_평점_분포.png", dpi=300)
plt.show()

mean_rating = df.groupby("app_name")["rating"].mean()

plt.figure(figsize=(8, 5))
mean_rating.plot(kind="bar")
plt.title("앱별 평균 평점")
plt.xlabel("앱")
plt.ylabel("평균 평점")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_앱별_평균_평점.png", dpi=300)
plt.show()

monthly_reviews = df.groupby("review_month").size()

plt.figure(figsize=(14, 6))
monthly_reviews.plot()
plt.title("월별 리뷰 수 추이")
plt.xlabel("월")
plt.ylabel("리뷰 개수")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_월별_리뷰_추이.png", dpi=300)
plt.show()

monthly_app = pd.crosstab(
    df["review_month"],
    df["app_name"]
)

monthly_app.plot(
    figsize=(14, 6)
)

plt.title("앱별 월별 리뷰 수 추이")
plt.xlabel("월")
plt.ylabel("리뷰 개수")
plt.xticks(rotation=45)
plt.legend(title="앱")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_앱별_월별_리뷰_추이.png", dpi=300)
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(
    df["review_length"],
    bins=50
)
plt.title("리뷰 글자 수 분포")
plt.xlabel("리뷰 글자 수")
plt.ylabel("리뷰 개수")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_리뷰_글자수_분포.png", dpi=300)
plt.show()

app_length = df.groupby("app_name")["review_length"].mean()

plt.figure(figsize=(8, 5))
app_length.plot(kind="bar")
plt.title("앱별 평균 리뷰 글자 수")
plt.xlabel("앱")
plt.ylabel("평균 글자 수")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/08_앱별_평균_리뷰_글자수.png", dpi=300)
plt.show()

rating_length = df.groupby("rating")["review_length"].mean()

plt.figure(figsize=(8, 5))
rating_length.plot(kind="bar")
plt.title("평점별 평균 리뷰 글자 수")
plt.xlabel("평점")
plt.ylabel("평균 글자 수")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/09_평점별_평균_리뷰_글자수.png", dpi=300)
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(
    df["thumbs_up_count"],
    bins=50
)
plt.title("리뷰 좋아요 수 분포")
plt.xlabel("좋아요 수")
plt.ylabel("리뷰 개수")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/10_좋아요수_분포.png", dpi=300)
plt.show()

app_likes = df.groupby("app_name")["thumbs_up_count"].mean()

plt.figure(figsize=(8, 5))
app_likes.plot(kind="bar")
plt.title("앱별 평균 리뷰 좋아요 수")
plt.xlabel("앱")
plt.ylabel("평균 좋아요 수")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/11_앱별_평균_좋아요수.png", dpi=300)
plt.show()

rating_ratio = (
    df["rating"]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)

plt.figure(figsize=(8, 5))
rating_ratio.plot(kind="bar")
plt.title("평점별 리뷰 비율")
plt.xlabel("평점")
plt.ylabel("비율 (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/12_평점별_리뷰_비율.png", dpi=300)
plt.show()

rating_ratio_app = pd.crosstab(
    df["app_name"],
    df["rating"],
    normalize="index"
) * 100

ax = rating_ratio_app.plot(
    kind="barh",
    stacked=True,
    figsize=(10, 4)
)
ax.invert_yaxis()

plt.title("앱별 평점 비율")
plt.xlabel("비율 (%)")
plt.ylabel("앱", rotation=0, labelpad=15)
plt.legend(
    title="평점",
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/13_앱별_평점_비율.png", dpi=300)
plt.show()

summary = pd.DataFrame({
    "리뷰수": df.groupby("app_name").size(),
    "평균평점": df.groupby("app_name")["rating"].mean(),
    "평균리뷰길이": df.groupby("app_name")["review_length"].mean(),
    "평균좋아요수": df.groupby("app_name")["thumbs_up_count"].mean()
})

summary.to_csv(
    f"{OUTPUT_DIR}/앱별_통계요약.csv",
    encoding="utf-8-sig"
)

print("\n" + "=" * 60)
print("EDA 완료")
print("=" * 60)

print(summary)

print("\n그래프 저장 위치:")
print(os.path.abspath(OUTPUT_DIR))