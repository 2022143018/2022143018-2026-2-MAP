import os
import time
import pandas as pd
from google_play_scraper import reviews, Sort

PER_APP_TARGET = 25000
BATCH_SIZE = 200
DELAY = 1.0

START_DATE = pd.Timestamp("2017-01-01")
END_DATE = pd.Timestamp("2026-01-01")

OUTPUT_FILE = "map_app_reviews_2024_2025_50000.csv"

APPS = {
    "카카오맵": {
        "url": "https://play.google.com/store/apps/details?id=net.daum.android.map",
        "app_id": "net.daum.android.map"
    },
    "네이버지도": {
        "url": "https://play.google.com/store/apps/details?id=com.nhn.android.nmap",
        "app_id": "com.nhn.android.nmap"
    }
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(BASE_DIR, OUTPUT_FILE)

print("=" * 70)
print("지도 앱 리뷰 수집 시작")
print("=" * 70)
print("수집 기간:", START_DATE.strftime("%Y-%m-%d"), "~", END_DATE.strftime("%Y-%m-%d"))
print("앱별 목표:", f"{PER_APP_TARGET:,}개")
print("전체 목표:", f"{PER_APP_TARGET * len(APPS):,}개")
print("저장 파일:", OUTPUT_PATH)
print()

existing_reviews = []
existing_ids = set()

if os.path.exists(OUTPUT_PATH):
    try:
        df_existing = pd.read_csv(OUTPUT_PATH)

        if "review_id" in df_existing.columns:
            existing_ids = set(
                df_existing["review_id"].astype(str)
            )

        existing_reviews = df_existing.to_dict("records")

        print("기존 저장 리뷰:", f"{len(existing_reviews):,}개")

    except Exception as e:
        print("기존 파일을 읽는 중 오류가 발생했습니다:", e)
        print("새로 수집합니다.")
        existing_reviews = []
        existing_ids = set()

else:
    print("기존 파일이 없습니다.")
    print("새로 수집합니다.")

print()

for app_name, app_info in APPS.items():

    app_id = app_info["app_id"]
    app_url = app_info["url"]

    current_count = sum(
        1
        for review in existing_reviews
        if review.get("app_name") == app_name
        and pd.notna(review.get("review_date"))
    )

    print("=" * 70)
    print("앱:", app_name)
    print("현재 저장:", f"{current_count:,}개")
    print("목표:", f"{PER_APP_TARGET:,}개")
    print("=" * 70)

    if current_count >= PER_APP_TARGET:
        print("이미 목표 수량에 도달했습니다.")
        print()
        continue

    continuation_token = None
    empty_result_count = 0
    total_requested = 0

    while current_count < PER_APP_TARGET:

        remaining = PER_APP_TARGET - current_count
        request_count = min(BATCH_SIZE, remaining)

        try:

            result, continuation_token = reviews(
                app_id,
                lang="ko",
                country="kr",
                sort=Sort.NEWEST,
                count=request_count,
                continuation_token=continuation_token
            )

            total_requested += len(result)

        except Exception as e:

            print("수집 오류:", e)
            print("5초 후 다시 시도합니다.")
            time.sleep(5)
            continue

        if not result:
            empty_result_count += 1

            print(
                f"{app_name}: Google Play에서 리뷰가 반환되지 않았습니다. "
                f"연속 {empty_result_count}회"
            )

            if empty_result_count >= 3:
                print("리뷰를 3회 연속 가져오지 못해 해당 앱 수집을 종료합니다.")
                break

            time.sleep(5)
            continue

        empty_result_count = 0

        new_count = 0
        duplicate_count = 0
        outside_count = 0
        invalid_count = 0

        for review in result:

            review_id = str(review.get("reviewId", ""))

            if not review_id:
                invalid_count += 1
                continue

            if review_id in existing_ids:
                duplicate_count += 1
                continue

            review_date = review.get("at")

            if review_date is None:
                invalid_count += 1
                continue

            review_date = pd.to_datetime(
                review_date,
                errors="coerce"
            )

            if pd.isna(review_date):
                invalid_count += 1
                continue

            if review_date < START_DATE or review_date >= END_DATE:
                outside_count += 1
                continue

            existing_reviews.append({
                "app_name": app_name,
                "app_url": app_url,
                "review_id": review_id,
                "rating": review.get("score"),
                "review_date": review_date.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "review_text": review.get("content", ""),
                "thumbs_up_count": review.get(
                    "thumbsUpCount",
                    0
                ),
                "app_version": review.get(
                    "appVersion",
                    ""
                )
            })

            existing_ids.add(review_id)

            current_count += 1
            new_count += 1

            if current_count >= PER_APP_TARGET:
                break

        print(
            f"{app_name}: "
            f"{current_count:,} / {PER_APP_TARGET:,} "
            f"| 저장: {new_count:,} "
            f"| 기간 밖: {outside_count:,} "
            f"| 중복: {duplicate_count:,} "
            f"| 오류: {invalid_count:,}"
        )

        if current_count >= PER_APP_TARGET:
            print(f"{app_name} 목표 25,000개에 도달했습니다.")
            break

        if continuation_token is None:
            print("다음 리뷰 페이지 토큰이 없어 수집을 종료합니다.")
            break

        time.sleep(DELAY)

    temp_df = pd.DataFrame(existing_reviews)

    if not temp_df.empty:
        temp_df.to_csv(
            OUTPUT_PATH,
            index=False,
            encoding="utf-8-sig"
        )

    print()
    print(
        app_name,
        "수집 종료:",
        f"{current_count:,}개"
    )
    print("현재까지 저장:", OUTPUT_PATH)
    print()

df = pd.DataFrame(existing_reviews)

if not df.empty:

    df["review_date"] = pd.to_datetime(
        df["review_date"],
        errors="coerce"
    )

    df = df[
        (df["review_date"] >= START_DATE) &
        (df["review_date"] < END_DATE)
    ].copy()

    df = df.drop_duplicates(
        subset=["review_id"]
    )

    df = df.sort_values(
        ["app_name", "review_date"],
        ascending=[True, False]
    )

    app_counts = df["app_name"].value_counts()

    print("=" * 70)
    print("최종 수집 결과")
    print("=" * 70)

    for app_name in APPS.keys():

        count = app_counts.get(
            app_name,
            0
        )

        print(
            f"{app_name}: {count:,}개"
        )

    print("-" * 70)

    print(
        f"전체 리뷰: {len(df):,}개"
    )

    if not df.empty:
        print(
            "실제 데이터 기간:",
            df["review_date"].min(),
            "~",
            df["review_date"].max()
        )

    print(
        "저장 파일:",
        OUTPUT_PATH
    )

    print("=" * 70)

    df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

else:

    print("수집된 리뷰가 없습니다.")