"""
검수가 끝난 감성라벨링_검수.xlsx 를 수업용 KoELECTRA 코드(KOELECTRA_Finetune.py)가 읽는 형식으로 변환

입력 : 감성라벨링_검수.xlsx  ([라벨링] 시트의 '최종 라벨', '확인' 열 사용)
출력 : map_reviews_train.txt  (탭 구분, 열: id, document, label — 수업 데이터 ratings_train2.txt 와 같은 형식)
       같은 파일을 ../../koelectra 폴더(수업 코드 폴더)에도 복사

라벨 번호 (수업 코드와 같은 뜻을 유지하도록 0=부정, 1=긍정)
    0 = 부정, 1 = 긍정, 2 = 중립, 3 = 관계없음

실행: python make_koelectra_data.py
"""
import os
import sys

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
XLSX = os.path.join(BASE_DIR, "감성라벨링_검수.xlsx")
OUT = os.path.join(BASE_DIR, "map_reviews_train.txt")
LABEL_TO_ID = {"부정": 0, "긍정": 1, "중립": 2, "관계없음": 3}

df = pd.read_excel(XLSX, sheet_name="라벨링")

# 1. 검수 진행 확인
checked = df["확인"].astype(str).str.strip().str.upper() == "O"
print(f"전체 {len(df):,}건 중 확인 완료 {checked.sum():,}건")
if not checked.all():
    print(f"아직 확인하지 않은 행이 {(~checked).sum():,}건 있습니다. 확인 완료(O)한 행만 변환합니다.")
df = df[checked].copy()
if len(df) < 2000:
    print(f"주의: 확인 완료한 데이터가 {len(df):,}건으로, 과제 기준(2,000건 이상)보다 적습니다.")
if df.empty:
    sys.exit("변환할 데이터가 없습니다.")

# 2. 라벨 확인
bad = ~df["최종 라벨"].isin(LABEL_TO_ID)
if bad.any():
    print("잘못된 라벨이 있는 행 번호:", df.loc[bad, "번호"].tolist())
    sys.exit("최종 라벨은 긍정, 부정, 중립, 관계없음 중 하나여야 합니다.")

# 3. 수업 코드 형식으로 저장 (탭·줄바꿈이 있으면 파일이 깨지므로 공백으로 바꿈)
out = pd.DataFrame({
    "id": df["번호"].astype(int),
    "document": df["리뷰 내용"].astype(str).str.replace(r"[\t\r\n]+", " ", regex=True).str.strip(),
    "label": df["최종 라벨"].map(LABEL_TO_ID).astype(int),
})
out.to_csv(OUT, sep="\t", index=False, encoding="utf-8")

print(f"\n저장 완료: {OUT} ({len(out):,}건)")
for name, i in LABEL_TO_ID.items():
    n = (out["label"] == i).sum()
    print(f"  {i} = {name:<4}: {n:>5,}건 ({n / len(out):.1%})")
changed = (df["AI 라벨"] != df["최종 라벨"]).sum()
print(f"\nAI 1차 라벨에서 직접 고친 행: {changed:,}건 ({changed / len(df):.1%})")
# 4. 수업 코드 폴더(PycharmProjects/koelectra)에도 같은 파일을 복사
KOELECTRA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "koelectra"))
if os.path.isdir(KOELECTRA_DIR):
    out.to_csv(os.path.join(KOELECTRA_DIR, "map_reviews_train.txt"), sep="\t", index=False, encoding="utf-8")
    print(f"\nkoelectra 폴더에도 복사했습니다: {KOELECTRA_DIR}")
    print("다음 단계: koelectra 폴더에서 python KOELECTRA_Finetune_map.py 를 실행하세요.")
else:
    print("\nkoelectra 폴더를 찾지 못했습니다. map_reviews_train.txt 를 KOELECTRA_Finetune_map.py 와 같은 폴더로 복사하세요.")
