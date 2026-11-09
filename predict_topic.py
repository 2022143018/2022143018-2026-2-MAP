"""
저장된 주제 분류 모델(models/topic_model.joblib)로 새 리뷰의 주제 예측

사용법:
    python predict_topic.py                          # 예시 리뷰로 예측
    python predict_topic.py "버스 도착 시간이 안 맞아요"   # 직접 입력한 리뷰 예측

주의: 모델 안에서 topic_keywords.py 의 keyword_features 함수를 사용하므로,
      이 파일과 topic_keywords.py 가 같은 폴더에 있어야 한다.
"""
import os
import sys
import json

import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
sys.path.insert(0, BASE_DIR)

try:
    model = joblib.load(os.path.join(MODEL_DIR, "topic_model.joblib"))
except Exception as e:
    # 모델을 저장한 scikit-learn 버전과 설치된 버전이 다르면 불러오지 못할 수 있다.
    print("모델을 불러오지 못했습니다:", e)
    print("python topic_model.py 를 실행하면 내 컴퓨터 환경에서 모델을 다시 만들 수 있습니다.")
    sys.exit(1)
with open(os.path.join(MODEL_DIR, "model_info.json"), encoding="utf-8") as f:
    info = json.load(f)

reviews = sys.argv[1:] or [
    "버스 도착 시간이 하나도 안 맞아요",
    "멀쩡한 길 두고 이상한 길로 안내해요",
    "업데이트 후에 화면이 너무 불편해졌어요",
    "앱이 자꾸 튕겨요",
    "즐겨찾기 개수 좀 늘려주세요",
    "잘 쓰고 있습니다",
]

print(f"모델: {info['model']} (학습 {info['train_size']:,}건, 교차검증 정확도 {info['cv_accuracy']:.1%})")
print("-" * 60)
proba = model.predict_proba(reviews)
classes = model.classes_
for text, p in zip(reviews, proba):
    order = np.argsort(p)[::-1]
    top = classes[order[0]]
    second = f"{classes[order[1]]} {p[order[1]]:.0%}"
    print(f"{top:<7} ({p[order[0]]:.0%}, 2순위 {second}) | {text}")
