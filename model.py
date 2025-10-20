
import pandas as pd
import numpy as np
import joblib
import json
import time
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    fbeta_score, confusion_matrix, classification_report
)

# ------------------------------------------------------------
# 🔧 basic_fix 함수 재정의 (모델과 동일하게)
# ------------------------------------------------------------
def basic_fix(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "tryshot_signal" in df.columns:
        df["tryshot_signal"] = df["tryshot_signal"].apply(lambda x: 1 if str(x).upper() == "D" else 0)
    if {"speed_ratio", "low_section_speed", "high_section_speed"} <= set(df.columns):
        df.loc[df["speed_ratio"] == float("inf"), "speed_ratio"] = -1
        df.loc[df["speed_ratio"] == -float("inf"), "speed_ratio"] = -1
        df.loc[(df["low_section_speed"] == 0) & (df["high_section_speed"] == 0), "speed_ratio"] = -2
    if "pressure_speed_ratio" in df.columns:
        df.loc[np.isinf(df["pressure_speed_ratio"]), "pressure_speed_ratio"] = -1
    return df

# ------------------------------------------------------------
# 1️⃣ 설정
# ------------------------------------------------------------
MODEL_PATH = "./models/fin_xgb_f20.pkl"
META_PATH = "./models/fin_xgb_meta_f20.json"
TEST_PATH = "./data/fin_test_kf_fixed.csv"
TARGET = "passorfail"
DROP_COLS = ["team", "real_time"]

# ------------------------------------------------------------
# 2️⃣ 모델 & 메타 불러오기
# ------------------------------------------------------------
print("🔍 모델 및 메타 정보 불러오는 중...")
model = joblib.load(MODEL_PATH)
with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

best_threshold = meta.get("best_threshold", 0.5)
print(f"✅ 모델 로드 완료 (threshold={best_threshold:.3f})")

# ------------------------------------------------------------
# 3️⃣ 테스트 데이터 로드
# ------------------------------------------------------------
def read_csv_safely(path):
    for enc in ["utf-8-sig", "cp949", "utf-8"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)

test = read_csv_safely(TEST_PATH)
print(f"✅ 테스트셋 로드 완료: {test.shape}")

# ------------------------------------------------------------
# 4️⃣ 예측 수행
# ------------------------------------------------------------
start = time.time()

X_test = test.drop(columns=[TARGET] + [c for c in DROP_COLS if c in test.columns], errors="ignore")
y_true = test[TARGET].values

y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= best_threshold).astype(int)

elapsed = time.time() - start

# ------------------------------------------------------------
# 5️⃣ 평가 지표 계산
# ------------------------------------------------------------
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f05 = fbeta_score(y_true, y_pred, beta=0.5)
f1 = fbeta_score(y_true, y_pred, beta=1.0)
f2 = fbeta_score(y_true, y_pred, beta=2.0)
cm = confusion_matrix(y_true, y_pred)

# ------------------------------------------------------------
# 6️⃣ 결과 출력
# ------------------------------------------------------------
print("\n📊 [테스트셋 평가 결과]")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F0.5     : {f05:.4f}")
print(f"F1       : {f1:.4f}")
print(f"F2       : {f2:.4f}")
print(f"⏱ 예측 소요 시간: {elapsed:.2f}초")

print("\nConfusion Matrix:")
print(cm)

print("\n세부 리포트:")
print(classification_report(y_true, y_pred, digits=4))
