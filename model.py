import pandas as pd
import numpy as np
import joblib
import json
import time
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    fbeta_score, confusion_matrix
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
# 5️⃣ 전체 평가 출력
# ------------------------------------------------------------
print("\n📊 [전체 테스트셋 평가]")
print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
print(f"Recall   : {recall_score(y_true, y_pred, zero_division=0):.4f}")
print(f"F0.5     : {fbeta_score(y_true, y_pred, beta=0.5):.4f}")
print(f"F1       : {fbeta_score(y_true, y_pred, beta=1.0):.4f}")
print(f"F2       : {fbeta_score(y_true, y_pred, beta=2.0):.4f}")
print(f"⏱ 예측 소요 시간: {elapsed:.2f}초")

# ------------------------------------------------------------
# 6️⃣ 몰드코드별 성능 분석
# ------------------------------------------------------------
if "mold_code" not in test.columns:
    print("\n⚠️ mold_code 컬럼이 없습니다.")
else:
    results = []
    for mold, group in test.groupby("mold_code"):
        y_t = group[TARGET].values
        X_t = group.drop(columns=[TARGET] + [c for c in DROP_COLS if c in group.columns], errors="ignore")
        y_p = model.predict_proba(X_t)[:, 1]
        y_p = (y_p >= best_threshold).astype(int)

        acc = accuracy_score(y_t, y_p)
        rec = recall_score(y_t, y_p, zero_division=0)
        prec = precision_score(y_t, y_p, zero_division=0)
        f1 = fbeta_score(y_t, y_p, beta=1.0)
        f2 = fbeta_score(y_t, y_p, beta=2.0)
        f05 = fbeta_score(y_t, y_p, beta=0.5)

        results.append({
            "mold_code": mold,
            "n_samples": len(group),
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "F0.5": f05,
            "F1": f1,
            "F2": f2
        })

    df_results = pd.DataFrame(results).sort_values("F1", ascending=False)
    print("\n📦 몰드코드별 성능 요약:")
    print(df_results.to_string(index=False, float_format="%.4f"))
