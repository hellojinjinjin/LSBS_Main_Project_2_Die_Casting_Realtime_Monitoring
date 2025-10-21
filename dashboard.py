import sys
import pandas as pd
import numpy as np
import joblib, json
from shiny import App, ui, render, reactive
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import asyncio

# ------------------------------------------------------------
# 1️⃣ 설정
# ------------------------------------------------------------
MODEL_PATH = "./models/fin_xgb_f20.pkl"
META_PATH  = "./models/fin_xgb_meta_f20.json"
DATA_PATH  = "./data/fin_test_kf_fixed.csv"

# ------------------------------------------------------------
# 2️⃣ 전처리 함수 정의
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
# 3️⃣ joblib이 함수를 인식할 수 있도록 등록
# ------------------------------------------------------------
import sys
sys.modules['__main__'].basic_fix = basic_fix
globals()['basic_fix'] = basic_fix

# ------------------------------------------------------------
# 4️⃣ 모델 및 메타 로드
# ------------------------------------------------------------
model = joblib.load(MODEL_PATH)
with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)
threshold = meta.get("best_threshold", 0.5)

# ------------------------------------------------------------
# 5️⃣ UI 구성
# ------------------------------------------------------------
app_ui = ui.page_fluid(
    ui.h3("🚗 다이캐스팅 실시간 불량 예측 대시보드", style="text-align:center; font-weight:bold;"),
    ui.hr(),
    ui.row(
        ui.column(6, ui.output_plot("perf_plot")),
        ui.column(6, ui.output_table("last_pred")),
    ),
    ui.hr(),
    ui.div(
        ui.input_action_button("start_btn", "▶ 시작", class_="btn btn-success"),
        ui.input_action_button("pause_btn", "⏸ 일시정지", class_="btn btn-warning ms-2"),
        ui.input_action_button("reset_btn", "🔄 리셋", class_="btn btn-secondary ms-2"),
        style="text-align:center; margin-top:15px;"
    )
)

# ------------------------------------------------------------
# 6️⃣ 서버 로직
# ------------------------------------------------------------
def server(input, output, session):
    global df, n_total   # ✅ 전역 변수 인식

    idx = reactive.Value(0)
    is_running = reactive.Value(False)
    results = reactive.Value([])

    async def stream_data():
        """5초마다 한 행씩 예측"""
        while is_running():
            reactive.invalidate_later(5)
            i = idx()
            if i >= n_total:
                is_running.set(False)
                break

            row = df.iloc[[i]]
            X = row.drop(columns=[TARGET], errors="ignore")
            y_true = int(row[TARGET].values[0])
            y_prob = model.predict_proba(X)[:, 1][0]
            y_pred = int(y_prob >= threshold)

            new_entry = {"index": i, "true": y_true, "pred": y_pred, "prob": y_prob}
            results().append(new_entry)
            idx.set(i + 1)
            await asyncio.sleep(5)

    # --- 버튼 이벤트 ---
    @reactive.effect
    @reactive.event(input.start_btn)
    async def _():
        is_running.set(True)
        await stream_data()

    @reactive.effect
    @reactive.event(input.pause_btn)
    def _():
        is_running.set(False)

    @reactive.effect
    @reactive.event(input.reset_btn)
    def _():
        idx.set(0)
        results.set([])
        is_running.set(False)

    # --- 최근 예측 테이블 ---
    @output
    @render.table
    def last_pred():
        data = results()
        if not data:
            return pd.DataFrame([{"상태": "대기 중..."}])
        df_res = pd.DataFrame(data).tail(5)
        df_res["정답 일치"] = (df_res["true"] == df_res["pred"]).astype(int)
        return df_res

    # --- 성능 플롯 ---
    @output
    @render.plot
    def perf_plot():
        import matplotlib.pyplot as plt

        data = results()
        if len(data) < 5:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "데이터 누적 중...", ha="center", va="center")
            ax.axis("off")
            return fig

        df_res = pd.DataFrame(data)
        acc = accuracy_score(df_res["true"], df_res["pred"])
        f1 = f1_score(df_res["true"], df_res["pred"])
        prec = precision_score(df_res["true"], df_res["pred"])
        rec = recall_score(df_res["true"], df_res["pred"])

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(["Accuracy", "Precision", "Recall", "F1"], [acc, prec, rec, f1], color="skyblue")
        ax.set_ylim(0, 1)
        ax.set_title(f"📈 실시간 모델 성능 (샘플 {len(df_res)})")
        for i, v in enumerate([acc, prec, rec, f1]):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold")
        return fig

# ------------------------------------------------------------
# 7️⃣ 앱 엔트리
# ------------------------------------------------------------
app = App(app_ui, server)
