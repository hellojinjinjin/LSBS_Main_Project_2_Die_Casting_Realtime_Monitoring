import pandas as pd
import joblib
import matplotlib.pyplot as plt
from shiny import App, ui, render, reactive, session
from shiny.reactive import invalidate_later
from shiny.ui import update_slider, update_numeric, update_select, update_navs
import seaborn as sns
import pathlib
import plotly.express as px
from shinywidgets import render_plotly, output_widget
import numpy as np
import matplotlib
from sklearn.metrics import pairwise_distances
import os
from matplotlib import font_manager
import plotly.io as pio
import calendar
import datetime
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy import stats
import json  # ✅ 1. 이 줄을 추가하세요
from sklearn.metrics import recall_score, fbeta_score # ✅ 2. 이 줄도 추가하세요
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()

# ======== 실시간 스트리밍 대시보드 (현장 메뉴) ========
from shared import streaming_df, RealTimeStreamer, KFStreamer
import plotly.express as px
import plotly.graph_objects as go
import datetime
from shiny import ui, render, reactive

# ==========================================
# 🔹 Baseline UCL 계산 함수 (고정형 관리도용)
# ==========================================
from scipy.stats import f
from collections import deque

data_queue = deque()  # 스트리밍 데이터 큐 (2초마다 1행씩 처리)
stream_speed = reactive.Value(2.0)  # 기본 2초 주기

# 🔧 basic_fix 함수 추가 (model.py와 동일하게)
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

def calc_baseline_ucl(train_df, cols):
    """Train 데이터 기반 UCL, mean, inv_cov 계산"""
    X = train_df[cols].dropna().values
    n, p = X.shape
    mean = np.mean(X, axis=0)
    cov = np.cov(X, rowvar=False)
    inv_cov = np.linalg.pinv(cov)
    UCL = p * (n - 1) * (n + 1) / (n * (n - p)) * f.ppf(0.99, p, n - p)
    print(f"✅ Baseline UCL({cols[0][:6]}...) 계산 완료: {UCL:.3f}")
    return UCL, mean, inv_cov


# ============================================================
# 🔹 X-R 관리도 단계별 변수 매핑
# ============================================================
XR_GROUPS = {
    "용융 단계": [
        ("molten_temp", "용융 온도"),
        ("molten_volume", "주입한 금속 양")
    ],
    "충진 단계": [
        ("sleeve_temperature", "주입 관 온도"),
        ("EMS_operation_time", "전자 교반(EMS) 가동 시간"),
        ("low_section_speed", "하위 구간 주입 속도"),
        ("high_section_speed", "상위 구간 주입 속도"),
        ("cast_pressure", "주입 압력")
    ],
    "냉각 단계": [
        ("upper_mold_temp1", "상부1 금형 온도"),
        ("upper_mold_temp2", "상부2 금형 온도"),
        # ("upper_mold_temp3", "상부3 금형 온도"),
        ("lower_mold_temp1", "하부1 금형 온도"),
        ("lower_mold_temp2", "하부2 금형 온도"),
        ("Coolant_temperature", "냉각수 온도")
    ],
    "생산 속도": [
        ("facility_operation_cycleTime", "장비 전체 사이클 시간"),
        ("production_cycletime", "실제 생산 사이클 시간")
    ],
    "제품 테스트": [
        ("biscuit_thickness", "주조물 두께"),
        ("physical_strength", "제품 강도")
    ]
}


# ============================================================
# 🔹 X-R 관리도용 기준선 계산
# ============================================================
def calc_baseline_xr(train_df, subgroup_size=5):
    """
    fin_train.csv 기반 Xbar-R 관리도용 기준값 계산
    - 각 변수별로 subgroup 단위로 평균(Xbar), 범위(R)을 계산
    - UCL/LCL은 표준식으로 계산 (A2=0.577, D3=0, D4=2.114)
    """
    import numpy as np

    XR_LIMITS = {}

    # subgroup 크기와 표준계수
    A2, D3, D4 = 0.577, 0, 2.114

    for col in train_df.columns:
        if train_df[col].dtype not in ["float64", "int64"]:
            continue  # 숫자형 데이터만 계산

        values = train_df[col].dropna().values

        # ✅ subgroup으로 데이터 나누기
        subgroups = [
            values[i:i + subgroup_size]
            for i in range(0, len(values), subgroup_size)
            if len(values[i:i + subgroup_size]) == subgroup_size
        ]

        if not subgroups:
            continue

        # 평균(Xbar)과 범위(R) 계산
        xbar = np.array([np.mean(sg) for sg in subgroups])
        rbar = np.array([np.ptp(sg) for sg in subgroups])  # ptp = max - min

        # 평균값
        Xbarbar = np.mean(xbar)
        Rbar = np.mean(rbar)

        # 관리한계 계산
        UCLx = Xbarbar + A2 * Rbar
        LCLx = Xbarbar - A2 * Rbar
        UCLr = D4 * Rbar
        LCLr = D3 * Rbar

        XR_LIMITS[col] = {
            "Xbar_bar": Xbarbar,
            "R_bar": Rbar,
            "UCLx": UCLx,
            "LCLx": LCLx,
            "UCLr": UCLr,
            "LCLr": LCLr,
        }

    return XR_LIMITS


# 🔸 baseline 미리 계산
fin_train = pd.read_csv("./data/train_raw.csv")
BASELINE_XR = calc_baseline_xr(fin_train)
print("✅ X-R Baseline 계산 완료:", len(BASELINE_XR), "개 변수")



# ✅ 표시에서 제외할 컬럼
EXCLUDE_COLS = ["id", "line", "name", "mold_name", "date", "time", "registration_time", "count"]

# ✅ 표시 대상: 위 제외 목록을 빼고 나머지 수치형 컬럼 자동 선택
display_cols = [
    c for c in streaming_df.columns
    if c not in EXCLUDE_COLS and pd.api.types.is_numeric_dtype(streaming_df[c])
]

# 스트리밍 초기 설정
streamer = reactive.Value(RealTimeStreamer(streaming_df))
current_data = reactive.Value(pd.DataFrame())
is_streaming = reactive.Value(False)
KF_PATH = pathlib.Path("./data/fin_test_kf_fixed.csv")
kf_streamer = reactive.Value(KFStreamer(KF_PATH))

# ===== 한글 변수명 매핑 =====
VAR_LABELS = {
    # 용융 단계
    "molten_temp": "용융 온도",
    "heating_furnace": "용해로 정보",

    # 충진 단계
    "sleeve_temperature": "슬리브 온도",
    "EMS_operation_time": "EMS 가동시간",
    "low_section_speed": "하부 주입속도",
    "high_section_speed": "상부 주입속도",
    "molten_volume": "주입 금속량",
    "cast_pressure": "주입 압력",

    # 냉각 단계
    "upper_mold_temp1": "상부1 금형온도",
    "upper_mold_temp2": "상부2 금형온도",
    "upper_mold_temp3": "상부3 금형온도",
    "lower_mold_temp1": "하부1 금형온도",
    "lower_mold_temp2": "하부2 금형온도",
    "lower_mold_temp3": "하부3 금형온도",
    "Coolant_temperature": "냉각수 온도",

    # 품질 및 속도
    "production_cycletime": "생산 사이클",
    "biscuit_thickness": "주조물 두께",
    "physical_strength": "제품 강도",
    
    "mold_code": "금형코드",
}

# ===== 센서 위치 (x, y) =====
VAR_POSITIONS = {
    # 용융부
    "molten_temp": (735, 250),
    # "heating_furnace": (735, 450),

    # 슬리브 / 주입
    "sleeve_temperature": (510, 325),
    "EMS_operation_time": (30, 340),
    "low_section_speed": (350, 390),
    "high_section_speed": (350, 135),
    "molten_volume": (700, 320),
    "cast_pressure": (520, 360),

    # 금형 냉각
    "upper_mold_temp1": (30, 30),
    "upper_mold_temp2": (30, 80),
    "upper_mold_temp3": (30, 130),
    "lower_mold_temp1": (530, 110),
    "lower_mold_temp2": (530, 160),
    "lower_mold_temp3": (530, 210),
    "Coolant_temperature": (30, 370),

    # 속도/품질
    "production_cycletime": (30, 460),
    "biscuit_thickness": (30, 430),
    "physical_strength": (30, 400),
    
    "mold_code": (350, 480),
}

# ==========================================
# 🔹 Train 데이터 로딩 및 공정별 UCL 기준 계산
# ==========================================
train_df = pd.read_csv("./data/fin_train.csv")
train_df.columns = [c.strip() for c in train_df.columns]


# ============================================================
# 🔹 X-R Baseline 로드
# ============================================================
fin_train = pd.read_csv("./data/fin_train.csv")
XR_COLS = [
    "molten_temp", "cast_pressure", "biscuit_thickness",
    "upper_mold_temp1", "lower_mold_temp1", "physical_strength"
]
BASELINE_XR = calc_baseline_xr(fin_train)
print("✅ X-R Baseline 계산 완료:", list(BASELINE_XR.keys()))


# 공정별 변수 리스트
melting_cols = ["molten_temp", "molten_volume"]
filling_cols = ["sleeve_temperature", "EMS_operation_time", "low_section_speed",
                "high_section_speed", "cast_pressure"]
cooling_cols = ["upper_mold_temp1", "upper_mold_temp2",
                # "upper_mold_temp3",
                "lower_mold_temp1", "lower_mold_temp2",
                "Coolant_temperature"]
speed_cols = ["facility_operation_cycleTime", "production_cycletime"]
quality_cols = ["biscuit_thickness", "physical_strength"]

# 단계별 기준값 계산 (한 번만 수행)
UCL_MELT, MEAN_MELT, INV_MELT = calc_baseline_ucl(train_df, melting_cols)
UCL_FILL, MEAN_FILL, INV_FILL = calc_baseline_ucl(train_df, filling_cols)
UCL_COOL, MEAN_COOL, INV_COOL = calc_baseline_ucl(train_df, cooling_cols)
UCL_SPEED, MEAN_SPEED, INV_SPEED = calc_baseline_ucl(train_df, speed_cols)
UCL_QUAL, MEAN_QUAL, INV_QUAL = calc_baseline_ucl(train_df, quality_cols)

# ===== 백엔드 및 폰트 설정 =====
matplotlib.use("Agg")  # Tkinter 대신 Agg backend 사용 (GUI 불필요)
app_dir = pathlib.Path(__file__).parent

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

APP_DIR = os.path.dirname(os.path.abspath(__file__))
font_path = os.path.join(APP_DIR, "www", "fonts", "NanumGothic-Regular.ttf")

if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "NanumGothic"
else:
    plt.rcParams["font.family"] = "sans-serif"
    print(f"⚠️ 한글 폰트 파일 없음 → {font_path}")

pio.templates["nanum"] = pio.templates["plotly_white"].update(
    layout_font=dict(family="NanumGothic")
)
pio.templates.default = "nanum"

# ===== 모델 불러오기 =====
MODEL_PATH = "./models/model_2.pkl"
model = joblib.load(MODEL_PATH)

# ✅ 추가: joblib이 basic_fix를 찾을 수 있게 __main__에 등록
import sys
sys.modules['__main__'].basic_fix = basic_fix

model = joblib.load(MODEL_PATH)

MODEL_XGB_PATH = "./models/fin_xgb_f20.pkl"
model_xgb = joblib.load(MODEL_XGB_PATH)

# ===== 데이터 불러오기 =====
df_raw = pd.read_csv("./data/train_raw.csv")

# ★ 특정 이상치 행 제거
df_raw = df_raw[
    (df_raw["low_section_speed"] != 65535) &
    (df_raw["lower_mold_temp3"] != 65503) &
    (df_raw["physical_strength"] != 65535)
]

# 예측용 데이터도 동일 처리
df_predict = pd.read_csv("./data/train.csv")
df_predict["pressure_speed_ratio"] = df_predict["pressure_speed_ratio"].replace([np.inf, -np.inf], np.nan)


# df_predict = df_predict[
#     (df_predict["low_section_speed"] != 65535) &
#     (df_predict["lower_mold_temp3"] != 65503) &
#     (df_predict["physical_strength"] != 65535)
# ]

# 탐색 탭용 (필터링/EDA)
drop_cols_explore = ["id","line","name","mold_name","date","time", "registration_time"]
df_explore = df_raw.drop(columns=drop_cols_explore, errors="ignore")  # ← 안전하게
# mold_code는 남김


# 전처리 후 데이터 (모델 학습용)
df_processed = pd.read_csv("./data/processed_train.csv")

# 컬럼 이름 표준화
df_processed.columns = df_processed.columns.str.strip().str.lower()
# 원본 탐색 데이터도 동일하게
df_explore.columns = df_explore.columns.str.strip().str.lower()

# 혹시 passorfail이 인덱스로 들어갔다면 컬럼으로 리셋
if "passorfail" not in df_processed.columns and "passorfail" in df_processed.index.names:
    df_processed = df_processed.reset_index()


# ✅ 파생 변수 자동 추가
derived_cols = ["speed_ratio", "pressure_speed_ratio"]
for col in derived_cols:
    if col in df_predict.columns:
        df_explore[col] = df_predict[col]

# 예측에서 제외할 컬럼
drop_cols = [
    "real_time",   # registration_time → real_time
    "passorfail",
    # "count",
    # "global_count",
    # "monthly_count",
    # "speed_ratio",
	# "pressure_speed_ratio",
    # "shift",
]
used_columns = df_predict.drop(columns=drop_cols).columns

# 그룹 분류
cat_cols = ["mold_code","working","emergency_stop","heating_furnace", "shift", "tryshot_signal"]
num_cols = [c for c in used_columns if c not in cat_cols]

# ===== 라벨 맵 =====
label_map = {
    # 기본 정보 관련
    "id": "고유 번호",
    "line": "생산 라인 이름",
    "name": "장비 이름",
    "mold_name": "금형 이름",
    "time": "측정 날짜",
    "date": "측정 시간",

    # 공정 상태 관련
    "count": "누적 제품 개수",
    "working": "장비 가동 여부 (가동 / 멈춤 등)",
    "emergency_stop": "비상 정지 여부 (ON / OFF)",
    "registration_time": "데이터 등록 시간",
    "tryshot_signal": "측정 딜레이 여부",

    # 용융 단계
    "molten_temp": "용융 온도",
    "heating_furnace": "용해로 정보",

    # 충진 단계
    "sleeve_temperature": "주입 관 온도",
    "ems_operation_time": "전자 교반(EMS) 가동 시간",
    "EMS_operation_time": "전자 교반(EMS) 가동 시간",
    "low_section_speed": "하위 구간 주입 속도",
    "high_section_speed": "상위 구간 주입 속도",
    "mold_code": "금형 코드",
    "molten_volume": "주입한 금속 양",
    "cast_pressure": "주입 압력",

    # 냉각 단계
    "upper_mold_temp1": "상부1 금형 온도",
    "upper_mold_temp2": "상부2 금형 온도",
    "upper_mold_temp3": "상부3 금형 온도",
    "lower_mold_temp1": "하부1 금형 온도",
    "lower_mold_temp2": "하부2 금형 온도",
    "lower_mold_temp3": "하부3 금형 온도",
    "coolant_temperature": "냉각수 온도",
    "Coolant_temperature": "냉각수 온도",

    # 공정 속도 관련
    "facility_operation_cycletime": "장비 전체 사이클 시간",
    "facility_operation_cycleTime": "장비 전체 사이클 시간",
    "production_cycletime": "실제 생산 사이클 시간",

    # 품질 및 성능
    "biscuit_thickness": "주조물 두께",
    "physical_strength": "제품 강도",

    # 평가
    "passorfail": "합격/불합격",

    "global_count": "전체 누적 개수",
    "monthly_count": "월간 누적 개수",
    "speed_ratio": "상/하부 주입 속도 비율",
	"pressure_speed_ratio": "주입 압력 비율",
    "shift": "주/야간 교대",
}

# ===== 라벨 정의 (표시 텍스트 = 한글, 실제 var = 변수명) =====
labels = [
    {"id": "label1", "text": label_map["upper_mold_temp1"], "var": "upper_mold_temp1",
     "x": 200, "y": 85, "w": 120, "h": 30,
     "arrow_from": (260, 115), "arrow_to": (400, 195)}, 

    {"id": "label2", "text": label_map["lower_mold_temp1"], "var": "lower_mold_temp1",
     "x": 650, "y": 85, "w": 120, "h": 30,
     "arrow_from": (710, 115), "arrow_to": (580, 195)}, 

    {"id": "label3", "text": label_map["cast_pressure"], "var": "cast_pressure",
     "x": 900, "y": 285, "w": 100, "h": 30,
     "arrow_from": (950, 315), "arrow_to": (780, 395)}, 

    {"id": "label4", "text": label_map["molten_volume"], "var": "molten_volume",
     "x": 700, "y": 185, "w": 120, "h": 30,
     "arrow_from": (760, 215), "arrow_to": (780, 315)}, 

    {"id": "label5", "text": label_map["sleeve_temperature"], "var": "sleeve_temperature",
     "x": 670, "y": 435, "w": 120, "h": 30,
     "arrow_from": (730, 435), "arrow_to": (600, 395)},  

    {"id": "label6", "text": label_map["high_section_speed"], "var": "high_section_speed",
     "x": 400, "y": 105, "w": 160, "h": 30,
     "arrow_from": (480, 135), "arrow_to": (510, 215)}, 

    {"id": "label7", "text": label_map["low_section_speed"], "var": "low_section_speed",
     "x": 400, "y": 455, "w": 160, "h": 30,
     "arrow_from": (480, 455), "arrow_to": (510, 355)},
]

def get_label(col): return label_map.get(col, col)

# ===== Helper: 슬라이더 + 인풋 =====
def make_num_slider(col):
    return ui.div(
        ui.input_slider(
            f"{col}_slider", get_label(col),
            min=int(df_predict[col].min()), max=int(df_predict[col].max()),
            value=int(df_predict[col].mean()), width="100%"
        ),
        ui.input_numeric(col, "", value=int(df_predict[col].mean()), width="110px"),
        style="display: flex; align-items: center; gap: 8px; justify-content: space-between;"
    )

# ===== 범주형 없음도 추가 ========
def make_select(col, label=None, width="100%"):
    label = label if label else get_label(col)
    if(col == "tryshot_signal"):
        choices = ["없음"] + sorted(df_predict[col].dropna().unique().astype(str))
    else:
        choices = sorted(df_predict[col].dropna().unique().astype(str)) + ["없음"]
    return ui.input_select(col, label, choices=choices, width=width)


def make_svg(labels):
    parts = []
    for lbl in labels:
        # 화살표 시작점: arrow_from 있으면 사용, 없으면 중앙
        if "arrow_from" in lbl:
            cx, cy = lbl["arrow_from"]
        else:
            cx = lbl["x"] + lbl["w"]/2
            cy = lbl["y"] + lbl["h"]/2

        x2, y2 = lbl["arrow_to"]
        text = label_map.get(lbl["var"], lbl["var"])

        parts.append(f"""
        <g>
        <rect x="{lbl['x']}" y="{lbl['y']}" width="{lbl['w']}" height="{lbl['h']}" 
                fill="#e0e6ef" stroke="black"/>
        <text x="{lbl['x'] + lbl['w']/2}" y="{lbl['y'] + lbl['h']/2}" 
                fill="black" font-size="14" font-weight="bold"
                text-anchor="middle" dominant-baseline="middle">{text}</text>
        <line x1="{cx}" y1="{cy}" x2="{x2}" y2="{y2}" 
                stroke="red" marker-end="url(#arrow)"/>
        </g>
        """)
    return "\n".join(parts)

svg_code = f"""
<svg width="1000" height="500" xmlns="http://www.w3.org/2000/svg"
     style="background:url('die-castings.gif'); background-size:cover;">
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L6,3 z" fill="red"/>
    </marker>
  </defs>
  {make_svg(labels)}
</svg>
"""

# ========== 데이터 준비 ==========
train = pd.read_csv("./data/train_raw.csv")
train["time"] = pd.to_datetime(train["time"], errors="coerce")
train["day"] = train["time"].dt.date

# 몰드코드별 요약
mold_cycle = (
    train.groupby("mold_code")["facility_operation_cycleTime"]
    .mean()
    .reset_index(name="avg_facility_cycleTime")
)
mold_cycle["daily_capacity"] = (86400 / mold_cycle["avg_facility_cycleTime"]).round()

daily_actual = train.groupby(["day", "mold_code"])["count"].agg(["min", "max"]).reset_index()
daily_actual["daily_actual"] = daily_actual["max"] - daily_actual["min"] + 1

mold_stats = daily_actual.groupby("mold_code")["daily_actual"].agg(
    min_prod="min", max_prod="max", avg_prod="mean"
).reset_index()

mold_summary = pd.merge(mold_cycle, mold_stats, on="mold_code")

# mold_code를 문자열로 변환
mold_summary["mold_code"] = mold_summary["mold_code"].astype(int).astype(str)
codes = list(mold_summary["mold_code"])
last_code = codes[-1]

# 색상 팔레트
cmap = cm.get_cmap("tab10", len(codes))
mold_colors = {code: mcolors.to_hex(cmap(i)) for i, code in enumerate(codes)}

# ================================
# 권장 세팅값 계산
# ================================
def smooth_series(series, window=5):
    smoothed = series.rolling(window=window, center=True, min_periods=1).mean()
    Q1, Q3 = smoothed.quantile(0.25), smoothed.quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    filtered = smoothed[(smoothed >= lower) & (smoothed <= upper)]
    return filtered.dropna()

setting_cols = [
    "molten_temp",
    "upper_mold_temp1","upper_mold_temp2","upper_mold_temp3",
    "lower_mold_temp1","lower_mold_temp2","lower_mold_temp3",
    "sleeve_temperature","cast_pressure","biscuit_thickness",
    "physical_strength","Coolant_temperature"
]

setting_table = {}
for code, df in train.groupby("mold_code"):
    settings = {}
    for col in setting_cols:
        smoothed = smooth_series(df[col].dropna())
        if len(smoothed) == 0:
            settings[col] = df[col].mean()
            continue
        try:
            mode_val = stats.mode(smoothed, keepdims=True)[0][0]
            settings[col] = mode_val
        except Exception:
            settings[col] = smoothed.mean()
    setting_table[str(code)] = settings  # 🔑 mold_code를 문자열로 저장

setting_df = pd.DataFrame(setting_table).T.reset_index().rename(columns={"index": "mold_code"})
setting_df["mold_code"] = setting_df["mold_code"].astype(str)  # 문자열로 통일

# ================================
# 생산 시뮬레이션 탭 비율 그래프
# ================================
train_raw = pd.read_csv("./data/train_raw.csv")

if "date" in train_raw.columns and "time" in train_raw.columns:
    train_raw["real_time"] = pd.to_datetime(
        train_raw["date"].astype(str) + " " + train_raw["time"].astype(str),
        errors="coerce"
    )
elif "registration_time" in train_raw.columns:
    train_raw["real_time"] = pd.to_datetime(train_raw["registration_time"], errors="coerce")
else:
    raise ValueError("date/time 또는 registration_time 컬럼을 확인해주세요.")

train_raw["date_only"] = train_raw["real_time"].dt.date

# 날짜별 mold_code 생산 개수
daily_mold = train_raw.groupby(["date_only", "mold_code"]).size().reset_index(name="count")
pivot_count = daily_mold.pivot(index="date_only", columns="mold_code", values="count").fillna(0)

years = list(range(2024, 2027))
months = list(range(1, 13))

# ======== 전역 HEAD (favicon, CSS 등) ========
global_head = ui.head_content(
    ui.tags.link(rel="icon", type="image/x-icon", href="favicon.ico"),
    ui.tags.link(rel="stylesheet", href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"),
    ui.tags.link(rel="stylesheet", type="text/css", href="custom.css"),
    ui.tags.title("주조 공정 불량 예측 대시보드"),

    # =====================================================
    # 📜 mold_code별 6시그마 + 실시간 색상 업데이트 스크립트
    # =====================================================
    ui.tags.script("""
        // =====================================================
        // 📘 mold_code별 6시그마 기준 로드
        // =====================================================
        let THRESHOLDS_BY_MOLD = {};

        fetch("sixsigma_thresholds_by_mold.json?t=" + Date.now())
            .then(res => res.json())
            .then(data => {
                THRESHOLDS_BY_MOLD = data;
            })
            .catch(err => console.error("⚠️ 6시그마 로드 실패:", err));

        // =====================================================
        // ⚙️ 단위 판정
        // =====================================================
        function unitFor(key) {
            const k = key.toLowerCase();
            if (k.includes("temp")) return " °C";
            if (k.includes("pressure")) return " bar";
            if (k.includes("speed")) return " cm/s";
            if (k.includes("volume")) return " cc";
            if (k.includes("thickness")) return " mm";
            if (k.includes("strength")) return " MPa";
            if (k.includes("cycle") || k.includes("time")) return " s";
            return "";
        }

        // =====================================================
        // 🎨 mold_code별 σ 색상 계산
        // =====================================================
        function colorBySigmaLevel(key, val, moldCode) {
            const k = key.toLowerCase();
            if (k.includes("mold_code")) return "#111827";

            const moldData = THRESHOLDS_BY_MOLD[moldCode];
            if (!moldData) return "#00C853";

            const info = moldData[key];
            if (!info || !info.sigma || info.sigma === 0) return "#00C853";

            const mu = info.mu;
            const sigma = info.sigma;
            const diff = Math.abs(val - mu);

            if (diff <= 1 * sigma) return "#00C853"; // 초록
            if (diff <= 2 * sigma) return "#FFD600"; // 노랑
            if (diff <= 3 * sigma) return "#FB8C00"; // 주황
            return "#E53935";                         // 빨강
        }

        // =====================================================
        // 🔹 실시간 센서 업데이트 (mold_code 기반)
        // =====================================================
        Shiny.addCustomMessageHandler("updateSensors", function(data) {
            const values = data.values;
            const moldCode = String(data.mold_code || "");

            for (const [key, val] of Object.entries(values)) {
                if (typeof val !== "number" || isNaN(val) || val === 0) continue;

                const valueNode = document.querySelector(`#var-${key} .value`);
                if (!valueNode) continue;

                const color = colorBySigmaLevel(key, val, moldCode);

                const isMold = key.toLowerCase().includes("mold_code");
                const txt = isMold
                    ? `${Math.round(val)}`
                    : `${val.toFixed(1)}${unitFor(key)}`;

                valueNode.textContent = txt;
                valueNode.setAttribute("fill", color);

                // 배경 테두리 색상 동기화
                const rectNode = document.querySelector(`#var-${key} rect`);
                if (rectNode) {
                    const strokeColor = color === "#00C853" ? "#ddd" : color;
                    rectNode.setAttribute("stroke", strokeColor);
                    rectNode.setAttribute("stroke-width",
                        color === "#00C853" ? "0.5" : "1.5");
                }
            }
        });

        // =====================================================
        // 🔹 센서 초기화 핸들러 (값 '—', 테두리 회색)
        // =====================================================
        Shiny.addCustomMessageHandler("resetSensors", function(message) {
            document.querySelectorAll("tspan.value").forEach(node => {
                node.textContent = "—";
                node.setAttribute("fill", "#111827");
                const parent = node.closest("text");
                if (parent) parent.setAttribute("fill", "#111827");
            });

            document.querySelectorAll("g[id^='var-'] rect").forEach(rect => {
                rect.setAttribute("stroke", "#ddd");
                rect.setAttribute("stroke-width", "0.5");
            });
        });
    """),

    # =====================================================
    # 🖼️ GIF 업데이트 스크립트
    # =====================================================
    ui.tags.script("""
        Shiny.addCustomMessageHandler("updateGif", function(data) {
            const img = document.getElementById("process_gif");
            if (!img) return;
            img.src = data.src + "?t=" + new Date().getTime();
        });
    """),
)

# ======== 상태 저장 ========
login_state = reactive.Value(False)
page_state = reactive.Value("login")   # login → menu → main


# ======== 1️⃣ 로그인 페이지 ========
def login_page():
    return ui.page_fillable(
        ui.div(
            {
                "style": (
                    "display:flex; flex-direction:column; justify-content:center; "
                    "align-items:center; height:100vh; background-color:#f8f9fa;"
                )
            },
            # ▼ 로고 이미지
            ui.img(
                src="LS_Logo.svg",   # www 폴더 안에 LS_Logo.svg 위치해야 함
                style="width:150px; margin-bottom:25px;"
            ),
            # ▼ 로그인 카드
            ui.card(
                {
                    "style": (
                        "width:350px; padding:20px; box-shadow:0 0 10px rgba(0,0,0,0.1);"
                    )
                },
                ui.h3("🔐 로그인", style="text-align:center; margin-bottom:20px;"),
                ui.input_text("user", "아이디", placeholder="아이디를 입력하세요"),
                ui.input_password("password", "비밀번호", placeholder="비밀번호를 입력하세요"),
                ui.input_action_button("login_btn", "로그인", class_="btn btn-primary w-100 mt-3"),
                ui.div(
                    ui.output_text("login_msg"),
                    style="color:red; margin-top:10px; text-align:center;",
                ),
            ),
        )
    )


# ======== 2️⃣ 카드 탭 선택 페이지 ========
def menu_page():
    return ui.page_fillable(
        ui.div(
            {
                "style": (
                    "min-height:100vh; background-color:#fdfdfd; padding:40px; "
                    "display:flex; flex-direction:column; align-items:center;"
                )
            },
            ui.h3("주조 공정 불량 예측 대시보드", style="margin-bottom:30px; font-weight:bold;"),
            ui.div(
                {
                    "style": (
                        "display:grid; grid-template-columns:repeat(auto-fit, minmax(250px, 1fr)); "
                        "gap:20px; width:80%; max-width:1800px;"
                    )
                },
                # 📊 현장 대시보드
                ui.card(
                    {"class": "overview-card",
                     "style": (
                         "border:2px solid #FFC966; color:#333; text-align:center; "
                         "cursor:pointer; box-shadow:0 2px 6px rgba(0,0,0,0.05);"
                     )},
                    ui.card_header(
                        "📊 현장 대시보드",
                        style=(
                            "background-color:#FFC966; color:#333; "
                            "font-weight:bold; font-size:20px; text-align:center; "
                            "padding-top:15px; padding-bottom:15px;"
                        ),
                    ),
                    ui.tags.img(
                        src="1.png",
                        style="width:100%; height:400px; object-fit:cover; margin-bottom:10px; border-radius:8px;"
                    ),
                    ui.p("현장별 주요 지표 및 트렌드"),
                    ui.input_action_button("goto_field", "이동", class_="btn btn-outline-primary mt-2"),
                ),

                # 🧭 품질 모니터링
                ui.card(
                    {"class": "overview-card",
                     "style": (
                         "border:2px solid #A5C16A; color:#333; text-align:center; "
                         "cursor:pointer; box-shadow:0 2px 6px rgba(0,0,0,0.05);"
                     )},
                    ui.card_header(
                        "🧭 품질 모니터링",
                        style=(
                            "background-color:#A5C16A; color:#333; "
                            "font-weight:bold; font-size:20px; text-align:center; "
                            "padding-top:15px; padding-bottom:15px;"
                        ),
                    ),
                    ui.tags.img(
                        src="3.png",
                        style="width:100%; height:400px; object-fit:cover; margin-bottom:10px; border-radius:8px;"
                    ),
                    ui.p("불량률, 센서 이상 감지, 예측 결과"),
                    ui.input_action_button("goto_quality", "이동", class_="btn btn-outline-success mt-2"),
                ),

                # 📈 데이터 분석
                ui.card(
                    {"class": "overview-card",
                     "style": (
                         "border:2px solid #80CBC4; color:#333; text-align:center; "
                         "cursor:pointer; box-shadow:0 2px 6px rgba(0,0,0,0.05);"
                     )},
                    ui.card_header(
                        "📈 데이터 분석",
                        style=(
                            "background-color:#80CBC4; color:#333; "
                            "font-weight:bold; font-size:20px; text-align:center; "
                            "padding-top:15px; padding-bottom:15px;"
                        ),
                    ),
                    ui.tags.img(
                        src="2.png",
                        style="width:100%; height:400px; object-fit:cover; margin-bottom:10px; border-radius:8px;"
                    ),
                    ui.p("주요 피처 분석 결과"),
                    ui.input_action_button("goto_analysis", "이동", class_="btn btn-outline-secondary mt-2"),
                ),
            ),
            ui.input_action_button("logout_btn", "로그아웃", class_="btn btn-light mt-5"),
        )
    )

def field_dashboard_ui():
    return ui.div(
        {"style": "display:flex; flex-direction:column; gap:20px;"},  # 🔹 세로 2행 구성
        # ──────────────── 1행: 제어 + 공정 상태 ────────────────
        ui.div(
            {
                "style": (
                    "display:grid; grid-template-columns:1fr 2fr; gap:20px;"
                )
            },
        # ───────────── 공정 상태 카드 ─────────────
        ui.card(
            ui.card_header("공정 상태"),
            ui.output_ui("process_status_card"),  # ✅ 추가
            ui.output_ui("realtime_predict_card"),  # 🧠 추가
        ),
            ui.card(
                ui.card_header(
                    ui.div(
                        {"style": "display:flex; justify-content:space-between; align-items:center;"},
                        ui.span("🧭 주조 공정 실시간 상태", style="font-weight:700; font-size:16px;"),
                        ui.div(
                            {"class": "legend-row"},
                            ui.div({"class": "legend-box"},
                                ui.div({"class": "legend-color", "style": "background:#00C853;"}),
                                "정상"
                            ),
                            ui.div({"class": "legend-box"},
                                ui.div({"class": "legend-color", "style": "background:#FFD600;"}),
                                "주의"
                            ),
                            ui.div({"class": "legend-box"},
                                ui.div({"class": "legend-color", "style": "background:#FB8C00;"}),
                                "경고"
                            ),
                            ui.div({"class": "legend-box"},
                                ui.div({"class": "legend-color", "style": "background:#E53935;"}),
                                "이상"
                            )
                        )
                    )
                ),
                ui.output_ui("process_svg_inline"),
                style="width:100%;"
            ),
        ),
        # ──────────────── 2행: 실시간 알림창 ────────────────
        ui.card(
            ui.card_header(ui.output_ui("alert_card_header")),
            ui.div(
                ui.output_ui("realtime_alert_box"),
                style=(
                    "max-height:300px; overflow-y:auto; background:white; "
                    "padding:10px; border-radius:8px; border:1px solid #eee;"
                ),
            ),
            style="width:100%; background-color:#fff8e1;"
        ),
    )

def floating_stream_bar():
    """헤더 바로 아래 탭 스타일 스트리밍 제어 바"""
    return ui.div(
        {
            "style": (
                "display:flex; align-items:center; gap:16px;"
                "background-color:#fef6ee; border:1px solid #e0c8a0;"
                "border-bottom:none; border-radius:8px 8px 0 0;"
                "padding:8px 16px; position:absolute; top:28px; right:40px;"
                # 🔽 z-index를 낮춤 (1500 → 900)
                "z-index:900; font-weight:bold; color:#5c4b3b;"
                "backdrop-filter:blur(2px);"  # 💡선택: 흐림 효과 보완
            )
        },
        ui.div("스트리밍 제어", style="font-weight:bold; font-size:15px;"),
        ui.output_ui("stream_status"),

        # 시간 표시 (고정폭)
        ui.div(
            ui.output_ui("stream_time_display"),
            style="font-size:14px; width:180px; text-align:center; white-space:nowrap;",
        ),

        # 버튼 그룹
        ui.div(
            {"style": "display:flex; gap:8px;"},
            ui.output_ui("stream_buttons")
        ),
    )

def load_svg_inline():
    svg_path = os.path.join(APP_DIR, "www", "diagram.svg")
    with open(svg_path, "r", encoding="utf-8") as f:
        return f.read()

def make_dynamic_svg(sensor_list: list[str]) -> str:
    """센서 목록을 받아 SVG 텍스트 노드를 자동 생성"""
    base_svg = [
        '<svg width="900" height="{}" xmlns="http://www.w3.org/2000/svg">'.format(100 + 30 * len(sensor_list)),
        '<rect width="100%" height="100%" fill="#f9f9f9"/>'
    ]
    for i, name in enumerate(sensor_list):
        y = 40 + i * 30
        base_svg.append(f'<text id="{name}" x="50" y="{y}" font-size="16" font-weight="bold" fill="#333">{name}: --</text>')
    base_svg.append('</svg>')
    return "\n".join(base_svg)




def plan_page_ui():
    """생산계획 탭의 UI를 반환하는 함수"""
    years = list(range(datetime.date(2019, 1, 19).year, datetime.date(2019, 1, 19).year + 3))
    months = list(range(1, 13))
    return ui.layout_sidebar(
        ui.sidebar(
            ui.input_numeric("monthly_target", "이달의 총 생산 목표 수", value=20000, min=1000, step=1000),
            ui.input_select("year", "연도 선택", {str(y): str(y) for y in years}, selected=str(datetime.date(2019, 1, 19).year)),
            ui.input_select("month", "월 선택", {str(m): f"{m}월" for m in months}, selected=str(datetime.date(2019, 1, 19).month)),
            ui.output_ui("mold_inputs"),
            ui.output_text("remaining_qty"),
            ui.input_action_button("run_plan", "시뮬레이션 실행", class_="btn btn-primary"),
        ),
        ui.card(ui.card_header("금형코드별 생산성 요약"), ui.output_data_frame("mold_summary_table")),
        ui.card(
            ui.card_header("달력형 계획표", ui.input_action_button("show_modal", "날짜별 금형 코드 생산 추이", class_="btn btn-sm btn-outline-primary", style="position:absolute; top:10px; right:10px;")),
            ui.output_ui("calendar_view"),
            ui.hr(),  
        )
    )

def analysis_page_ui():
    """스케치 기반의 '데이터 분석 / 모델 모니터링' 탭 UI 생성"""
    return ui.navset_tab(
        ui.nav_panel(
            "모델 모니터링",
            ui.layout_sidebar(
                # === 1. 사이드바 (제어 패널) ===
                ui.sidebar(
                    {"title": "모델 제어"},
                    ui.input_select(
                         "analysis_mold_select", "Mold Code 선택", 
                           choices={
                            "all": "전체", 
                            "8412": "Mold Code 8412", # ✅ 키: 값 형태로 수정
                            "8413": "Mold Code 8413", # ✅ 키: 값 형태로 수정
                            "8576": "Mold Code 8576", # ✅ 키: 값 형태로 수정
                            "8722": "Mold Code 8722", # ✅ 키: 값 형태로 수정
                            "8917": "Mold Code 8917"  # ✅ 키: 값 형태로 수정
                        }, 
                         selected="all"
                    ),
                    ui.input_slider(
                        "analysis_threshold", "Threshold 조정",
                        min=0, max=1, value=0.5, step=0.01
                    ),
                    ui.hr(),
                    ui.h5("스트리밍 제어"),
                    ui.output_ui("stream_control_ui"),
                    ui.br(),
                    ui.output_ui("comm_status"),
                ),

                # === 2. 메인 컨텐츠 ===
                ui.card(
                    ui.card_header("실시간 예측 확률"),
                    # 스케치의 '들어오는 데이터' 그래프
                    ui.output_plot("main_analysis_plot") 
                ),
                ui.layout_columns(
                    ui.card(
                        ui.card_header("모델 응답 지연 (Latency)"),
                        # 스케치의 'Latency' 그래프
                        ui.output_plot("latency_plot") 
                    ),
                    ui.card(
                        ui.card_header("누적 성능 지표"),
                        # 스케치의 'Accuracy' 등 4개 카드
                        ui.output_ui("metric_cards") 
                    ),
                    col_widths=[6, 6]
                ),
                ui.card(
                    ui.card_header("실시간 예측 로그"),
                    # 스케치의 '로그 뷰어'
                    ui.output_ui("log_viewer") 
                )
            )
        )
    )
# ======== 3️⃣ 본문 페이지 ========
def main_page(selected_tab: str):
    # --- 메뉴별 제목 및 본문 내용 ---
    tab_titles = {
        "field": "📊 현장 대시보드",
        "quality": "🧭 품질 모니터링",
        "analysis": "📈 데이터 분석"
    }
    tab_contents = {
        "field": ui.navset_tab(
    ui.nav_panel("실시간 대시보드", field_dashboard_ui()),

    # ───────── 이번달 생산목표 ─────────
    ui.nav_panel(
    "생산현황",
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_date("ref_date", "조회 기준일", value=datetime.date(2019, 1, 19)),
            style="background-color:#fffaf2; padding:20px; border-radius:10px;"
        ),
        ui.card(
            ui.card_header("📅 생산 현황"),
            ui.output_ui("calendar_view_current"),
            ui.hr(),
            ui.output_text("daily_summary"),   # ← 누적/예상 표시
            style="background-color:white; padding:20px;"
        )
    )
),

    # ───────── 다음달 생산목표 ─────────
    ui.nav_panel(
        "생산목표",
        plan_page_ui()  # ✅ 기존의 시뮬레이션 탭
    ),
    id="field_tabs"
),

        
        # 🧭 품질 모니터링 (예측 시뮬레이션 UI 포함)
        "quality": ui.navset_tab(
            ui.nav_panel("실시간 관리도",
                ui.card(
                    ui.navset_tab(
                        # ─────────────── 다변량 관리도 ───────────────
                        ui.nav_panel("다변량 관리도",
                            ui.input_select(
                                "mv_group",
                                "관리 구분 선택",
                                choices=["공정 관리", "생산 관리", "제품 관리"],
                                selected="공정 관리"
                            ),
                            ui.output_ui("mv_group_ui")
                        ),

                        # ─────────────── X-R 관리도 ───────────────
                        ui.nav_panel("X-R 관리도",
                            ui.input_select(
                                "xr_select",
                                "단계 선택",
                                choices=list(XR_GROUPS.keys()),
                                selected="용융 단계"
                            ),
                            ui.div(
                                ui.output_plot("xr_chart", height="1200px"),
                                style=(
                                    "height:1200px;"
                                    "overflow-y:auto;"
                                    "overflow-x:hidden;"
                                    "padding:10px;"
                                    "background-color:#fff;"
                                    "border-radius:8px;"
                                    "box-shadow:0 1px 3px rgba(0,0,0,0.1);"
                                )
                            )
                        ),
                    )
                ),
            ),



            ui.nav_panel("원인 분석",

                ui.card(
                    ui.card_header("불량 및 공정 에러 발생 조건", style="text-align:center;"),
                    ui.output_plot("local_factor_plot", click=True),   # 클릭 가능한 그래프
                    ui.hr(),
                    ui.output_ui("local_factor_desc"),      # 텍스트 설명
                    ui.output_ui("sensor_detail_modal")     # 클릭 시 뜨는 모달창
                ),

                # ──────────────── 2행: 실시간 데이터 표 ────────────────
                ui.card(
                    ui.card_header("📊 실시간 데이터"),
                    ui.div(
                        ui.output_data_frame("recent_data_table"),
                        # 🔹 스크롤이 생기도록 wrapping div에 명시적 width/overflow 지정
                        style=(
                            "width:100%; "
                            "overflow-x:auto; overflow-y:auto; "  # 가로/세로 스크롤 모두 허용
                            "max-height:500px; "  # 너무 길면 세로 스크롤
                            "display:block;"
                        )
                    ),
                    style="width:100%;"
                ),

            ),

            # =========================================
            # 기존 코드 최대한 유지 + 탭 통합 버전
            # =========================================
            ui.nav_panel("예측 및 개선",
                # 입력 변수 카드
                ui.div(
                    ui.card(
                        ui.card_header("입력 변수", style="background-color:#f8f9fa; text-align:center;"),

                        # 생산 환경 정보 카드 (최상단)
                        ui.card(
                            ui.card_header("생산 환경 정보", style="text-align:center;"),
                            ui.layout_columns(
                                ui.div(
                                    f"생산 라인: {df_raw['line'].iloc[0]}",
                                    style="background-color:#e9ecef; padding:8px 12px; border-radius:6px; text-align:center; font-weight:bold;"
                                ),
                                ui.div(
                                    f"장비 이름: {df_raw['name'].iloc[0]}",
                                    style="background-color:#e9ecef; padding:8px 12px; border-radius:6px; text-align:center; font-weight:bold;"
                                ),
                                ui.div(
                                    f"금형 이름: {df_raw['mold_name'].iloc[0]}",
                                    style="background-color:#e9ecef; padding:8px 12px; border-radius:6px; text-align:center; font-weight:bold;"
                                ),
                                col_widths=[4,4,4]
                            )
                        ),

                        # === 공정 상태 관련 ===
                        ui.card(
                            ui.card_header("공정 상태 관련"),
                            ui.layout_columns(
                                ui.input_numeric("count", "일조 누적 제품 개수", value=int(df_predict["count"].mean())),
                                # ui.input_numeric("monthly_count", "월간 누적 제품 개수", value=int(df_predict["monthly_count"].mean())),
                                # ui.input_numeric("global_count", "전체 누적 제품 개수", value=int(df_predict["global_count"].mean())),
                                ui.input_numeric("speed_ratio", "상하 구역 속도 비율", value=int(df_predict["speed_ratio"].mean())),
                                ui.input_numeric("pressure_speed_ratio", "주조 압력 속도 비율", value=int(df_predict["pressure_speed_ratio"].mean())),
                                make_select("working", "장비 가동 여부"),
                                make_select("emergency_stop", "비상 정지 여부"),
                                make_select("tryshot_signal", "측정 딜레이 여부"),
                                make_select("shift", "주, 야간 조"),
                                col_widths=[3,3,3,3]
                            )
                        ),

                        # === 용융 단계 ===
                        ui.card(
                            ui.card_header("용융 단계"),
                            ui.layout_columns(
                                make_num_slider("molten_temp"),
                                make_select("heating_furnace", "용해로"),
                                col_widths=[6,6]
                            )
                        ),

                        # === 충진 단계 ===
                        ui.card(
                            ui.card_header("충진 단계"),
                            ui.layout_columns(
                                make_num_slider("sleeve_temperature"),
                                make_num_slider("EMS_operation_time"),
                                make_num_slider("low_section_speed"),
                                make_num_slider("high_section_speed"),
                                make_num_slider("molten_volume"),
                                make_num_slider("cast_pressure"),
                                ui.input_select("mold_code", "금형 코드", choices=sorted(df_predict["mold_code"].dropna().unique().astype(str))),
                                col_widths=[3,3,3,3]
                            )
                        ),

                        # === 냉각 단계 ===
                        ui.card(
                            ui.card_header("냉각 단계"),
                            ui.layout_columns(
                                make_num_slider("upper_mold_temp1"),
                                make_num_slider("upper_mold_temp2"),
                                # make_num_slider("upper_mold_temp3"),
                                make_num_slider("lower_mold_temp1"),
                                make_num_slider("lower_mold_temp2"),
                                # make_num_slider("lower_mold_temp3"),
                                make_num_slider("Coolant_temperature"),
                                col_widths=[3,3,3,3]
                            )
                        ),

                        # === 공정 속도 관련 ===
                        ui.card(
                            ui.card_header("공정 속도 관련"),
                            ui.layout_columns(
                                make_num_slider("facility_operation_cycleTime"),
                                make_num_slider("production_cycletime"),
                                col_widths=[6,6]
                            )
                        ),

                        # === 품질 및 성능 ===
                        ui.card(
                            ui.card_header("품질 및 성능"),
                            ui.layout_columns(
                                make_num_slider("biscuit_thickness"),
                                make_num_slider("physical_strength"),
                                col_widths=[6,6]
                            )
                        )
                    ),
                    style="max-width: 1200px; margin: 0 auto;"
                ),

                ui.br(),

                # 예측 실행 + 결과 카드
                ui.div(
                    ui.card(
                        ui.card_header(
                            ui.div(
                                [
                                    ui.input_action_button(
                                        "predict_btn", "예측 실행",
                                        class_="btn btn-primary btn-lg",
                                        style="flex:1;"
                                    ),
                                    ui.input_action_button(
                                        "reset_btn", ui.HTML('<i class="fa-solid fa-rotate-left"></i>'),
                                        class_="btn btn-secondary btn-lg",
                                        style="margin-left:10px; width:60px;"
                                    )
                                ],
                                style="display:flex; align-items:center; width:100%;"
                            ),
                            style="background-color:#f8f9fa; text-align:center;"
                        ),
                        ui.output_ui("prediction_result")
                    ),
                    style="""
                        position: -webkit-sticky;
                        position: sticky;
                        bottom: 1px;
                        z-index: 1000;
                        max-width: 1200px;
                        margin: 0 auto;
                    """
                ),

                ui.br(),

                # === 개선 방안 섹션 (조건부 렌더링 추가) ===
                ui.output_ui("improvement_section")

            ),
        ),
        "analysis": analysis_page_ui()
    }

    current_title = tab_titles.get(selected_tab, "")

    # === 상단 바 ===
    header_bar = ui.div(
        {
            "class": "app-title bg-primary text-white",
            "style": (
                "display:flex; justify-content:space-between; align-items:center; "
                "padding:10px 20px;"
            ),
        },
        # 왼쪽: 뒤로가기 버튼
        ui.input_action_button(
            "back_btn",
            "← 뒤로가기",
            class_="btn btn-light btn-sm",
            style="font-weight:bold; min-width:100px; height:34px;"
        ),

        # 중앙: 타이틀 + 메뉴명 + 드롭다운 버튼
        ui.div(
            {"style": "display:flex; align-items:center; gap:8px;"},
            ui.h4(
                [
                    "🏭 주조 공정 불량 예측 대시보드",
                    ui.span(
                        f" — {current_title}",
                        style="font-weight:normal; font-size:17px; margin-left:6px; color:#ffffff;"
                    ),
                ],
                style="margin:0; font-weight:bold;"
            ),

            # ▼ 드롭다운 메뉴 버튼
            ui.tags.div(
                {"class": "dropdown"},
                ui.tags.button(
                    "",
                    {
                        "class": "btn btn-outline-light btn-sm dropdown-toggle",
                        "type": "button",
                        "data-bs-toggle": "dropdown",
                        "aria-expanded": "false",
                        "style": (
                            "padding:2px 10px; font-weight:bold; font-size:16px; line-height:1;"
                        ),
                    },
                ),
                ui.tags.ul(
                    {"class": "dropdown-menu dropdown-menu-end"},
                    ui.tags.li(
                        ui.input_action_button(
                            "goto_field",
                            "📊 현장 대시보드",
                            class_=(
                                "dropdown-item w-100 text-start "
                                + ("active-menu" if selected_tab == "field" else "")
                            ),
                        )
                    ),
                    ui.tags.li(
                        ui.input_action_button(
                            "goto_quality",
                            "🧭 품질 모니터링",
                            class_=(
                                "dropdown-item w-100 text-start "
                                + ("active-menu" if selected_tab == "quality" else "")
                            ),
                        )
                    ),
                    ui.tags.li(
                        ui.input_action_button(
                            "goto_analysis",
                            "📈 데이터 분석",
                            class_=(
                                "dropdown-item w-100 text-start "
                                + ("active-menu" if selected_tab == "analysis" else "")
                            ),
                        )
                    ),
                ),
            ),
        ),

        # 오른쪽: 로그아웃 버튼
        ui.input_action_button(
            "logout_btn",
            "🔓 로그아웃",
            class_="btn btn-light btn-sm",
            style="font-weight:bold; min-width:100px; height:34px;"
        ),
    )

    # === 본문 영역 ===
    content_area = ui.div(
        {
            "style": (
                "padding:30px 40px; background-color:#f8f9fa; "
                "min-height:calc(100vh - 80px);"
            )
        },
        # ui.h4(current_title),
        ui.div(tab_contents.get(selected_tab, ui.p("페이지 없음"))),
    )

    return ui.page_fluid(
        header_bar,
        ui.div(
            {"style": "position:relative;"},
            floating_stream_bar(),  # ✅ 새로운 탭 형태 바 적용
            content_area
        )
    )

# ======== 전체 UI ========
app_ui = ui.page_fluid(global_head, ui.output_ui("main_ui"))


# ======== 서버 로직 ========
def server(input, output, session):
# ============================================================
# 🟢 로그인 페이지
# ============================================================

    # 로그인 처리
    @reactive.effect
    @reactive.event(input.login_btn)
    def _login():
        if input.user() == "admin" and input.password() == "1234":
            login_state.set(True)
            page_state.set("menu")
        else:
            login_state.set(False)
            page_state.set("login")

    # 카드 선택 → 해당 본문으로 이동
    @reactive.effect
    @reactive.event(input.goto_field)
    def _go_field():
        page_state.set("field")

    @reactive.effect
    @reactive.event(input.goto_quality)
    def _go_quality():
        page_state.set("quality")

    @reactive.effect
    @reactive.event(input.goto_analysis)
    def _go_analysis():
        page_state.set("analysis")

    # 로그아웃 버튼 클릭 → 확인 모달 표시
    @reactive.effect
    @reactive.event(input.logout_btn)
    def _logout_confirm():
        if login_state():
            m = ui.modal(
                ui.p("정말 로그아웃 하시겠습니까?"),
                title="로그아웃 확인",
                easy_close=False,
                footer=ui.div(
                    ui.input_action_button("confirm_logout", "확인", class_="btn btn-danger"),
                    ui.input_action_button("cancel_logout", "취소", class_="btn btn-secondary ms-2"),
                ),
            )
            ui.modal_show(m)

    # 로그아웃 확인 / 취소
    @reactive.effect
    @reactive.event(input.confirm_logout)
    def _logout_ok():
        login_state.set(False)
        page_state.set("login")
        ui.modal_remove()

    @reactive.effect
    @reactive.event(input.cancel_logout)
    def _logout_cancel():
        ui.modal_remove()
    
    # ===== 뒤로가기 버튼: 카드 선택 페이지로 복귀 ===== 
    @reactive.effect 
    @reactive.event(input.back_btn) 
    def _go_back(): 
        page_state.set("menu")

    # 페이지 상태에 따라 UI 전환
    @output
    @render.ui
    def main_ui():
        state = page_state()
        if state == "login":
            return login_page()
        elif state == "menu":
            return menu_page()
        elif state in ["field", "quality", "analysis"]:
            return main_page(state)
        else:
            return ui.p("⚠️ 알 수 없는 페이지 상태입니다.")

    # 로그인 실패 메시지 출력
    @output
    @render.text
    def login_msg():
        if input.login_btn() > 0 and not login_state():
            return "아이디 또는 비밀번호가 올바르지 않습니다."
        return ""

# 🟢 로그인 페이지 끝
# ============================================================



    
        
# ============================================================
# 🟢 TAB1. 현장 관리 (최신 Shiny 버전 호환)
# ============================================================

    # =====================================================
    # ✅ 데이터 로드 및 전처리
    # =====================================================
    train_raw = pd.read_csv("./data/train_raw.csv", low_memory=False)
    fin_test = pd.read_csv("./data/fin_test.csv", low_memory=False)

    for df in [train_raw, fin_test]:
        if "real_time" not in df.columns and "date" in df.columns and "time" in df.columns:
            df["real_time"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce")

    common_cols = [c for c in train_raw.columns if c in fin_test.columns]
    fin_all = pd.concat([train_raw[common_cols], fin_test[common_cols]], ignore_index=True)

    # real_time 변환 (NaT 제거)
    fin_all["real_time"] = pd.to_datetime(fin_all["real_time"], errors="coerce")
    fin_all = fin_all.dropna(subset=["real_time"]).copy()

    # 날짜 변환 (2019 → 2025년 10월)
    fin_all["real_time"] = fin_all["real_time"] 
    fin_all["date"] = fin_all["real_time"].dt.floor("D")

    # =====================================================
    # 📅 달력 렌더링 (선택한 달 기준으로 표시)
    # =====================================================
    @render.ui
    def calendar_view_current():
        ref_date_str = input.ref_date() or None
        if not ref_date_str:
            ref_date = datetime.date(2019, 1, 19)
        else:
            ref_date = pd.to_datetime(ref_date_str).date()

        year, month = ref_date.year, ref_date.month
        total_days_in_month = calendar.monthrange(year, month)[1]

        # 이번 달 데이터 필터링
        df_month = fin_all[
            (fin_all["real_time"].dt.year == year) &
            (fin_all["real_time"].dt.month == month)
        ].copy()

        if df_month.empty:
            return ui.HTML(f"<p>⚠️ {year}년 {month}월 데이터 없음</p>")

        # 날짜별 생산량 계산
        daily_df = df_month.groupby("date").size().reset_index(name="daily_prod")

        # 하루 평균 및 목표 계산
        total_rows = daily_df["daily_prod"].sum()
        unique_days = daily_df["date"].nunique()
        avg_daily = total_rows / unique_days
        daily_target = avg_daily
        monthly_target = daily_target * total_days_in_month 

        # 누적 계산
        daily_df = daily_df.sort_values("date")
        daily_df["cum_prod"] = daily_df["daily_prod"].cumsum()
        daily_df["achieve_rate(%)"] = (daily_df["cum_prod"] / monthly_target * 100).round(1)

        produced = daily_df[daily_df["date"] <= pd.Timestamp(ref_date)]["daily_prod"].sum()
        achieve_rate = (produced / monthly_target) * 100
        remaining = max(monthly_target - produced, 0)
        last_day = datetime.date(year, month, total_days_in_month)
        remaining_days = max((last_day - ref_date).days, 0)
        daily_need = round(remaining / remaining_days, 1) if remaining_days > 0 else 0

        # 달력 UI
        cal = calendar.Calendar(firstweekday=6)
        month_days = cal.monthdatescalendar(year, month)

        html = ["<table style='width:100%; text-align:center; border-collapse:collapse;'>"]
        html.append("<tr>" + "".join(f"<th>{d}</th>" for d in ["일","월","화","수","목","금","토"]) + "</tr>")

        for week in month_days:
            html.append("<tr>")
            for day in week:
                if day.month != month:
                    html.append("<td style='background:#efefef;'></td>")
                    continue

                row = daily_df[daily_df["date"].dt.date == day]
                if day <= ref_date:
                    if not row.empty:
                        prod = int(row["daily_prod"].values[0])
                        rate = (prod / daily_target) * 100
                        bg = "#c7f9cc" if rate >= 100 else "#fff3b0" if rate >= 80 else "#ffcccb"
                        html.append(
                            f"<td style='border:1px solid #ddd; height:70px; background:{bg};'>"
                            f"<b>{day.day}</b><br>{prod:,}ea<br>({rate:.1f}%)</td>"
                        )
                    else:
                        html.append(f"<td style='border:1px solid #ddd; background:#f9f9f9;'><b>{day.day}</b><br>-</td>")
                else:
                    html.append(
                        f"<td style='border:1px solid #ddd; background:#f0f8ff;'>"
                        f"<b>{day.day}</b><br>{daily_need:,.0f}ea 예정</td>"
                    )
            html.append("</tr>")
        html.append("</table>")

        legend_ui = ui.div(
            ui.div(
                ui.span(style="display:inline-block; width:18px; height:18px; background-color:#c7f9cc; border:1px solid #aaa; margin-right:6px; vertical-align:middle;"),
                ui.span("달성률 100% 이상", style="vertical-align:middle; font-size:13px;"),
                style="display:flex; align-items:center;"
            ),
            ui.div(
                ui.span(style="display:inline-block; width:18px; height:18px; background-color:#fff3b0; border:1px solid #aaa; margin-right:6px; vertical-align:middle;"),
                ui.span("달성률 80% ~ 100%", style="vertical-align:middle; font-size:13px;"),
                style="display:flex; align-items:center;"
            ),
            ui.div(
                ui.span(style="display:inline-block; width:18px; height:18px; background-color:#ffcccb; border:1px solid #aaa; margin-right:6px; vertical-align:middle;"),
                ui.span("달성률 80% 미만", style="vertical-align:middle; font-size:13px;"),
                style="display:flex; align-items:center;"
            ),
            style="""
            display: flex; 
            justify-content: center; 
            gap: 20px; 
            margin-top: 15px; 
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 8px;
            border: 1px solid #eee;
            """
        )

        return ui.div(
                ui.HTML("".join(html)),
                legend_ui,  # ✅ 범례 UI 추가
                ui.br(),
            )

    # =====================================================
    # 🧮 하단 요약 텍스트
    # =====================================================
    @render.text
    def daily_summary():
        ref_date_str = input.ref_date() or None
        if not ref_date_str:
            ref_date = datetime.date(2019, 1, 19)

        else:
            ref_date = pd.to_datetime(ref_date_str).normalize()

        year, month = ref_date.year, ref_date.month
        total_days_in_month = calendar.monthrange(year, month)[1]

        df_month = fin_all[
            (fin_all["real_time"].dt.year == year) &
            (fin_all["real_time"].dt.month == month)
        ].copy()

        if df_month.empty:
            return f"⚠️ {year}년 {month}월 데이터 없음"

        daily_df = df_month.groupby("date").size().reset_index(name="daily_prod")
        total_rows = daily_df["daily_prod"].sum()
        unique_days = daily_df["date"].nunique()
        avg_daily = total_rows / unique_days
        monthly_target = avg_daily * total_days_in_month 

        produced = daily_df[daily_df["date"] <= ref_date]["daily_prod"].sum()
        achieve_rate = round(produced / monthly_target * 100, 1)
        remaining = max(monthly_target - produced, 0)
        last_day = datetime.date(year, month, total_days_in_month)
        remaining_days = max((last_day - ref_date.date()).days, 0)
        daily_need = round(remaining / remaining_days, 1) if remaining_days > 0 else 0

        return (
            f"📆 {ref_date.strftime('%Y년 %m월 %d일')} 기준 누적 생산량: {produced:,.0f}ea "
            f"({achieve_rate:.1f}%) 🎯 남은 목표: {remaining:,.0f}ea / 남은 {remaining_days}일 → "
            f"하루 평균 {daily_need:,.0f}ea 필요"
        )



    @reactive.effect
    @reactive.event(input.ref_date)  # ✅ 날짜 변경 감지
    def _():
        
        # --- 🐞 디버깅 코드 시작 ---
        # 날짜가 바뀔 때마다 현재 페이지 상태를 터미널(콘솔)에 출력합니다.
        current_state = page_state()
        active_tab = input.field_tabs()
        print(f"===== 날짜 변경 감지됨 ===== 현재 페이지: {current_state}")
        # --- 🐞 디버깅 코드 끝 ---

        # ✅ 현재 페이지가 "field"일 때만 팝업 실행
        if current_state == "field" and active_tab == "생산현황":
            
            print(f"===== '{current_state}' 페이지이므로 팝업을 실행합니다.")
            
            # --- 여기부터 기존 팝업 로직 ---
            ref_date_str = input.ref_date() or "2019-01-19"
            ref_date = pd.to_datetime(ref_date_str).normalize()
            year, month = ref_date.year, ref_date.month
            total_days_in_month = calendar.monthrange(year, month)[1]

            # 월별 데이터 필터링
            df_month = fin_all[
                (fin_all["real_time"].dt.year == year)
                & (fin_all["real_time"].dt.month == month)
            ].copy()

            if df_month.empty:
                ui.modal_show(
                    ui.modal(
                        ui.p(f"⚠️ {year}년 {month}월 데이터가 없습니다."),
                        title="⚠️ 알림",
                        easy_close=True,
                        footer=ui.modal_button("닫기"),
                    )
                )
                return

            # 날짜별 생산량 계산
            daily_df = df_month.groupby("date").size().reset_index(name="daily_prod")
            daily_df["date"] = pd.to_datetime(daily_df["date"]).dt.normalize()
            past_df = daily_df[daily_df["date"] <= ref_date]

            if past_df.empty:
                ui.modal_show(
                    ui.modal(
                        ui.p(f"⚠️ {ref_date.strftime('%Y-%m-%d')} 이전 데이터가 없습니다."),
                        title="⚠️ 알림",
                        easy_close=True,
                        footer=ui.modal_button("닫기"),
                    )
                )
                return

            # === 기준일까지의 통계 계산 ===
            total_prod = past_df["daily_prod"].sum()
            avg_daily = past_df["daily_prod"].mean()
            # ... (이하 계산 로직 동일) ...
            total_days_so_far = past_df["date"].nunique()
            monthly_target = avg_daily * total_days_in_month
            achieve_rate = (total_prod / monthly_target) * 100
    
            remaining = max(monthly_target - total_prod, 0)
            last_day = datetime.date(year, month, total_days_in_month)
            remaining_days = max((last_day - ref_date.date()).days, 0)
            daily_need = round(remaining / remaining_days, 1) if remaining_days > 0 else 0
    
            # 최고 / 최저 생산일
            best_row = past_df.loc[past_df["daily_prod"].idxmax()]
            worst_row = past_df.loc[past_df["daily_prod"].idxmin()]
            best_day = best_row["date"]
            worst_day = worst_row["date"]
            best_val = best_row["daily_prod"]
            worst_val = worst_row["daily_prod"]

            # === HTML 팝업 구성 ===
            html = f"""
            <div style='font-size:15px; line-height:1.6;'>
                <h4>📘 {year}년 {month}월 생산 계획 달성률 보고서</h4>
                <hr>
                <p>📅 기준일: <b>{ref_date.strftime('%Y-%m-%d')}</b></p>
                <ul>
                    <li>누적 생산량: <b>{total_prod:,.0f}ea</b></li>
                    <li>평균 일일 생산량: <b>{avg_daily:,.0f}ea</b></li>
                    <li>달성률: <b>{achieve_rate:.1f}%</b></li>
                    <li>남은 목표: <b>{remaining:,.0f}ea</b></li>
                    <li>남은 기간: <b>{remaining_days}일</b></li>
                    <li>하루 평균 필요 생산량: <b>{daily_need:,.0f}ea</b></li>
                </ul>
                <hr>
                <p>
                🏆 최고 생산일: <b>{best_day.strftime('%Y-%m-%d')}</b> ({best_val:,}ea)<br>
                ⚠️ 최저 생산일: <b>{worst_day.strftime('%Y-%m-%d')}</b> ({worst_val:,}ea)
                </p>
            </div>
            """

            # 팝업 표시
            ui.modal_show(
                ui.modal(
                    ui.HTML(html),
                    title=f"📊 {year}년 {month}월 보고서",
                    easy_close=True,
                    footer=ui.modal_button("닫기"),
                    size="xl",
                )
            )
        
        else:
             print(f"===== '{current_state}' 페이지이므로 팝업을 표시하지 않습니다.")













    # ======== 📈 데이터 분석 탭 ========
   # --- 생산계획 탭 서버 로직 ---
    @render.ui
    def mold_inputs():
        if not codes: return ui.p("금형코드 데이터 없음")
        inputs = []
        for code in codes[:-1]:
            inputs.append(ui.input_numeric(f"target_{code}", ui.HTML(f"<span style='color:{mold_colors.get(code, '#000')}; font-weight:bold;'>금형코드 {code}</span>"), value=0, min=0, step=100))
        return ui.div(*inputs)
    
    DATA_PATH = pathlib.Path("./data/train_raw.csv")
    try:
        df_raw = pd.read_csv(DATA_PATH)
        print(f"✅ 데이터 로드 완료: {df_raw.shape}")
    except Exception as e:
        print("⚠️ 데이터 로드 실패:", e)
        df_raw = pd.DataFrame()

    @render.text
    def remaining_qty():
        if not codes: return ""
        total_target = input.monthly_target() or 0
        user_sum = sum(input[f"target_{code}"]() or 0 for code in codes[:-1])
        remaining = total_target - user_sum
        if user_sum > total_target:
            return f"⚠️ 목표 초과: {user_sum-total_target:,}개"
        return f"남은 생산량 ({last_code}): {remaining:,}개"

    @output
    @render.data_frame
    def mold_summary_table():
        if mold_summary.empty: return pd.DataFrame()
        df = mold_summary.rename(columns={
            "mold_code": "금형코드", "avg_facility_cycleTime": "평균사이클(초)",
            "daily_capacity": "일일생산능력", "min_prod": "최소일일생산량",
            "max_prod": "최대일일생산량", "avg_prod": "평균일일생산량"
        })
        return df.round(2)

    plan_df = reactive.Value(pd.DataFrame())
    @reactive.effect
    @reactive.event(input.run_plan)
    def _():
        if not codes: 
            plan_df.set(pd.DataFrame())
            return
        
        total_target = input.monthly_target() or 0
        year, month = int(input.year()), int(input.month())
        targets = {code: input[f"target_{code}"]() or 0 for code in codes[:-1]}
        user_sum = sum(targets.values())
        targets[last_code] = max(total_target - user_sum, 0)
        
        if sum(targets.values()) == 0: # If all targets are 0, distribute by capacity
            total_capacity = mold_summary["daily_capacity"].sum()
            if total_capacity > 0:
                for code in codes:
                    ratio = mold_summary.loc[mold_summary.mold_code == code, "daily_capacity"].iloc[0] / total_capacity
                    targets[code] = int(total_target * ratio)

        _, last_day = calendar.monthrange(year, month)
        schedule = []
        # (This is a simplified scheduling logic)
        for day in range(1, last_day + 1):
            for code in codes:
                daily_plan = int(targets[code] / last_day) if last_day > 0 else 0
                schedule.append({"date": datetime.date(year, month, day), "mold_code": code, "plan_qty": daily_plan})
        plan_df.set(pd.DataFrame(schedule))


    DATA_PATH = pathlib.Path("./data/train_raw.csv")
    try:
        df_raw = pd.read_csv(DATA_PATH)
        print(f"✅ 데이터 로드 완료: {df_raw.shape}")
    except Exception as e:
        print("⚠️ 데이터 로드 실패:", e)
        df_raw = pd.DataFrame()

    

    # -------- UI 내용 --------

    plan_df = reactive.Value(pd.DataFrame())

    @reactive.effect
    @reactive.event(input.run_plan)
    def _make_plan_df():
        total_target = input.monthly_target()
        year, month = int(input.year()), int(input.month())

        targets = {}
        user_sum = 0
        for code in codes[:-1]:
            qty = input[f"target_{code}"]()
            targets[code] = qty
            user_sum += qty
        targets[last_code] = max(total_target - user_sum, 0)

        if sum(targets.values()) == 0:
            for _, row in mold_summary.iterrows():
                code = row["mold_code"]
                ratio = row["daily_capacity"] / mold_summary["daily_capacity"].sum()
                targets[code] = int(total_target * ratio)

        _, last_day = calendar.monthrange(year, month)

        weeks = ["3종류", "2종류", "3종류", "2종류"]
        codes_3, codes_2 = codes[:3], codes[3:5]

        schedule = []
        day_counter = 0
        for week_num, mode in enumerate(weeks, start=1):
            if day_counter >= last_day:
                break
            selected = codes_3 if mode == "3종류" else codes_2
            daily_sum = sum(
                mold_summary.loc[mold_summary["mold_code"] == c, "daily_capacity"].values[0]
                for c in selected
            )
            ratios = {
                c: mold_summary.loc[mold_summary["mold_code"] == c, "daily_capacity"].values[0] / daily_sum
                for c in selected
            }
            for day in range(1, 8):
                day_counter += 1
                if day_counter > last_day:
                    break
                for code in codes:
                    if code in selected:
                        total_target_code = targets[code]
                        daily_plan = int((total_target_code / last_day) * ratios[code] * len(selected))
                    else:
                        daily_plan = 0
                    schedule.append({
                        "date": datetime.date(year, month, day_counter),
                        "week": week_num,
                        "day": day,
                        "mold_code": code,
                        "plan_qty": daily_plan
                    })

        df = pd.DataFrame(schedule)
        plan_df.set(df)   # ✅ reactive.Value 객체 업데이트



    # 달력형 뷰 (버튼 클릭 시에만 갱신)
    @render.ui
    @reactive.event(input.run_plan)
    def calendar_view():
        df = plan_df()
        year, month = int(input.year()), int(input.month())
        calendar.setfirstweekday(calendar.SUNDAY)
        days_kr = ["일", "월", "화", "수", "목", "금", "토"]
        cal = calendar.monthcalendar(year, month)

        html = '<div style="display:grid; grid-template-columns: 80px repeat(7, 1fr); gap:4px;">'
        html += '<div></div>' + "".join([f"<div style='font-weight:bold; text-align:center;'>{d}</div>" for d in days_kr])

        for w_i, week in enumerate(cal, start=1):
            html += f"<div style='font-weight:bold;'>{w_i}주</div>"
            for d in week:
                if d == 0:
                    html += "<div style='border:1px solid #ccc; min-height:80px; background:#f9f9f9;'></div>"
                else:
                    cell_date = datetime.date(year, month, d)
                    cell_df = df[df["date"] == cell_date]

                    cell_html = ""
                    for _, r in cell_df.iterrows():
                        if r["plan_qty"] > 0:
                            code = str(r["mold_code"])

                            # 세팅값 조회
                            row = setting_df[setting_df["mold_code"] == code]
                            if row.empty:
                                tooltip_html = "<p>세팅값 없음</p>"
                            else:
                                settings = row.to_dict("records")[0]

                            # HTML 표 생성
                            rows_html = "".join([
                                f"<tr><td>{label_map.get(k, k)}</td><td>{f'{v:.2f}' if isinstance(v, (int, float)) else v}</td></tr>"
                                for k, v in settings.items() if k != "mold_code"
                            ])
                            tooltip_html = f"""
                            <table class="table table-sm table-bordered" style="font-size:11px; background:white; color:black;">
                                <thead><tr><th>변수</th><th>값</th></tr></thead>
                                <tbody>{rows_html}</tbody>
                            </table>
                            """

                            # 툴팁 적용
                            cell_html += str(
                                ui.tooltip(
                                    ui.span(
                                        f"{code}: {r['plan_qty']}",
                                        style=f"color:{mold_colors[code]}; font-weight:bold;"
                                    ),
                                    ui.HTML(tooltip_html),  # 표 형태 툴팁
                                    placement="right"
                                )
                            ) + "<br>"

                    html += f"<div style='border:1px solid #ccc; min-height:80px; padding:4px; font-size:12px;'>{d}<br>{cell_html}</div>"
        html += "</div>"

        notes_ui = ui.div(
            ui.p("※ 몰드코드에 따른 공정 조건을 확인하세요!", style="margin-bottom: 5px;"),
            ui.p("※ 선택한 연월의 금형 계획과 공정 조건을 확인 가능 합니다. 몰드별 최대 생산량을 고려한 조건임을 유의하세요.", style="margin-bottom: 0;"),
            style="""
            margin-top: 15px;
            font-size: 13px;
            color: #495057;
            background-color: #f8f9fa;
            border: 1px dashed #ced4da;
            border-radius: 8px;
            padding: 12px 15px;
            """
        )

        return ui.div(
             ui.HTML(html),
                notes_ui  # ✅ 안내 문구 추가
             )
    


    @output
    @render.plot
    def mold_plot():
        fig, ax = plt.subplots(figsize=(12, 6))
        if not pivot_count.empty:
            pivot_count.plot(kind="bar", stacked=True, ax=ax, color=[mold_colors.get(str(int(c))) for c in pivot_count.columns])
        ax.set_title("날짜별 금형 코드 생산 추이")
        ax.set_xlabel("날짜")
        ax.set_ylabel("생산 개수")
        ax.legend(title="금형 코드")
        plt.tight_layout()
        return fig

    @reactive.effect
    @reactive.event(input.show_modal)
    def _():
        ui.modal_show(ui.modal(ui.output_plot("mold_plot"), title="날짜별 금형 코드 생산 추이", size="xl", easy_close=True))

    
    # ===== 실시간 스트리밍 로직 =====
    @output
    @render.ui
    def stream_status():
        color = "green" if is_streaming() else "gray"
        return ui.span(f"{'🟢' if is_streaming() else '🔴'}", style=f"color:{color};")

    @output
    @render.plot
    def stream_plot():
        df = current_data()
        fig, ax = plt.subplots(figsize=(10, 4))
        if df.empty:
            ax.text(0.5, 0.5, "▶ Start Streaming", ha="center", va="center", fontsize=14)
            ax.set_xticks([]); ax.set_yticks([])
            return fig
        for col in display_cols:
            ax.plot(df[col].values, label=col)
        ax.legend(); ax.grid(True)
        ax.set_title("Real Time Sensor Data")
        return fig
    
    

    
   
        
    # ===== 품질 모니터링용 관리도 출력 =====
    # @output
    # @render.plot
    # @reactive.calc
    # def xr_chart_quality():
    #     df = current_data.get()
    #     if df is None or df.empty:
    #         fig, ax = plt.subplots()
    #         ax.axis("off")
    #         ax.text(0.5, 0.5, "데이터 수신 대기 중...", ha="center", va="center")
    #         return fig
    
    #     var = input.spc_var() or "cast_pressure"
    #     if var not in df.columns:
    #         fig, ax = plt.subplots()
    #         ax.axis("off")
    #         ax.text(0.5, 0.5, f"{var} 데이터 없음", ha="center", va="center")
    #         return fig
    
    #     xbar, R, limits = calc_xr_chart(df, var=var)
    #     fig = plot_xr_chart_matplotlib(xbar, R, limits)
    #     return fig


    # @output
    # @render.plot
    # @reactive.calc
    # def p_chart_quality():
    #     df = current_data.get()
    #     if df is None or df.empty:
    #         fig, ax = plt.subplots()
    #         ax.axis("off")
    #         ax.text(0.5, 0.5, "데이터 수신 대기 중...", ha="center", va="center")
    #         return fig

    #     if "passorfail" not in df.columns:
    #         fig, ax = plt.subplots()
    #         ax.axis("off")
    #         ax.text(0.5, 0.5, "passorfail 데이터 없음", ha="center", va="center")
    #         return fig

    #     p_bar, UCL, LCL = calc_p_chart(df, var="passorfail")
    #     return plot_p_chart_matplotlib(p_bar, UCL, LCL)
    
    # ============================================================
    # 🧭 다변량 관리도 (Hotelling’s T²) 계산 함수
    # ============================================================
    def calc_hotelling_t2(df, cols):
        """Hotelling’s T² (고정 UCL 적용)"""
        df = df.dropna(subset=cols)
        if len(df) == 0:
            return None, None, None

        X = df[cols].values

        # ✅ 공정별 baseline 매칭
        if set(cols) == set(melting_cols):
            mean, inv_cov, UCL = MEAN_MELT, INV_MELT, UCL_MELT
        elif set(cols) == set(filling_cols):
            mean, inv_cov, UCL = MEAN_FILL, INV_FILL, UCL_FILL
        elif all(c in cooling_cols for c in cols):
            mean, inv_cov, UCL = MEAN_COOL, INV_COOL, UCL_COOL
        elif set(cols) == set(speed_cols):
            mean, inv_cov, UCL = MEAN_SPEED, INV_SPEED, UCL_SPEED
        elif set(cols) == set(quality_cols):
            mean, inv_cov, UCL = MEAN_QUAL, INV_QUAL, UCL_QUAL
        else:
            print("⚠ 알 수 없는 컬럼 세트, 실시간 UCL 계산으로 fallback")
            mean = np.mean(X, axis=0)
            cov = np.cov(X, rowvar=False)
            inv_cov = np.linalg.pinv(cov)
            from scipy.stats import f
            n, p = len(df), len(cols)
            UCL = p * (n - 1) * (n + 1) / (n * (n - p)) * f.ppf(0.99, p, n - p)

        # ✅ T² 계산
        T2 = np.array([(x - mean) @ inv_cov @ (x - mean).T for x in X])
        return df.index, T2, UCL
    
    
    # ============================================================
    # ✅ X-R 관리도 계산 (고정형 그룹, xr_chart() 호환 버전)
    # ============================================================
    def calc_realtime_xr(df, col, baseline_xr, subgroup_size=5):
        """
        고정된 서브그룹 단위로 Xbar-R 관리도를 계산하고,
        xr_chart()에서 사용 가능한 (idx, xbars, ranges, baseline) 형태로 반환
        """

        # 데이터가 부족하면 None 반환
        if len(df) < subgroup_size:
            return None

        # 완전한 그룹 수 계산
        num_groups = len(df) // subgroup_size
        used_df = df.iloc[:num_groups * subgroup_size].copy()
        groups = np.array_split(used_df[col].values, num_groups)

        # 그룹별 평균과 범위 계산
        xbars = [g.mean() for g in groups]
        ranges = [g.max() - g.min() for g in groups]
        idx = list(range(1, num_groups + 1))  # x축 = 그룹 번호

        # 중심선 계산
        xbar_bar = np.mean(xbars)
        r_bar = np.mean(ranges)

        # 표준 관리도 상수 (n=5 기준)
        A2, D3, D4 = 0.577, 0, 2.115

        # UCL/LCL 계산
        UCLx = xbar_bar + A2 * r_bar
        LCLx = xbar_bar - A2 * r_bar
        UCLr = D4 * r_bar
        LCLr = D3 * r_bar

        # baseline dict 구성 (xr_chart에서 baseline["UCLx"] 식으로 사용)
        baseline = {
            "UCLx": UCLx, "LCLx": LCLx, "Xbar_bar": xbar_bar,
            "UCLr": UCLr, "LCLr": LCLr, "R_bar": r_bar
        }

        return idx, xbars, ranges, baseline


    # ───────────────────────────────
    # 공통 함수: 관리도 그리기
    # ───────────────────────────────
    def plot_t2_chart(index, T2, UCL, title):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3))
        if T2 is None or len(T2) == 0:
            ax.text(0.5, 0.5, "데이터 부족", ha="center", va="center")
            ax.axis("off")
            return fig

        # 데이터 선
        ax.plot(index, T2, marker="o", color="steelblue", label="T²", alpha=0.8)
        ax.axhline(UCL, color="red", linestyle="--", label="UCL(99%)")

        # y축 범위 반영 후 UCL 이상 배경 붉게 표시
        ax.figure.canvas.draw()
        y_min, y_max = ax.get_ylim()
        ax.axhspan(UCL, y_max, color="lightcoral", alpha=0.25, zorder=0)

        ax.set_title(title)
        ax.set_ylabel("T²")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    
    
    # ============================================================
    # 🔹 X-R 관리도 시각화 함수
    # ============================================================
    def plot_xr_chart(idx, xbars, ranges, baseline, col):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(6, 5), sharex=True)

        # Xbar Chart
        axes[0].plot(idx, xbars, marker="o", color="steelblue")
        axes[0].axhline(baseline["UCLx"], color="red", linestyle="--", label="UCLx")
        axes[0].axhline(baseline["LCLx"], color="red", linestyle="--", label="LCLx")
        axes[0].axhline(baseline["Xbar_bar"], color="green", linestyle=":")
        axes[0].set_title(f"{col} – X-bar 관리도")
        axes[0].legend(); axes[0].grid(True, alpha=0.3)

        # R Chart
        axes[1].plot(idx, ranges, marker="o", color="darkorange")
        axes[1].axhline(baseline["UCLr"], color="red", linestyle="--", label="UCLr")
        axes[1].axhline(baseline["LCLr"], color="red", linestyle="--", label="LCLr")
        axes[1].axhline(baseline["R_bar"], color="green", linestyle=":")
        axes[1].set_title(f"{col} – R 관리도")
        axes[1].legend(); axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        plt.close(fig)
        return fig


    # ───────────────────────────────
    # 공통 함수: 로그 데이터프레임 생성
    # ───────────────────────────────
    def make_overlog(df, cols):
        idx, T2, UCL = calc_hotelling_t2(df, cols)
        over_mask = T2 > UCL
        over_data = df.loc[over_mask, cols].copy()
        over_data["T2"] = T2[over_mask]

        if over_data.empty:
            return pd.DataFrame({"메시지": ["모든 데이터가 UCL 이하입니다."]})
        else:
            over_data = over_data.reset_index()

            # 🔹 시간 컬럼 추가
            if "registration_time" in df.columns:
                over_data["시간"] = df.loc[over_mask, "registration_time"].values
            elif "datetime" in df.columns:
                over_data["시간"] = df.loc[over_mask, "datetime"].values

            # 🔹 한글 컬럼명 매핑 (공정별 전체 반영)
            col_name_map = {
                "T2": "T²",
                # 용융 단계
                "molten_temp": "용융 온도",
                "molten_volume": "주입한 금속 양",

                # 충진 단계
                "sleeve_temperature": "주입 관 온도",
                "EMS_operation_time": "전자 교반(EMS) 가동 시간",
                "low_section_speed": "하위 구간 주입 속도",
                "high_section_speed": "상위 구간 주입 속도",
                "cast_pressure": "주입 압력",

                # 냉각 단계
                "upper_mold_temp1": "상부1 금형 온도",
                "upper_mold_temp2": "상부2 금형 온도",
                # "upper_mold_temp3": "상부3 금형 온도",
                "lower_mold_temp1": "하부1 금형 온도",
                "lower_mold_temp2": "하부2 금형 온도",
                # "lower_mold_temp3": "하부3 금형 온도",
                "Coolant_temperature": "냉각수 온도",

                # 생산 속도
                "facility_operation_cycleTime": "장비 전체 사이클 시간",
                "production_cycletime": "실제 생산 사이클 시간",

                # 제품 테스트
                "biscuit_thickness": "주조물 두께",
                "physical_strength": "제품 강도",

                # 공통
                "시간": "시간",
            }

            # 🔹 표시 컬럼 순서
            display_cols = ["시간", "T2"] + cols if "시간" in over_data.columns else ["T2"] + cols

            # 🔹 매핑 적용
            over_data = over_data[display_cols].round(3)
            over_data.rename(columns=col_name_map, inplace=True)

            return over_data.tail(10)

    # ===============================
    # 🔹 공통 에러 처리용 함수
    # ===============================
    def make_placeholder_chart(title):
        """데이터 없을 때 표시되는 안내 그래프"""
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5,
                f"📡 {title}\n데이터 수집 중입니다. 잠시만 기다려주세요.",
                ha="center", va="center", color="gray", fontsize=11)
        ax.axis("off")
        return fig
    
    # ───────────────────────────────
    # 🔹 용융 단계
    # ───────────────────────────────
    @output
    @render.plot
    def mv_chart_melting():
        try:
            df = current_data().tail(50)
            cols = ["molten_temp", "molten_volume"]
            idx, T2, UCL = calc_hotelling_t2(df, cols)
            return plot_t2_chart(idx, T2, UCL, "용융 단계")
        except Exception:
            return make_placeholder_chart("용융 단계")


    @output
    @render.table
    def mv_log_melting():
        try:
            df = current_data().tail(50)
            cols = ["molten_temp", "molten_volume"]
            return make_overlog(df, cols)
        except Exception:
            return pd.DataFrame({"메시지": ["데이터 수집 중입니다. 잠시만 기다려주세요."]})

    # ───────────────────────────────
    # 🔹 충진 단계
    # ───────────────────────────────
    @output
    @render.plot
    def mv_chart_filling():
        try:
            df = current_data().tail(50)
            cols = ["sleeve_temperature", "EMS_operation_time",
                    "low_section_speed", "high_section_speed", "cast_pressure"]
            idx, T2, UCL = calc_hotelling_t2(df, cols)
            return plot_t2_chart(idx, T2, UCL, "충진 단계")
        except Exception:
            return make_placeholder_chart("충진 단계")


    @output
    @render.table
    def mv_log_filling():
        try:
            df = current_data().tail(50)
            cols = ["sleeve_temperature", "EMS_operation_time",
                    "low_section_speed", "high_section_speed", "cast_pressure"]
            return make_overlog(df, cols)
        except Exception:
            return pd.DataFrame({"메시지": ["데이터 수집 중입니다. 잠시만 기다려주세요."]})

    # ───────────────────────────────
    # 🔹 냉각 단계
    # ───────────────────────────────
    @output
    @render.plot
    def mv_chart_cooling():
        try:
            df = current_data().tail(50)

            # ✅ (추가) 한글 컬럼명을 영어로 자동 되돌리기
            reverse_map = {v: k for k, v in label_map.items()}
            df = df.rename(columns=reverse_map)

            cols = [
                "upper_mold_temp1", "upper_mold_temp2", 
                # "upper_mold_temp3",
                "lower_mold_temp1", "lower_mold_temp2", 
                "Coolant_temperature"
            ]

            # ✅ 실제 존재하는 컬럼만 필터
            cols = [c for c in cols if c in df.columns]

            if len(cols) < 2:
                print("⚠ 냉각 단계 컬럼 부족:", cols)
                return make_placeholder_chart("냉각 단계")

            idx, T2, UCL = calc_hotelling_t2(df, cols)
            return plot_t2_chart(idx, T2, UCL, "냉각 단계")

        except Exception as e:
            print("❌ 냉각 단계 에러:", e)
            return make_placeholder_chart("냉각 단계")


    @output
    @render.table
    def mv_log_cooling():
        try:
            df = current_data().tail(50)
            reverse_map = {v: k for k, v in label_map.items()}
            df = df.rename(columns=reverse_map)   # ✅ inplace=False로 안전하게
    
            cols = [
                "upper_mold_temp1", "upper_mold_temp2", 
                # "upper_mold_temp3",
                "lower_mold_temp1", "lower_mold_temp2",
                "Coolant_temperature"
            ]
    
            available_cols = [c for c in cols if c in df.columns]
    
            if not available_cols:
                return pd.DataFrame({"메시지": ["냉각 단계 데이터가 존재하지 않습니다."]})
    
            log_df = make_overlog(df, available_cols)
    
            if log_df is None or log_df.empty:
                return pd.DataFrame({"메시지": ["모든 데이터가 UCL 이하입니다."]})
            return log_df
    
        except Exception as e:
            import traceback
            print("❌ 냉각 단계 로그 생성 오류:", e)
            traceback.print_exc()
            return pd.DataFrame({"메시지": ["데이터 수집 중입니다. 잠시만 기다려주세요."]})


    # ───────────────────────────────
    # 🔹 생산 속도
    # ───────────────────────────────
    @output
    @render.plot
    def mv_chart_speed():
        try:
            df = current_data().tail(50)
            cols = ["facility_operation_cycleTime", "production_cycletime"]
            idx, T2, UCL = calc_hotelling_t2(df, cols)
            return plot_t2_chart(idx, T2, UCL, "생산 속도")
        except Exception:
            return make_placeholder_chart("생산 속도")


    @output
    @render.table
    def mv_log_speed():
        try:
            df = current_data().tail(50)
            cols = ["facility_operation_cycleTime", "production_cycletime"]
            return make_overlog(df, cols)
        except Exception:
            return pd.DataFrame({"메시지": ["데이터 수집 중입니다. 잠시만 기다려주세요."]})

    # ───────────────────────────────
    # 🔹 제품 테스트
    # ───────────────────────────────
    @output
    @render.plot
    def mv_chart_quality():
        try:
            df = current_data().tail(50)
            cols = ["biscuit_thickness", "physical_strength"]
            idx, T2, UCL = calc_hotelling_t2(df, cols)
            return plot_t2_chart(idx, T2, UCL, "제품 테스트")
        except Exception:
            return make_placeholder_chart("제품 테스트")
    
    # ============================================================
    # 🔹 X-R 관리도 출력 (Shiny 렌더) — 슬라이딩형 버전
    # ============================================================
    @output
    @render.plot
    def xr_chart():
        import matplotlib.pyplot as plt

        # ✅ 전체 누적 데이터 사용 (tail 제한 없음)
        df = current_data()
        stage = input.xr_select()

        if stage not in XR_GROUPS:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "단계를 선택하세요.", ha="center", va="center")
            plt.close(fig)
            return fig

        cols = XR_GROUPS[stage]
        n = len(cols)
        n_rows = max(n, 2)
        fig, axes = plt.subplots(n_rows, 2, figsize=(12, 8))
        if n_rows == 1:
            axes = np.array([axes])

        MAX_POINTS = 30       # ✅ 최근 30개 그룹만 표시
        SUBGROUP_SIZE = 5     # ✅ 그룹 크기 고정

        for i, (col, label) in enumerate(cols):
            result = calc_realtime_xr(df, col, BASELINE_XR, subgroup_size=SUBGROUP_SIZE)
            ax1, ax2 = axes[i]

            if result is None:
                for ax, title in zip([ax1, ax2], [f"{label} – Xbar", f"{label} – R"]):
                    ax.set_title(f"{title} (데이터 없음)", pad=15)
                    ax.axis("off")
                continue

            idx, xbars, ranges, baseline = result
            total_points = len(idx)

            # ✅ 최근 30개만 표시 (슬라이딩 윈도우)
            if total_points > MAX_POINTS:
                idx = idx[-MAX_POINTS:]
                xbars = xbars[-MAX_POINTS:]
                ranges = ranges[-MAX_POINTS:]

            # ✅ Xbar 관리도
            ax1.plot(idx, xbars, "o-", color="steelblue")
            ax1.axhline(baseline["UCLx"], color="red", ls="--")
            ax1.axhline(baseline["LCLx"], color="red", ls="--")
            ax1.axhline(baseline["Xbar_bar"], color="green", ls=":")
            ax1.set_title(f"{label} – Xbar")
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(max(1, total_points - MAX_POINTS + 1) - 0.5, total_points + 0.5)

            # ✅ R 관리도
            ax2.plot(idx, ranges, "o-", color="darkorange")
            ax2.axhline(baseline["UCLr"], color="red", ls="--")
            ax2.axhline(baseline["LCLr"], color="red", ls="--")
            ax2.axhline(baseline["R_bar"], color="green", ls=":")
            ax2.set_title(f"{label} – R")
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(max(1, total_points - MAX_POINTS + 1) - 0.5, total_points + 0.5)

        plt.tight_layout()
        plt.close(fig)
        return fig

    
    # ============================================================
    # 🔹 용융 단계 X-R 관리도
    # ============================================================
    @output
    @render.plot
    def xr_chart_melting():
        try:
            df = current_data().tail(200)
            cols = ["molten_temp", "molten_volume"]
            labels = ["용융 온도", "주입 금속량"]
            return xr_chart("용융 단계", df, cols, labels, BASELINE_XR)
        except Exception as e:
            print("❌ XR 용융 단계 오류:", e)
            return make_placeholder_chart("용융 단계 X-R 관리도")


    # ============================================================
    # 🔹 충진 단계 X-R 관리도
    # ============================================================
    @output
    @render.plot
    def xr_chart_filling():
        try:
            df = current_data().tail(200)
            cols = [
                "sleeve_temperature", "EMS_operation_time",
                "low_section_speed", "high_section_speed", "cast_pressure"
            ]
            labels = ["슬리브 온도", "EMS 가동시간", "하위속도", "상위속도", "주입압력"]
            return xr_chart("충진 단계", df, cols, labels, BASELINE_XR)
        except Exception as e:
            print("❌ XR 충진 단계 오류:", e)
            return make_placeholder_chart("충진 단계 X-R 관리도")


    # ============================================================
    # 🔹 냉각 단계 X-R 관리도
    # ============================================================
    @output
    @render.plot
    def xr_chart_cooling():
        try:
            df = current_data().tail(200)
            reverse_map = {v: k for k, v in label_map.items()}
            df = df.rename(columns=reverse_map)

            cols = [
                "upper_mold_temp1", "upper_mold_temp2",
                "lower_mold_temp1", "lower_mold_temp2",
                "Coolant_temperature"
            ]
            cols = [c for c in cols if c in df.columns]
            labels = ["상부1 금형", "상부2 금형", "하부1 금형", "하부2 금형", "냉각수 온도"]

            if len(cols) < 2:
                return make_placeholder_chart("냉각 단계 X-R 관리도")

            return xr_chart("냉각 단계", df, cols, labels, BASELINE_XR)
        except Exception as e:
            print("❌ XR 냉각 단계 오류:", e)
            return make_placeholder_chart("냉각 단계 X-R 관리도")


    # ============================================================
    # 🔹 생산 속도 X-R 관리도
    # ============================================================
    @output
    @render.plot
    def xr_chart_speed():
        try:
            df = current_data().tail(200)
            cols = ["facility_operation_cycleTime", "production_cycletime"]
            labels = ["설비 사이클", "생산 사이클"]
            return xr_chart("생산 속도", df, cols, labels, BASELINE_XR)
        except Exception as e:
            print("❌ XR 속도 단계 오류:", e)
            return make_placeholder_chart("생산 속도 X-R 관리도")


    # ============================================================
    # 🔹 제품 테스트 X-R 관리도
    # ============================================================
    @output
    @render.plot
    def xr_chart_quality():
        try:
            df = current_data().tail(200)
            cols = ["biscuit_thickness", "physical_strength"]
            labels = ["비스킷 두께", "제품 강도"]
            return xr_chart("제품 테스트", df, cols, labels, BASELINE_XR)
        except Exception as e:
            print("❌ XR 품질 단계 오류:", e)
            return make_placeholder_chart("제품 테스트 X-R 관리도")



    # ============================================================
    # 🔹 실시간 관리도 갱신 주기 제어
    # ============================================================
    @reactive.effect
    def refresh_control_charts():
        # 🔸 주기(초) 설정 — 2.0이면 2초마다 다시 그림
        invalidate_later(5.0)
        pass


    @output
    @render.table
    def mv_log_quality():
        try:
            df = current_data().tail(50)
            cols = ["biscuit_thickness", "physical_strength"]
            return make_overlog(df, cols)
        except Exception:
            return pd.DataFrame({"메시지": ["데이터 수집 중입니다. 잠시만 기다려주세요."]})
    
    
    @output
    @render.ui
    def mv_group_ui():
        group = input.mv_group()
    
        if group == "공정 관리":
            return ui.layout_columns(
                ui.card(
                    ui.output_plot("mv_chart_melting"),
                    ui.div(
                        ui.output_table("mv_log_melting"),
                        style=(
                            "max-height:200px;"
                            "overflow-y:auto;"
                            "overflow-x:auto;"
                            "white-space:nowrap;"
                            "table-layout:fixed;"
                            "word-break:keep-all;"
                            "border-top:1px solid #ccc;"
                        )
                    ),
                ),
                ui.card(
                    ui.output_plot("mv_chart_filling"),
                    ui.div(
                        ui.output_table("mv_log_filling"),
                        style=(
                            "max-height:200px;"
                            "overflow-y:auto;"
                            "overflow-x:auto;"
                            "white-space:nowrap;"
                            "table-layout:fixed;"
                            "word-break:keep-all;"
                            "border-top:1px solid #ccc;"
                        )
                    ),
                ),
                ui.card(
                    ui.output_plot("mv_chart_cooling"),
                    ui.div(
                        ui.output_table("mv_log_cooling"),
                        style=(
                            "max-height:200px;"
                            "overflow-y:auto;"
                            "overflow-x:auto;"
                            "white-space:nowrap;"
                            "table-layout:fixed;"
                            "word-break:keep-all;"
                            "border-top:1px solid #ccc;"
                        )
                    ),
                ),
                col_widths=[4, 4, 4]
            )
    
        elif group == "생산 관리":
            return ui.layout_columns(
                ui.card(
                    ui.output_plot("mv_chart_speed"),
                    ui.div(
                        ui.output_table("mv_log_speed"),
                        style=(
                            "max-height:200px;"
                            "overflow-y:auto;"
                            "overflow-x:auto;"
                            "white-space:nowrap;"
                            "table-layout:fixed;"
                            "word-break:keep-all;"
                            "border-top:1px solid #ccc;"
                        )
                    ),
                ),
                col_widths=[12]
            )
    
        elif group == "제품 관리":
            return ui.layout_columns(
                ui.card(
                    ui.output_plot("mv_chart_quality"),
                    ui.div(
                        ui.output_table("mv_log_quality"),
                        style=(
                            "max-height:200px;"
                            "overflow-y:auto;"
                            "overflow-x:auto;"
                            "white-space:nowrap;"
                            "table-layout:fixed;"
                            "word-break:keep-all;"
                            "border-top:1px solid #ccc;"
                        )
                    ),
                ),
                col_widths=[12]
            )



    # ---------- 버튼 렌더링 ----------
    @output
    @render.ui
    def stream_buttons():
        """스트리밍 상태에 따라 버튼 표시 전환 (Font Awesome 아이콘 + 가로 배속 표시)"""
        btn_base = (
            "min-width:32px; height:32px; display:flex; align-items:center; justify-content:center;"
            "border:none; border-radius:6px; font-size:14px; color:white; font-weight:bold;"
            "box-shadow:0 1px 3px rgba(0,0,0,0.15); padding:0 6px;"
            "transition:all 0.2s ease;"
        )

        # 현재 배속 표시
        speed = stream_speed()
        speed_map = {2.0: "1x", 1.0: "2x", 0.5: "4x", 0.1: "20x", 0.05: "40x"}
        label = speed_map.get(speed, "1x")

        # 색상 (스트리밍 중: 파랑 / 정지 시: 회색)
        fast_color = "#60a5fa" if is_streaming() else "#9ca3af"

        return ui.div(
            {"style": "display:flex; gap:6px; align-items:center;"},
            # ▶ / ⏸ 버튼
            ui.input_action_button(
                "pause_stream" if is_streaming() else "start_stream",
                ui.HTML(
                    '<i class="fa-solid fa-pause"></i>'
                    if is_streaming()
                    else '<i class="fa-solid fa-play"></i>'
                ),
                style=btn_base + (
                    "background-color:#fbbf24;" if is_streaming() else "background-color:#f59e0b;"
                ),
                title="재생/일시정지",
            ),

            # ✅ 빨리감기 버튼 (Font Awesome 아이콘 + 배속 가로 배치)
            ui.input_action_button(
                "fast_stream",
                ui.HTML(
                    f"<div style='display:flex; align-items:center; gap:3px;'>"
                    f"<i class='fa-solid fa-forward'></i>"
                    f"<span style='font-size:11px;'>{label}</span>"
                    f"</div>"
                ),
                style=btn_base + f"background-color:{fast_color}; "
                                f"opacity:{1 if is_streaming() else 0.5}; "
                                f"cursor:{'pointer' if is_streaming() else 'not-allowed'};",
                disabled=not is_streaming(),
                title="빨리감기",
            ),

            # 🔄 초기화 버튼
            ui.input_action_button(
                "reset_stream",
                ui.HTML('<i class="fa-solid fa-rotate-right"></i>'),
                style=btn_base + "background-color:#d97706;",
                title="리셋",
            ),
        )

    # ---------- 버튼 동작 ----------
    @reactive.effect
    @reactive.event(input.start_stream)
    def _start_stream():
        is_streaming.set(True)

    @reactive.effect
    @reactive.event(input.pause_stream)
    def _pause_stream():
        is_streaming.set(False)

    @reactive.effect
    @reactive.event(input.reset_stream)
    async def _reset_stream():
        streamer().reset_stream()
        kf_streamer().reset_stream()
        current_data.set(pd.DataFrame())
        is_streaming.set(False)
        stream_speed.set(2.0)  # ✅ 배속 기본값으로 초기화
        alerts.set([])
        alert_buffer.set([])

        # ✅ 1️⃣ 먼저 표시 리셋
        await session.send_custom_message("resetSensors", True)

        # ✅ 2️⃣ mold_code 구조에 맞는 빈 값 전달 (오류 방지)
        await session.send_custom_message("updateSensors", {
            "values": {},
            "mold_code": ""
        })

    # 빨리감기 버튼 클릭 → 속도 순환 변경
    @reactive.effect
    @reactive.event(input.fast_stream)
    def _fast_stream():
        current = stream_speed()
        # 단계별 속도 순환
        next_speed = {2.0: 1.0, 1.0: 0.5, 0.5: 0.1, 0.1: 0.05, 0.05: 2.0}.get(current, 2.0)
        stream_speed.set(next_speed)

    # === GIF 표시 제어 (스트리밍 상태 연동) ===

    # ▶ 시작 시 GIF 표시
    @reactive.effect
    @reactive.event(input.start_stream)
    async def _gif_start():
        await session.send_custom_message("updateGif", {"src": "die-castings.gif"})


    # ⏸ 일시정지 시 PNG 표시
    @reactive.effect
    @reactive.event(input.pause_stream)
    async def _gif_pause():
        await session.send_custom_message("updateGif", {"src": "die-castings.png"})


    # 🔄 리셋 시 PNG 표시
    @reactive.effect
    @reactive.event(input.reset_stream)
    async def _gif_reset():
        await session.send_custom_message("updateGif", {"src": "die-castings.png"})


    # ✅ 스트리밍이 중단 상태일 때도 자동으로 PNG 표시 유지
    @reactive.effect
    async def _sync_gif_state():
        if not is_streaming():
            await session.send_custom_message("updateGif", {"src": "die-castings.png"})

    # ======================================================
    # ① 스트리머에서 데이터 수집 (배치 단위)
    # ======================================================
    @reactive.effect
    async def _collect_stream():
        """스트리머에서 여러 행을 받아 큐에 쌓기만 하는 역할"""
        if not is_streaming():
            return

        reactive.invalidate_later(stream_speed())

        page = page_state()
        if page == "field":
            s = streamer()
        elif page == "quality":
            s = kf_streamer()
        else:
            return

        # 여러 행 들어올 수 있음 → 전부 큐에 적재
        next_batch = s.get_next_batch(1)
        if next_batch is not None and not next_batch.empty:
            for _, row in next_batch.iterrows():
                data_queue.append(row.to_dict())

    # ======================================================
    # ② 큐에서 한 행씩 소비 (2초마다 한 건씩 처리)
    # ======================================================
    @reactive.effect
    async def _consume_stream():
        """2초마다 큐에서 한 행만 꺼내 공정상태 + 알림 처리"""
        if not is_streaming():
            return

        reactive.invalidate_later(stream_speed())

        if not data_queue:
            return

        latest = data_queue.popleft()

        # ✅ 기존 데이터 가져와서 누적
        df_old = current_data()
        if df_old is None or df_old.empty:
            df_new = pd.DataFrame([latest])
        else:
            df_new = pd.concat([df_old, pd.DataFrame([latest])], ignore_index=True)

        current_data.set(df_new)

        # === 🚨 불량 감지 ===
        if latest.get("passorfail", 0) == 1:
            mold = latest.get("mold_code", "-")
            time_str = str(latest.get("real_time", ""))
            push_alert(f" 불량 발생 — 금형 {mold}, 시각 {time_str}", defer=True)

        # === ⚠️ 이상치 감지 (Z-score 기반) ===
        numeric_keys = [
            k for k, v in latest.items()
            if isinstance(v, (int, float)) and not pd.isna(v) and k != "passorfail"
        ]
        if numeric_keys:
            try:
                df_check = current_data()
                if df_check is not None and len(df_check) > 10:
                    df_num = df_check[numeric_keys].select_dtypes(include="number")
                    means = df_num.mean()
                    stds = df_num.std().replace(0, np.nan)
                    z_scores = (pd.Series(latest)[numeric_keys] - means) / stds

                    # 🚨 |z|>3 : 심각 이상치
                    severe_cols = [c for c in z_scores.index if abs(z_scores[c]) > 3]
                    # ⚠️ 2<|z|≤3 : 경고 수준 이상치
                    warn_cols = [c for c in z_scores.index if 2 < abs(z_scores[c]) <= 3]

                    mold = latest.get("mold_code", "-")
                    time_str = str(latest.get("real_time", ""))

                    # ⚠️ 경고 수준 알림
                    if warn_cols:
                        cols_kor = [label_map.get(c, c) for c in warn_cols]
                        cols_str = ", ".join(cols_kor)
                        push_alert(
                            f" 경고 구간 감지 — 금형 {mold}, 시각 {time_str}, 변수: {cols_str}",
                            level="warning",
                            defer=True
                        )

                    # 🚨 심각 수준 알림
                    if severe_cols:
                        cols_kor = [label_map.get(c, c) for c in severe_cols]
                        cols_str = ", ".join(cols_kor)
                        push_alert(
                            f" 이상치 감지 — 금형 {mold}, 시각 {time_str}, 변수: {cols_str}",
                            level="danger2",
                            defer=True
                        )

            except Exception as e:
                print("⚠️ 이상치 감지 중 오류:", e)

        # === JS 업데이트 ===
        clean_values = {}
        for k, v in latest.items():
            if any(sub in str(k).lower() for sub in ["unnamed", "index"]):
                continue
            if isinstance(v, (int, float)) and not pd.isna(v):
                clean_values[str(k).replace(":", "_").replace(" ", "_")] = float(v)

        mold_code = str(latest.get("mold_code", ""))
        await session.send_custom_message("updateSensors", {
            "values": clean_values,
            "mold_code": mold_code
        })

        # 🔚 루프 끝: 버퍼 → alerts 반영 (한 번만)
        buf = list(alert_buffer())
        if buf:
            lst = list(alerts())
            lst.extend(buf)
            alerts.set(lst[-100:])
            alert_buffer.set([])

    @output
    @render.ui
    def process_svg_inline():
        svg_items = []
        for key, label in VAR_LABELS.items():
            if key not in VAR_POSITIONS:
                continue
            x, y = VAR_POSITIONS[key]
            svg_items.append(make_item_with_bg(key, label, x, y))

        svg_html = "\n".join(svg_items)

        return ui.HTML(f"""
            <div style='position:relative;width:900px;height:500px;margin:auto;'>
                <!-- ✅ 초기 상태는 PNG (정지 상태) -->
                <img id='process_gif' src='die-castings.png'
                    style='position:absolute;width:100%;height:100%;object-fit:contain;z-index:1;'/>
                <svg xmlns='http://www.w3.org/2000/svg'
                    width='100%' height='100%'
                    viewBox='0 0 900 500'
                    preserveAspectRatio='xMidYMid meet'
                    style='position:absolute;z-index:2;pointer-events:none;'>
                    {svg_html}
                </svg>
            </div>
        """)
    
    def make_item_with_bg(key: str, label: str, x: int, y: int) -> str:
        return f"""
        <g id='var-{key}'>
            <rect x='{x - 5}' y='{y - 18}' rx='4' ry='4'
                width='200' height='24'
                fill='rgba(255,255,255,0.75)' stroke='#ddd' stroke-width='0.5'/>
            <text x='{x}' y='{y}' fill='#111827'
                font-size='15' font-weight='700'>
                <tspan class='label'>{label}: </tspan>
                <tspan class='value'>—</tspan>
            </text>
        </g>
        """
    
    # ======================================================
    # 🎯 목표 계산 (앱 실행 시 1회 수행)
    # ======================================================

    def _get_prod_date(t):
        return (t - datetime.timedelta(days=1)).date() if t.time() < datetime.time(8,0) else t.date()

    def _get_shift(t):
        if datetime.time(8,0) <= t.time() < datetime.time(20,0):
            return "Day"
        else:
            return "Night"

    streaming_df["prod_date"] = streaming_df["real_time"].apply(_get_prod_date)
    streaming_df["shift"] = streaming_df["real_time"].apply(_get_shift)

    # === 조별 목표량 (row 수 × 1.1)
    shift_target_df = (
        streaming_df.groupby(["prod_date","shift"])
        .size().reset_index(name="shift_target")
    )
    shift_target_df["shift_target"] = (shift_target_df["shift_target"] * 1.1).round().astype(int)

    # === 일일 목표량 (08~익일08시 row 수 × 1.1)
    daily_target_df = (
        streaming_df.groupby("prod_date")
        .size().reset_index(name="daily_target")
    )
    daily_target_df["daily_target"] = (daily_target_df["daily_target"] * 1.1).round().astype(int)


    # ======================================================
    # ⚙️ 실시간 달성률 계산 함수
    # ======================================================
    def calc_achievements(df_live):
        if df_live is None or df_live.empty:
            return 0, 0

        df = df_live.copy()
        df["real_time"] = pd.to_datetime(df["real_time"], errors="coerce")
        df = df.dropna(subset=["real_time"]).sort_values("real_time")

        now = df["real_time"].iloc[-1]
        prod_date = (now - datetime.timedelta(days=1)).date() if now.time() < datetime.time(8,0) else now.date()

        # --- 현재 교대 구간 ---
        if datetime.time(8,0) <= now.time() < datetime.time(20,0):
            current_shift = "Day"
            shift_start = datetime.datetime.combine(now.date(), datetime.time(8,0))
        else:
            current_shift = "Night"
            if now.time() >= datetime.time(20,0):
                shift_start = datetime.datetime.combine(now.date(), datetime.time(20,0))
            else:
                shift_start = datetime.datetime.combine(now.date()-datetime.timedelta(days=1), datetime.time(20,0))

        # --- 현재 구간별 누적 row 수 ---
        df_shift = df[df["real_time"] >= shift_start]
        shift_count = len(df_shift)

        day_start = datetime.datetime.combine(now.date(), datetime.time(8,0))
        if now.time() < datetime.time(8,0):
            day_start -= datetime.timedelta(days=1)
        df_day = df[df["real_time"] >= day_start]
        day_count = len(df_day)

        # --- 목표량 조회 ---
        shift_target_row = shift_target_df.query(
            "(prod_date == @prod_date) & (shift == @current_shift)"
        )["shift_target"]
        daily_target_row = daily_target_df.query(
            "prod_date == @prod_date"
        )["daily_target"]

        shift_target = int(shift_target_row.iloc[0]) if not shift_target_row.empty else 1
        daily_target = int(daily_target_row.iloc[0]) if not daily_target_row.empty else 1

        # --- 달성률 계산 ---
        shift_rate = min((shift_count / shift_target) * 100, 100)
        daily_rate = min((day_count / daily_target) * 100, 100)

        return round(daily_rate, 1), round(shift_rate, 1)


    # ======================================================
    # 🧩 실시간 달성률 카드 (UI) — 빈 상태 포함
    # ======================================================
    @output
    @render.ui
    def process_status_card():
        import datetime

        df_live = current_data()

        if df_live is None or df_live.empty:
            return ui.div(
                {
                    "style": (
                        "border:2px solid #ccc; border-radius:12px; padding:14px; "
                        "box-shadow:0 2px 6px rgba(0,0,0,0.1); background-color:white; "
                        "font-family:'NanumGothic'; width:100%; max-width:100%;"
                    )
                },
                ui.h4("📅 -", style="margin-bottom:8px; text-align:center; color:#aaa;"),
                ui.div("⏸ 데이터 대기 중...", 
                    style="text-align:center; font-size:18px; color:gray; font-weight:bold;"),
                ui.hr(),
                ui.div(
                    {"style": "padding:4px 8px;"},
                    ui.span("조별 달성률", style="font-weight:bold; color:#777;"),
                    ui.div("0.0%", style="text-align:right; color:#999; font-weight:bold;"),
                    ui.div(
                        {"style": (
                            "background-color:#e9ecef; border-radius:8px; height:18px; width:100%; margin-top:4px;"
                        )}
                    ),
                ),
                ui.div(
                    {"style": "padding:4px 8px; margin-top:6px;"},
                    ui.span("일일 달성률", style="font-weight:bold; color:#777;"),
                    ui.div("0.0%", style="text-align:right; color:#999; font-weight:bold;"),
                    ui.div(
                        {"style": (
                            "background-color:#e9ecef; border-radius:8px; height:18px; width:100%; margin-top:4px;"
                        )}
                    ),
                )
            )

        # 데이터 존재 시
        latest = df_live.iloc[-1]
        daily_rate, shift_rate = calc_achievements(df_live)

        shift_icon = "🌞" if datetime.time(8, 0) <= latest["real_time"].time() < datetime.time(20, 0) else "🌙"

        def progress_bar(value, color):
            return ui.div(
                {
                    "style": (
                        "background-color:#e9ecef; border-radius:8px; height:18px; width:100%; margin-top:4px;"
                    )
                },
                ui.div(
                    {
                        "style": (
                            f"width:{min(value, 100):.1f}%; background-color:{color}; height:100%; "
                            f"border-radius:8px; transition:width 0.3s;"
                        )
                    }
                )
            )

        return ui.div(
            {
                "style": (
                    "border:2px solid #e0e0e0; border-radius:12px; padding:14px; "
                    "box-shadow:0 2px 6px rgba(0,0,0,0.1); background-color:white; "
                    "font-family:'NanumGothic'; width:100%; max-width:100%;"
                )
            },
            ui.h4(f"📅 {latest['real_time']:%Y-%m-%d %H:%M:%S}",
                style="margin-bottom:8px; text-align:center; color:#333;"),
            ui.div(
                {"style": "text-align:center; font-size:18px; font-weight:bold; color:#555;"},
                f"{shift_icon} {latest.get('shift','-')}조  (Team {latest.get('team','-')})"
            ),
            ui.hr(),
            ui.div(
                {"style": "padding:4px 8px;"},
                ui.span("조별 달성률", style="font-weight:bold; color:#444;"),
                ui.div(f"{shift_rate:.1f}%", style="text-align:right; color:#0d6efd; font-weight:bold;"),
                progress_bar(shift_rate, "#0d6efd"),
            ),
            ui.div(
                {"style": "padding:4px 8px; margin-top:6px;"},
                ui.span("일일 달성률", style="font-weight:bold; color:#444;"),
                ui.div(f"{daily_rate:.1f}%", style="text-align:right; color:#198754; font-weight:bold;"),
                progress_bar(daily_rate, "#198754"),
            ),
        )


    # ======================================================
    # 🧠 실시간 품질 판정 + 누적 불량률 (깜빡임 애니메이션 포함)
    # ======================================================
    @output
    @render.ui
    def realtime_predict_card():
        df_live = current_data()

        if df_live is None or df_live.empty:
            return ui.div(
                {
                    "style": (
                        "border:2px solid #ccc; border-radius:12px; padding:16px; "
                        "background-color:white; box-shadow:0 2px 6px rgba(0,0,0,0.1); "
                        "text-align:center; font-family:'NanumGothic'; width:100%; max-width:100%;"
                    )
                },
                ui.h4("🤖 실시간 품질 판정", style="margin-bottom:10px; color:#333;"),
                ui.h3("⏸ 데이터 대기 중...", style="color:gray; margin-bottom:6px;"),
                ui.h5("누적 불량률: -%", style="color:#888; margin-bottom:6px;"),
                ui.p("데이터 시각: -", style="color:#aaa; font-size:14px; margin-top:6px;"),
            )

        latest = df_live.tail(1).iloc[0]
        if "passorfail" not in latest:
            return ui.div("⚠️ passorfail 컬럼이 없습니다.", style="color:red; text-align:center;")

        result = int(latest["passorfail"])
        label = "✅ 양품" if result == 0 else "❌ 불량"
        color = "#28a745" if result == 0 else "#dc3545"
        emoji = "🟢" if result == 0 else "🔴"
        anim_color = "rgba(40,167,69,0.25)" if result == 0 else "rgba(220,53,69,0.25)"

        total_count = len(df_live)
        fail_count = df_live["passorfail"].sum()
        fail_rate = (fail_count / total_count) * 100 if total_count > 0 else 0

        return ui.div(
            {
                "style": (
                    f"border:2px solid {color}; border-radius:12px; padding:16px; "
                    f"background-color:white; box-shadow:0 2px 6px rgba(0,0,0,0.1); "
                    f"text-align:center; font-family:'NanumGothic'; width:100%; max-width:100%; "
                    f"animation: flash-bg 0.8s ease;"
                )
            },
            ui.tags.style(f"""
                @keyframes flash-bg {{
                    0%   {{ box-shadow: 0 0 0px {anim_color}; }}
                    30%  {{ box-shadow: 0 0 30px {anim_color}; }}
                    60%  {{ box-shadow: 0 0 20px {anim_color}; }}
                    100% {{ box-shadow: 0 0 0px transparent; }}
                }}
            """),
            ui.h4("🤖 실시간 품질 판정", style="margin-bottom:10px; color:#333;"),
            ui.h3(f"{emoji} {label}", style=f"color:{color}; font-weight:bold; margin-bottom:6px;"),
            ui.h5(f"누적 불량률: {fail_rate:.1f}%", style="color:#555; margin-bottom:6px;"),
            ui.p(f"데이터 시각: {latest['real_time']}", style="color:#777; font-size:14px; margin-top:6px;"),
        )

    @output
    @render.ui
    def stream_time_display():
        df = current_data()
        if df is None or df.empty:
            time_str = "-------- --:--:--"
        else:
            latest_time = pd.to_datetime(df["real_time"].iloc[-1], errors="coerce")
            if pd.isna(latest_time):
                time_str = "-------- --:--:--"
            else:
                time_str = f"{latest_time:%Y-%m-%d %H:%M:%S}"

        color = "#16a34a" if is_streaming() else "#6b4f2a"

        return ui.HTML(
            f"<span style='color:{color}; font-weight:bold;'>🕒 {time_str}</span>"
        )
    
    @output
    @render.ui
    def stream_speed_badge():
        speed = stream_speed()

        # 배속 매핑
        speed_map = {2.0: "1x", 1.0: "2x", 0.5: "4x", 0.1: "20x", 0.05: "40x"}
        label = speed_map.get(speed, "1x")

        # 색상: 속도에 따라 강조
        color_map = {2.0: "#6b4f2a", 1.0: "#f59e0b", 0.5: "#f97316", 0.1: "#ef4444", 0.05: "#dc2626"}
        bg_color = color_map.get(speed, "#6b4f2a")

        return ui.HTML(
            f"<span style='background:{bg_color}; color:white; padding:3px 10px; border-radius:10px; font-size:13px;'>⏩ {label}</span>"
        )

    # ============================================
    # 🔔 실시간 불량 알림 시스템
    # ============================================
    alerts = reactive.Value([])          # 기존 알림 표시용
    alert_buffer = reactive.Value([])    # 🚀 버퍼를 reactive로 (전역 리스트 사용 X)

    def push_alert(message, level="danger", defer=True):
        color_map = {
            "info": "#2196F3",
            "success": "#4CAF50",
            "warning": "#FB8C00",
            "danger": "#AD0603",
            "danger2": "#E53935",
        }
        icon_map = {
            "info": "fa-circle-info",
            "success": "fa-check-circle",
            "warning": "fa-triangle-exclamation",
            "danger": "fa-xmark",
            "danger2": "fa-circle-exclamation",
        }
        now = datetime.datetime.now().strftime("%H:%M:%S")
        item = {
            "msg": message,
            "level": level,
            "color": color_map.get(level, "#2196F3"),
            "icon": icon_map.get(level, "fa-circle-info"),
            # "time": now,
        }

        if defer:
            buf = list(alert_buffer())    # ✅ 가져오고
            buf.append(item)              #   추가한 뒤
            alert_buffer.set(buf)         #   다시 set (reactive 업데이트)
        else:
            lst = list(alerts())
            lst.append(item)
            alerts.set(lst[-100:])

    @output
    @render.ui
    def realtime_alert_box():
        items = list(reversed(alerts()))
        if not items:
            return ui.div("알림 없음", style="color:gray; text-align:center;")

        html = ""
        for a in items:
            html += f"""
            <div style="
                margin-bottom:6px; border-left:4px solid {a['color']};
                padding-left:8px; background:rgba(255,0,0,0.02);
                border-radius:4px;">
                <i class="fa-solid {a['icon']}" style="color:{a['color']};"></i>
                <span style="margin-left:6px;">{a['msg']}</span>
            </div>
            """
        return ui.HTML(html)


    # ======================================================
    # 🧩 알람 카드 제목 렌더링 (알람 개수 표시)
    # ======================================================
    @output
    @render.ui
    def alert_card_header():
        count = len(alerts()) if alerts() else 0

        # 빨간 배지 스타일
        badge_style = (
            "background-color:#dc3545; color:white; font-weight:bold; "
            "border-radius:50%; width:22px; height:22px; "
            "display:flex; align-items:center; justify-content:center; "
            "font-size:13px; margin-left:8px;"
        )

        return ui.div(
            {
                "style": (
                    "display:flex; align-items:center; gap:8px; "
                    "font-weight:bold; font-size:16px; color:#5c4b3b;"
                )
            },
            "📢 실시간 알림",
            ui.div(str(count), style=badge_style)  # 🔴 빨간 동그라미 숫자
        )

# 🟢 TAB1. 끝
# ============================================================



# ============================================================
# 🟢 TAB2. 품질
# ============================================================

    selected_row = reactive.Value(None)

    last_proba = reactive.value(None)
    loading = reactive.value(False)
    local_factors = reactive.value(None)




    @output
    @render.data_frame
    def recent_data_table():
        df = current_data()
        if df is None or df.empty:
            return pd.DataFrame({"알림": ["현재 수신된 데이터가 없습니다."]})

        data = df.copy()

        # ✅ 2.5) passorfail 컬럼을 사람이 보기 좋게 한글 변환
        if "passorfail" in data.columns:
            data["passorfail"] = data["passorfail"].map({0: "양품", 1: "불량"}).fillna(data["passorfail"])

        # 1) 3시그마 이상치 행 찾기
        numeric_cols = data.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            means = data[numeric_cols].mean()
            stds = data[numeric_cols].std().replace(0, np.nan)
            z = (data[numeric_cols] - means) / stds
            mask_3sigma = (z.abs() > 3).any(axis=1)
        else:
            mask_3sigma = pd.Series(False, index=data.index)

        # 2) 불량 행(passorfail==1) 찾기
        if "passorfail" in data.columns:
            mask_fail = data["passorfail"] == 1
        else:
            mask_fail = pd.Series(False, index=data.index)

        # 3) 두 조건 중 하나라도 맞는 행만 필터
        flagged = data[mask_3sigma | mask_fail].copy()

        # 4) 없으면 “이상 행 없음” 표시(표는 1행 안내)
        if flagged.empty:
            return pd.DataFrame({"알림": ["현재 3σ 이상치나 불량 행이 없습니다."]})

        # 5) 보기 좋게 정리
        #    - 최근 것부터 최대 200행
        flagged = flagged.tail(200).round(2)

        # 6) 한글 컬럼명으로 매핑(네가 선언한 label_map 재사용)
        #    label_map에 없는 건 원래 이름 유지
        def to_kor(col):
            return label_map.get(col, col)
        flagged.rename(columns={c: to_kor(c) for c in flagged.columns}, inplace=True)

        # 7) 자주 보는 컬럼 앞으로 배치
        prefer = [to_kor(c) for c in ["real_time", "passorfail"] if c in df.columns]
        other_cols = [c for c in flagged.columns if c not in prefer]
        flagged = flagged[prefer + other_cols] if prefer else flagged

        return flagged.reset_index(drop=True)






    @reactive.effect
    @reactive.event(input.predict_btn)
    def _():
     loading.set(True)
     try:
        X = get_input_data()
        proba = model.predict_proba(X)[0, 1]
        last_proba.set(proba)

        # === 불량 기여 요인 계산 ===
        # 1) 누적형 변수 제거
        exclude_vars = ["count", "monthly_count", "global_count"]
        use_num_cols = [c for c in num_cols if c not in exclude_vars]

        baseline = df_predict[df_predict["passorfail"] == 0][use_num_cols].mean()
        current = X[use_num_cols].iloc[0]

        # 2) 표준화 거리 (표준편차로 나눔)
        stds = df_predict[use_num_cols].std().replace(0, 1)  # 분모=0 방지
        diffs = ((current - baseline) / stds) ** 2

        # 3) 기여도 계산
        if diffs.sum() > 0:
            contrib = (diffs / diffs.sum()).sort_values(ascending=False)
            local_factors.set(
                pd.DataFrame({
                    "feature": [get_label(c) for c in contrib.index],
                    "importance": contrib.values
                })
            )
        else:
            local_factors.set(
                pd.DataFrame({"feature": [], "importance": []})
            )

     except Exception as e:
        last_proba.set(f"error:{e}")
     finally:
        loading.set(False)
    
    # @reactive.effect
    # @reactive.event(input.apply_suggestions)
    # def _():
    #     factors = local_factors()
    #     if factors is None or factors.empty:
    #         return

    #     top = factors.head(5).copy()
    #     exclude_vars = ["count", "monthly_count", "global_count"]
    #     use_num_cols = [c for c in num_cols if c not in exclude_vars]

    #     baseline = df_predict[df_predict["passorfail"] == 0][use_num_cols].mean()
    #     current = get_input_data().iloc[0][use_num_cols]

    #     for _, row in top.iterrows():
    #         feat = row["feature"]
    #         col = [k for k, v in label_map.items() if v == feat]
    #         if not col: 
    #             continue
    #         col = col[0]

    #         if col in current.index:
    #             diff = current[col] - baseline[col]
    #             if abs(diff) > 1e-6:
    #                 new_val = current[col] - diff/2   # 현재값과 baseline 사이 중간으로 이동
    #                 update_slider(f"{col}_slider", value=float(new_val))
    #                 update_numeric(col, value=float(new_val))
    #                 print(f"[반영됨] {col}: {current[col]} → {new_val} (baseline {baseline[col]})")

    #     # 🔹 자동 예측 실행
    #     session.send_input_message("predict_btn", 1)


    # ============================================================
    # 개선 방안 반영 후 즉시 재예측 + 최종 판정 표시
    # ============================================================
    @reactive.effect
    @reactive.event(input.apply_suggestions)
    def _():
        factors = local_factors()
        if factors is None or factors.empty:
            return

        top = factors.head(5).copy()
        exclude_vars = ["count", "monthly_count", "global_count"]
        use_num_cols = [c for c in num_cols if c not in exclude_vars]

        baseline = df_predict[df_predict["passorfail"] == 0][use_num_cols].mean()
        current = get_input_data().iloc[0][use_num_cols]

        # === ① 개선값 반영 ===
        for _, row in top.iterrows():
            feat = row["feature"]
            col = [k for k, v in label_map.items() if v == feat]
            if not col: 
                continue
            col = col[0]

            if col in current.index:
                diff = current[col] - baseline[col]
                if abs(diff) > 1e-6:
                    new_val = current[col] - diff / 2  # baseline 쪽으로 50% 이동
                    update_slider(f"{col}_slider", value=float(new_val))
                    update_numeric(col, value=float(new_val))
                    print(f"[반영됨] {col}: {current[col]} → {new_val} (baseline {baseline[col]})")

        # === ② 개선 후 자동 예측 ===
        try:
            X_new = get_input_data()
            proba_new = model.predict_proba(X_new)[0, 1]
            last_proba.set(proba_new)
            prediction_done.set(True)  # 개선된 판정 결과 섹션 표시용

            # === ③ 개선된 결과 저장용 상태값 추가 ===
            session.send_custom_message("scroll_to_bottom", {})  # 하단 자동 스크롤

        except Exception as e:
            last_proba.set(f"error:{e}")




    @render.ui
    def prediction_result():
        if loading():
            return ui.div(
                ui.div(class_="spinner-border text-primary", role="status"),
                ui.HTML("<div style='margin-top:10px;'>예측 실행 중...</div>"),
                style="text-align:center; padding:20px;"
            )

        proba = last_proba()
        if proba is None:
            return ui.div(
                ui.HTML("<span style='color:gray; font-size:18px;'>아직 예측을 실행하지 않았습니다.</span>"),
                style="text-align:center; padding:20px;"
            )

        if isinstance(proba, str) and proba.startswith("error:"):
            return ui.div(
                ui.HTML(f"<span style='color:red;'>예측 중 오류 발생: {proba[6:]}</span>")
            )

        if proba < 0.02:
            style = "background-color:#d4edda; color:#155724; font-size:18px; font-weight:bold; padding:15px; text-align:center; border-radius:8px;"
        elif proba < 0.04:
            style = "background-color:#fff3cd; color:#856404; font-size:18px; font-weight:bold; padding:15px; text-align:center; border-radius:8px;"
        else:
            style = "background-color:#f8d7da; color:#721c24; font-size:18px; font-weight:bold; padding:15px; text-align:center; border-radius:8px;"

        judgment = "불량품" if proba >= 0.2 else "양품"

        return ui.div(
            [
                ui.HTML(f"예상 불량률: {proba*100:.2f}%"),
                ui.br(),
                ui.HTML(f"최종 판정: <span style='font-size:22px;'>{judgment}</span>")
            ],
            style=style
        )

    @render.plot
    def feature_importance_plot():
        try:
            importances = model.named_steps["model"].feature_importances_
            feat_names = model.named_steps["preprocessor"].get_feature_names_out()
            imp_df = pd.DataFrame({"feature": feat_names, "importance": importances})
            imp_df = imp_df.sort_values("importance", ascending=False).head(10)

            plt.figure(figsize=(8,5))
            plt.barh(imp_df["feature"], imp_df["importance"])
            plt.gca().invert_yaxis()
            plt.title("변수 중요도 Top 10")
            plt.tight_layout()
        except Exception:
            plt.figure()
            plt.text(0.5,0.5,"변수 중요도 계산 불가",ha="center",va="center")

    @render.plot
    def distribution_plot():
        try:
            plt.figure(figsize=(8,5))
            df_good = df_predict[df_predict["passorfail"]==0]["biscuit_thickness"]
            df_bad = df_predict[df_predict["passorfail"]==1]["biscuit_thickness"]

            plt.hist(df_good, bins=30, alpha=0.6, label="양품")
            plt.hist(df_bad, bins=30, alpha=0.6, label="불량품")

            plt.axvline(df_predict["biscuit_thickness"].mean(), color="red", linestyle="--", label="평균")
            plt.legend()
            plt.title("비스킷 두께 분포 (양품 vs 불량)")
            plt.tight_layout()
        except Exception:
            plt.figure()
            plt.text(0.5,0.5,"분포 그래프 생성 불가",ha="center",va="center")

    @render.plot
    def process_trend_plot():
        try:
            mold_trend = df_predict.groupby("mold_code")["passorfail"].mean().sort_values(ascending=False)
            plt.figure(figsize=(8,5))
            mold_trend.plot(kind="bar")
            plt.ylabel("불량률")
            plt.title("금형 코드별 불량률")
            plt.tight_layout()
        except Exception:
            plt.figure()
            plt.text(0.5,0.5,"공정별 그래프 생성 불가",ha="center",va="center")
            
    # ===== 품질 모니터링용 SPC 관리도 =====
    def calc_xr_chart(df, var='cast_pressure', subgroup_size=5):
        if df.empty:
            return None, None, (None, None, None, None)
        df = df.tail(subgroup_size * 10).copy()
        df['group'] = np.floor(np.arange(len(df)) / subgroup_size)
        grouped = df.groupby('group')[var]
        xbar = grouped.mean()
        R = grouped.max() - grouped.min()
        Xbar_bar, R_bar = xbar.mean(), R.mean()
        A2, D3, D4 = 0.577, 0, 2.114   # n=5 기준
        return xbar, R, (
            Xbar_bar + A2 * R_bar, Xbar_bar - A2 * R_bar,
            D4 * R_bar, D3 * R_bar
        )


    def calc_p_chart(df, var='passorfail', window=50):
        if df.empty or var not in df:
            return None, None, None
        df = df.tail(window)
        p_bar = df[var].mean()
        n = len(df)
        sigma_p = np.sqrt(p_bar * (1 - p_bar) / n)
        return p_bar, p_bar + 3*sigma_p, p_bar - 3*sigma_p


    def plot_xr_chart_matplotlib(xbar, R, limits):
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        if xbar is None or R is None:
            for ax in axes: ax.axis("off")
            axes[0].text(0.5,0.5,"데이터 부족",ha="center",va="center")
            return fig
        UCLx,LCLx,UCLr,LCLr = limits
        axes[0].plot(xbar.index,xbar.values,marker='o'); axes[0].axhline(xbar.mean(),c='g')
        axes[0].axhline(UCLx,c='r',ls='--'); axes[0].axhline(LCLx,c='r',ls='--')
        axes[0].set_title("X-bar 관리도"); axes[0].grid(True,ls='--',alpha=.5)
        axes[1].plot(R.index,R.values,marker='o'); axes[1].axhline(R.mean(),c='g')
        axes[1].axhline(UCLr,c='r',ls='--'); axes[1].axhline(LCLr,c='r',ls='--')
        axes[1].set_title("R 관리도"); axes[1].grid(True,ls='--',alpha=.5)
        plt.tight_layout(); return fig


    def plot_p_chart_matplotlib(p_bar, UCL, LCL):
        fig, ax = plt.subplots(figsize=(8,4))
        if p_bar is None:
            ax.axis("off"); ax.text(0.5,0.5,"데이터 부족",ha="center",va="center"); return fig
        ax.hlines([p_bar,UCL,LCL],0,1,colors=['g','r','r'],linestyles=['-','--','--'])
        ax.text(0.5,p_bar,f"불량률 {p_bar*100:.2f}%",ha='center',va='bottom',fontsize=12)
        ax.set_ylim(0,max(1,UCL*1.2)); ax.set_title("P 관리도 (실시간 불량률)")
        ax.grid(True,ls='--',alpha=.5); return fig
    
    def get_input_data():
        data = {}
        for col in cat_cols + num_cols:
            data[col] = [input[col]()]

        return pd.DataFrame(data)

    for col in num_cols:
        @reactive.effect
        @reactive.event(input[col])
        def _(col=col):
            update_slider(f"{col}_slider", value=input[col]())
        @reactive.effect
        @reactive.event(input[f"{col}_slider"])
        def _(col=col):
            update_numeric(col, value=input[f"{col}_slider"]())

    @reactive.effect
    @reactive.event(input.reset_btn)
    def _():
        # 범주형 변수: 첫 번째 값으로 초기화
        for col in cat_cols:
            first_val = str(sorted(df_predict[col].dropna().unique())[0])
            if(col == "tryshot_signal"):
                first_val = "없음"
            ui.update_select(col, selected=first_val)

        # 수치형 변수: 안전하게 숫자 변환 후 평균값으로 초기화
        for col in num_cols:
            series = pd.to_numeric(df_predict[col], errors="coerce")       # 문자열 → 숫자 (에러시 NaN)
            series = series.replace([np.inf, -np.inf], np.nan)             # inf → NaN
            mean_val = series.dropna().mean()                              # NaN 제거 후 평균
            default_val = int(mean_val) if pd.notna(mean_val) else 0       # fallback: 0
            update_slider(f"{col}_slider", value=default_val)
            update_numeric(col, value=default_val)

        # 예측 결과 초기화
        last_proba.set(None)

    @output
    @render.plot
    def local_factor_plot():
     factors = local_factors()
     if factors is None or factors.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "아직 예측을 실행하지 않았습니다.", ha="center", va="center")
        ax.axis("off")
        return fig

     top = factors.head(5).copy()
     top["importance"] = top["importance"] * 100  # % 변환

     fig, ax = plt.subplots(figsize=(8, 4))
     bars = ax.barh(top["feature"], top["importance"], color="tomato")

    # 각 막대 끝에 % 숫자 표시
     for bar, val in zip(bars, top["importance"]):
        ax.text(bar.get_width() + 0.5,
                bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%",
                va="center")

        ax.invert_yaxis()
        ax.set_xlabel("기여도 (%)")
        ax.set_title("이번 케이스 불량 기여 요인 Top 5")
        plt.tight_layout()
     return fig

    # === 여기에 local_factor_desc() 붙여넣기 ===
    @output
    @render.ui
    def local_factor_desc():
     factors = local_factors()
     if factors is None or factors.empty:
        return ui.markdown("아직 예측을 실행하지 않았습니다.")

     top = factors.head(5).copy()
     top["importance"] = top["importance"] * 100

     exclude_vars = ["count", "monthly_count", "global_count"]
     use_num_cols = [c for c in num_cols if c not in exclude_vars]
     baseline = df_predict[df_predict["passorfail"] == 0][use_num_cols].mean()
     current = get_input_data().iloc[0][use_num_cols]

     rows_html = []
     for _, row in top.iterrows():
        feat = row["feature"]
        importance = row["importance"]

        col = [k for k, v in label_map.items() if v == feat]
        if not col: 
            continue
        col = col[0]

        left_text = f"{feat}: {importance:.1f}%"

        if col in current.index:
            diff = current[col] - baseline[col]
            if abs(diff) > 1e-6:
                direction = "낮추세요" if diff > 0 else "올리세요"
                adj_val = abs(diff) / 2
                right_text = f"{adj_val:.1f} 단위 {direction} (현재 {current[col]:.1f}, 기준 {baseline[col]:.1f})"
            else:
                right_text = "-"
        else:
            right_text = "-"

        row_html = f"""
        <div style='display:flex; align-items:center; margin-bottom:8px; font-size:15px;'>
            <div style='flex:1; text-align:left;'>{left_text}</div>
            <div style='flex:0.2; text-align:center;'>
                <i class="fa-solid fa-arrow-right fa-beat" style="color:#007bff;"></i>
            </div>
            <div style='flex:2; text-align:left; color:#444;'>{right_text}</div>
        </div>
        """
        rows_html.append(row_html)

    # 🔹 for문 끝난 뒤에 return 실행
     return ui.div(
        [
            ui.markdown("**이번 예측에서 불량률은 아래 요인들의 영향을 많이 받습니다:**"),
            ui.HTML("".join(rows_html)),
            ui.input_action_button(
                "apply_suggestions", "반영하고 다시 예측하기",
                class_="btn btn-warning", style="margin-top:15px;"
            )
        ]
    )

# ================================================
# 개선 방안 섹션 조건부 표시
# ================================================

    @output
    @render.ui
    def improvement_section():
        # 예측 결과가 존재할 때만 개선 방안 섹션 렌더링
        if not prediction_done.get():   # 예: prediction_done은 reactive.Value(True/False)
            return None

        return ui.card(
            ui.card_header("불량 기여 요인 Top 5", style="text-align:center; background-color:#f8f9fa; font-weight:bold;"),
            ui.output_plot("local_factor_plot"),
            ui.hr(),
            ui.output_ui("local_factor_desc")
        )
    prediction_done = reactive.Value(False)

    @reactive.effect
    @reactive.event(input.predict_btn)
    def _():
        # ... 기존 예측 로직 ...
        prediction_done.set(True)

    @reactive.effect
    @reactive.event(input.reset_btn)
    def _():
        prediction_done.set(False)


    # ================================================
    # 개선 방안 섹션 조건부 표시 (양품이면 숨김)
    # ================================================
    @output
    @render.ui
    def improvement_section():
        # 예측 결과가 존재하지 않으면 아무것도 표시하지 않음
        if not prediction_done.get():
            return None

        proba = last_proba()
        if proba is None:
            return None

        # ✅ 판정이 양품이면 Top5 숨기기
        if proba < 0.2:  # 양품 기준: 20% 미만
            return None

        # 불량인 경우만 표시
        return ui.card(
            ui.card_header("불량 기여 요인 Top 5", style="text-align:center; background-color:#f8f9fa; font-weight:bold;"),
            ui.output_plot("local_factor_plot"),
            ui.hr(),
            ui.output_ui("local_factor_desc")
        )





    ##### 원인 분석 - 불량 및 공정 에러 발생 조건

    # 선택된 센서 & 현재 그래프의 y라벨 순서 저장
    selected_sensor = reactive.Value(None)
    plot_labels = reactive.Value([])   # ← barh에 그려진 y축 카테고리 순서

    @output
    @render.plot
    def local_factor_plot():
        df = current_data()
        if df is None or df.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "실시간 데이터 수신 대기 중...", ha="center", va="center", fontsize=13)
            ax.axis("off")
            return fig

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if not numeric_cols:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "수치형 데이터 없음", ha="center", va="center")
            ax.axis("off")
            return fig

        mean_std = df[numeric_cols].describe().T[["mean", "std"]]
        latest = df.iloc[-1]
        z_scores = (latest - mean_std["mean"]) / mean_std["std"]
        z_scores = z_scores.dropna().sort_values(ascending=True)

        # ⬇️ 현재 그래프의 y축 카테고리 순서 저장 (index가 레이블 순서)
        plot_labels.set(list(z_scores.index))

        colors = ["#e74c3c" if abs(z) > 3 else "#95a5a6" for z in z_scores]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(range(len(z_scores)), z_scores.values, color=colors)  # ← 정수 y위치로 그림
        ax.set_yticks(range(len(z_scores)))
        ax.set_yticklabels(list(z_scores.index))
        ax.set_xlabel("Z-score (표준편차 기준)")
        ax.set_title("실시간 이상 감지 센서 (클릭 시 상세보기)")
        ax.grid(True, axis="x", linestyle="--", alpha=0.5)
        plt.tight_layout()
        return fig


    @output
    @render.ui
    def local_factor_desc():
        df = current_data()
        if df is None or df.empty:
            return ui.p("⚪ 실시간 데이터 수신 중이 아닙니다.", style="color:gray;")

        # === 1️⃣ 사용할 주요 컬럼만 선택 ===
        selected_cols = [
            # 공정 상태 관련
            "count", "speed_ratio", "pressure_speed_ratio",
            # 용융 단계
            "molten_temp",
            # 충진 단계
            "sleeve_temperature", "EMS_operation_time",
            "low_section_speed", "high_section_speed",
            "molten_volume", "cast_pressure", "mold_code",
            # 냉각 단계
            "upper_mold_temp1", "upper_mold_temp2", "upper_mold_temp3",
            "lower_mold_temp1", "lower_mold_temp2", "Coolant_temperature",
            # 공정 속도 관련
            "facility_operation_cycleTime", "production_cycletime",
            # 품질 및 성능
            "biscuit_thickness", "physical_strength",
        ]

        numeric_cols = [c for c in selected_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            return ui.p("⚪ 표시할 수치형 센서가 없습니다.", style="color:gray;")

        # === 2️⃣ Z-score 기반 이상 감지 ===
        mean_std = df[numeric_cols].describe().T[["mean", "std"]]
        latest = df.iloc[-1]

        anomalies = []
        for col in numeric_cols:
            val, mean, std = latest[col], mean_std.loc[col, "mean"], mean_std.loc[col, "std"]
            if pd.notna(std) and std > 0 and abs(val - mean) > 3 * std:
                anomalies.append((col, val, mean, std))

        if not anomalies:
            return ui.p("✅ 현재 이상 조건이 없습니다.", style="color:green;")

        # === 3️⃣ 한글 라벨 적용 ===
        alerts = [
            f"<li><b>{label_map.get(col, col)}</b>: 현재 {val:.2f} "
            f"(평균 {mean:.2f} ± {3*std:.2f}) → <span style='color:red;'>이상 감지</span></li>"
            for col, val, mean, std in anomalies
        ]

        # === 4️⃣ UI 렌더링 ===
        return ui.HTML(f"""
            <div style="background:#fff7f7; padding:10px; border-radius:8px;">
                <p><b>⚠ 공정 이상 감지 항목 ({len(anomalies)}개)</b></p>
                <ul>{''.join(alerts)}</ul>
                <p style='color:gray;font-size:13px;'>그래프를 클릭하면 상세 추이를 볼 수 있습니다.</p>
            </div>
        """)




    # 클릭 이벤트 처리 (y좌표 → 레이블로 변환)
    @reactive.effect
    @reactive.event(input.local_factor_plot_click)
    def _handle_click():
        click = input.local_factor_plot_click()
        if not click:
            return

        # y좌표값(실수형)을 가져오기
        y_val = None
        if isinstance(click, dict):
            y_val = (click.get("domain", {}) or {}).get("y", None)
            if y_val is None:
                y_val = click.get("y", None)
        if y_val is None:
            return

        # 그래프의 y라벨 순서와 매칭
        labels = plot_labels() or []
        idx = int(round(float(y_val)))
        if idx < 0 or idx >= len(labels):
            return

        sensor = labels[idx]
        selected_sensor.set(sensor)

        df = current_data()
        if df is None or df.empty or sensor not in df.columns:
            return

        # 한글 센서명으로 제목 표시
        sensor_name = label_map.get(sensor, sensor)

        ui.modal_show(
            ui.modal(
                ui.output_plot("sensor_detail_plot"),
                title=f"🔍 {sensor_name} 센서 상세 그래프",
                size="l",
                easy_close=True,
            )
        )

    @output
    @render.plot
    def sensor_detail_plot():
        sensor = selected_sensor.get()
        df = current_data()
        if not sensor or df is None or df.empty or sensor not in df.columns:
            fig, ax = plt.subplots()
            ax.axis("off")
            ax.text(0.5, 0.5, "선택된 센서 데이터가 없습니다.", ha="center", va="center")
            return fig

        # 한글 센서명 매핑
        sensor_name = label_map.get(sensor, sensor)

        y = pd.to_numeric(df[sensor], errors="coerce")
        y = y.dropna()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(y.values[-100:], marker="o", linestyle="-", alpha=0.7)

        m, s = y.mean(), y.std()
        if pd.notna(m):
            ax.axhline(m, color="green", linestyle="--", label="평균")
        if pd.notna(m) and pd.notna(s):
            ax.axhline(m + 3*s, color="red", linestyle="--", alpha=0.5, label="+3σ")
            ax.axhline(m - 3*s, color="red", linestyle="--", alpha=0.5, label="-3σ")

        ax.legend()
        ax.set_title(f"📈 '{sensor_name}' 최근 추이 (최근 100개)")
        ax.set_xlabel("시간순")
        ax.set_ylabel(sensor_name)
        ax.grid(True)
        return fig

    @output
    @render.ui
    def sensor_detail_modal():
        return None



    ##### 실시간 이상 데이터 테이블 (3시그마 or 불량만 강조 표시, 클릭 시 조건 카드 열림)
    # @output
    # @render.plot
    # def local_factor_plot():
    #     df = current_data()
    #     if df is None or df.empty:
    #         fig, ax = plt.subplots()
    #         ax.text(0.5, 0.5, "실시간 데이터 수신 대기 중...", ha="center", va="center", fontsize=13)
    #         ax.axis("off")
    #         return fig

    #     # === 1️⃣ 사용할 주요 컬럼만 선택 ===
    #     selected_cols = [
    #         # 공정 상태 관련
    #         "count", "speed_ratio", "pressure_speed_ratio",
    #         # 용융 단계
    #         "molten_temp",
    #         # 충진 단계
    #         "sleeve_temperature", "EMS_operation_time",
    #         "low_section_speed", "high_section_speed",
    #         "molten_volume", "cast_pressure", "mold_code",
    #         # 냉각 단계
    #         "upper_mold_temp1", "upper_mold_temp2", "upper_mold_temp3",
    #         "lower_mold_temp1", "lower_mold_temp2", "Coolant_temperature",
    #         # 공정 속도 관련
    #         "facility_operation_cycleTime", "production_cycletime",
    #         # 품질 및 성능
    #         "biscuit_thickness", "physical_strength",
    #     ]

    #     # 실제 df에 존재하고 수치형인 컬럼만 남김
    #     numeric_cols = [c for c in selected_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    #     if not numeric_cols:
    #         fig, ax = plt.subplots()
    #         ax.text(0.5, 0.5, "표시할 수치형 센서 데이터가 없습니다.", ha="center", va="center")
    #         ax.axis("off")
    #         return fig

    #     # === 2️⃣ Z-score 계산 ===
    #     mean_std = df[numeric_cols].describe().T[["mean", "std"]]
    #     latest = df.iloc[-1]
    #     z_scores = (latest - mean_std["mean"]) / mean_std["std"]
    #     z_scores = z_scores.dropna().sort_values(ascending=True)

    #     # 현재 그래프의 y축 카테고리 순서 저장
    #     plot_labels.set(list(z_scores.index))

    #     # === 3️⃣ 한글 라벨 매핑 ===
    #     labels = [label_map.get(col, col) for col in z_scores.index]

    #     # === 4️⃣ 그래프 ===
    #     colors = ["#e74c3c" if abs(z) > 3 else "#95a5a6" for z in z_scores]
    #     fig, ax = plt.subplots(figsize=(7, 5))
    #     ax.barh(range(len(z_scores)), z_scores.values, color=colors)
    #     ax.set_yticks(range(len(z_scores)))
    #     ax.set_yticklabels(labels)
    #     ax.set_xlabel("Z-score (표준편차 기준)")
    #     ax.set_title("실시간 이상 감지 센서 (클릭 시 상세보기)")
    #     ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    #     plt.tight_layout()
    #     return fig


    # ✅ 기존 local_factor_plot(실시간용) 교체
    @output
    @render.plot
    def local_factor_plot():
        df = current_data()
        if df is None or df.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "실시간 데이터 수신 대기 중...", ha="center", va="center", fontsize=13)
            ax.axis("off")
            return fig

        # 1) 사용할 주요 컬럼만 (UI에 있는 것만)
        selected_cols = [
            "count", "speed_ratio", "pressure_speed_ratio",
            "molten_temp",
            "sleeve_temperature", "EMS_operation_time",
            "low_section_speed", "high_section_speed",
            "molten_volume", "cast_pressure", "mold_code",
            "upper_mold_temp1", "upper_mold_temp2", "upper_mold_temp3",
            "lower_mold_temp1", "lower_mold_temp2", "Coolant_temperature",
            "facility_operation_cycleTime", "production_cycletime",
            "biscuit_thickness", "physical_strength",
        ]
        numeric_cols = [c for c in selected_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "표시할 수치형 센서 데이터가 없습니다.", ha="center", va="center")
            ax.axis("off")
            return fig

        # 2) Z-score
        mean_std = df[numeric_cols].describe().T[["mean", "std"]]
        latest = df.iloc[-1]
        z_scores = (latest[numeric_cols] - mean_std["mean"]) / mean_std["std"]
        z_scores = z_scores.dropna().sort_values(ascending=True)

        # y축 라벨 순서 저장(클릭 처리용)
        plot_labels.set(list(z_scores.index))

        # 3) 강도별 색상: |z|>2.5=빨강, |z|>1.5=노랑, else=회색
        colors = []
        for z in z_scores.values:
            if abs(z) > 2.5:
                colors.append("#e74c3c")   # 강한 이상
            elif abs(z) > 1.5:
                colors.append("#f1c40f")   # 주의
            else:
                colors.append("#95a5a6")   # 정상

        # 4) 한글 레이블
        ylabels = [label_map.get(c, c) for c in z_scores.index]

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(range(len(z_scores)), z_scores.values, color=colors)
        ax.set_yticks(range(len(z_scores)))
        ax.set_yticklabels(ylabels)
        ax.set_xlabel("Z-score (표준편차 기준)")
        ax.set_title("실시간 이상 감지 센서 (클릭 시 상세보기)")
        ax.grid(True, axis="x", linestyle="--", alpha=0.5)
        plt.tight_layout()
        return fig




    @reactive.effect
    @reactive.event(input.selected_row)
    def _show_snapshot():
        idx = input.selected_row()
        if idx is None:
            return

        df = current_data()
        if df is None or df.empty:
            return

        # ✅ 클릭된 시점 데이터 스냅샷 저장
        snapshot = df.iloc[:idx + 1].copy()
        snapshot_file = "/tmp/snapshot.csv"
        snapshot.to_csv(snapshot_file, index=False)
        selected_row.set(idx)

        ui.modal_show(
            ui.modal(
                ui.div(
                    ui.card(
                        ui.card_header(
                            "⚙ 선택된 시점의 공정 상태",
                            style="text-align:center; font-size:20px; font-weight:bold; color:#333;"
                        ),
                        ui.output_plot("local_factor_plot"),
                        ui.hr(),
                        ui.output_ui("local_factor_desc"),
                        ui.input_action_button(
                            "resume_realtime", "🔄 실시간 보기로 돌아가기",
                            class_="btn btn-outline-primary", style="margin-top:10px;"
                        )
                    )
                ),
                title="📋 상세 보기 (고정)",
                size="l",
                easy_close=True
            )
        )



    @reactive.effect
    @reactive.event(input.selected_row)
    def _show_condition_card():
        idx = input.selected_row()
        if idx is None:
            return

        # 클릭 시 '불량 및 공정 에러 발생 조건' 카드 모달로 표시
        ui.modal_show(
            ui.modal(
                ui.card(
                    ui.card_header(
                        "⚙ 불량 및 공정 에러 발생 조건",
                        style="text-align:center; font-size:20px; font-weight:bold; color:#333;"
                    ),
                    ui.output_plot("local_factor_plot"),
                    ui.hr(),
                    ui.output_ui("local_factor_desc"),
                    easy_close=True,
                ),
                title="📋 상세 조건 보기",
                size="xl",
                easy_close=True
            )
        )


    @reactive.effect
    @reactive.event(input.resume_realtime)
    def _resume_realtime():
        selected_row.set(None)
        ui.modal_remove()








# 🟢 TAB2. 품질 끝
# ============================================================




# ============================================================
# 🟢 TAB3. 데이터 분석
# ============================================================




# ============================================================
# 🟢 TAB3. 데이터 분석 (서버 로직)
# ============================================================

    # ------------------------------------------------------------
    # ⚙️ 1. 스트리밍 주기 (초)
    # ------------------------------------------------------------
    def stream_speed2() -> float:
        """루프 실행 주기 (초)"""
        return 1.0


    # ------------------------------------------------------------
    # ⚙️ 2. 전처리 함수 basic_fix (모델 pickle 참조용)
    # ------------------------------------------------------------
    def basic_fix(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # tryshot_signal 변환
        if "tryshot_signal" in df.columns:
            df["tryshot_signal"] = df["tryshot_signal"].apply(
                lambda x: 1 if str(x).upper() == "D" else 0
            )

        # speed_ratio 관련 처리
        if {"speed_ratio", "low_section_speed", "high_section_speed"} <= set(df.columns):
            df.loc[df["speed_ratio"].isin([np.inf, -np.inf]), "speed_ratio"] = -1
            df.loc[
                (df["low_section_speed"] == 0) & (df["high_section_speed"] == 0),
                "speed_ratio"
            ] = -2

        # pressure_speed_ratio 처리
        if "pressure_speed_ratio" in df.columns:
            df.loc[np.isinf(df["pressure_speed_ratio"]), "pressure_speed_ratio"] = -1

        return df


    # joblib 모델이 이 함수를 찾을 수 있게 등록
    import sys
    sys.modules["__main__"].basic_fix = basic_fix


    # ------------------------------------------------------------
    # ⚙️ 3. CSV 스트리머 클래스
    # ------------------------------------------------------------
    class MyStreamer:
        """CSV 파일을 한 줄씩 스트리밍 (앱 시작 시 미리 로드)"""

        def __init__(self, path, chunk_size=1, loop=True):
            self.path = pathlib.Path(path)
            self.chunk_size = chunk_size
            self.loop = loop
            self.index = 0
            self.df = None  # 처음에 None으로 초기화

            # ✅✅✅ 앱 시작 시 파일을 미리 로드합니다. ✅✅✅
            try:
                if not self.path.exists():
                    print(f"⚠️ [Streamer Init] 파일 없음: {self.path}")
                    return  # self.df는 None으로 유지됨

                print(f"⏳ [Streamer Init] {self.path.name} 로드 중...")
                self.df = pd.read_csv(self.path, low_memory=False)

                if self.df.empty:
                    print("⚠️ [Streamer Init] CSV 파일이 비어있음")
                    self.df = None  # 비어있으면 None으로 다시 설정
                else:
                    print(f"✅ [Streamer Init] MyStreamer 로드 완료 ({len(self.df)}행)")
            
            except Exception as e:
                print(f"⚠️ [Streamer Init] MyStreamer 로드 실패: {e}")
                self.df = None # 로드 실패 시 None으로 유지
            # ✅✅✅ 여기까지 수정 ✅✅✅


        def reset(self):
            self.index = 0
            print("🔄 MyStreamer 리셋")

        def stream(self):
            try:
                # ✅✅✅ 'self.df is None' 검사 로직 수정 ✅✅✅
                # (파일 로딩 코드를 __init__으로 옮겼습니다)
                if self.df is None:
                    print("⚠️ MyStreamer.df가 None입니다. (파일 로드 실패 또는 비어있음)")
                    return None
                # ✅✅✅ 여기까지 수정 ✅✅✅

                if self.index >= len(self.df):
                    if self.loop:
                        print("🔁 EOF → 루프 재시작")
                        self.index = 0
                    else:
                        print("🏁 스트리밍 종료")
                        return None

                chunk = self.df.iloc[self.index : self.index + self.chunk_size].copy()
                self.index += self.chunk_size
                return chunk

            except Exception as e:
                print(f"⚠️ MyStreamer 오류: {e}")
                return None

    # ------------------------------------------------------------
    # ⚙️ 4. 모델 및 메타 로드
    # ------------------------------------------------------------
    MODEL_PATH = "./models/fin_xgb_f20.pkl"
    META_PATH = "./models/fin_xgb_meta_f20.json"
    TARGET = "passorfail"

    try:
        print("🔍 모델 로드 중…")
        model = joblib.load(MODEL_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            META = json.load(f)
        print("✅ 모델 및 메타 로드 완료")
    except Exception as e:
        print(f"⚠️ 모델 로드 실패: {e}")
        model, META = None, {}

    model_features = META.get("features", [])
    best_threshold = float(META.get("best_threshold", 0.5))


    # ------------------------------------------------------------
    # ⚙️ 5. 상태 변수 정의
    # ------------------------------------------------------------
    analy_streamer = MyStreamer("./data/fin_test_kf_fixed.csv", chunk_size=1, loop=True)
    is_analysis_streaming = reactive.Value(False)
    analysis_data = reactive.Value(pd.DataFrame())
    log_df = reactive.Value(pd.DataFrame(columns=["time", "prob", "pred", "true", "result"]))
    latency_list = reactive.Value([])


    # ------------------------------------------------------------
    # ▶ 6. 스트리밍 제어 버튼
    # ------------------------------------------------------------
    @render.ui
    def stream_control_ui():
        btn_text = "⏹ 스트리밍 중지" if is_analysis_streaming() else "▶ 스트리밍 시작"
        color = "#d9534f" if is_analysis_streaming() else "#5cb85c"
        return ui.input_action_button(
            "toggle_stream", btn_text, style=f"background-color:{color};color:white;"
        )


    @reactive.effect
    @reactive.event(input.toggle_stream)
    def _toggle_stream():
        current = is_analysis_streaming()
        is_analysis_streaming.set(not current)
        if not current:
            analy_streamer.reset()
            print("▶ 스트리밍 시작됨")
        else:
            print("⏹ 스트리밍 중지됨")


    # ------------------------------------------------------------
    # ▶ 7. 스트리밍 루프
    # ------------------------------------------------------------
    @reactive.effect
    def _stream_loop():
        invalidate_later(stream_speed2())  # 주기적 실행
        if not is_analysis_streaming():
            return
        try:
            chunk = analy_streamer.stream()
            if chunk is not None and not chunk.empty:
                old = analysis_data()
                new_df = pd.concat([old, chunk], ignore_index=True).tail(500)
                analysis_data.set(new_df)
                print(f"📦 새 데이터 수신 ({len(chunk)}행)")
        except Exception as e:
            print(f"⚠️ 스트리밍 오류: {e}")


    # ------------------------------------------------------------
    # 🧠 8. 실시간 예측 루프
    # ------------------------------------------------------------
    @reactive.effect
    def _predict_loop():
        invalidate_later(stream_speed2())
        if not is_analysis_streaming() or model is None:
            return
        df = analysis_data()
        if df.empty:
            return

        try:
            latest = df.iloc[-1:].copy()
            try:
                latest = basic_fix(latest)
            except Exception:
                pass

            X = latest.drop(columns=[TARGET], errors="ignore")
            for col in model_features:
                if col not in X.columns:
                    X[col] = 0
            if model_features:
                X = X[model_features]

            if hasattr(model, "predict_proba"):
                prob = float(model.predict_proba(X)[0, 1])
            else:
                prob = float(model.predict(X)[0])

            pred = int(prob >= best_threshold)
            y_true = None
            if TARGET in latest.columns:
                try:
                    y_true = int(latest[TARGET].values[0])
                except Exception:
                    y_true = None

            result = "✅ 정상" if y_true is not None and pred == y_true else "❌ 불일치"
            ts = latest["real_time"].iloc[0] if "real_time" in latest.columns else datetime.datetime.now()

            latency = np.random.uniform(10, 50)
            latency_list.set((latency_list.get() + [latency])[-30:])

            new_row = pd.DataFrame([{
                "time": ts, "prob": prob, "pred": pred, "true": y_true, "result": result
            }])
            log_df.set(pd.concat([log_df(), new_row], ignore_index=True).tail(500))

        except Exception as e:
            print(f"⚠️ 예측 오류: {e}")


    # ------------------------------------------------------------
    # 📡 9. 통신 상태 표시
    # ------------------------------------------------------------
    @render.ui
    def comm_status():
        color = "green" if is_analysis_streaming() else "red"
        text = "정상 연결" if is_analysis_streaming() else "연결 끊김"
        return ui.HTML(f"<b>📡 통신 상태:</b> <span style='color:{color}'>{text}</span>")


    # ------------------------------------------------------------
    # 📈 10. Latency 그래프 (기존 코드)
    # ------------------------------------------------------------
    @render.plot
    def latency_plot():
        # ... (기존 latency_plot 코드) ...
        return fig

    # ✅✅✅ 10-B. [신규] 메인 예측 확률 그래프 ✅✅✅
    # ( latency_plot 함수 뒤에 추가하세요 )
    @render.plot
    def main_analysis_plot():
        df = log_df() # 실시간 로그 데이터 사용
        
        # 1단계에서 추가한 슬라이더 값 가져오기
        thresh = input.analysis_threshold() or 0.5 

        fig, ax = plt.subplots(figsize=(10, 4))
        
        if df.empty:
            ax.text(0.5, 0.5, "▶ 스트리밍을 시작하세요", ha="center", va="center", fontsize=14)
            ax.axis("off")
            return fig

        # 최근 100개 데이터만 표시
        df_tail = df.tail(100).reset_index(drop=True) 
        
        # 1. 예측 확률 라인 그래프 (스케치의 파란색 물결)
        ax.plot(df_tail.index, df_tail["prob"], marker='o', linestyle='-', label="예측 확률 (Prob)", zorder=2)
        
        # 2. Threshold 라인 (스케치의 빨간색 점선)
        ax.axhline(y=thresh, color='r', linestyle='--', label=f"Threshold ({thresh:.2f})", zorder=3)
        
        # 3. Threshold 상회 값 강조
        above = df_tail[df_tail["prob"] >= thresh]
        ax.scatter(above.index, above["prob"], color='red', zorder=5, label="불량 예측")

        ax.set_title("실시간 불량 예측 확률 (최근 100건)")
        ax.set_xlabel("Data Point (Recent)")
        ax.set_ylabel("Probability (0:양품 ~ 1:불량)")
        ax.set_ylim(0, 1) # Y축 0~1 고정
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig


   # ------------------------------------------------------------
    # 📈 10. Latency 그래프
    # ------------------------------------------------------------
    @render.plot
    def latency_plot():
        lst = latency_list.get()
        
        # ✅✅✅ `fig`와 `ax`를 if 문보다 먼저 정의합니다. ✅✅✅
        fig, ax = plt.subplots(figsize=(5, 3))
        
        if not lst:
            ax.text(0.5, 0.5, "Latency 데이터 없음", ha="center", va="center")
            ax.axis("off")
            return fig  # 👈 데이터가 없어도 `fig`를 반환
        
        # --- 데이터가 있을 때 그리는 로직 ---
        ax.plot(lst, marker="o", color="#5cb85c")
        ax.set_title("모델 응답 지연 (ms)")
        ax.set_ylabel("ms")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        
        return fig


    # ------------------------------------------------------------
    # 📜 12. 로그 뷰어
    # ------------------------------------------------------------
    @render.ui
    def log_viewer():
        # ... (기존 log_viewer 함수 코드) ...
        return ui.HTML(f"<div style='max-height:300px;overflow-y:auto;font-size:13px'>{html}</div>")

    # ✅✅✅ 여기부터 붙여넣기 시작 ✅✅✅
    # ------------------------------------------------------------
    # ⚙️ [신규] Mold Code 드롭다운 업데이트
    # ------------------------------------------------------------
    @reactive.effect
    def _update_mold_select():
        try:
            if analy_streamer.df is not None:
                # 스트리머 데이터에서 고유한 mold_code 목록 추출
                mold_codes = sorted(analy_streamer.df['mold_code'].unique().astype(str))
                
                # 드롭다운 선택지 생성 ({"all": "전체", "8412": "Mold Code 8412", ...})
                choices = {"all": "전체"}
                choices.update({code: f"Mold Code {code}" for code in mold_codes})
                
                # UI의 input_select 업데이트
                ui.update_select(
                    "analysis_mold_select",
                    choices=choices,
                    selected="all"
                )
                print("✅ Mold code 드롭다운 메뉴가 업데이트되었습니다.")
            else:
                print("⚠️ Mold code를 업데이트하기 위한 스트리머 데이터가 없습니다.")
        except Exception as e:
            print(f"❌ Mold code 드롭다운 업데이트 실패: {e}")

    # ------------------------------------------------------------
    # ⚙️ [신규] 선택된 Mold Code로 데이터 필터링
    # ------------------------------------------------------------
    @reactive.calc
    def filtered_log_df():
        df = log_df()
        selected_mold = input.analysis_mold_select()
        
        # '전체'가 선택되거나 데이터가 없으면 원본 반환
        if df.empty or selected_mold == "all":
            return df
        
        # 선택된 mold_code로 데이터 필터링하여 반환
        return df[df["mold_code"] == selected_mold].copy()
    # ✅✅✅ 여기까지 붙여넣기 끝 ✅✅✅

# =====================================================
# 📘 mold_code별 6시그마 계산
# =====================================================
# INPUT_FILE = "./data/fin_train.csv"
# OUTPUT_FILE = "./www/sixsigma_thresholds_by_mold.json"

# os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# df = pd.read_csv(INPUT_FILE)

# # 숫자형 컬럼만 선택 (mold_code 제외)
# num_cols = df.select_dtypes(include=["number"]).columns
# if "mold_code" in num_cols:
#     num_cols = num_cols.drop("mold_code")

# thresholds = {}

# for mold, group in df.groupby("mold_code"):
#     mold_dict = {}
#     for col in num_cols:
#         mu = group[col].mean()
#         sigma = group[col].std()

#         # NaN이나 비정상 값 처리
#         if pd.isna(mu) or pd.isna(sigma):
#             continue

#         mu = float(np.nan_to_num(mu, nan=0.0))
#         sigma = float(np.nan_to_num(sigma, nan=0.0))
#         mold_dict[col] = {"mu": round(mu, 4), "sigma": round(sigma, 4)}

#     thresholds[str(mold)] = mold_dict

# # 저장
# with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
#     json.dump(thresholds, f, ensure_ascii=False, indent=2, allow_nan=False)

# print(f"✅ mold_code별 6시그마 저장 완료: {len(thresholds)}개 금형 → {OUTPUT_FILE}")

# ======== 앱 실행 ========
app = App(app_ui, server, static_assets=app_dir / "www")
