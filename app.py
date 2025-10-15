import pandas as pd
import joblib
import matplotlib.pyplot as plt
from shiny import App, ui, render, reactive, session
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
# ======== 실시간 스트리밍 대시보드 (현장 메뉴) ========
from shared import streaming_df, RealTimeStreamer
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF

# ✅ 표시에서 제외할 컬럼
EXCLUDE_COLS = ["id", "line", "name", "mold_name", "date", "time", "registration_time", "count"]

# ✅ 표시 대상: 위 제외 목록을 빼고 나머지 수치형 컬럼 자동 선택
display_cols = [
    c for c in streaming_df.columns
    if c not in EXCLUDE_COLS and pd.api.types.is_numeric_dtype(streaming_df[c])
]

# 스트리밍 초기 설정
streamer = reactive.Value(RealTimeStreamer(streaming_df[display_cols]))
current_data = reactive.Value(pd.DataFrame())
is_streaming = reactive.Value(False)

# ===== 한글 변수명 매핑 =====
VAR_LABELS = {
    # 용융 단계
    "molten_temp": "용융 온도(℃)",
    "heating_furnace": "용해로 정보",

    # 충진 단계
    "sleeve_temperature": "슬리브 온도(℃)",
    "EMS_operation_time": "EMS 가동시간(s)",
    "low_section_speed": "하부 주입속도(cm/s)",
    "high_section_speed": "상부 주입속도(cm/s)",
    "molten_volume": "주입 금속량(cc)",
    "cast_pressure": "주입 압력(bar)",

    # 냉각 단계
    "upper_mold_temp1": "상부1 금형온도(℃)",
    "upper_mold_temp2": "상부2 금형온도(℃)",
    "upper_mold_temp3": "상부3 금형온도(℃)",
    "lower_mold_temp1": "하부1 금형온도(℃)",
    "lower_mold_temp2": "하부2 금형온도(℃)",
    "lower_mold_temp3": "하부3 금형온도(℃)",
    "Coolant_temperature": "냉각수 온도(℃)",

    # 품질 및 속도
    "production_cycletime": "생산 사이클(sec)",
    "biscuit_thickness": "주조물 두께(mm)",
    "physical_strength": "제품 강도(MPa)",
}

# ===== 센서 위치 (x, y) =====
VAR_POSITIONS = {
    # 용융부
    "molten_temp": (750, 360),
    "heating_furnace": (810, 380),

    # 슬리브 / 주입
    "sleeve_temperature": (650, 330),
    "EMS_operation_time": (620, 280),
    "low_section_speed": (580, 250),
    "high_section_speed": (580, 210),
    "molten_volume": (620, 160),
    "cast_pressure": (590, 120),

    # 금형 냉각
    "upper_mold_temp1": (430, 180),
    "upper_mold_temp2": (400, 230),
    "upper_mold_temp3": (370, 280),
    "lower_mold_temp1": (430, 330),
    "lower_mold_temp2": (400, 380),
    "lower_mold_temp3": (370, 430),
    "Coolant_temperature": (300, 350),

    # 속도/품질
    "production_cycletime": (200, 460),
    "biscuit_thickness": (220, 420),
    "physical_strength": (220, 380),
}




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


# 예측 탭용 (모델 input 그대로)
df_predict = pd.read_csv("./data/train.csv")
df_predict["pressure_speed_ratio"] = df_predict["pressure_speed_ratio"].replace([np.inf, -np.inf], np.nan)

df_predict = df_predict[
    (df_predict["low_section_speed"] != 65535) &
    (df_predict["lower_mold_temp3"] != 65503) &
    (df_predict["physical_strength"] != 65535)
]

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

# ===== CSS (카드 전체 클릭영역) =====
card_click_css = """
<style>
/* 개요 전용 카드만 hover 효과 */
.overview-card {
    transition: transform 0.2s ease, box-shadow 0.2s ease, background-color 0.2s ease;
    position: relative;
}

.overview-card:hover {
    background-color: #f8f9fa !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    transform: translateY(-2px);
}

/* 카드 전체를 클릭 가능하게 하는 투명 버튼 */
.card-link {
    position: absolute;
    inset: 0;
    z-index: 10;
    cursor: pointer;
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
.card-link:hover,
.card-link:focus,
.card-link:active {
    background: transparent !important;
    box-shadow: none !important;
}
</style>
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



# ======== 전역 HEAD (favicon, CSS 등) ========
global_head = ui.head_content(
    ui.tags.link(rel="icon", type="image/x-icon", href="favicon.ico"),
    ui.tags.link(rel="stylesheet", href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"),
    ui.tags.link(rel="stylesheet", type="text/css", href="custom.css"),
    ui.tags.title("주조 공정 불량 예측 대시보드"),
    ui.tags.script("""
      Shiny.addCustomMessageHandler("updateSensors", function(values) {
        const units = {
          temp: "°C", Temp: "°C",
          pressure: "bar", Pressure: "bar",
          speed: "cm/s", Speed: "cm/s",
          volume: "cc", thickness: "mm",
          strength: "MPa", Strength: "MPa",
          cycle: "sec", time: "s"
        };

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

        function colorFor(key, val) {
          const k = key.toLowerCase();
          if (k.includes("temp")) {
            const c = Math.min(255, Math.max(0, Math.round(val*1.5)));
            return `rgb(${c},50,50)`;
          }
          if (k.includes("pressure")) {
            const c = Math.min(255, Math.max(0, Math.round(val*8)));
            return `rgb(50,${c},80)`;
          }
          if (k.includes("speed")) {
            const c = Math.min(255, Math.max(0, Math.round(val*6)));
            return `rgb(40,100,${c})`;
          }
          if (k.includes("strength")) {
            const c = Math.min(255, Math.max(0, Math.round(val*5)));
            return `rgb(${120+c/4},${80+c/5},${150+c/2})`;
          }
          return "#111827";
        }

        for (const [key, val] of Object.entries(values)) {
          if (typeof val !== "number" || isNaN(val)) continue;

          // ✅ 값 노드를 정확히 찾음: #var-<key> .value
          const valueNode = document.querySelector(`#var-${key} .value`);
          if (!valueNode) {
            console.log(`⚠️ '#var-${key} .value' 노드를 찾을 수 없습니다.`);
            continue;
          }

          const txt = `${val.toFixed(1)}${unitFor(key)}`;
          valueNode.textContent = txt;

          // 색상 반영
          valueNode.setAttribute("fill", colorFor(key, val));

          // 갱신 애니메이션
          valueNode.animate([{opacity:.3},{opacity:1}], {duration:350, iterations:1});
        }
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
            ui.h3("메뉴 선택", style="margin-bottom:30px; font-weight:bold;"),
            ui.div(
                {
                    "style": (
                        "display:grid; grid-template-columns:repeat(auto-fit, minmax(250px, 1fr)); "
                        "gap:20px; width:80%; max-width:800px;"
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
                    ui.p("EDA 및 주요 피처 분석 결과"),
                    ui.input_action_button("goto_analysis", "이동", class_="btn btn-outline-secondary mt-2"),
                ),
            ),
            ui.input_action_button("logout_btn", "로그아웃", class_="btn btn-light mt-5"),
        )
    )

def field_dashboard_ui():
    return ui.div(
        {"style": "display:grid; grid-template-columns:1fr 2fr; gap:20px;"},
        ui.card(
            ui.card_header("스트리밍 제어"),
            ui.input_action_button("start_stream", "▶ 시작", class_="btn btn-success me-1"),
            ui.input_action_button("pause_stream", "⏸ 일시정지", class_="btn btn-warning me-1"),
            ui.input_action_button("reset_stream", "🔄 리셋", class_="btn btn-secondary"),
            ui.hr(),
            ui.output_ui("stream_status"),
        ),
        ui.div(
            {"style": "display:flex; flex-direction:column; gap:20px;"},
            ui.card(
                ui.card_header("🧩 주조 공정 실시간 상태"),
                # ✅ PNG 그림 삽입
                # ui.tags.img(
                #     {
                #         "src": "diecast.png",  # ./www/diecast.png 경로
                #         "style": (
                #             "width:100%; max-width:900px; height:auto; "
                #             "border:2px solid #d0d7de; border-radius:8px; "
                #             "box-shadow:0 0 6px rgba(0,0,0,0.1);"
                #         )
                #     }
                # ),
                ui.output_ui("process_svg_inline")  # SVG와 병행 표시 가능
            ),
        )
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

# ======== 3️⃣ 본문 페이지 ========
def main_page(selected_tab: str):
    # --- 메뉴별 제목 및 본문 내용 ---
    tab_titles = {
        "field": "📊 현장 대시보드",
        "quality": "🧭 품질 모니터링",
        "analysis": "📈 데이터 분석"
    }
    tab_contents = {
        "field": field_dashboard_ui(),  # ✅ 실시간 대시보드 삽입
        # "quality": ui.h5("여기에 품질 모니터링 내용을 표시합니다."),

        # 🧭 품질 모니터링 (예측 시뮬레이션 UI 포함)
        "quality": ui.navset_tab(
            ui.nav_panel("예측",
                ui.div(
                    ui.card(
                        ui.card_header("입력 변수", style="background-color:#f8f9fa; text-align:center;"),

                        # 생산 환경 정보 카드
                        ui.card(
                            ui.card_header("생산 환경 정보", style="text-align:center;"),
                            ui.layout_columns(
                                ui.div(
                                    "생산 라인: A라인",
                                    style="background-color:#e9ecef; padding:8px 12px; border-radius:6px; text-align:center; font-weight:bold;"
                                ),
                                ui.div(
                                    "장비 이름: DC Machine 01",
                                    style="background-color:#e9ecef; padding:8px 12px; border-radius:6px; text-align:center; font-weight:bold;"
                                ),
                                ui.div(
                                    "금형 이름: Mold-01",
                                    style="background-color:#e9ecef; padding:8px 12px; border-radius:6px; text-align:center; font-weight:bold;"
                                ),
                                col_widths=[4, 4, 4]
                            )
                        ),

                        # === 공정 상태 관련 ===
                        ui.card(
                            ui.card_header("공정 상태 관련", style=""),
                            ui.layout_columns(
                                ui.input_numeric("count", "일조 누적 제품 개수", value=1000),
                                ui.input_numeric("monthly_count", "월간 누적 제품 개수", value=20000),
                                ui.input_numeric("global_count", "전체 누적 제품 개수", value=100000),
                                ui.input_numeric("speed_ratio", "상하 구역 속도 비율", value=95),
                                ui.input_numeric("pressure_speed_ratio", "주조 압력 속도 비율", value=90),
                                ui.input_select("working", "장비 가동 여부", choices=["가동", "정지"]),
                                ui.input_select("emergency_stop", "비상 정지 여부", choices=["정상", "비상정지"]),
                                ui.input_select("tryshot_signal", "측정 딜레이 여부", choices=["없음", "있음"]),
                                ui.input_select("shift", "근무조", choices=["주간", "야간"]),
                                col_widths=[3,3,3,3]
                            )
                        ),

                        # === 용융 단계 ===
                        ui.card(
                            ui.card_header("용융 단계", style=""),
                            ui.layout_columns(
                                ui.input_slider("molten_temp", "용융 온도(℃)", 600, 750, 680),
                                ui.input_select("heating_furnace", "용해로", choices=["F1", "F2", "F3"]),
                                col_widths=[6,6]
                            )
                        ),

                        # === 충진 단계 ===
                        ui.card(
                            ui.card_header("충진 단계", style=""),
                            ui.layout_columns(
                                ui.input_slider("sleeve_temperature", "슬리브 온도", 100, 200, 150),
                                ui.input_slider("EMS_operation_time", "EMS 작동 시간", 0, 10, 5),
                                ui.input_slider("low_section_speed", "저속 구간 속도", 0, 2, 1),
                                ui.input_slider("high_section_speed", "고속 구간 속도", 0, 5, 3),
                                ui.input_slider("molten_volume", "용탕량", 0, 100, 50),
                                ui.input_slider("cast_pressure", "주조 압력", 0, 200, 100),
                                ui.input_select("mold_code", "금형 코드", choices=["M1", "M2", "M3"]),
                                col_widths=[3,3,3,3]
                            )
                        ),

                        # === 냉각 단계 ===
                        ui.card(
                            ui.card_header("냉각 단계", style=""),
                            ui.layout_columns(
                                ui.input_slider("upper_mold_temp1", "상형 온도1", 0, 300, 150),
                                ui.input_slider("upper_mold_temp2", "상형 온도2", 0, 300, 160),
                                ui.input_slider("upper_mold_temp3", "상형 온도3", 0, 300, 155),
                                ui.input_slider("lower_mold_temp1", "하형 온도1", 0, 300, 140),
                                ui.input_slider("lower_mold_temp2", "하형 온도2", 0, 300, 145),
                                ui.input_slider("lower_mold_temp3", "하형 온도3", 0, 300, 150),
                                ui.input_slider("Coolant_temperature", "냉각수 온도", 0, 100, 25),
                                col_widths=[3,3,3,3]
                            )
                        ),

                        # === 공정 속도 관련 ===
                        ui.card(
                            ui.card_header("공정 속도 관련", style=""),
                            ui.layout_columns(
                                ui.input_slider("facility_operation_cycleTime", "설비 주기", 0, 100, 50),
                                ui.input_slider("production_cycletime", "생산 주기", 0, 100, 55),
                                col_widths=[6,6]
                            )
                        ),

                        # === 품질 및 성능 ===
                        ui.card(
                            ui.card_header("품질 및 성능", style=""),
                            ui.layout_columns(
                                ui.input_slider("biscuit_thickness", "비스킷 두께", 0, 10, 5),
                                ui.input_slider("physical_strength", "물리적 강도", 0, 100, 70),
                                col_widths=[6,6]
                            )
                        )
                    ),
                    style="max-width:1200px; margin:0 auto;"
                ),

                ui.br(),

                # === 예측 실행 카드 (하단 고정) ===
                ui.div(
                    ui.card(
                        ui.card_header(
                            ui.div(
                                [
                                    ui.input_action_button("predict_btn", "예측 실행", class_="btn btn-primary btn-lg", style="flex:1;"),
                                    ui.input_action_button("reset_btn", ui.HTML('<i class="fa-solid fa-rotate-left"></i>'),
                                                           class_="btn btn-secondary btn-lg",
                                                           style="margin-left:10px; width:60px;")
                                ],
                                style="display:flex; align-items:center; width:100%;"
                            ),
                            style="background-color:#f8f9fa; text-align:center;"
                        ),
                        ui.output_ui("prediction_result")
                    ),
                    style="""
                        position: sticky;
                        bottom: 1px;
                        z-index: 1000;
                        max-width: 1200px;
                        margin: 0 auto;
                        width: 100%;
                    """
                ),
            ),

            ui.nav_panel("개선 방안",
                ui.card(
                    ui.card_header("불량 기여 요인 Top 5", style="text-align:center;"),
                    ui.output_plot("local_factor_plot"),
                    ui.hr(),
                    ui.output_ui("local_factor_desc")
                )
            )
        ),




        "analysis": ui.h5("여기에 데이터 분석 결과를 표시합니다.")
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
        ui.h4(current_title),
        ui.div(tab_contents.get(selected_tab, ui.p("페이지 없음"))),
    )

    return ui.page_fluid(header_bar, content_area)



# ======== 전체 UI ========
app_ui = ui.page_fluid(global_head, ui.output_ui("main_ui"))

# ======== 서버 로직 ========
def server(input, output, session):

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

    # ======== 📈 데이터 분석 탭 ========
    DATA_PATH = pathlib.Path("./data/train_raw.csv")
    try:
        df_raw = pd.read_csv(DATA_PATH)
        print(f"✅ 데이터 로드 완료: {df_raw.shape}")
    except Exception as e:
        print("⚠️ 데이터 로드 실패:", e)
        df_raw = pd.DataFrame()

    # PDF 리포트 생성
    def generate_report(df):
        report_dir = os.path.join(APP_DIR, "report")
        os.makedirs(report_dir, exist_ok=True)
        pdf_path = os.path.join(report_dir, "Production_Achievement_Report.pdf")

        pdf = FPDF()
        pdf.add_page()
        pdf.add_font("Nanum", "", font_path, uni=True)
        pdf.set_font("Nanum", size=12)
        pdf.cell(0, 10, "📑 생산 계획 달성률 보고서", ln=True, align="C")
        pdf.ln(10)

        target = 1000
        achieved = len(df)
        rate = achieved / target * 100
        pdf.multi_cell(0, 8, f"이번 기간 달성률: {rate:.1f}%")
        pdf.multi_cell(0, 8, "주요 저하 원인:\n - 설비 온도 불안정\n - 냉각수 지연\n - 교대 시 세팅 시간 증가")

        if "mold_code" in df.columns:
            pdf.ln(5)
            pdf.cell(0, 8, "공정별 달성률:", ln=True)
            for m, v in (df["mold_code"].value_counts(normalize=True) * 100).items():
                pdf.cell(0, 8, f" - Mold {m}: {v:.1f}%", ln=True)

        pdf.ln(8)
        pdf.cell(0, 8, f"설비 가동률: {np.random.uniform(85,97):.1f}%", ln=True)
        pdf.output(pdf_path)
        return pdf_path

    # -------- UI 내용 --------
    @output
    @render.ui
    def analysis_content():
        return ui.div(
            ui.h4("📊 생산 계획 달성률 분석"),
            output_widget("ach_rate"),
            output_widget("mold_pie"),
            output_widget("delay_pie"),
            output_widget("cond_box"),
            ui.input_action_button("make_report", "📑 PDF 리포트 생성", class_="btn btn-primary mt-4"),
            ui.output_text("report_msg")
        )

    # -------- 그래프들 --------
    @output
    @render_plotly
    def ach_rate():
        if df_raw.empty:
            return go.Figure()
        df_raw["idx"] = range(1, len(df_raw) + 1)
        fig = px.line(df_raw, x="idx", y=df_raw.columns[1], title="📈 생산 달성률 추이")
        return fig

    @output
    @render_plotly
    def mold_pie():
        if "mold_code" not in df_raw.columns:
            return go.Figure()
        share = df_raw["mold_code"].value_counts(normalize=True) * 100
        fig = go.Figure(go.Pie(labels=share.index, values=share.values, textinfo="label+percent"))
        fig.update_layout(title="몰드별 생산 비율")
        return fig

    @output
    @render_plotly
    def delay_pie():
        labels = ["냉각수 지연", "작업자 교대", "금형 세정", "설비 점검"]
        values = np.random.randint(5, 15, len(labels))
        fig = go.Figure(go.Pie(labels=labels, values=values, textinfo="label+value"))
        fig.update_layout(title="딜레이 요인 분석")
        return fig

    @output
    @render_plotly
    def cond_box():
        cols = [c for c in ["molten_temp", "injection_pressure", "upper_plunger_speed", "cooling_temp"] if c in df_raw.columns]
        if not cols:
            return go.Figure()
        dfm = df_raw[cols].melt()
        fig = px.box(dfm, x="variable", y="value", title="생산 컨디션 분포", points="all")
        return fig

    @output
    @render.text
    @reactive.event(input.make_report)
    def report_msg():
        if df_raw.empty:
            return "⚠️ 데이터가 없습니다."
        path = generate_report(df_raw)
        return f"✅ 리포트 생성 완료: {path}"

    # ===== 실시간 스트리밍 로직 =====
    @output
    @render.ui
    def stream_status():
        return ui.div("🟢 스트리밍 중" if is_streaming() else "🔴 정지됨")

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

    @output
    @render.table
    def recent_data_table():
        df = current_data()
        if df.empty:
            return pd.DataFrame({"상태": ["데이터 없음"]})
        return df.tail(10).round(2)

    # 버튼 동작
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
    def _reset_stream():
        streamer().reset_stream()
        current_data.set(pd.DataFrame())
        is_streaming.set(False)

    # 주기적 업데이트
    @reactive.effect
    async def _auto_update():
        if not is_streaming():
            return

        reactive.invalidate_later(1)
        s = streamer()
        next_batch = s.get_next_batch(1)
        if next_batch is not None:
            current_data.set(s.get_current_data())

            latest = next_batch.iloc[-1].to_dict()
            # ✅ NaN → None 으로 바꿔서 JSON 직렬화 가능하게 함
            clean_values = {}
            for k, v in latest.items():
                if isinstance(v, (int, float)):
                    if pd.isna(v):
                        clean_values[k] = 0.0   # 또는 None, 0.0 중 선택 가능
                    else:
                        clean_values[k] = float(v)
            await session.send_custom_message("updateSensors", clean_values)
        else:
            is_streaming.set(False)

    @output
    @render.ui
    def process_svg_inline():
        def make_item(key: str, label: str, x: int, y: int) -> str:
            return (
                f"<text id='var-{key}' x='{x}' y='{y}'>"
                f"  <tspan class='label'>{label}: </tspan>"
                f"  <tspan class='value'>—</tspan>"
                f"</text>"
            )

        svg_items = []
        for key, label in VAR_LABELS.items():
            if key not in VAR_POSITIONS:
                continue
            x, y = VAR_POSITIONS[key]
            svg_items.append(make_item(key, label, x, y))

        svg_html = "\n".join(svg_items)

        return ui.HTML(f"""
            <div style="
                position:relative;
                width:900px;
                height:500px;
                margin:auto;
                border:1px solid #ccc;
                border-radius:8px;
                overflow:hidden;
                background-color:#f8f9fa;">
                
                <!-- 배경 이미지 -->
                <img src="diecast.png" 
                    style="
                        position:absolute;
                        top:0; left:0;
                        width:100%; height:100%;
                        object-fit:contain;
                        z-index:1;"/>

                <!-- SVG 오버레이 -->
                <div style="
                    position:absolute; top:0; left:0;
                    width:100%; height:100%;
                    z-index:2; pointer-events:none;">
                    <svg xmlns='http://www.w3.org/2000/svg'
                        width='100%' height='100%'
                        viewBox='0 0 900 500'
                        preserveAspectRatio='xMidYMid meet'>
                        <style>
                            text {{
                                font-family: 'NanumGothic','Malgun Gothic',sans-serif;
                                font-weight: 700;
                                font-size: 15px;
                                fill: #111827;
                                stroke: #fff;
                                stroke-width: .6px;
                                paint-order: stroke;
                            }}
                            .value {{ fill:#111827; }}
                        </style>
                        {svg_html}
                    </svg>
                </div>
            </div>
        """)

# ======== 앱 실행 ========
app = App(app_ui, server, static_assets=app_dir / "www")
