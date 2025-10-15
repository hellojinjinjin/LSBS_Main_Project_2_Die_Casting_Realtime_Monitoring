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
        "quality": ui.h5("여기에 품질 모니터링 내용을 표시합니다."),
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
    
    # ===== 드롭다운 메뉴 항목 클릭 시 페이지 전환 =====
    @reactive.effect
    @reactive.event(input.goto_field)
    def _goto_field():
        page_state.set("field")

    @reactive.effect
    @reactive.event(input.goto_quality)
    def _goto_quality():
        page_state.set("quality")

    @reactive.effect
    @reactive.event(input.goto_analysis)
    def _goto_analysis():
        page_state.set("analysis")

    # ===== 뒤로가기 버튼: 카드 선택 페이지로 복귀 =====
    @reactive.effect
    @reactive.event(input.back_btn)
    def _go_back():
        page_state.set("menu")

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
