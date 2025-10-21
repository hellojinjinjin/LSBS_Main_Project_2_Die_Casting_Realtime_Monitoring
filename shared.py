# shared.py
import pandas as pd
from pathlib import Path

app_dir = Path(__file__).parent
try:
    streaming_df = pd.read_csv(app_dir / "./data/test.csv", encoding="utf-8")
except UnicodeDecodeError:
    streaming_df = pd.read_csv(app_dir / "./data/test.csv", encoding="cp949")

try:
    kf_streaming_df = pd.read_csv(app_dir / "./data/ffin_test_kf_fixed.csv", encoding="utf-8")
except UnicodeDecodeError:
    kf_streaming_df = pd.read_csv(app_dir / "./data/ffin_test_kf_fixed.csv", encoding="cp949")

# real_time 컬럼을 datetime 형식으로 변환
streaming_df["real_time"] = pd.to_datetime(streaming_df["real_time"], errors="coerce")
kf_streaming_df["real_time"] = pd.to_datetime(kf_streaming_df["real_time"], errors="coerce")

# # 오름차순 정렬
# streaming_df = streaming_df.sort_values(by="real_time", ascending=True).reset_index(drop=True)
# streaming_df.to_csv("./data/test.csv", encoding="utf-8-sig", index=False)

class RealTimeStreamer:
    def __init__(self, data: pd.DataFrame):
        self.full_data = data.reset_index(drop=True).copy()
        self.current_index = 0

    def get_next_batch(self, batch_size: int = 1):
        if self.current_index >= len(self.full_data):
            return None
        end_index = min(self.current_index + batch_size, len(self.full_data))
        batch = self.full_data.iloc[self.current_index:end_index].copy()
        self.current_index = end_index
        return batch

    def get_current_data(self):
        if self.current_index == 0:
            return pd.DataFrame()
        return self.full_data.iloc[: self.current_index].copy()

    def reset_stream(self):
        self.current_index = 0

# ======================================
# ✅ Kalman Filter 기반 실시간 스트리머
# ======================================
import pandas as pd
from pathlib import Path

class KFStreamer:
    def __init__(self, data: pd.DataFrame):
        self.full_data = data.reset_index(drop=True).copy()
        self.current_index = 0

    def get_next_batch(self, batch_size: int = 1):
        if self.current_index >= len(self.full_data):
            return None
        end_index = min(self.current_index + batch_size, len(self.full_data))
        batch = self.full_data.iloc[self.current_index:end_index].copy()
        self.current_index = end_index
        return batch

    def get_current_data(self):
        if self.current_index == 0:
            return pd.DataFrame()
        return self.full_data.iloc[:self.current_index].copy()

    def reset_stream(self):
        self.current_index = 0
