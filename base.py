#######################################################
# 몰튼 탬프 끝

import pandas as pd
import plotly.graph_objects as go
from pykalman import KalmanFilter

# 데이터 불러오기
train = pd.read_csv('./data/train.csv')

# tryshot_signal D → 1, 아니면 0
train['tryshot_signal_D'] = train['tryshot_signal'].apply(lambda x: 1 if x == 'D' else 0)

# real_time → datetime
train['real_time'] = pd.to_datetime(train['real_time'])

# 결측값 여부
train['is_nan'] = train['molten_temp'].isna()

# Kalman Filter 적용
kf = KalmanFilter(
    transition_matrices=[1],
    observation_matrices=[1],
    initial_state_mean=train['molten_temp'].dropna().iloc[0],
    observation_covariance=1,
    transition_covariance=0.1
)
state_means, _ = kf.filter(train['molten_temp'].fillna(method='ffill').values)
train['molten_temp_kf'] = state_means.flatten()

# 스무싱
train['molten_temp_kf_smooth'] = train['molten_temp_kf'].rolling(window=10, min_periods=1).mean()

# 시각화
fig = go.Figure()

# KF 보정 선 (진한 빨강)
fig.add_trace(go.Scatter(
    x=train['real_time'],
    y=train['molten_temp_kf'],
    mode='lines',
    name='Kalman Filtered',
    line=dict(color='#b30000', width=2.5)
))

# 스무싱 선 (진한 검정)
fig.add_trace(go.Scatter(
    x=train['real_time'],
    y=train['molten_temp_kf_smooth'],
    mode='lines',
    name='Smoothed (Rolling 10)',
    line=dict(color='black', width=2.5)
))

# passorfail=1 점 (연한 빨강)
fig.add_trace(go.Scatter(
    x=train.loc[train['passorfail']==1, 'real_time'],
    y=train.loc[train['passorfail']==1, 'molten_temp_kf'],
    mode='markers',
    name='passorfail=1',
    marker=dict(color='#ff9999', size=6, opacity=0.7)
))

# passorfail=0 점 (연한 파랑)
fig.add_trace(go.Scatter(
    x=train.loc[train['passorfail']==0, 'real_time'],
    y=train.loc[train['passorfail']==0, 'molten_temp_kf'],
    mode='markers',
    name='passorfail=0',
    marker=dict(color='#99ccff', size=6, opacity=0.7)
))

# 결측값 위치 X 마커 (검정)
fig.add_trace(go.Scatter(
    x=train.loc[train['is_nan'], 'real_time'],
    y=train.loc[train['is_nan'], 'molten_temp_kf'],
    mode='markers',
    name='NaN',
    marker=dict(color='black', size=6, symbol='x', opacity=0.9)
))

# tryshot_signal D 점 (초록)
fig.add_trace(go.Scatter(
    x=train.loc[train['tryshot_signal_D']==1, 'real_time'],
    y=train.loc[train['tryshot_signal_D']==1, 'molten_temp_kf'],
    mode='markers',
    name='tryshot_signal=D',
    marker=dict(color='green', size=5, opacity=0.7)
))

fig.update_layout(
    title='Molten Temp: KF + Smoothed with passorfail and NaN Highlighted',
    xaxis_title='Real Time',
    yaxis_title='Molten Temperature',
    template='plotly_white',
    height=600
)

fig.show()




##################################################################
train.isna().sum()

import pandas as pd
import plotly.graph_objects as go

# 데이터 불러오기
train = pd.read_csv('./data/train.csv')

# real_time → datetime
train['real_time'] = pd.to_datetime(train['real_time'])

# 결측값 여부
train['is_nan'] = train['low_section_speed'].isna()

# 시계열 라인
fig = go.Figure()

# 원본 시계열 선
fig.add_trace(go.Scatter(
    x=train['real_time'],
    y=train['low_section_speed'],
    mode='lines+markers',
    name='low_section_speed',
    line=dict(color='#1f77b4', width=2),
    marker=dict(size=4, opacity=0.7)
))

# 결측값 X 표시
fig.add_trace(go.Scatter(
    x=train.loc[train['is_nan'], 'real_time'],
    y=[0]*train['is_nan'].sum(),  # 시각적으로 0 위치
    mode='markers',
    name='Missing Value',
    marker=dict(color='red', size=6, symbol='x', opacity=0.9)
))

fig.update_layout(
    title='Low Section Speed over Time',
    xaxis_title='Real Time',
    yaxis_title='Low Section Speed',
    template='plotly_white',
    height=500
)

fig.show()
train.isna().sum()


#################################################################################

import pandas as pd
import plotly.graph_objects as go
from pykalman import KalmanFilter

# 데이터 불러오기
train = pd.read_csv('./data/train.csv')

# tryshot_signal D → 1, 아니면 0
train['tryshot_signal_D'] = train['tryshot_signal'].apply(lambda x: 1 if x == 'D' else 0)

# real_time → datetime
train['real_time'] = pd.to_datetime(train['real_time'])

# molten_volume 결측 여부
train['is_nan_volume'] = train['molten_volume'].isna()

# Kalman Filter 적용
kf_volume = KalmanFilter(
    transition_matrices=[1],
    observation_matrices=[1],
    initial_state_mean=train['molten_volume'].dropna().iloc[0],
    observation_covariance=1,
    transition_covariance=0.1
)
state_means_vol, _ = kf_volume.filter(train['molten_volume'].fillna(method='ffill').values)
train['molten_volume_kf'] = state_means_vol.flatten()

# 스무싱
train['molten_volume_kf_smooth'] = train['molten_volume_kf'].rolling(window=10, min_periods=1).mean()

# 시각화
fig = go.Figure()

# KF 보정 선 (진한 빨강)
fig.add_trace(go.Scatter(
    x=train['real_time'],
    y=train['molten_volume_kf'],
    mode='lines',
    name='Kalman Filtered',
    line=dict(color='#b30000', width=2.5)
))

# 스무싱 선 (진한 검정)
fig.add_trace(go.Scatter(
    x=train['real_time'],
    y=train['molten_volume_kf_smooth'],
    mode='lines',
    name='Smoothed (Rolling 10)',
    line=dict(color='black', width=2.5)
))

# passorfail=1 점 (연한 빨강)
fig.add_trace(go.Scatter(
    x=train.loc[train['passorfail']==1, 'real_time'],
    y=train.loc[train['passorfail']==1, 'molten_volume_kf'],
    mode='markers',
    name='passorfail=1',
    marker=dict(color='#ff9999', size=6, opacity=0.7)
))

# passorfail=0 점 (연한 파랑)
fig.add_trace(go.Scatter(
    x=train.loc[train['passorfail']==0, 'real_time'],
    y=train.loc[train['passorfail']==0, 'molten_volume_kf'],
    mode='markers',
    name='passorfail=0',
    marker=dict(color='#99ccff', size=6, opacity=0.7)
))

# 결측값 위치 X 마커 (검정)
fig.add_trace(go.Scatter(
    x=train.loc[train['is_nan_volume'], 'real_time'],
    y=train.loc[train['is_nan_volume'], 'molten_volume_kf'],
    mode='markers',
    name='NaN',
    marker=dict(color='black', size=6, symbol='x', opacity=0.9)
))

# tryshot_signal D 점 (초록)
fig.add_trace(go.Scatter(
    x=train.loc[train['tryshot_signal_D']==1, 'real_time'],
    y=train.loc[train['tryshot_signal_D']==1, 'molten_volume_kf'],
    mode='markers',
    name='tryshot_signal=D',
    marker=dict(color='green', size=5, opacity=0.7)
))

fig.update_layout(
    title='Molten Volume: KF + Smoothed with passorfail and NaN Highlighted',
    xaxis_title='Real Time',
    yaxis_title='Molten Volume',
    template='plotly_white',
    height=600
)

fig.show()

#################################################################

import numpy as np
import pandas as pd
from pykalman import KalmanFilter

# 1️⃣ 데이터 불러오기
df = pd.read_csv('./data/train.csv')

# 2️⃣ KF로 결측치 채울 칼럼
kf_columns = ['molten_temp', 'molten_volume', 'upper_mold_temp3']

# 3️⃣ 각 칼럼마다 KF 적용 후 바로 기존 칼럼에 결측치 채우기
for col in kf_columns:
    if col in df.columns:
        kf = KalmanFilter(
            transition_matrices=[1],
            observation_matrices=[1],
            initial_state_mean=df[col].dropna().iloc[0],
            observation_covariance=1,
            transition_covariance=0.1
        )
        state_means, _ = kf.filter(df[col].fillna(method='ffill').values)
        mask = df[col].isna()
        df.loc[mask, col] = state_means.flatten()[mask]

# 4️⃣ 삭제할 칼럼만 제거
if 'lower_mold_temp3' in df.columns:
    df = df.drop(columns=['lower_mold_temp3'])


df.isna().sum()

df.dropna(subset= 'working', inplace=True)

# 5️⃣ CSV 저장
df.to_csv('./data/train_res.csv', index=False)

print("✅ KF로 결측치 바로 채우고, lower_mold_temp3만 제거 완료, train_res.csv 저장됨")


