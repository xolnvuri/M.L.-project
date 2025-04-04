import requests
import json
import pandas as pd
from prophet import Prophet
import pyarrow.parquet as pq
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import matplotlib
import platform

# 한글 폰트 설정 (운영체제별)
if platform.system() == 'Windows':
    matplotlib.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':  # macOS
    matplotlib.rc('font', family='AppleGothic')
else:  # Linux
    matplotlib.rc('font', family='NanumGothic')

# 음수 깨짐 방지
matplotlib.rcParams['axes.unicode_minus'] = False

# API 호출 함수
def fetch_bitcoin_data(days):
    url = f"https://api.upbit.com/v1/candles/days?market=KRW-BTC&count={days}"
    headers = {"Accept": "application/json"}
    response = requests.get(url, headers=headers)
    data = json.loads(response.text)
    df = pd.DataFrame([{
        'datetime': item['candle_date_time_kst'],
        'price': item['trade_price'],
        'volume': item['candle_acc_trade_price']
    } for item in data])

    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime')

    # 이동평균 및 노이즈 감소
    df['price'] = df['price'].rolling(window=5, min_periods=1).mean()
    df['volume'] = df['volume'].rolling(window=5, min_periods=1).mean()

    # RSI 계산
    df['delta'] = df['price'].diff()
    df['gain'] = df['delta'].where(df['delta'] > 0, 0)
    df['loss'] = -df['delta'].where(df['delta'] < 0, 0)
    df['avg_gain'] = df['gain'].rolling(window=14, min_periods=1).mean()
    df['avg_loss'] = df['loss'].rolling(window=14, min_periods=1).mean()
    df['rs'] = df['avg_gain'] / df['avg_loss']
    df['rsi'] = 100 - (100 / (1 + df['rs']))

    df = df.dropna(subset=['rsi', 'price', 'volume'])
    return df[['datetime', 'price', 'volume', 'rsi']]

# 데이터 가져오기 (3년치: 약 1095일)
bitcoin_data = fetch_bitcoin_data(1095)

# 데이터 전처리
bitcoin_data = bitcoin_data[bitcoin_data['price'] > 0]

# Prophet 모델 학습 준비
df = bitcoin_data.rename(columns={'datetime': 'ds', 'price': 'y'})
df['volume'] = bitcoin_data['volume']
df['rsi'] = bitcoin_data['rsi']

# 이벤트 날짜 예시 (비트코인 반감기 등)
events = pd.DataFrame({
    'ds': pd.to_datetime(['2020-05-11', '2024-04-20']),
    'event': ['halving_2020', 'halving_2024']
})

# events 데이터 중 df에 있는 날짜만 유지
events = events[events['ds'].between(df['ds'].min(), df['ds'].max())]
df = df.merge(events, on='ds', how='left')
df['event'] = df['event'].notna().astype(int)

# Prophet 모델 생성 및 외부 요인 추가
model = Prophet(
    yearly_seasonality=True,
    daily_seasonality=True,
    seasonality_mode='additive',
    changepoint_prior_scale=0.05,
    n_changepoints=150
)
model.add_seasonality(name='weekly', period=7, fourier_order=10)
model.add_seasonality(name='monthly', period=30.5, fourier_order=10)
model.add_regressor('volume')
model.add_regressor('rsi')
model.add_regressor('event', standardize=False)
model.fit(df)

# 예측 기간 설정
future = model.make_future_dataframe(periods=180)
future['ds'] = pd.to_datetime(future['ds'])

# 예측용 데이터 채우기
last_volume = df['volume'].iloc[-1]
last_rsi = df['rsi'].iloc[-1]
future['volume'] = future['ds'].apply(lambda x: df.loc[df['ds'] <= x, 'volume'].iloc[-1] if any(df['ds'] <= x) else last_volume)
future['rsi'] = future['ds'].apply(lambda x: df.loc[df['ds'] <= x, 'rsi'].iloc[-1] if any(df['ds'] <= x) else last_rsi)

# future 데이터에 이벤트 추가
events_future = events[events['ds'].between(future['ds'].min(), future['ds'].max())]
future = future.merge(events_future, on='ds', how='left')
future['event'] = future['event'].notna().astype(int)

# 예측 수행
forecast = model.predict(future)

# 저장
forecast.to_parquet('bitcoin_forecast.parquet')

# 출력 포맷 수정 (전체 예측 결과 출력)
formatted_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
formatted_forecast['ds'] = formatted_forecast['ds'].dt.strftime('%Y-%m-%d')
for col in ['yhat', 'yhat_lower', 'yhat_upper']:
    formatted_forecast[col] = formatted_forecast[col].map(lambda x: f'{x:,.0f}')
print(formatted_forecast)

# 시각화 (예측만 미래구간에 표시)
plt.figure(figsize=(12, 6))
plt.plot(df['ds'], df['y'], label='과거 가격', color='blue')
future_only = forecast[forecast['ds'] > df['ds'].max()]
plt.plot(future_only['ds'], future_only['yhat'], label='예측 가격', color='red')
plt.fill_between(future_only['ds'], future_only['yhat_lower'], future_only['yhat_upper'], color='pink', alpha=0.3)

plt.xlabel('날짜')
plt.ylabel('가격 (₩)')
plt.title('비트코인 가격 예측 (거래량, RSI, 이벤트 포함)', fontsize=14)
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
plt.tight_layout()
plt.show()
