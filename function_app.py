import logging
import requests
import psycopg2
from datetime import date, timedelta, datetime
import os
import azure.functions as func
from azure.functions import FunctionApp, HttpResponse

app = FunctionApp()

def safe_round(val, digits=2):
    import math
    if val is None:
        return None
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    try:
        return round(val, digits)
    except Exception:
        return None

# -------- 天気データの一括取得 --------
def fetch_weather_features_bulk():
    import pandas as pd

    logging.info("Fetching historical and forecast weather data")

    latitude, longitude = 35.6895, 139.6917
    today = datetime.now().date()
    logging.info(today)

    # 過去データ
    past_start = (today - timedelta(days=10)).isoformat()
    past_end = (today - timedelta(days=1)).isoformat()

    archive_url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={latitude}&longitude={longitude}"
        f"&start_date={past_start}&end_date={past_end}"
        f"&daily=temperature_2m_mean,precipitation_sum,wind_speed_10m_mean,"
        f"weather_code,shortwave_radiation_sum"
        f"&timezone=Asia%2FTokyo"
    )

    archive_resp = requests.get(archive_url)
    if archive_resp.status_code != 200:
        raise Exception("過去天気データの取得に失敗しました。")
    archive_data = archive_resp.json()["daily"]
    df_archive = pd.DataFrame(archive_data)

    # 未来データ
    future_start = today.isoformat()
    future_end = (today + timedelta(days=8)).isoformat()

    forecast_url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={latitude}&longitude={longitude}"
        f"&start_date={future_start}&end_date={future_end}"
        f"&daily=temperature_2m_mean,temperature_2m_max,temperature_2m_min,"
        f"precipitation_sum,wind_speed_10m_mean,wind_speed_10m_max,"
        f"relative_humidity_2m_max,relative_humidity_2m_min,"
        f"weather_code,shortwave_radiation_sum"
        f"&timezone=Asia%2FTokyo"
    )

    forecast_resp = requests.get(forecast_url)
    if forecast_resp.status_code != 200:
        raise Exception("未来天気データの取得に失敗しました。")
    forecast_data = forecast_resp.json()["daily"]
    df_forecast = pd.DataFrame(forecast_data)

    # 結合と前処理
    df_all = pd.concat([df_archive, df_forecast], ignore_index=True)
    df_all["time"] = pd.to_datetime(df_all["time"])
    df_all.set_index("time", inplace=True)

    df_all["temp_10d_avg"] = df_all["temperature_2m_mean"].rolling(window=10, min_periods=1).mean()
    df_all["temp_diff"] = df_all["temperature_2m_mean"] - df_all["temp_10d_avg"]
    df_all["weekday"] = df_all.index.dayofweek

    target_range = (df_all.index.date >= today) & (df_all.index.date <= today + timedelta(days=16))
    return df_all.loc[target_range]

# -------- ビール販売量の一括予測 --------
def predict_beer_sales_bulk():
    import lightgbm as lgb
    import pandas as pd

    logging.info("Starting bulk prediction")

    feature_df_full = fetch_weather_features_bulk()
    feature_cols = [
        "temperature_2m_mean", "precipitation_sum", "wind_speed_10m_mean",
        "weekday", "temp_diff", "temp_10d_avg", "weather_code", "shortwave_radiation_sum"
    ]
    feature_df = feature_df_full[feature_cols]

    target_cols = ["pale_ale", "lager", "ipa", "white", "dark", "fruit"]
    predictions = {}

    for col in target_cols:
        model_filename = f"model_{col}.txt"
        model_path = os.path.join(os.path.dirname(__file__), model_filename)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"モデルファイル {model_filename} が存在しません。")

        model = lgb.Booster(model_file=model_path)
        pred_values = model.predict(feature_df)
        predictions[col] = [round(p, 2) for p in pred_values]

    result = []
    for i, date in enumerate(feature_df.index):
        day_prediction = {"date": date.date().isoformat()}
        for col in target_cols:
            day_prediction[col] = predictions[col][i]

        weather_data = feature_df_full.iloc[i].to_dict()
        include_weather_keys = [
            "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
            "relative_humidity_2m_max", "relative_humidity_2m_min", "wind_speed_10m_max",
            "weather_code", "weekday"
        ]
        for key in include_weather_keys:
            value = weather_data.get(key)
            if value is not None:
                day_prediction[key] = safe_round(value, 2) if isinstance(value, (int, float)) else value

        result.append(day_prediction)

    return result

# -------- HTTPルート関数 (GET) --------
@app.function_name(name="pred")
@app.route(route="pred", methods=["GET"])
def pred(req: func.HttpRequest) -> func.HttpResponse:
    import json
    logging.basicConfig(level=logging.INFO)
    logging.info("beerAPI bulk prediction has started")

    try:
        results = predict_beer_sales_bulk()
        return HttpResponse(json.dumps(results, ensure_ascii=False), mimetype="application/json")
    except Exception as e:
        logging.error(f"bulk_pred_error: {str(e)}")
        return HttpResponse(f"予測に失敗しました: {str(e)}", status_code=500)

# -------- タイマートリガーでDBへ天気保存 --------
@app.function_name(name="timer_trigger_weather_to_db")
@app.schedule(schedule="0 1 * * *", arg_name="mytimer", run_on_startup=False, use_monitor=True)
def timer_trigger_weather_to_db(mytimer: func.TimerRequest) -> None:
    logging.info('Timer triggered!')

    yesterday = (date.today() - timedelta(days=1)).isoformat()

    res = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
        "latitude": 35.6812,
        "longitude": 139.7671,
        "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,weathercode,windspeed_10m_max,humidity_2m_max,humidity_2m_min",
        "start_date": yesterday,
        "end_date": yesterday,
        "timezone": "Asia/Tokyo"
    })

    if res.status_code != 200:
        logging.error("❌ 天気APIの取得に失敗しました")
        return

    data = res.json().get("daily")
    if not data or not data.get("weathercode"):
        logging.error("❌ 天気データが不足しています")
        return

    try:
        conn = psycopg2.connect(
            host=os.environ["DB_HOST"],
            dbname=os.environ["DB_NAME"],
            user=os.environ["DB_USER"],
            password=os.environ["DB_PASSWORD"],
            port="5432"
        )
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO sales_weather (
                date, weather_code_id, temperature_max, temperature_min,
                temperature_mean, windspeed_max, humidity_max, humidity_min,
                created_at, updated_at, is_deleted
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, now(), now(), false)
            ON CONFLICT (date) DO NOTHING;
        """, (
            yesterday,
            data["weathercode"][0],
            data["temperature_2m_max"][0],
            data["temperature_2m_min"][0],
            data["temperature_2m_mean"][0],
            data["windspeed_10m_max"][0],
            data["humidity_2m_max"][0],
            data["humidity_2m_min"][0]
        ))

        conn.commit()
        cur.close()
        conn.close()
        logging.info("✅ 天気データの登録完了")
    except Exception as e:
        logging.error(f"❌ データベース処理エラー: {e}")
