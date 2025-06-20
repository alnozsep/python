import logging
import os
import math
import json
from datetime import date, timedelta, datetime
features = []
import azure.functions as func
from azure.functions import FunctionApp, HttpResponse
import requests
import pandas as pd
import joblib
import lightgbm as lgb
import pickle
import jpholiday
import numpy as np

url = (
    f"https://archive-api.open-meteo.com/v1/archive?"
    f"latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}"
    f"&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
    f"precipitation_sum,precipitation_hours,relative_humidity_2m_max,relative_humidity_2m_min,"
    f"windspeed_10m_max,windspeed_10m_mean,windgusts_10m_max,"
    f"shortwave_radiation_sum,et0_fao_evapotranspiration,"
    f"sunset,"
    f"uv_index_max&timezone=Asia%2FTokyo"
)
features = [
    "temperature_2m_max", "temperature_2m_min", "precipitation_sum", "precipitation_hours",
    "windspeed_10m_mean", "windspeed_10m_max", "shortwave_radiation_sum",
    "temp_10d_avg", 
    "is_weekend", "is_holiday_flag", "is_day_before_holiday",
    "relative_humidity_2m_max", "relative_humidity_2m_min", "et0_fao_evapotranspiration",
    "windgusts_10m_max", "weekday", "month","temp2","prc2","temperature_2m_mean","tempprc","shiny_holiday",
    "et02"
] 
app = FunctionApp()

def safe_round(val, digits=2):
    if val is None:
        return None
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    try:
        return round(val, digits)
    except Exception:
        return None

def fetch_weather_features_bulk():
    latitude, longitude = 35.6895, 139.6917
    today = datetime.now().date()

    past_start = (today - timedelta(days=14)).isoformat()  # 14日前から
    past_end = (today - timedelta(days=1)).isoformat()

    archive_url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={latitude}&longitude={longitude}"
        f"&start_date={past_start}&end_date={past_end}"
        f"&daily=temperature_2m_mean,precipitation_sum,wind_speed_10m_mean,"
        f"weather_code,shortwave_radiation_sum,"
        f"temperature_2m_max,temperature_2m_min,"
        f"precipitation_hours,relative_humidity_2m_max,relative_humidity_2m_min,"
        f"windspeed_10m_max,windspeed_10m_mean,windgusts_10m_max,"
        f"et0_fao_evapotranspiration,uv_index_max"
        f"&timezone=Asia%2FTokyo"
    )

    archive_resp = requests.get(archive_url)
    if archive_resp.status_code != 200:
        raise Exception("過去天気データの取得に失敗しました。")
    archive_data = archive_resp.json()["daily"]
    df_archive = pd.DataFrame(archive_data)

    future_start = today.isoformat()
    future_end = (today + timedelta(days=8)).isoformat()

    forecast_url = (
        f"https://archive-api.open-meteo.com/v1/forecast?"
        f"latitude={latitude}&longitude={longitude}"
        f"&start_date={future_start}&end_date={future_end}"
        f"&daily=temperature_2m_mean,precipitation_sum,wind_speed_10m_mean,"
        f"weather_code,shortwave_radiation_sum,"
        f"temperature_2m_max,temperature_2m_min,"
        f"precipitation_hours,relative_humidity_2m_max,relative_humidity_2m_min,"
        f"windspeed_10m_max,windspeed_10m_mean,windgusts_10m_max,"
        f"et0_fao_evapotranspiration,uv_index_max"
        f"&timezone=Asia%2FTokyo"
    )

    features += [
    "temperature_2m_max", "temperature_2m_min", "precipitation_sum", "precipitation_hours",
    "windspeed_10m_mean", "windspeed_10m_max", "shortwave_radiation_sum",
    "temp_10d_avg", 
    "is_weekend", "is_holiday_flag", "is_day_before_holiday",
    "relative_humidity_2m_max", "relative_humidity_2m_min", "et0_fao_evapotranspiration",
    "windgusts_10m_max", "weekday", "month", "temp2", "prc2", "temperature_2m_mean", "tempprc", "shiny_holiday",
    "et02"]


    forecast_resp = requests.get(forecast_url)
    if forecast_resp.status_code != 200:
        raise Exception("未来天気データの取得に失敗しました。")
    forecast_data = forecast_resp.json()["daily"]
    df_forecast = pd.DataFrame(forecast_data)

    merged_df = pd.concat([df_archive, df_forecast], ignore_index=True)
    merged_df["time"] = pd.to_datetime(merged_df["time"])
    merged_df.set_index("time", inplace=True)

    # ■ 平均気温との乖離
    for window_size in range(1, 15):  # 1日から14日まで
        col_avg = f"temp_{window_size}d_avg"
        col_diff = f"temp_diff_{window_size}d"
        merged_df[col_avg] = merged_df["temperature_2m_mean"].rolling(window=window_size, min_periods=1).mean()
        merged_df[col_diff] = merged_df["temperature_2m_mean"] - merged_df[col_avg]
    
    # precipitation_sumの移動平均と差分
    for window_size in range(1, 15):
        prec_avg_col = f"precip_{window_size}d_avg"
        prec_diff_col = f"precip_diff_{window_size}d"
        merged_df[prec_avg_col] = merged_df["precipitation_sum"].rolling(window=window_size, min_periods=1).mean()
        merged_df[prec_diff_col] = merged_df["precipitation_sum"] - merged_df[prec_avg_col]
        features.append(prec_diff_col)
        
    for window_size in range(1, 15):  # temperature差分の2乗列
        col_diff = f"temp_diff_{window_size}d"
        col_diff_sq = f"{col_diff}_squared"
        merged_df[col_diff_sq] = merged_df[col_diff] ** 2
    
    for window_size in range(1, 15):  # precipitation差分の2乗列
        prec_diff_col = f"precip_diff_{window_size}d"
        prec_diff_sq_col = f"{prec_diff_col}_squared"
        merged_df[prec_diff_sq_col] = merged_df[prec_diff_col] ** 2
        features.append(prec_diff_sq_col)  # 必要ならfeaturesにも追加

    # tempprc, temp2, prc2, et02 計算
    merged_df["tempprc"] = merged_df["temperature_2m_mean"] * merged_df["precipitation_sum"]
    merged_df["temp2"] = merged_df["temperature_2m_mean"] ** 2
    merged_df["prc2"] = merged_df["precipitation_sum"] ** 2
    merged_df["et02"] = merged_df["et0_fao_evapotranspiration"] ** 2

    merged_df["weekday"] = merged_df.index.dayofweek
    merged_df["is_weekend"] = (merged_df["weekday"] >= 5).astype(int)
    merged_df["is_newyear"] = ((merged_df.index.month == 12) & (merged_df.index.day == 31)) | \
                             ((merged_df.index.month == 1) & (merged_df.index.day.isin([1, 2, 3])))
    merged_df["is_holiday"] = merged_df.index.to_series().apply(jpholiday.is_holiday)
    merged_df["is_holiday_flag"] = (merged_df["is_weekend"] | merged_df["is_holiday"] | merged_df["is_newyear"]).astype(int)
    merged_df["shiny_holiday"] = merged_df["shortwave_radiation_sum"] * merged_df["is_holiday_flag"]
    for window_size in range(1, 15):
    col_diff = f"temp_diff_{window_size}d"
    features.append(col_diff)


    date_start = datetime.now().date()
    date_end = date_start + timedelta(days=16)
    target_range = (merged_df.index.date >= date_start) & (merged_df.index.date <= date_end)

    return merged_df.loc[target_range]

def load_models(target_col):
    model_dir = os.path.join(os.path.dirname(__file__), f"saved_models/{target_col}")
    lgb_model_path = os.path.join(model_dir, "lightgbm.txt")
    lr_model_path = os.path.join(model_dir, "linear_regression.pkl")
    prophet_model_path = os.path.join(model_dir, "prophet.pkl")

    if not os.path.exists(lgb_model_path) or not os.path.exists(lr_model_path) or not os.path.exists(prophet_model_path):
        raise FileNotFoundError(f"{target_col} のモデルファイルが不足しています。")

    model_lgb = lgb.Booster(model_file=lgb_model_path)
    model_lr = joblib.load(lr_model_path)
    with open(prophet_model_path, "rb") as f:
        model_prophet = pickle.load(f)

    return model_lgb, model_lr, model_prophet

def predict_beer_sales_bulk():
    logging.info("Starting bulk prediction")

    feature_df_full = fetch_weather_features_bulk()
    target_cols = ["pale_ale", "lager", "ipa", "white", "dark", "fruit"]
    results = []

    # 使用特徴量（モデルに合わせて必要なものをセット）
    feature_cols = features

    feature_df = feature_df_full.reindex(columns=feature_cols, fill_value=0)

    for target_col in target_cols:
        model_lgb, model_lr, model_prophet = load_models(target_col)

        # LightGBM予測
        pred_lgb_log = model_lgb.predict(feature_df)
        pred_lgb = np.expm1(pred_lgb_log)

        # Linear Regression予測
        pred_lr_log = model_lr.predict(feature_df)
        pred_lr = np.expm1(pred_lr_log)

        # Prophet予測
        prophet_df = pd.DataFrame({"ds": feature_df_full.index})
        prophet_forecast = model_prophet.predict(prophet_df)
        pred_prophet_log = prophet_forecast["yhat"].values
        pred_prophet = np.expm1(pred_prophet_log)

        # 3モデル平均
        pred_ensemble = (pred_lgb + pred_lr + pred_prophet) / 3

        if len(results) == 0:
            for dt in feature_df_full.index:
                results.append({"date": dt.date().isoformat()})

        for idx in range(len(feature_df_full)):
            results[idx][target_col] = safe_round(pred_ensemble[idx], 2)

    weather_keys = [
        "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
        "relative_humidity_2m_max", "relative_humidity_2m_min",
        "wind_speed_10m_max", "weather_code", "weekday"
    ]

    for idx, dt in enumerate(feature_df_full.index):
        weather_row = feature_df_full.iloc[idx]
        for key in weather_keys:
            val = weather_row.get(key)
            results[idx][key] = safe_round(val, 2) if isinstance(val, (int, float)) else val

    return results


@app.function_name(name="pred")
@app.route(route="pred", methods=["GET"])
def pred(req: func.HttpRequest) -> func.HttpResponse:
    logging.basicConfig(level=logging.INFO)
    logging.info("beerAPI bulk prediction started")

    try:
        results = predict_beer_sales_bulk()
        return HttpResponse(json.dumps(results, ensure_ascii=False), mimetype="application/json")
    except Exception as e:
        logging.error(f"bulk_pred_error: {str(e)}")
        return HttpResponse(f"予測に失敗しました: {str(e)}", status_code=500)


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
        import psycopg2
        conn = psycopg2.connect(os.environ["CONNECTION_STRING"])
        cur = conn.cursor()

        weather_code = data["weathercode"][0]
        temp_max = data["temperature_2m_max"][0]
        temp_min = data["temperature_2m_min"][0]
        temp_mean = data["temperature_2m_mean"][0]
        wind_max = data["windspeed_10m_max"][0]
        hum_max = data["humidity_2m_max"][0]
        hum_min = data["humidity_2m_min"][0]

        cur.execute(
            """
            INSERT INTO weathercode(date, weathercode, temperature_2m_max, temperature_2m_min,
            temperature_2m_mean, windspeed_10m_max, humidity_2m_max, humidity_2m_min)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (date) DO UPDATE SET
            weathercode=EXCLUDED.weathercode,
            temperature_2m_max=EXCLUDED.temperature_2m_max,
            temperature_2m_min=EXCLUDED.temperature_2m_min,
            temperature_2m_mean=EXCLUDED.temperature_2m_mean,
            windspeed_10m_max=EXCLUDED.windspeed_10m_max,
            humidity_2m_max=EXCLUDED.humidity_2m_max,
            humidity_2m_min=EXCLUDED.humidity_2m_min;
            """,
            (yesterday, weather_code, temp_max, temp_min, temp_mean, wind_max, hum_max, hum_min)
        )
        conn.commit()
        cur.close()
        conn.close()
        logging.info("天気データをDBに保存しました。")
    except Exception as e:
        logging.error(f"DB保存エラー: {str(e)}")
