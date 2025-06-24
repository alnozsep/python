import azure.functions as func
from azure.functions import FunctionApp, HttpResponse
app = FunctionApp()

from azure.storage.blob import BlobServiceClient
from datetime import datetime, timedelta
import os
import json
import numpy as np
import pandas as pd
import logging

def run_prediction_and_return():
    import json
    results = predict_beer_sales_bulk()
    return json.dumps(convert_np_types(results), ensure_ascii=False)


def fetch_weather_features_bulk():
    import requests
    import jpholiday

    latitude, longitude = 35.6895, 139.6917
    today = datetime.now().date()
    past_start = (today - timedelta(days=15)).isoformat()
    past_end = (today - timedelta(days=1)).isoformat()
    future_start = today.isoformat()
    future_end = (today + timedelta(days=14)).isoformat()

    # Archive
    archive_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={latitude}&longitude={longitude}&start_date={past_start}&end_date={past_end}&daily=temperature_2m_mean,precipitation_sum,wind_speed_10m_mean,weather_code,shortwave_radiation_sum,temperature_2m_max,temperature_2m_min,precipitation_hours,relative_humidity_2m_max,relative_humidity_2m_min,windspeed_10m_max,windspeed_10m_mean,windgusts_10m_max,et0_fao_evapotranspiration,uv_index_max&timezone=Asia%2FTokyo"
    df_archive = pd.DataFrame(requests.get(archive_url).json()["daily"])

    # Forecast
    forecast_url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&start_date={future_start}&end_date={future_end}&daily=temperature_2m_mean,precipitation_sum,weather_code,shortwave_radiation_sum,temperature_2m_max,temperature_2m_min,precipitation_hours,relative_humidity_2m_max,relative_humidity_2m_min,windspeed_10m_max,windspeed_10m_mean,windgusts_10m_max,et0_fao_evapotranspiration&timezone=Asia%2FTokyo"
    df_forecast = pd.DataFrame(requests.get(forecast_url).json()["daily"])

    merged_df = pd.concat([df_archive, df_forecast], ignore_index=True)
    merged_df["wind_speed_10m_max"] = merged_df["windspeed_10m_max"]
    merged_df["wind_speed_10m_mean"] = merged_df["windspeed_10m_mean"]
    merged_df["time"] = pd.to_datetime(merged_df["time"])
    merged_df.set_index("time", inplace=True)

    merged_df["weekday"] = merged_df.index.dayofweek
    merged_df["month"] = merged_df.index.month
    merged_df["tempprc"] = merged_df["temperature_2m_mean"] * merged_df["precipitation_sum"]
    merged_df["temp2"] = merged_df["temperature_2m_mean"] ** 2
    merged_df["prc2"] = merged_df["precipitation_sum"] ** 2
    merged_df["et02"] = merged_df["et0_fao_evapotranspiration"] ** 2

    for window_size in range(1, 15):
        merged_df[f"temp_{window_size}d_avg"] = merged_df["temperature_2m_mean"].rolling(window=window_size, min_periods=1).mean()
        merged_df[f"temp_diff_{window_size}d"] = merged_df["temperature_2m_mean"] - merged_df[f"temp_{window_size}d_avg"]
        merged_df[f"precip_{window_size}d_avg"] = merged_df["precipitation_sum"].rolling(window=window_size, min_periods=1).mean()
        merged_df[f"precip_diff_{window_size}d"] = merged_df["precipitation_sum"] - merged_df[f"precip_{window_size}d_avg"]
        merged_df[f"temp_diff_{window_size}d_squared"] = merged_df[f"temp_diff_{window_size}d"] ** 2
        merged_df[f"precip_diff_{window_size}d_squared"] = merged_df[f"precip_diff_{window_size}d"] ** 2

    merged_df["is_holiday"] = merged_df.index.to_series().apply(jpholiday.is_holiday)
    merged_df["is_weekend"] = (merged_df["weekday"] >= 5).astype(int)
    merged_df["is_newyear"] = ((merged_df.index.month == 12) & (merged_df.index.day == 31)) | ((merged_df.index.month == 1) & (merged_df.index.day.isin([1, 2, 3])))
    merged_df["is_holiday_flag"] = (merged_df["is_weekend"] | merged_df["is_holiday"] | merged_df["is_newyear"]).astype(int)

    def is_day_before_holiday(ts):
        next_day = ts.date() + timedelta(days=1)
        return jpholiday.is_holiday(next_day) or next_day.weekday() >= 5 or (next_day.month == 12 and next_day.day == 31)

    merged_df["is_day_before_holiday"] = merged_df.index.to_series().apply(is_day_before_holiday)
    merged_df["shiny_holiday"] = merged_df["shortwave_radiation_sum"] * merged_df["is_holiday_flag"]

    # pressure
    pressure_urls = [
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&start_date={future_start}&end_date={future_end}&hourly=surface_pressure&timezone=Asia%2FTokyo",
        f"https://archive-api.open-meteo.com/v1/archive?latitude={latitude}&longitude={longitude}&start_date={past_start}&end_date={past_end}&hourly=surface_pressure&timezone=Asia%2FTokyo"
    ]

    pressure_df_all = pd.DataFrame()
    for url in pressure_urls:
        data = requests.get(url).json()
        df = pd.DataFrame({
            "datetime": pd.to_datetime(data["hourly"]["time"]),
            "mean_sealevel_pressure": data["hourly"]["surface_pressure"]
        })
        df["time"] = df["datetime"].dt.floor("D")
        daily_pressure = df.groupby("time")["mean_sealevel_pressure"].mean().reset_index()
        pressure_df_all = pd.concat([pressure_df_all, daily_pressure])

    pressure_df_all = pressure_df_all.drop_duplicates(subset="time")
    merged_df = merged_df.reset_index()
    merged_df["time"] = pd.to_datetime(merged_df["time"]).dt.floor("D")
    pressure_df_all["time"] = pd.to_datetime(pressure_df_all["time"]).dt.floor("D")
    merged_df = pd.merge(merged_df, pressure_df_all, on="time", how="left")
    merged_df.set_index("time", inplace=True)

    date_start = today - timedelta(days=3)
    date_end = today + timedelta(days=14)
    target_range = (merged_df.index.date >= date_start) & (merged_df.index.date <= date_end)

    merged_df.fillna(0, inplace=True)
    return merged_df.loc[target_range]
import azure.functions as func
from azure.functions import FunctionApp, HttpResponse
from azure.storage.blob import BlobServiceClient

from datetime import datetime, timedelta
import os
import json
import numpy as np
import logging

app = FunctionApp()

# ---------------- 共通ユーティリティ ----------------
def convert_np_types(obj):
    if isinstance(obj, dict):
        return {k: convert_np_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_types(i) for i in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def save_to_blob(results, filename=None):
    try:
        connect_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        container_name = "beer-cache"
        if filename is None:
            filename = f"beer_pred_{datetime.today().date()}.json"

        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        container_client = blob_service_client.get_container_client(container_name)
        blob_client = container_client.get_blob_client(filename)

        json_data = json.dumps(convert_np_types(results), ensure_ascii=False)
        blob_client.upload_blob(json_data, overwrite=True)
        logging.info(f"キャッシュ保存成功: {filename}")
        return True
    except Exception as e:
        logging.error(f"キャッシュ保存失敗: {str(e)}")
        return False

def load_from_blob(filename=None):
    try:
        connect_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        container_name = "beer-cache"
        if filename is None:
            filename = f"beer_pred_{datetime.today().date()}.json"

        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        container_client = blob_service_client.get_container_client(container_name)
        blob_client = container_client.get_blob_client(filename)

        if not blob_client.exists():
            logging.info(f"キャッシュが存在しません: {filename}")
            return None

        download_stream = blob_client.download_blob()
        json_data = download_stream.readall().decode('utf-8')
        logging.info(f"キャッシュ読み込み成功: {filename}")
        return json.loads(json_data)

    except Exception as e:
        logging.error(f"キャッシュ読み込み失敗: {str(e)}")
        return None

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


# ---------------- メイン予測処理（キャッシュ対応） ----------------
def predict_beer_sales_bulk():
    import joblib
    from prophet.serialize import model_from_json
    import pandas as pd

    logging.info("Starting bulk prediction")

    cache_filename = f"beer_pred_{datetime.today().date()}.json"
    cached_result = load_from_blob(cache_filename)
    if cached_result:
        return cached_result

    feature_df_full = fetch_weather_features_bulk()
    target_cols = ["pale_ale", "lager", "ipa", "white", "dark", "fruit"]
    results = []

    for target_col in target_cols:
        model_dir = os.path.join(os.path.dirname(__file__), f"final_models/{target_col}")
        lr_model_path = os.path.join(model_dir, "linear_reg.pkl")
        prophet_model_path = os.path.join(model_dir, "prophet_model.json")
        selected_features_path = os.path.join(model_dir, "selected_features.pkl")

        if not os.path.exists(selected_features_path):
            raise FileNotFoundError(f"{target_col} の特徴量定義が見つかりません")

        selected_features = joblib.load(selected_features_path)
        feature_df = feature_df_full[selected_features]

        # LightGBM アンサンブル
        lgb_preds = []
        for seed in range(41, 51):
            lgb_path = os.path.join(model_dir, f"lgb_seed_{seed}.pkl")
            model_lgb = joblib.load(lgb_path)
            X = feature_df[model_lgb.feature_name()].astype(np.float32).values
            pred = model_lgb.predict(X)
            lgb_preds.append(np.expm1(pred))
        pred_lgb = np.mean(lgb_preds, axis=0)

        # 線形モデル
        model_lr = joblib.load(lr_model_path)
        pred_lr_log = model_lr.predict(feature_df)
        pred_lr = np.expm1(pred_lr_log)

        # Prophet
        with open(prophet_model_path, "r") as f:
            model_prophet = model_from_json(json.load(f))
        prophet_df = pd.DataFrame({"ds": feature_df_full.index})
        pred_prophet_log = model_prophet.predict(prophet_df)["yhat"].values
        pred_prophet = np.expm1(pred_prophet_log)

        if target_col == "white":
            pred_ensemble = (pred_lgb*0.1+ pred_lr*0.5 + pred_prophet*0.4)
        else:
            pred_ensemble = (pred_lgb*0.3+ pred_lr*0.6 + pred_prophet*0.1)

        if not results:
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

    save_to_blob(results, cache_filename)
    return results


@app.function_name(name="pred_http")
@app.route(route="pred", methods=["GET"])
def pred_http(req: func.HttpRequest) -> func.HttpResponse:
    logging.basicConfig(level=logging.INFO)
    try:
        json_result = run_prediction_and_return()
        return func.HttpResponse(json_result, mimetype="application/json")
    except Exception as e:
        logging.error(f"HTTP予測失敗: {str(e)}")
        return func.HttpResponse(f"予測に失敗しました: {str(e)}", status_code=500)
    


@app.function_name(name="pred_timer")
@app.schedule(schedule="0 0 0 * * *", arg_name="mytimer", run_on_startup=False, use_monitor=True)
def pred_timer(mytimer: func.TimerRequest) -> None:
    logging.basicConfig(level=logging.INFO)
    try:
        logging.info("Timer triggered - Running prediction")
        _ = run_prediction_and_return()
    except Exception as e:
        logging.error(f"Timer予測失敗: {str(e)}")
