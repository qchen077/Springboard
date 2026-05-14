import pandas as pd
import numpy as np


def haversine_vectorized(lat1, lon1, lat2, lon2, radius_km=6371.0):
    lat1 = np.array(lat1, dtype="float64")
    lon1 = np.array(lon1, dtype="float64")
    lat2 = np.array(lat2, dtype="float64")
    lon2 = np.array(lon2, dtype="float64")

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return radius_km * c


def add_distance_feature(
    df,
    lat_col="lat",
    lon_col="long",
    merch_lat_col="merch_lat",
    merch_lon_col="merch_long",
    out_col="txn_distance_km"
):
    df = df.copy()

    df[out_col] = haversine_vectorized(
        df[lat_col],
        df[lon_col],
        df[merch_lat_col],
        df[merch_lon_col]
    )

    return df


def add_time_features(df, id_col="cc_num", dt_col="trans_date_trans_time"):
    df = df.copy()

    df[dt_col] = pd.to_datetime(df[dt_col])
    df = df.sort_values([id_col, dt_col])

    dt = df[dt_col]

    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["hour"] = dt.dt.hour
    df["dayofweek"] = dt.dt.dayofweek

    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_night"] = ((df["hour"] <= 5) | (df["hour"] >= 22)).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    return df


def add_time_gap_and_speed(
    df,
    id_col="cc_num",
    time_col="unix_time",
    dist_col="txn_distance_km",
    speed_clip_upper=None
):
    df = df.copy()
    df = df.sort_values([id_col, time_col])

    df["time_gap_sec"] = df.groupby(id_col)[time_col].diff()
    df["time_gap_hr"] = df["time_gap_sec"] / 3600

    gap_hr = df["time_gap_hr"].clip(lower=1 / 3600)
    df["txn_speed_kmh"] = df[dist_col] / gap_hr

    if speed_clip_upper is None:
        speed_clip_upper = df["txn_speed_kmh"].quantile(0.99)

    df["txn_speed_kmh_clip"] = df["txn_speed_kmh"].clip(upper=speed_clip_upper)

    return df, speed_clip_upper


def add_log_features(
    df,
    amt_col="amt",
    time_gap_col="time_gap_sec",
    speed_kmh_col="txn_speed_kmh_clip",
    city_pop_col="city_pop"
):
    df = df.copy()

    df["time_gap_log"] = np.log1p(df[time_gap_col].clip(lower=0))
    df["txn_speed_kmh_clip_log"] = np.log1p(df[speed_kmh_col].clip(lower=0))
    df["amt_log"] = np.log1p(df[amt_col].clip(lower=0))
    df["city_pop_log"] = np.log1p(df[city_pop_col].clip(lower=0))

    return df


def demo_feature(df, dob_col="dob", date_time_col="trans_date_trans_time"):
    df = df.copy()

    df[dob_col] = pd.to_datetime(df[dob_col])
    df[date_time_col] = pd.to_datetime(df[date_time_col])

    t = df[date_time_col]
    b = df[dob_col]

    before_bday = (
        (t.dt.month < b.dt.month)
        | ((t.dt.month == b.dt.month) & (t.dt.day < b.dt.day))
    )

    df["age"] = (t.dt.year - b.dt.year) - before_bday.astype(int)

    return df


def merchant_feature_engineering(df, merchant_col="merchant"):
    df = df.copy()

    df["merchant_clean"] = (
        df[merchant_col]
        .str.replace("fraud_", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.lower()
        .str.strip()
    )

    def extract_type(x):
        if x.endswith("llc"):
            return "llc"
        elif x.endswith("inc"):
            return "inc"
        elif x.endswith("ltd"):
            return "ltd"
        elif "group" in x:
            return "group"
        elif "and sons" in x:
            return "family_business"
        else:
            return "other"

    df["merchant_type"] = df["merchant_clean"].apply(extract_type)

    if merchant_col in df.columns:
        df = df.drop(columns=[merchant_col])

    return df


def drop_unused_columns(df):
    df = df.copy()

    drop_cols = [
        "unix_time",
        "trans_num",
        "time_gap_sec",
        "time_gap_hr",
        "txn_speed_kmh",
        "hour",
        "dayofweek",
        "city_pop",
        "dob",
        "lat",
        "long",
        "merch_lat",
        "merch_long",
        "Unnamed: 0",
        "first",
        "last",
        "street"
    ]

    existing_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=existing_cols)

    return df


def clean_and_engineer_base_features(df, speed_clip_upper=None):
    df = df.copy()

    df = add_distance_feature(df)
    df = add_time_features(df)
    df, speed_clip_upper = add_time_gap_and_speed(
        df,
        speed_clip_upper=speed_clip_upper
    )
    df = add_log_features(df)
    df = demo_feature(df)
    df = merchant_feature_engineering(df)
    df = drop_unused_columns(df)

    return df, speed_clip_upper