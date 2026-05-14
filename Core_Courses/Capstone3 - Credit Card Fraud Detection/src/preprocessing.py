import pandas as pd


def fit_train_artifacts(
    train_df,
    target_col="is_fraud",
    merchant_col="merchant_clean",
    category_col="category",
    impute_cols=None,
    amt_col="amt",
    high_amt_quantile=0.996
):
    """
    Fit all preprocessing artifacts using the full training data only.

    Artifacts include:
    - global fraud rate
    - merchant target encoding map
    - category one-hot feature space
    - median values for imputation
    - high amount threshold
    """

    if impute_cols is None:
        impute_cols = ["txn_speed_kmh_clip_log", "time_gap_log"]

    # Global fraud rate from train only
    global_mean = train_df[target_col].mean()

    # Merchant target encoding map from train only
    merchant_te_map = (
        train_df
        .groupby(merchant_col, observed=False)[target_col]
        .mean()
    )

    # Category one-hot columns from train only
    train_cat = pd.get_dummies(
        train_df[category_col],
        prefix=category_col,
        dtype="int8"
    )

    category_cols = train_cat.columns.tolist()

    # Train-only median values for missing value imputation
    train_medians = {
        col: train_df[col].median()
        for col in impute_cols
    }

    # Train-only high amount threshold
    amt_threshold = train_df[amt_col].quantile(high_amt_quantile)

    artifacts = {
        "global_mean": global_mean,
        "merchant_te_map": merchant_te_map,
        "category_cols": category_cols,
        "train_medians": train_medians,
        "impute_cols": impute_cols,
        "amt_threshold": amt_threshold,
        "high_amt_quantile": high_amt_quantile
    }

    return artifacts


def transform_with_train_artifacts(
    df,
    artifacts,
    merchant_col="merchant_clean",
    category_col="category",
    amt_col="amt"
):
    """
    Transform train or test data using preprocessing artifacts learned from train only.
    This avoids data leakage when applying the pipeline to test data.
    """

    df = df.copy()

    global_mean = artifacts["global_mean"]
    merchant_te_map = artifacts["merchant_te_map"]
    category_cols = artifacts["category_cols"]
    train_medians = artifacts["train_medians"]
    impute_cols = artifacts["impute_cols"]
    amt_threshold = artifacts["amt_threshold"]

    # Merchant target encoding
    df["merchant_clean_te"] = (
        df[merchant_col]
        .map(merchant_te_map)
        .fillna(global_mean)
    )

    # High amount flag based on train-only threshold
    df["high_amt_flag"] = (
        df[amt_col] >= amt_threshold
    ).astype(int)

    # Gender binary encoding
    df["gender_bin"] = df["gender"].map({"M": 0, "F": 1})

    # Category one-hot encoding using train category columns
    cat_df = pd.get_dummies(
        df[category_col],
        prefix=category_col,
        dtype="int8"
    )

    cat_df = cat_df.reindex(columns=category_cols, fill_value=0)

    df = pd.concat([df, cat_df], axis=1)

    # Missing indicators + train-median imputation
    for col in impute_cols:
        df[f"{col}_missing"] = df[col].isna().astype(int)
        df[col] = df[col].fillna(train_medians[col])

    return df


def get_final_features(category_cols, impute_cols):
    """
    Define the final model feature list.
    Only columns listed here will enter the model.
    """

    base_features = [
        "year", "month",
        "is_weekend", "is_night",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",

        "txn_distance_km",
        "txn_speed_kmh_clip_log",
        "time_gap_log",

        "amt_log",

        "city_pop_log",
        "age"
    ]

    final_features = (
        base_features
        + ["merchant_clean_te", "high_amt_flag", "gender_bin"]
        + category_cols
        + [f"{col}_missing" for col in impute_cols]
    )

    return final_features