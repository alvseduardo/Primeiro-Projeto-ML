import math
from datetime import date as date_type
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

MIN_QTY_FOR_LOG = 0.1

try:
    from xgboost import XGBRegressor  # type: ignore

    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


def prepare_daily(rows):
    if not rows:
        return pd.DataFrame()

    base_cols = ["siglaloja", "dtmovimento", "quantidade", "custo", "venda"]
    extra_cols = []
    if rows and len(rows[0]) >= 6:
        extra_cols.append("ruptura")
    if rows and len(rows[0]) >= 7:
        extra_cols.append("promocao")

    df = pd.DataFrame(rows, columns=base_cols + extra_cols)
    df["dtmovimento"] = pd.to_datetime(df["dtmovimento"])
    df["quantidade"] = pd.to_numeric(df["quantidade"], errors="coerce").fillna(0)
    df["custo"] = pd.to_numeric(df["custo"], errors="coerce").fillna(0)
    df["venda"] = pd.to_numeric(df["venda"], errors="coerce").fillna(0)
    if "ruptura" in df.columns:
        df["ruptura"] = pd.to_numeric(df["ruptura"], errors="coerce").fillna(0)
    if "promocao" in df.columns:
        df["promocao"] = pd.to_numeric(df["promocao"], errors="coerce").fillna(0)

    agg = {"quantidade": "sum", "custo": "sum", "venda": "sum"}
    if "ruptura" in df.columns:
        agg["ruptura"] = "max"
    if "promocao" in df.columns:
        agg["promocao"] = "max"

    daily = df.groupby("dtmovimento", as_index=False).agg(agg)

    daily = daily[daily["quantidade"] > 0]
    daily["preco_unit"] = daily["venda"] / daily["quantidade"]
    daily["custo_unit"] = daily["custo"] / daily["quantidade"]
    daily = daily[(daily["preco_unit"] > 0) & (daily["custo_unit"] >= 0)]

    return daily.sort_values("dtmovimento")


def fill_missing_days(daily, start_date, end_date):
    if daily is None or daily.empty:
        return daily

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    full = pd.DataFrame({"dtmovimento": pd.date_range(start, end, freq="D")})
    merged = full.merge(daily, on="dtmovimento", how="left")

    if "quantidade" in merged.columns:
        merged["quantidade"] = merged["quantidade"].fillna(0)
    if "custo" in merged.columns:
        merged["custo"] = merged["custo"].fillna(0)
    if "venda" in merged.columns:
        merged["venda"] = merged["venda"].fillna(0)
    if "ruptura" in merged.columns:
        merged["ruptura"] = merged["ruptura"].fillna(0)
    if "promocao" in merged.columns:
        merged["promocao"] = merged["promocao"].fillna(0)

    if "preco_unit" in merged.columns:
        merged["preco_unit"] = merged["preco_unit"].ffill().bfill()
    if "custo_unit" in merged.columns:
        merged["custo_unit"] = merged["custo_unit"].ffill().bfill()

    return merged.sort_values("dtmovimento")


def apply_qty_outlier_filter(daily, method="winsorize", upper_quantile=0.98):
    if daily is None or daily.empty:
        return daily
    if "quantidade" not in daily.columns:
        return daily

    try:
        q = float(upper_quantile)
    except (TypeError, ValueError):
        q = 0.98

    if q <= 0 or q >= 1:
        return daily

    cutoff = daily["quantidade"].quantile(q)
    if cutoff is None or not np.isfinite(cutoff):
        return daily

    cleaned = daily.copy()
    if method == "remove":
        cleaned = cleaned[cleaned["quantidade"] <= cutoff].copy()
    else:
        cleaned["quantidade"] = cleaned["quantidade"].clip(upper=cutoff)
    return cleaned


def _trend_values(dates, start_date, scale_days):
    if scale_days <= 0:
        scale_days = 1.0
    return (dates - start_date).dt.days.astype(float) / float(scale_days)


def build_feature_matrix(
    daily,
    trend_start=None,
    trend_scale=None,
    include_promo=True,
    include_promo_price_interaction=False,
    include_price=True,
):
    if daily.empty:
        return None, None, None

    dates = pd.to_datetime(daily["dtmovimento"])
    if trend_start is None:
        trend_start = dates.min()
    if trend_scale is None:
        span_days = (dates.max() - trend_start).days
        trend_scale = max(1, span_days)

    trend = _trend_values(dates, trend_start, trend_scale).values
    weekdays = dates.dt.weekday.values

    log_price = None
    if include_price:
        log_price = np.log(daily["preco_unit"].values)
    promo_flag = None
    if include_promo:
        if "promocao" in daily.columns:
            promo_flag = (
                pd.to_numeric(daily["promocao"], errors="coerce")
                .fillna(0)
                .values
            )
        else:
            promo_flag = np.zeros(len(daily))
    dummies = np.zeros((len(daily), 6))
    for i in range(6):
        dummies[:, i] = (weekdays == i).astype(float)

    feature_blocks = []
    feature_names = []
    if include_price:
        feature_blocks.append(log_price)
        feature_names.append("log_price")
    if include_promo:
        feature_blocks.append(promo_flag)
        feature_names.append("promo_flag")
        if include_promo_price_interaction and include_price and log_price is not None:
            feature_blocks.append(promo_flag * log_price)
            feature_names.append("promo_log_price")

    feature_blocks.append(trend)
    feature_names.append("trend")
    for i in range(6):
        feature_blocks.append(dummies[:, i])
        feature_names.append(f"wd_{i}")

    X = np.column_stack(feature_blocks)
    qty = daily["quantidade"].values
    y = np.log(np.maximum(qty, MIN_QTY_FOR_LOG))
    weekday_probs = dummies.mean(axis=0) if len(dummies) else np.zeros(6)
    trend_last = float(trend[-1]) if len(trend) else 0.0

    return X, y, {
        "trend_start": trend_start,
        "trend_scale": trend_scale,
        "feature_names": feature_names,
        "weekday_probs": weekday_probs,
        "trend_last": trend_last,
        "include_promo": bool(include_promo),
        "promo_interaction": bool(include_promo_price_interaction),
        "include_price": bool(include_price),
    }


def fit_demand_model(
    daily,
    include_promo=True,
    include_promo_price_interaction=False,
    price_mode="two_stage",
    min_price_range_pct=0.02,
    min_price_points=3,
):
    if daily.empty or len(daily) < 8:
        return None

    daily_sorted = daily.sort_values("dtmovimento")
    price_min = float(daily_sorted["preco_unit"].min())
    price_max = float(daily_sorted["preco_unit"].max())
    price_mean = float(daily_sorted["preco_unit"].mean())
    price_range_pct = (
        (price_max - price_min) / price_mean if price_mean > 0 else 0.0
    )
    price_points = int(daily_sorted["preco_unit"].nunique())

    log_price = np.log(daily_sorted["preco_unit"].values)
    log_qty = np.log(
        np.maximum(daily_sorted["quantidade"].values, MIN_QTY_FOR_LOG)
    )
    dlog_price = np.diff(log_price)
    dlog_qty = np.diff(log_qty)
    change_mask = np.abs(dlog_price) > 1e-6
    price_changes = int(change_mask.sum())

    price_learned = (
        price_range_pct >= float(min_price_range_pct)
        and price_points >= int(min_price_points)
        and price_changes >= 5
    )

    if price_mode == "full":
        X, y, meta = build_feature_matrix(
            daily,
            include_promo=include_promo,
            include_promo_price_interaction=include_promo_price_interaction,
            include_price=True,
        )
        if X is None:
            return None

        model = LinearRegression()
        model.fit(X, y)
        preds = model.predict(X)
        r2 = r2_score(y, preds)

        n, k = X.shape
        elasticity_se = None
        elasticity_ci_low = None
        elasticity_ci_high = None
        dof = n - (k + 1)
        if dof > 0:
            X_ext = np.column_stack([np.ones(n), X])
            beta = np.concatenate([[model.intercept_], model.coef_])
            resid = y - (X_ext @ beta)
            sigma2 = float((resid**2).sum() / dof)
            try:
                xtx_inv = np.linalg.inv(X_ext.T @ X_ext)
            except np.linalg.LinAlgError:
                xtx_inv = np.linalg.pinv(X_ext.T @ X_ext)
            cov = sigma2 * xtx_inv
            se = np.sqrt(np.diag(cov))
            if len(se) > 1:
                elasticity_se = float(se[1])
                elasticity_ci_low = float(model.coef_[0] - 1.96 * elasticity_se)
                elasticity_ci_high = float(model.coef_[0] + 1.96 * elasticity_se)

        feature_names = meta.get("feature_names") if meta else None
        coef_list = model.coef_.tolist()
        coef_map = {}
        if feature_names and len(feature_names) == len(coef_list):
            coef_map = dict(zip(feature_names, coef_list))

        elasticity_base = coef_map.get("log_price", float(model.coef_[0]))
        promo_log_price = coef_map.get("promo_log_price")
        elasticity_promo = (
            float(elasticity_base + promo_log_price)
            if promo_log_price is not None
            else None
        )
        promo_coef = coef_map.get("promo_flag")

        return {
            "model": model,
            "r2": float(r2),
            "elasticidade": float(elasticity_base),
            "elasticidade_promo": elasticity_promo,
            "promo_coef": promo_coef,
            "intercept": float(model.intercept_),
            "coef": coef_list,
            "elasticity_se": elasticity_se,
            "elasticity_ci_low": elasticity_ci_low,
            "elasticity_ci_high": elasticity_ci_high,
            "price_min": price_min,
            "price_max": price_max,
            "price_range_pct": price_range_pct,
            "price_points": price_points,
            "price_learned": price_learned,
            "price_mode": "full",
            **(meta or {}),
        }

    X_base, y, meta = build_feature_matrix(
        daily_sorted,
        include_promo=include_promo,
        include_promo_price_interaction=False,
        include_price=False,
    )
    if X_base is None or meta is None:
        return None

    base_model = LinearRegression()
    base_model.fit(X_base, y)
    base_pred = base_model.predict(X_base)

    price_coef = 0.0
    if price_learned:
        x = dlog_price[change_mask]
        y_diff = dlog_qty[change_mask]
        if len(x) > 0:
            x_centered = x - x.mean()
            denom = float((x_centered**2).sum())
            if denom > 0:
                price_coef = float(
                    (x_centered * (y_diff - y_diff.mean())).sum() / denom
                )

    price_ref = float(np.mean(log_price)) if len(log_price) else 0.0
    combined_intercept = float(base_model.intercept_ - price_coef * price_ref)
    combined_coef = [float(price_coef)] + base_model.coef_.tolist()
    combined_names = ["log_price"] + (meta.get("feature_names") or [])

    y_hat = combined_intercept + price_coef * log_price + (
        X_base @ base_model.coef_
    )
    r2 = r2_score(y, y_hat)

    coef_map = dict(zip(meta.get("feature_names") or [], base_model.coef_))
    promo_coef = coef_map.get("promo_flag")

    elasticity_value = float(price_coef) if price_learned else 0.0

    return {
        "model": base_model,
        "r2": float(r2),
        "elasticidade": elasticity_value,
        "elasticidade_promo": None,
        "promo_coef": promo_coef,
        "intercept": combined_intercept,
        "coef": combined_coef,
        "elasticity_se": None,
        "elasticity_ci_low": None,
        "elasticity_ci_high": None,
        "price_min": price_min,
        "price_max": price_max,
        "price_range_pct": price_range_pct,
        "price_points": price_points,
        "price_changes": price_changes,
        "price_learned": price_learned,
        "price_mode": "two_stage",
        **(meta or {}),
        "feature_names": combined_names,
    }


def _weekday_features(weekday, weekday_probs):
    vec = np.zeros(6)
    if weekday is None:
        if weekday_probs is not None:
            return np.array(weekday_probs, dtype=float)
        return vec
    try:
        wd = int(weekday)
    except (TypeError, ValueError):
        return vec
    if 0 <= wd <= 5:
        vec[wd] = 1.0
    return vec


def _trend_from_date(day, trend_start, trend_scale):
    if not day or not trend_start:
        return 0.0
    if isinstance(day, datetime):
        day = day.date()
    if isinstance(trend_start, datetime):
        trend_start = trend_start.date()
    if isinstance(day, date_type) and isinstance(trend_start, date_type):
        delta = (day - trend_start).days
        scale = float(trend_scale) if trend_scale else 1.0
        if scale <= 0:
            scale = 1.0
        return float(delta) / scale
    return 0.0


def _feature_row(
    model_info,
    price,
    date=None,
    trend=None,
    weekday=None,
    promo_flag=0,
):
    feature_names = model_info.get("feature_names") or []
    if not feature_names:
        return None

    needs_price = any(
        name in ("log_price", "promo_log_price") for name in feature_names
    )
    if needs_price:
        if price is None or price <= 0:
            return None
        log_price = math.log(price)
    else:
        log_price = None
    if trend is None:
        if date is not None:
            trend = _trend_from_date(
                date,
                model_info.get("trend_start"),
                model_info.get("trend_scale"),
            )
        else:
            trend = model_info.get("trend_last", 0.0)

    if weekday is None and date is not None:
        try:
            weekday = date.weekday()
        except Exception:
            weekday = None

    weekday_vec = _weekday_features(weekday, model_info.get("weekday_probs"))

    values = []
    for name in feature_names:
        if name == "log_price":
            values.append(log_price)
        elif name == "promo_flag":
            values.append(float(promo_flag))
        elif name == "promo_log_price":
            if log_price is None:
                values.append(0.0)
            else:
                values.append(float(promo_flag) * log_price)
        elif name == "trend":
            values.append(float(trend))
        elif name.startswith("wd_"):
            try:
                idx = int(name.split("_", 1)[1])
            except (IndexError, ValueError):
                idx = None
            if idx is None or idx < 0 or idx >= len(weekday_vec):
                values.append(0.0)
            else:
                values.append(float(weekday_vec[idx]))
        else:
            values.append(0.0)
    return np.array(values, dtype=float)


def predict_qty(
    model_info,
    price,
    date=None,
    trend=None,
    weekday=None,
    elasticity=None,
    promo_flag=0,
):
    if not model_info or price is None or price <= 0:
        return None
    coef = model_info.get("coef") or []
    if len(coef) < 2:
        return None
    feature_names = model_info.get("feature_names") or []
    intercept = float(model_info.get("intercept", 0.0))
    vec = _feature_row(
        model_info,
        price,
        date=date,
        trend=trend,
        weekday=weekday,
        promo_flag=promo_flag,
    )
    if vec is None or len(vec) != len(coef):
        return None
    coef_arr = np.array(coef, dtype=float)
    if elasticity is not None and feature_names:
        for idx, name in enumerate(feature_names):
            if name == "log_price":
                try:
                    coef_arr[idx] = float(elasticity)
                except (TypeError, ValueError):
                    pass
                break
    linear = intercept + float(np.dot(coef_arr, vec))
    return float(math.exp(linear))


def round_qty(value):
    if value is None:
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def weekday_factors(daily):
    if daily.empty:
        return {i: 1.0 for i in range(7)}

    df = daily.copy()
    df["weekday"] = df["dtmovimento"].dt.weekday
    overall = df["quantidade"].mean()
    if overall <= 0:
        return {i: 1.0 for i in range(7)}

    by_day = df.groupby("weekday")["quantidade"].mean().to_dict()
    factors = {}
    for i in range(7):
        factor = by_day.get(i, overall) / overall
        factor = max(0.3, min(2.0, float(factor)))
        factors[i] = factor
    return factors


def forecast_series(base_qty, start_date, days, factors):
    if base_qty is None:
        return [None] * days

    series = []
    for i in range(days):
        day = start_date + timedelta(days=i)
        factor = factors.get(day.weekday(), 1.0)
        value = max(0.0, base_qty * factor)
        series.append(int(round(value)))
    return series


def forecast_qty_series(
    model_info,
    price,
    start_date,
    days,
    elasticity=None,
    promo_flag=0,
):
    if model_info is None or price is None or price <= 0:
        return [None] * days
    def _promo_for_day(idx, day):
        if promo_flag is None:
            return 0
        if callable(promo_flag):
            try:
                return promo_flag(day)
            except Exception:
                return 0
        if isinstance(promo_flag, (list, tuple, np.ndarray, pd.Series)):
            if len(promo_flag) == 0:
                return 0
            if idx < len(promo_flag):
                return promo_flag[idx]
            return promo_flag[-1]
        return promo_flag
    series = []
    for i in range(days):
        day = start_date + timedelta(days=i)
        qty = predict_qty(
            model_info,
            price,
            date=day,
            elasticity=elasticity,
            promo_flag=_promo_for_day(i, day),
        )
        if qty is None:
            series.append(None)
        else:
            try:
                series.append(int(round(float(qty))))
            except (TypeError, ValueError):
                series.append(None)
    return series


def suggest_price(daily, model_info, avg_cost=None, promo_flag=0):
    if daily.empty or not model_info:
        return None

    min_p = float(daily["preco_unit"].min())
    max_p = float(daily["preco_unit"].max())
    if model_info.get("price_min") is not None:
        try:
            min_p = float(model_info.get("price_min"))
        except (TypeError, ValueError):
            pass
    if model_info.get("price_max") is not None:
        try:
            max_p = float(model_info.get("price_max"))
        except (TypeError, ValueError):
            pass
    if not model_info.get("price_learned", True):
        return None
    if avg_cost is None:
        avg_cost = float(daily["custo_unit"].mean())
    else:
        try:
            avg_cost = float(avg_cost)
        except (TypeError, ValueError):
            avg_cost = float(daily["custo_unit"].mean())

    lower = max(min_p, avg_cost * 1.02)
    upper = max_p
    if upper <= lower:
        return None

    grid = np.linspace(lower, upper, 40)
    best_price = None
    best_profit = -1e18

    for p in grid:
        qty = predict_qty(model_info, p, promo_flag=promo_flag)
        if qty is None:
            continue
        profit = (p - avg_cost) * qty
        if profit > best_profit:
            best_profit = profit
            best_price = p

    return float(best_price) if best_price else float(max_p)


def risk_level(
    n_points,
    r2,
    elasticity,
    elasticity_ci=None,
    elasticity_instability=None,
    price_range_pct=None,
):
    if elasticity is not None:
        try:
            if abs(float(elasticity)) > 10:
                return "Alto"
        except (TypeError, ValueError):
            pass

    score = 0
    if n_points < 30:
        score += 2
    elif n_points < 60:
        score += 1

    if n_points < 20:
        score += 1

    if r2 < 0.3:
        score += 2
    elif r2 < 0.5:
        score += 1

    if elasticity >= -0.2:
        score += 2
    elif elasticity > -0.6:
        score += 1

    if elasticity_ci and len(elasticity_ci) == 2:
        ci_low, ci_high = elasticity_ci
        try:
            width = abs(float(ci_high) - float(ci_low))
        except (TypeError, ValueError):
            width = None
        if width is not None:
            try:
                denom = max(abs(float(elasticity)), 0.1)
            except (TypeError, ValueError):
                denom = 0.1
            rel_width = width / denom if denom else width
            if rel_width > 1.5:
                score += 2
            elif rel_width > 0.8:
                score += 1
        if ci_low <= 0 <= ci_high:
            score += 1

    if elasticity_instability is not None:
        try:
            instability = float(elasticity_instability)
        except (TypeError, ValueError):
            instability = None
        if instability is not None and instability > 0.3:
            score += 1

    if price_range_pct is not None:
        try:
            price_range_pct = float(price_range_pct)
        except (TypeError, ValueError):
            price_range_pct = None
        if price_range_pct is not None and price_range_pct < 0.05:
            score += 1

    if score >= 4:
        level = "Alto"
    elif score >= 2:
        level = "Médio"
    else:
        level = "Baixo"

    if n_points < 10:
        return "Alto"
    if n_points < 15 and level == "Baixo":
        return "Médio"
    return level


def margin(price, cost):
    if price <= 0:
        return None
    return (price - cost) / price
