from datetime import date, timedelta
import math
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, Form, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.db import (
    calc_margin,
    fetch_product_cost,
    fetch_products,
    fetch_promotions,
    fetch_sales_last_date,
    fetch_sales,
    fetch_store_price,
)
from app.ml import (
    XGBOOST_AVAILABLE,
    apply_qty_outlier_filter,
    build_feature_matrix,
    fill_missing_days,
    fit_demand_model,
    forecast_qty_series,
    predict_qty,
    prepare_daily,
    risk_level,
    round_qty,
    suggest_price,
)

STORE_CODES = [f"{i:03d}" for i in range(1, 27)]
OUTLIER_METHOD = "winsorize"
OUTLIER_UPPER_Q = 0.98
PROMO_SHARE_THRESHOLD = 0.60
PROMO_SHARE_WARN_HIGH = 0.80
PROMO_SHARE_WARN_LOW = 0.05
PROMO_INTERACTION_MIN_PROMO_DAYS = 8
PROMO_BASE_MIN_POINTS = 60
PROMO_BASE_MIN_POINTS_FALLBACK = 30
PROMO_MIN_SHARE = 0.10
PROMO_MIN_DAYS = 10


def _resolve_promo_stores(selected_stores):
    if selected_stores:
        return selected_stores
    return STORE_CODES

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

app = FastAPI(title="Projeto ML - Precificacao")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def parse_float(value):
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    if "," in raw and "." in raw:
        raw = raw.replace(".", "").replace(",", ".")
    else:
        raw = raw.replace(",", ".")
    try:
        return float(raw)
    except ValueError:
        return None


def load_products_safe():
    try:
        return fetch_products(), None
    except Exception as exc:
        return [], str(exc)


def product_display(codigoint: Optional[str], products: List[dict]) -> str:
    if not codigoint:
        return ""
    for p in products:
        if str(p.get("codigoint")) == str(codigoint):
            return f'{p.get("descricao")} ({p.get("codigoint")})'
    return ""


def revenue_series(qty_series, price):
    if price is None:
        return [None for _ in qty_series]
    return [None if q is None else round(q * price, 2) for q in qty_series]


def sum_series(values):
    if not values:
        return None
    total = 0.0
    has_value = False
    for value in values:
        if value is None:
            continue
        has_value = True
        total += float(value)
    return float(total) if has_value else None


def filter_training_daily(
    daily,
    min_points=8,
    remove_promo=True,
    remove_rupture=True,
):
    info = {
        "total_days": 0,
        "sales_days": 0,
        "zero_sales_days": 0,
        "promo_days": 0,
        "promo_removed_days": 0,
        "rupture_days": 0,
        "remove_promo": bool(remove_promo),
        "remove_rupture": bool(remove_rupture),
        "used_promo_filter": False,
        "used_rupture_filter": False,
        "skipped_promo_filter": False,
        "skipped_rupture_filter": False,
    }
    if daily is None or daily.empty:
        return daily, info

    info["total_days"] = len(daily)
    if "quantidade" in daily.columns:
        info["sales_days"] = int((daily["quantidade"] > 0).sum())
        info["zero_sales_days"] = int((daily["quantidade"] <= 0).sum())
    if "promocao" in daily.columns:
        promo_vals = daily["promocao"].fillna(0)
        try:
            promo_vals = promo_vals.astype(float)
        except (TypeError, ValueError):
            pass
        promo_vals = promo_vals.clip(lower=0, upper=1)
        info["promo_days"] = float(promo_vals.sum())
    if "ruptura" in daily.columns:
        info["rupture_days"] = int((daily["ruptura"] > 0).sum())

    filtered = daily.copy()
    if remove_rupture and "ruptura" in filtered.columns:
        candidate = filtered[filtered["ruptura"] <= 0]
        if len(candidate) >= min_points:
            filtered = candidate
            info["used_rupture_filter"] = True
        else:
            info["skipped_rupture_filter"] = True

    if remove_promo and "promocao" in filtered.columns:
        before_len = len(filtered)
        candidate = filtered[filtered["promocao"] <= 0]
        removed_days = max(0, before_len - len(candidate))
        if len(candidate) >= min_points:
            filtered = candidate
            info["used_promo_filter"] = True
            info["promo_removed_days"] = removed_days
        else:
            info["skipped_promo_filter"] = True

    info["filtered_days"] = max(0, len(daily) - len(filtered))
    info["filtered_total_days"] = len(filtered)
    return filtered, info


def apply_promo_flags(daily, promo_ranges, stores=None):
    if daily is None or daily.empty:
        return daily
    if "promocao" in daily.columns:
        return daily
    daily = daily.copy()
    if not promo_ranges:
        daily["promocao"] = 0
        return daily

    def _as_date(value):
        try:
            return value.date()
        except AttributeError:
            return value

    dates = daily["dtmovimento"].dt.date
    base_mask = dates.isna() & False

    if isinstance(promo_ranges, dict):
        store_list = []
        if stores:
            for store in stores:
                if store is None:
                    continue
                raw = str(store).strip()
                if raw:
                    store_list.append(raw)
        if not store_list:
            daily["promocao"] = 0
            return daily

        counts = base_mask.astype(float)
        for store in store_list:
            ranges = promo_ranges.get(store)
            if not ranges:
                continue
            store_mask = base_mask.copy()
            for start, end in ranges:
                if start is None or end is None:
                    continue
                start = _as_date(start)
                end = _as_date(end)
                store_mask = store_mask | ((dates >= start) & (dates <= end))
            counts = counts + store_mask.astype(float)
        daily["promocao"] = counts / float(len(store_list))
        return daily

    mask = base_mask.copy()
    for start, end in promo_ranges:
        if start is None or end is None:
            continue
        start = _as_date(start)
        end = _as_date(end)
        mask = mask | ((dates >= start) & (dates <= end))
    daily["promocao"] = mask.astype(int)
    return daily


def compute_regime_stats(daily):
    stats = {
        "days_total_period": 0,
        "days_with_sales": 0,
        "days_promo": 0,
        "days_non_promo": 0,
        "share_non_promo": None,
        "days_after_filter": 0,
    }
    if daily is None or daily.empty:
        return stats

    stats["days_total_period"] = len(daily)
    if "quantidade" in daily.columns:
        stats["days_with_sales"] = int((daily["quantidade"] > 0).sum())

    promo_vals = None
    if "promocao" in daily.columns:
        promo_vals = daily["promocao"].fillna(0)
        try:
            promo_vals = promo_vals.astype(float)
        except (TypeError, ValueError):
            pass
        promo_vals = promo_vals.clip(lower=0, upper=1)
        promo_sum = float(promo_vals.sum())
        stats["days_promo"] = promo_sum
        stats["days_non_promo"] = max(
            0.0, float(stats["days_total_period"]) - promo_sum
        )
    else:
        stats["days_non_promo"] = stats["days_total_period"]

    denom = stats["days_promo"] + stats["days_non_promo"]
    if denom > 0:
        stats["share_non_promo"] = stats["days_non_promo"] / float(denom)

    if "ruptura" in daily.columns and promo_vals is not None:
        rupture_ok = (daily["ruptura"] <= 0)
        stats["days_after_filter"] = float(
            ((1 - promo_vals) * rupture_ok).sum()
        )
    elif promo_vals is not None:
        stats["days_after_filter"] = float((1 - promo_vals).sum())
    elif "ruptura" in daily.columns:
        stats["days_after_filter"] = int((daily["ruptura"] <= 0).sum())
    else:
        stats["days_after_filter"] = stats["days_total_period"]

    return stats


def choose_training_mode(
    stats,
    min_points=PROMO_BASE_MIN_POINTS,
    fallback_min_points=PROMO_BASE_MIN_POINTS_FALLBACK,
):
    share = stats.get("share_non_promo")
    days_after = int(stats.get("days_after_filter") or 0)
    if share is None:
        share = 1.0

    enough_points = days_after >= min_points or days_after >= fallback_min_points
    if share >= PROMO_SHARE_THRESHOLD and enough_points:
        mode = "BASE"
        if days_after >= min_points:
            reason = (
                f"{int(round(share * 100))}% dos dias sem promoção; "
                f"{days_after} dias disponíveis sem promo/ruptura."
            )
        else:
            reason = (
                f"{int(round(share * 100))}% dos dias sem promoção; "
                f"treino base com {days_after} dias (fallback)."
            )
    else:
        mode = "PROMO_DOMINANTE"
        promo_share = int(round((1 - share) * 100))
        if share < PROMO_SHARE_THRESHOLD:
            reason = (
                f"Promoções em {promo_share}% dos dias; "
                f"treino sem promo ficaria com {days_after} dias."
            )
        else:
            reason = (
                f"Treino sem promo ficaria com {days_after} dias; "
                "usando modo com promo."
            )

    return mode, reason


def recent_slice(daily, days=14):
    if daily is None or daily.empty:
        return daily
    if len(daily) >= days:
        return daily.tail(days)
    return daily


def prefer_non_promo(daily):
    if daily is None or daily.empty:
        return daily
    filtered = daily
    if "ruptura" in filtered.columns:
        candidate = filtered[filtered["ruptura"] <= 0]
        if not candidate.empty:
            filtered = candidate
    if "promocao" in filtered.columns:
        candidate = filtered[filtered["promocao"] <= 0]
        if not candidate.empty:
            filtered = candidate
    return filtered


def safe_margin(codigoint, venda, mode):
    if venda is None or venda <= 0:
        return None
    try:
        value = calc_margin(codigoint, mode, venda)
    except Exception:
        return None
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_margin_pct(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if 0 <= value <= 1.5:
        return value * 100
    return value


def margin_from_price(codigoint, price, qty):
    if price is None:
        return None, None
    pct_raw = safe_margin(codigoint, price, "P")
    val_unit = safe_margin(codigoint, price, "V")
    pct = normalize_margin_pct(pct_raw)
    total_value = None
    if val_unit is not None and qty is not None:
        total_value = float(val_unit) * qty
    return pct, total_value


def fmt_money(value):
    if value is None:
        return None
    text = f"{float(value):,.2f}"
    return f"R$ {text.replace(',', 'X').replace('.', ',').replace('X', '.')}"


def fmt_qty(value):
    if value is None:
        return None
    try:
        value = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return f"{value:,}".replace(",", ".")


def pct_change(new, old):
    if new is None or old in (None, 0):
        return None
    try:
        return (new - old) / old * 100
    except ZeroDivisionError:
        return None


def elasticity_vs_base(model_info, base_price, target_price, promo_flag=0):
    if not model_info:
        return None
    try:
        base = float(base_price)
        target = float(target_price)
    except (TypeError, ValueError):
        return None
    if base <= 0 or target <= 0:
        return None
    if base == target:
        return float(model_info["elasticidade"])
    qty_base = predict_qty(model_info, base, promo_flag=promo_flag)
    qty_target = predict_qty(model_info, target, promo_flag=promo_flag)
    if qty_base in (None, 0) or qty_target is None:
        return None
    pct_q = (qty_target - qty_base) / qty_base
    pct_p = (target - base) / base
    if pct_p == 0:
        return float(model_info["elasticidade"])
    return float(pct_q / pct_p)


def profit_total(price, qty_total, avg_cost):
    if price is None or qty_total is None or avg_cost is None:
        return None
    try:
        return float((float(price) - float(avg_cost)) * float(qty_total))
    except (TypeError, ValueError):
        return None


def _compute_metrics(y_true, y_pred):
    n_eval = len(y_true)
    if n_eval == 0:
        return None
    abs_errors = [abs(p - t) for t, p in zip(y_true, y_pred)]
    sq_errors = [(p - t) ** 2 for t, p in zip(y_true, y_pred)]
    mae = sum(abs_errors) / n_eval
    rmse = (sum(sq_errors) / n_eval) ** 0.5
    mape_vals = [abs(p - t) / t for t, p in zip(y_true, y_pred) if t != 0]
    mape = (sum(mape_vals) / len(mape_vals)) * 100 if mape_vals else None

    mean_y = sum(y_true) / n_eval
    denom = sum((t - mean_y) ** 2 for t in y_true)
    r2 = None
    if denom > 0:
        r2 = 1 - (sum(sq_errors) / denom)

    return {
        "mae": mae,
        "rmse": rmse,
        "mape_pct": mape,
        "r2": r2,
    }


def run_backtest(
    daily,
    min_train_ratio=0.6,
    test_size=7,
    step=7,
    include_promo=True,
    include_promo_price_interaction=False,
):
    total_points = len(daily)
    if total_points < 9:
        return {
            "ok": False,
            "message": "Dados insuficientes para backtest (mínimo 9 dias).",
            "total_points": total_points,
        }

    min_train = max(30, int(total_points * min_train_ratio))
    if total_points < min_train + test_size:
        return {
            "ok": False,
            "message": "Dados insuficientes para walk-forward. "
            "É necessário mais histórico para gerar janelas de teste.",
            "total_points": total_points,
            "train_len": min_train,
            "test_len": max(0, total_points - min_train),
        }

    y_true = []
    y_pred = []
    y_pred_xgb = []
    y_true_xgb = []
    folds = 0
    elasticities = []

    idx = min_train
    while idx + test_size <= total_points:
        train = daily.iloc[:idx]
        test = daily.iloc[idx : idx + test_size]
        model_info = fit_demand_model(
            train,
            include_promo=include_promo,
            include_promo_price_interaction=include_promo_price_interaction,
        )
        if model_info:
            try:
                elasticities.append(float(model_info.get("elasticidade")))
            except (TypeError, ValueError):
                pass
            for _, row in test.iterrows():
                price = row.get("preco_unit")
                qty = row.get("quantidade")
                dt = row.get("dtmovimento")
                promo_flag = row.get("promocao", 0)
                if price is None or qty is None or dt is None:
                    continue
                pred = predict_qty(
                    model_info,
                    float(price),
                    date=dt,
                    promo_flag=promo_flag,
                )
                if pred is None:
                    continue
                y_true.append(float(qty))
                y_pred.append(float(pred))

        if XGBOOST_AVAILABLE:
            try:
                from xgboost import XGBRegressor  # type: ignore
            except Exception:
                XGBRegressor = None
            if XGBRegressor is not None:
                X_train, y_train, meta = build_feature_matrix(
                    train,
                    include_promo=include_promo,
                    include_promo_price_interaction=include_promo_price_interaction,
                )
                if X_train is not None and meta is not None:
                    xgb = XGBRegressor(
                        n_estimators=200,
                        max_depth=3,
                        learning_rate=0.1,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective="reg:squarederror",
                        random_state=42,
                    )
                    xgb.fit(X_train, y_train)
                    X_test, y_test, _ = build_feature_matrix(
                        test,
                        trend_start=meta["trend_start"],
                        trend_scale=meta["trend_scale"],
                        include_promo=include_promo,
                        include_promo_price_interaction=include_promo_price_interaction,
                    )
                    if X_test is not None and y_test is not None:
                        preds_log = xgb.predict(X_test)
                        preds = [math.exp(float(p)) for p in preds_log]
                        actuals = test["quantidade"].astype(float).tolist()
                        if len(actuals) == len(preds):
                            y_true_xgb.extend(actuals)
                            y_pred_xgb.extend(preds)

        folds += 1
        idx += step

    metrics = _compute_metrics(y_true, y_pred)
    if not metrics:
        return {
            "ok": False,
            "message": "Sem pontos válidos para avaliar no walk-forward.",
            "total_points": total_points,
        }

    elasticity_stats = None
    if elasticities:
        mean_el = sum(elasticities) / len(elasticities)
        var = sum((e - mean_el) ** 2 for e in elasticities) / len(elasticities)
        std_el = math.sqrt(var)
        elasticity_stats = {
            "count": len(elasticities),
            "mean": mean_el,
            "std": std_el,
            "min": min(elasticities),
            "max": max(elasticities),
            "range": max(elasticities) - min(elasticities),
        }

    result = {
        "ok": True,
        "total_points": total_points,
        "train_len": min_train,
        "test_len": total_points - min_train,
        "n_eval": len(y_true),
        "folds": folds,
        "metrics": metrics,
        "elasticity_stats": elasticity_stats,
        "validation": {
            "method": "walk-forward",
            "min_train_ratio": min_train_ratio,
            "test_size": test_size,
            "step": step,
        },
    }

    xgb_metrics = _compute_metrics(y_true_xgb, y_pred_xgb)
    if xgb_metrics:
        result["benchmark"] = {"xgboost": xgb_metrics}

    return result


def _promo_days_share(daily):
    if daily is None or daily.empty or "promocao" not in daily.columns:
        return 0, None
    promo_vals = daily["promocao"].fillna(0)
    try:
        promo_vals = promo_vals.astype(float)
    except (TypeError, ValueError):
        pass
    promo_vals = promo_vals.clip(lower=0, upper=1)
    promo_days = float(promo_vals.sum())
    total_days = len(daily)
    if total_days <= 0:
        return promo_days, None
    return promo_days, promo_days / float(total_days)


def should_try_promo_interaction(daily):
    promo_days, promo_share = _promo_days_share(daily)
    if promo_days < PROMO_INTERACTION_MIN_PROMO_DAYS:
        return False
    if promo_share is None:
        return False
    if promo_share <= PROMO_SHARE_WARN_LOW:
        return False
    if promo_share >= (1.0 - PROMO_SHARE_WARN_LOW):
        return False
    return True


def select_backtest_variant(daily):
    base_bt = run_backtest(
        daily,
        min_train_ratio=0.7,
        include_promo=True,
        include_promo_price_interaction=False,
    )
    best_bt = base_bt
    use_interaction = False
    interaction_bt = None

    if should_try_promo_interaction(daily):
        interaction_bt = run_backtest(
            daily,
            min_train_ratio=0.7,
            include_promo=True,
            include_promo_price_interaction=True,
        )
        base_mape = (base_bt.get("metrics") or {}).get("mape_pct")
        inter_mape = (interaction_bt.get("metrics") or {}).get("mape_pct")
        if interaction_bt.get("ok"):
            if (not base_bt.get("ok")) or (
                inter_mape is not None
                and (base_mape is None or inter_mape < base_mape)
            ):
                best_bt = interaction_bt
                use_interaction = True

    return best_bt, use_interaction, base_bt, interaction_bt


def _metrics_text(backtest_result):
    if not backtest_result or not backtest_result.get("ok"):
        return None
    metrics = backtest_result.get("metrics") or {}
    parts = []
    mape = metrics.get("mape_pct")
    r2 = metrics.get("r2")
    if mape is not None:
        parts.append(f"MAPE {mape:.1f}%")
    if r2 is not None:
        parts.append(f"R² {r2:.2f}")
    return " · ".join(parts) if parts else None


def select_training_window(codigoint, end_dt, selected_stores):
    windows = [90, 180, 365]
    candidates = []
    promo_stores = _resolve_promo_stores(selected_stores)

    max_days = max(windows)
    max_start_dt = end_dt - timedelta(days=max_days - 1)
    max_start_date = max_start_dt.isoformat()
    end_date = end_dt.isoformat()
    rows = fetch_sales(codigoint, max_start_date, end_date, selected_stores)
    daily_all = prepare_daily(rows)
    daily_all = fill_missing_days(daily_all, max_start_date, end_date)
    try:
        promo_ranges = fetch_promotions(
            codigoint, max_start_date, end_date, promo_stores
        )
    except Exception:
        promo_ranges = []
    daily_all = apply_promo_flags(daily_all, promo_ranges, promo_stores)

    for days in windows:
        start_dt = end_dt - timedelta(days=days - 1)
        start_date = start_dt.isoformat()
        if daily_all.empty:
            daily_raw = daily_all
        else:
            mask = (
                daily_all["dtmovimento"].dt.date >= start_dt
            ) & (daily_all["dtmovimento"].dt.date <= end_dt)
            daily_raw = daily_all.loc[mask].copy()
        regime_stats = compute_regime_stats(daily_raw)
        daily_filtered, filter_info = filter_training_daily(
            daily_raw,
            remove_promo=False,
            remove_rupture=True,
        )
        daily_model = apply_qty_outlier_filter(
            daily_filtered, method=OUTLIER_METHOD, upper_quantile=OUTLIER_UPPER_Q
        )
        backtest, promo_interaction, base_bt, inter_bt = select_backtest_variant(
            daily_model
        )
        candidates.append(
            {
                "days": days,
                "start_date": start_date,
                "end_date": end_date,
                "daily": daily_raw,
                "daily_model": daily_model,
                "filter_info": filter_info,
                "mode_stats": regime_stats,
                "backtest": backtest,
                "promo_interaction": promo_interaction,
                "backtest_base": base_bt,
                "backtest_interaction": inter_bt,
            }
        )

    def _mape_value(item):
        backtest = item.get("backtest") or {}
        metrics = backtest.get("metrics") or {}
        value = metrics.get("mape_pct")
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _elasticity_std(item):
        backtest = item.get("backtest") or {}
        stats = backtest.get("elasticity_stats") or {}
        value = stats.get("std")
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def score(item):
        backtest = item.get("backtest") or {}
        metrics = backtest.get("metrics") or {}
        mape = metrics.get("mape_pct")
        r2 = metrics.get("r2")
        mape_score = mape if mape is not None else 1e9
        r2_score = r2 if r2 is not None else -1e9
        return (mape_score, -r2_score, -(backtest.get("total_points") or 0))

    valid = [c for c in candidates if c["backtest"].get("ok")]
    chosen_by_stability = False
    chosen_by_window = False
    if valid:
        mape_values = [v for v in (_mape_value(c) for c in valid) if v is not None]
        best_mape = min(mape_values) if mape_values else None
        if best_mape is not None:
            tol = 0.03 * best_mape
            shortlist = []
            for item in valid:
                mape = _mape_value(item)
                if mape is None:
                    continue
                if (mape - best_mape) <= tol:
                    shortlist.append(item)
            if not shortlist:
                shortlist = [item for item in valid if _mape_value(item) == best_mape]

            def std_score(item):
                value = _elasticity_std(item)
                return value if value is not None else 1e9

            min_std = min(std_score(item) for item in shortlist)
            std_candidates = [
                item for item in shortlist if std_score(item) == min_std
            ]
            if len(std_candidates) > 1:
                best = max(std_candidates, key=lambda item: item["days"])
                chosen_by_window = True
            else:
                best = std_candidates[0]
            chosen_by_stability = len(shortlist) > 1
            reason_mode = "quality"
        else:
            best = min(valid, key=score)
            reason_mode = "quality"
    else:
        best = max(candidates, key=lambda c: c["backtest"].get("total_points") or 0)
        reason_mode = "data"

    notes = []
    if reason_mode == "quality":
        metrics_text = _metrics_text(best["backtest"]) or "melhores métricas"
        if chosen_by_stability:
            stability = _elasticity_std(best)
            stability_text = (
                f"desvio de elasticidade {stability:.3f}"
                if stability is not None
                else "elasticidade mais estável"
            )
            reason = (
                f"MAPE similar (<3%) e {stability_text}"
                + (" com janela maior." if chosen_by_window else ".")
            )
            notes.append(
                f"Usado: últimos {best['days']} dias "
                f"({best['start_date']} a {best['end_date']}) "
                f"porque teve {reason}"
            )
        else:
            notes.append(
                f"Usado: últimos {best['days']} dias "
                f"({best['start_date']} a {best['end_date']}) "
                f"porque teve {metrics_text} entre as janelas testadas."
            )
    else:
        notes.append(
            f"Usado: últimos {best['days']} dias "
            f"({best['start_date']} a {best['end_date']}) "
            "porque as outras janelas não tinham dados suficientes para backtest."
        )

    if best.get("promo_interaction"):
        notes.append(
            "Interação promoção x preço ativada porque melhorou o backtest."
        )

    for item in candidates:
        if item is best:
            continue
        bt = item["backtest"]
        if not bt.get("ok"):
            total_points = bt.get("total_points", 0)
            notes.append(
                f"Não usado: {item['days']} dias "
                f"({item['start_date']} a {item['end_date']}) — "
                f"dados insuficientes ({total_points} dias válidos)."
            )
        else:
            metrics_text = _metrics_text(bt) or "métricas piores"
            notes.append(
                f"Não usado: {item['days']} dias "
                f"({item['start_date']} a {item['end_date']}) — "
                f"{metrics_text} pior."
            )

    return best, notes


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    products, error = load_products_safe()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "products": products,
            "stores": STORE_CODES,
            "result": None,
            "error": error,
            "today": date.today().isoformat(),
            "selected": None,
            "selected_display": "",
        },
    )


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request,
    codigoint: str = Form(...),
    start_date: Optional[str] = Form(None),
    end_date: Optional[str] = Form(None),
    stores: Optional[List[str]] = Form(None),
    custom_price: Optional[str] = Form(None),
):
    products, error = load_products_safe()
    if error:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "products": products,
                "stores": STORE_CODES,
                "result": None,
                "error": error,
                "today": date.today().isoformat(),
                "selected": None,
                "selected_display": "",
            },
        )

    selected_stores = None
    if stores:
        if "ALL" not in stores:
            selected_stores = stores

    end_dt = date.today()
    if end_date:
        try:
            end_dt = date.fromisoformat(end_date)
        except ValueError:
            end_dt = date.today()

    last_date = None
    try:
        last_date = fetch_sales_last_date(codigoint, selected_stores)
    except Exception:
        last_date = None
    if last_date:
        try:
            last_date = last_date.date()
        except AttributeError:
            pass
        if last_date < end_dt:
            end_dt = last_date

    best_window, period_notes = select_training_window(
        codigoint, end_dt, selected_stores
    )
    start_date = best_window["start_date"]
    end_date = best_window["end_date"]
    daily_raw = best_window["daily"]
    daily_model = best_window.get("daily_model")
    filter_info = best_window.get("filter_info") or {}
    mode_stats = best_window.get("mode_stats") or {}
    model_mode, model_mode_reason = choose_training_mode(mode_stats)
    if model_mode == "BASE":
        model_mode_label = "BASE"
        model_mode_text = "Treinado sem promoções; recomendado para preço normal."
    else:
        model_mode_label = "PROMO DOMINANTE"
        model_mode_text = (
            "Promoções são frequentes; o modelo já incorpora esse efeito."
        )
    default_promo_flag = 1 if model_mode == "PROMO_DOMINANTE" else 0
    promo_interaction = bool(best_window.get("promo_interaction"))
    if daily_model is None:
        daily_model = daily_raw

    result = {
        "has_data": not daily_model.empty,
        "message": None,
        "history_labels": [],
        "history_qty": [],
        "history_price": [],
        "history_cost": [],
        "forecast_labels": [],
        "forecast_qty_current": [],
        "forecast_qty_suggested": [],
        "forecast_qty_custom": [],
        "forecast_revenue_current": [],
        "forecast_revenue_suggested": [],
        "forecast_revenue_custom": [],
        "warnings": [],
        "promo_days": 0,
        "promo_total_days": 0,
        "promo_share_pct": None,
        "promo_share": None,
        "promo_scenario_current_qty_30d": None,
        "promo_scenario_current_qty_30d_fmt": None,
        "promo_scenario_current_revenue_30d": None,
        "promo_scenario_current_revenue_30d_fmt": None,
        "promo_scenario_current_profit_30d": None,
        "promo_scenario_current_profit_30d_fmt": None,
        "promo_impact_current_qty_30d": None,
        "promo_impact_current_qty_30d_fmt": None,
        "promo_impact_current_revenue_30d": None,
        "promo_impact_current_revenue_30d_fmt": None,
        "promo_impact_current_profit_30d": None,
        "promo_impact_current_profit_30d_fmt": None,
        "promo_scenario_suggested_qty_30d": None,
        "promo_scenario_suggested_qty_30d_fmt": None,
        "promo_scenario_suggested_revenue_30d": None,
        "promo_scenario_suggested_revenue_30d_fmt": None,
        "promo_scenario_suggested_profit_30d": None,
        "promo_scenario_suggested_profit_30d_fmt": None,
        "promo_impact_suggested_qty_30d": None,
        "promo_impact_suggested_qty_30d_fmt": None,
        "promo_impact_suggested_revenue_30d": None,
        "promo_impact_suggested_revenue_30d_fmt": None,
        "promo_impact_suggested_profit_30d": None,
        "promo_impact_suggested_profit_30d_fmt": None,
        "promo_scenario_custom_qty_30d": None,
        "promo_scenario_custom_qty_30d_fmt": None,
        "promo_scenario_custom_revenue_30d": None,
        "promo_scenario_custom_revenue_30d_fmt": None,
        "promo_scenario_custom_profit_30d": None,
        "promo_scenario_custom_profit_30d_fmt": None,
        "promo_impact_custom_qty_30d": None,
        "promo_impact_custom_qty_30d_fmt": None,
        "promo_impact_custom_revenue_30d": None,
        "promo_impact_custom_revenue_30d_fmt": None,
        "promo_impact_custom_profit_30d": None,
        "promo_impact_custom_profit_30d_fmt": None,
        "promo_scenario_note": None,
        "model_mode": model_mode,
        "model_mode_label": model_mode_label,
        "model_mode_text": model_mode_text,
        "show_base_forecast": True,
        "show_promo_forecast": False,
        "show_promo_block": False,
        "model_mode_reason": model_mode_reason,
        "model_mode_share_non_promo": mode_stats.get("share_non_promo"),
        "model_mode_days_after_filter": mode_stats.get("days_after_filter"),
        "promo_interaction": promo_interaction,
        "summary": [],
        "period_notes": period_notes,
        "period_window_days": best_window["days"],
        "period_start_date": best_window["start_date"],
        "period_end_date": best_window["end_date"],
    }

    if daily_model.empty or len(daily_model) < 8:
        result["message"] = "Dados insuficientes para treinar o modelo (mínimo 8 dias)."
    else:
        filter_info = best_window.get("filter_info") or {}
        model_info_full = fit_demand_model(
            daily_model,
            include_promo=True,
            include_promo_price_interaction=promo_interaction,
        )

        if not model_info_full:
            result["message"] = "Não foi possível ajustar o modelo com os dados atuais."
        else:
            result["show_base_forecast"] = True
            base_model_note = None
            promo_model_missing = False
            model_info_promo = None
            model_info_base = None

            model_info = None
            if "promocao" in daily_model.columns:
                promo_vals = daily_model["promocao"].fillna(0)
                try:
                    promo_vals = promo_vals.astype(float)
                except (TypeError, ValueError):
                    pass
                promo_vals = promo_vals.clip(lower=0, upper=1)

                daily_base_model = daily_model[promo_vals <= 0]
                daily_promo_model = daily_model[promo_vals > 0]

                if len(daily_base_model) >= 8:
                    model_info_base = fit_demand_model(
                        daily_base_model,
                        include_promo=False,
                        include_promo_price_interaction=False,
                    )
                if len(daily_promo_model) >= 8:
                    model_info_promo = fit_demand_model(
                        daily_promo_model,
                        include_promo=False,
                        include_promo_price_interaction=False,
                    )

            if model_mode == "PROMO_DOMINANTE":
                if model_info_promo:
                    model_info = model_info_promo
                else:
                    model_info = model_info_full
                    base_model_note = (
                        "Promo dominante sem histórico suficiente de promoção; "
                        "usando todos os dias."
                    )
            else:
                if model_info_base:
                    model_info = model_info_base
                else:
                    model_info = model_info_full
                    base_model_note = (
                        "Base treinada com todos os dias por falta de histórico "
                        "sem promoção suficiente."
                    )

            recent = recent_slice(daily_raw, 14)
            recent = prefer_non_promo(recent)
            store_price = fetch_store_price(codigoint, selected_stores)
            try:
                current_price = float(store_price) if store_price is not None else None
            except (TypeError, ValueError):
                current_price = None
            if current_price is None:
                current_price = float(recent["preco_unit"].mean())
            product_cost = fetch_product_cost(codigoint)
            try:
                avg_cost = float(product_cost) if product_cost is not None else None
            except (TypeError, ValueError):
                avg_cost = None
            if avg_cost is None:
                avg_cost = float(recent["custo_unit"].mean())

            elasticity_value = model_info.get("elasticidade")
            try:
                elasticity_value = float(elasticity_value)
            except (TypeError, ValueError):
                elasticity_value = None
            promo_el = model_info.get("elasticidade_promo")
            if promo_el is not None and elasticity_value is not None:
                try:
                    promo_el = float(promo_el)
                    elasticity_value = max(
                        (elasticity_value, promo_el), key=lambda v: abs(v)
                    )
                except (TypeError, ValueError):
                    pass

            suggested_price = suggest_price(
                daily_model,
                model_info,
                avg_cost,
                promo_flag=default_promo_flag,
            )
            custom_price_value = parse_float(custom_price)

            last_date = daily_model["dtmovimento"].iloc[-1].date()
            qty_current_raw = predict_qty(
                model_info, current_price, date=last_date, promo_flag=default_promo_flag
            )
            qty_suggested_raw = predict_qty(
                model_info, suggested_price, date=last_date, promo_flag=default_promo_flag
            )
            qty_custom_raw = (
                predict_qty(
                    model_info,
                    custom_price_value,
                    date=last_date,
                    promo_flag=default_promo_flag,
                )
                if custom_price_value
                else None
            )

            qty_current = round_qty(qty_current_raw)
            qty_suggested = round_qty(qty_suggested_raw)
            qty_custom = round_qty(qty_custom_raw)

            revenue_current = (
                float(qty_current * current_price)
                if qty_current is not None
                else None
            )
            revenue_suggested = (
                float(qty_suggested * suggested_price)
                if qty_suggested is not None
                else None
            )
            revenue_custom = (
                float(qty_custom * custom_price_value)
                if qty_custom is not None and custom_price_value is not None
                else None
            )

            margin_current_pct, margin_current_value = margin_from_price(
                codigoint, current_price, qty_current
            )
            margin_suggested_pct, margin_suggested_value = margin_from_price(
                codigoint, suggested_price, qty_suggested
            )
            if custom_price_value is not None:
                margin_custom_pct, margin_custom_value = margin_from_price(
                    codigoint, custom_price_value, qty_custom
                )
            else:
                margin_custom_pct = None
                margin_custom_value = None

            elasticity_current = elasticity_vs_base(
                model_info, current_price, current_price, promo_flag=default_promo_flag
            )
            elasticity_suggested = elasticity_vs_base(
                model_info, current_price, suggested_price, promo_flag=default_promo_flag
            )
            elasticity_custom = (
                elasticity_vs_base(
                    model_info,
                    current_price,
                    custom_price_value,
                    promo_flag=default_promo_flag,
                )
                if custom_price_value is not None
                else None
            )

            elasticity_ci = None
            if (
                model_info.get("elasticity_ci_low") is not None
                and model_info.get("elasticity_ci_high") is not None
            ):
                elasticity_ci = (
                    model_info.get("elasticity_ci_low"),
                    model_info.get("elasticity_ci_high"),
                )

            price_range_pct = None
            try:
                price_mean = float(daily_model["preco_unit"].mean())
                if price_mean > 0:
                    price_range_pct = float(
                        (
                            daily_model["preco_unit"].max()
                            - daily_model["preco_unit"].min()
                        )
                        / price_mean
                    )
            except (TypeError, ValueError):
                price_range_pct = None

            elasticity_instability = None
            if best_window.get("backtest") and best_window["backtest"].get(
                "elasticity_stats"
            ):
                elasticity_instability = best_window["backtest"]["elasticity_stats"].get(
                    "std"
                )

            trained_days = len(daily_model) if daily_model is not None else 0
            elasticity_for_risk = model_info.get("elasticidade")
            promo_el = model_info.get("elasticidade_promo")
            if promo_el is not None and elasticity_for_risk is not None:
                try:
                    elasticity_for_risk = max(
                        (float(elasticity_for_risk), float(promo_el)),
                        key=lambda v: abs(v),
                    )
                except (TypeError, ValueError):
                    pass

            risk_value = risk_level(
                trained_days,
                model_info["r2"],
                elasticity_for_risk,
                elasticity_ci,
                elasticity_instability=elasticity_instability,
                price_range_pct=price_range_pct,
            )
            if trained_days < 30:
                risk_value = "Alto"
            elif trained_days < 60 and risk_value == "Baixo":
                risk_value = "Médio"

            result.update(
                {
                    "current_price": current_price,
                    "suggested_price": suggested_price,
                    "custom_price": custom_price_value,
                    "qty_current": qty_current,
                    "qty_suggested": qty_suggested,
                    "qty_custom": qty_custom,
                    "elasticity": model_info["elasticidade"],
                    "r2": model_info["r2"],
                    "risk": risk_value,
                    "trained_days": trained_days,
                    "elasticity_ci_low": model_info.get("elasticity_ci_low"),
                    "elasticity_ci_high": model_info.get("elasticity_ci_high"),
                    "elasticity_se": model_info.get("elasticity_se"),
                    "elasticity_instability": elasticity_instability,
                    "price_range_pct": price_range_pct,
                    "margin_current_pct": margin_current_pct,
                    "margin_suggested_pct": margin_suggested_pct,
                    "margin_custom_pct": margin_custom_pct,
                    "margin_current_value": margin_current_value,
                    "margin_suggested_value": margin_suggested_value,
                    "margin_custom_value": margin_custom_value,
                    "avg_cost": avg_cost,
                    "elasticity_current": elasticity_current,
                    "elasticity_suggested": elasticity_suggested,
                    "elasticity_custom": elasticity_custom,
                    "promo_coef": model_info.get("promo_coef"),
                    "promo_interaction": promo_interaction,
                    "price_learned": model_info.get("price_learned"),
                    "price_points": model_info.get("price_points"),
                    "price_range_pct": model_info.get("price_range_pct"),
                }
            )

            result["history_labels"] = [
                d.strftime("%Y-%m-%d") for d in daily_raw["dtmovimento"].tolist()
            ]
            result["history_qty"] = [
                int(round(x)) for x in daily_raw["quantidade"].tolist()
            ]
            result["history_price"] = [
                float(x) for x in daily_raw["preco_unit"].tolist()
            ]
            result["history_cost"] = [
                float(x) for x in daily_raw["custo_unit"].tolist()
            ]

            forecast_start = date.today() + timedelta(days=1)
            days = 30
            result["forecast_labels"] = [
                (forecast_start + timedelta(days=i)).isoformat() for i in range(days)
            ]

            result["forecast_qty_current"] = forecast_qty_series(
                model_info,
                current_price,
                forecast_start,
                days,
                promo_flag=default_promo_flag,
            )
            result["forecast_qty_suggested"] = forecast_qty_series(
                model_info,
                suggested_price,
                forecast_start,
                days,
                promo_flag=default_promo_flag,
            )
            result["forecast_qty_custom"] = forecast_qty_series(
                model_info,
                custom_price_value,
                forecast_start,
                days,
                promo_flag=default_promo_flag,
            )

            result["forecast_revenue_current"] = revenue_series(
                result["forecast_qty_current"], current_price
            )
            result["forecast_revenue_suggested"] = revenue_series(
                result["forecast_qty_suggested"], suggested_price
            )
            result["forecast_revenue_custom"] = revenue_series(
                result["forecast_qty_custom"], custom_price_value
            )

            result["forecast_qty_current_total"] = sum_series(
                result["forecast_qty_current"]
            )
            result["forecast_qty_suggested_total"] = sum_series(
                result["forecast_qty_suggested"]
            )
            result["forecast_qty_custom_total"] = sum_series(
                result["forecast_qty_custom"]
            )
            result["forecast_revenue_current_total"] = sum_series(
                result["forecast_revenue_current"]
            )
            result["forecast_revenue_suggested_total"] = sum_series(
                result["forecast_revenue_suggested"]
            )
            result["forecast_revenue_custom_total"] = sum_series(
                result["forecast_revenue_custom"]
            )

            result["qty_current_30d"] = round_qty(
                result["forecast_qty_current_total"]
            )
            result["qty_suggested_30d"] = round_qty(
                result["forecast_qty_suggested_total"]
            )
            result["qty_custom_30d"] = round_qty(
                result["forecast_qty_custom_total"]
            )

            if elasticity_ci:
                ci_low, ci_high = elasticity_ci
                result["forecast_qty_current_low"] = forecast_qty_series(
                    model_info,
                    current_price,
                    forecast_start,
                    days,
                    elasticity=ci_low,
                    promo_flag=default_promo_flag,
                )
                result["forecast_qty_suggested_low"] = forecast_qty_series(
                    model_info,
                    suggested_price,
                    forecast_start,
                    days,
                    elasticity=ci_low,
                    promo_flag=default_promo_flag,
                )
                result["forecast_qty_custom_low"] = forecast_qty_series(
                    model_info,
                    custom_price_value,
                    forecast_start,
                    days,
                    elasticity=ci_low,
                    promo_flag=default_promo_flag,
                )
                result["forecast_qty_current_high"] = forecast_qty_series(
                    model_info,
                    current_price,
                    forecast_start,
                    days,
                    elasticity=ci_high,
                    promo_flag=default_promo_flag,
                )
                result["forecast_qty_suggested_high"] = forecast_qty_series(
                    model_info,
                    suggested_price,
                    forecast_start,
                    days,
                    elasticity=ci_high,
                    promo_flag=default_promo_flag,
                )
                result["forecast_qty_custom_high"] = forecast_qty_series(
                    model_info,
                    custom_price_value,
                    forecast_start,
                    days,
                    elasticity=ci_high,
                    promo_flag=default_promo_flag,
                )

                result["qty_current_30d_low"] = sum_series(
                    result["forecast_qty_current_low"]
                )
                result["qty_suggested_30d_low"] = sum_series(
                    result["forecast_qty_suggested_low"]
                )
                result["qty_custom_30d_low"] = sum_series(
                    result["forecast_qty_custom_low"]
                )
                result["qty_current_30d_high"] = sum_series(
                    result["forecast_qty_current_high"]
                )
                result["qty_suggested_30d_high"] = sum_series(
                    result["forecast_qty_suggested_high"]
                )
                result["qty_custom_30d_high"] = sum_series(
                    result["forecast_qty_custom_high"]
                )
            result["qty_current_30d_fmt"] = fmt_qty(result["qty_current_30d"])
            result["qty_suggested_30d_fmt"] = fmt_qty(result["qty_suggested_30d"])
            result["qty_custom_30d_fmt"] = fmt_qty(result["qty_custom_30d"])
            result["revenue_current_30d"] = result["forecast_revenue_current_total"]
            result["revenue_suggested_30d"] = result[
                "forecast_revenue_suggested_total"
            ]
            result["revenue_custom_30d"] = result["forecast_revenue_custom_total"]
            result["revenue_current_30d_fmt"] = fmt_money(
                result["revenue_current_30d"]
            )
            result["revenue_suggested_30d_fmt"] = fmt_money(
                result["revenue_suggested_30d"]
            )
            result["revenue_custom_30d_fmt"] = fmt_money(
                result["revenue_custom_30d"]
            )

            promo_days, promo_share = _promo_days_share(daily_model)
            valid_days = len(daily_model) if daily_model is not None else 0
            show_promo_block = (
                model_mode == "BASE"
                and promo_share is not None
                and promo_share >= PROMO_MIN_SHARE
                and promo_days >= PROMO_MIN_DAYS
            )
            if show_promo_block and model_info_promo is None:
                promo_model_missing = True
                show_promo_block = False
            result["promo_days"] = promo_days
            result["promo_total_days"] = valid_days
            result["promo_share"] = promo_share
            if promo_share is not None:
                result["promo_share_pct"] = int(round(promo_share * 100))
            else:
                result["promo_share_pct"] = None
            result["show_promo_block"] = show_promo_block
            result["show_promo_forecast"] = show_promo_block

            promo_forecast_current = None
            promo_forecast_suggested = None
            promo_forecast_custom = None
            if show_promo_block and model_info_promo is not None:
                promo_forecast_current = forecast_qty_series(
                    model_info_promo,
                    current_price,
                    forecast_start,
                    days,
                    promo_flag=0,
                )
                promo_forecast_suggested = forecast_qty_series(
                    model_info_promo,
                    suggested_price,
                    forecast_start,
                    days,
                    promo_flag=0,
                )
                promo_forecast_custom = forecast_qty_series(
                    model_info_promo,
                    custom_price_value,
                    forecast_start,
                    days,
                    promo_flag=0,
                )

            def _apply_promo_model(series, price):
                total = sum_series(series)
                if total is None:
                    return (None, None, None, None, None, None)
                qty = round_qty(total)
                revenue = float(total * price) if price is not None else None
                profit = profit_total(price, total, avg_cost)
                return (
                    qty,
                    fmt_qty(qty),
                    revenue,
                    fmt_money(revenue),
                    profit,
                    fmt_money(profit),
                )

            (
                result["promo_scenario_current_qty_30d"],
                result["promo_scenario_current_qty_30d_fmt"],
                result["promo_scenario_current_revenue_30d"],
                result["promo_scenario_current_revenue_30d_fmt"],
                result["promo_scenario_current_profit_30d"],
                result["promo_scenario_current_profit_30d_fmt"],
            ) = _apply_promo_model(promo_forecast_current, current_price)

            (
                result["promo_scenario_suggested_qty_30d"],
                result["promo_scenario_suggested_qty_30d_fmt"],
                result["promo_scenario_suggested_revenue_30d"],
                result["promo_scenario_suggested_revenue_30d_fmt"],
                result["promo_scenario_suggested_profit_30d"],
                result["promo_scenario_suggested_profit_30d_fmt"],
            ) = _apply_promo_model(promo_forecast_suggested, suggested_price)

            (
                result["promo_scenario_custom_qty_30d"],
                result["promo_scenario_custom_qty_30d_fmt"],
                result["promo_scenario_custom_revenue_30d"],
                result["promo_scenario_custom_revenue_30d_fmt"],
                result["promo_scenario_custom_profit_30d"],
                result["promo_scenario_custom_profit_30d_fmt"],
            ) = _apply_promo_model(promo_forecast_custom, custom_price_value)
            result["promo_scenario_note"] = None

            result["profit_current_30d"] = profit_total(
                current_price, result["qty_current_30d"], avg_cost
            )
            result["profit_suggested_30d"] = profit_total(
                suggested_price, result["qty_suggested_30d"], avg_cost
            )
            result["profit_custom_30d"] = profit_total(
                custom_price_value, result["qty_custom_30d"], avg_cost
            )
            if elasticity_ci:
                result["profit_current_30d_low"] = profit_total(
                    current_price, result.get("qty_current_30d_low"), avg_cost
                )
                result["profit_suggested_30d_low"] = profit_total(
                    suggested_price, result.get("qty_suggested_30d_low"), avg_cost
                )
                result["profit_custom_30d_low"] = profit_total(
                    custom_price_value, result.get("qty_custom_30d_low"), avg_cost
                )
                result["profit_current_30d_high"] = profit_total(
                    current_price, result.get("qty_current_30d_high"), avg_cost
                )
                result["profit_suggested_30d_high"] = profit_total(
                    suggested_price, result.get("qty_suggested_30d_high"), avg_cost
                )
                result["profit_custom_30d_high"] = profit_total(
                    custom_price_value, result.get("qty_custom_30d_high"), avg_cost
                )
            result["profit_current_30d_fmt"] = fmt_money(
                result["profit_current_30d"]
            )
            result["profit_suggested_30d_fmt"] = fmt_money(
                result["profit_suggested_30d"]
            )
            result["profit_custom_30d_fmt"] = fmt_money(result["profit_custom_30d"])

            def scenario_line(
                label,
                price,
                qty_total,
                profit_value,
                elasticity_value,
                risk,
                is_best,
                show_elasticity_risk=True,
            ):
                if price is None:
                    if show_elasticity_risk:
                        return (
                            f"{label}: informe um preço para calcular. "
                            f"Elasticidade — · Risco {risk}."
                        )
                    return f"{label}: informe um preço para calcular."
                price_text = fmt_money(price) or f"{price:.2f}"
                if qty_total is None or profit_value is None:
                    if show_elasticity_risk:
                        return (
                            f"{label} ({price_text}): sem previsão. "
                            f"Elasticidade — · Risco {risk}."
                        )
                    return f"{label} ({price_text}): sem previsão."
                qty_text = f"{fmt_qty(qty_total) or qty_total} un"
                profit_text = fmt_money(profit_value) or f"{profit_value:.2f}"
                elasticity_text = (
                    f"{elasticity_value:.2f}"
                    if elasticity_value is not None
                    else "—"
                )
                elasticity_suffix = (
                    f" Elasticidade {elasticity_text} · Risco {risk}."
                    if show_elasticity_risk
                    else ""
                )
                if is_best:
                    return (
                        f"Recomendado: {label} ({price_text}) porque gera maior lucro em 30 dias: "
                        f"{qty_text} · Lucro {profit_text}."
                        f"{elasticity_suffix}"
                    )
                return (
                    f"{label} ({price_text}): {qty_text} em 30 dias · Lucro {profit_text}."
                    f"{elasticity_suffix}"
                )

            scenarios = [
                {
                    "key": "current",
                    "label": "Preço atual",
                    "price": current_price,
                    "qty": result["qty_current_30d"],
                    "profit": result["profit_current_30d"],
                    "elasticity": elasticity_current,
                    "show_elasticity_risk": False,
                },
                {
                    "key": "suggested",
                    "label": "Preço sugerido",
                    "price": suggested_price,
                    "qty": result["qty_suggested_30d"],
                    "profit": result["profit_suggested_30d"],
                    "elasticity": elasticity_suggested,
                    "show_elasticity_risk": True,
                },
                {
                    "key": "custom",
                    "label": "Preço escolhido",
                    "price": custom_price_value,
                    "qty": result["qty_custom_30d"],
                    "profit": result["profit_custom_30d"],
                    "elasticity": elasticity_custom,
                    "show_elasticity_risk": custom_price_value is not None,
                },
            ]

            best = None
            candidates = [s for s in scenarios if s["profit"] is not None]
            if candidates:
                best = max(candidates, key=lambda item: item["profit"])

            scenario_lines = {}
            for s in scenarios:
                scenario_lines[s["key"]] = scenario_line(
                    s["label"],
                    s["price"],
                    s["qty"],
                    s["profit"],
                    s["elasticity"],
                    result["risk"],
                    is_best=(best is not None and s is best),
                    show_elasticity_risk=s.get("show_elasticity_risk", True),
                )

            summary = []
            if best:
                summary.append(scenario_lines[best["key"]])

            diff_line = None
            if (
                result["profit_current_30d"] is not None
                and result["profit_suggested_30d"] is not None
            ):
                diff_profit = result["profit_suggested_30d"] - result["profit_current_30d"]
                diff_text = fmt_money(abs(diff_profit)) or f"{abs(diff_profit):.2f}"
                if diff_profit > 0:
                    diff_line = (
                        f"Diferença de lucro (sugerido vs atual): +{diff_text} em 30 dias."
                    )
                elif diff_profit < 0:
                    diff_line = (
                        f"Diferença de lucro (sugerido vs atual): -{diff_text} em 30 dias."
                    )
                else:
                    diff_line = (
                        "Diferença de lucro (sugerido vs atual): igual em 30 dias."
                    )

            if diff_line:
                summary.append(diff_line)

            if best is None or best["key"] != "current":
                summary.append(scenario_lines["current"])

            if custom_price_value is not None and (
                best is None or best["key"] != "custom"
            ):
                summary.append(scenario_lines["custom"])

            rupture_days = int(filter_info.get("rupture_days") or 0)
            trained_days = len(daily_model) if daily_model is not None else 0
            warnings = []
            if base_model_note:
                warnings.append(
                    {
                        "title": "Base sem histórico suficiente",
                        "body": base_model_note,
                    }
                )
            if promo_model_missing:
                warnings.append(
                    {
                        "title": "Promoção sem histórico suficiente",
                        "body": "Não há dias promocionais suficientes para "
                        "treinar um modelo específico. O cenário promocional "
                        "foi ocultado.",
                    }
                )

            include_rupture = (
                rupture_days > 0 and filter_info.get("skipped_rupture_filter")
            )
            if include_rupture:
                warnings.append(
                    {
                        "title": "Treino com ruptura",
                        "body": "Modelo treinado com dias de ruptura por falta de "
                        "histórico suficiente para removê-los. Resultados podem "
                        "estar distorcidos.",
                    }
                )

            extreme_elasticity = False
            if elasticity_value is not None:
                try:
                    extreme_elasticity = abs(float(elasticity_value)) > 10
                except (TypeError, ValueError):
                    extreme_elasticity = False
            if extreme_elasticity:
                warnings.append(
                    {
                        "title": "Elasticidade muito elevada",
                        "body": "Elasticidade muito elevada — possível instabilidade "
                        "estatística ou baixa variação de preço. Interprete a "
                        "recomendação com cautela.",
                    }
                )

            share_non_promo = result.get("model_mode_share_non_promo")
            promo_share = None
            if share_non_promo is not None:
                try:
                    promo_share = 1.0 - float(share_non_promo)
                except (TypeError, ValueError):
                    promo_share = None
            if promo_share is not None and promo_share >= PROMO_SHARE_WARN_HIGH:
                warnings.append(
                    {
                        "title": "Promo dominante",
                        "body": "Promoções aparecem em mais de 80% dos dias do histórico. "
                        "O comparativo promo vs normal pode refletir um regime quase "
                        "sempre promocional.",
                    }
                )
            if promo_share is not None and promo_share <= PROMO_SHARE_WARN_LOW:
                warnings.append(
                    {
                        "title": "Poucos dias promocionais",
                        "body": "Há poucos dias com promoção no histórico. O efeito "
                        "da promoção pode estar impreciso.",
                    }
                )

            promo_coef = result.get("promo_coef")
            if promo_coef is not None:
                try:
                    if float(promo_coef) < 0:
                        warnings.append(
                            {
                                "title": "Coeficiente de promoção negativo",
                                "body": "O coeficiente de promoção ficou negativo. "
                                "Isso sugere inconsistência no flag de promoção ou "
                                "dados atípicos.",
                            }
                        )
                except (TypeError, ValueError):
                    pass

            promo_drop = False
            for base_val, promo_val in (
                (result.get("qty_current_30d"), result.get("promo_scenario_current_qty_30d")),
                (result.get("qty_suggested_30d"), result.get("promo_scenario_suggested_qty_30d")),
                (result.get("qty_custom_30d"), result.get("promo_scenario_custom_qty_30d")),
            ):
                if base_val is not None and promo_val is not None and promo_val < base_val:
                    promo_drop = True
                    break

            promo_hist_ratio = None
            if daily_model is not None and not daily_model.empty:
                if "promocao" in daily_model.columns and "quantidade" in daily_model.columns:
                    data = daily_model
                    if "ruptura" in data.columns:
                        no_rupture = data[data["ruptura"] <= 0]
                        if not no_rupture.empty:
                            data = no_rupture
                    promo_vals = data["promocao"].fillna(0)
                    try:
                        promo_vals = promo_vals.astype(float)
                    except (TypeError, ValueError):
                        pass
                    promo_vals = promo_vals.clip(lower=0, upper=1)
                    promo_mask = promo_vals > 0
                    if promo_mask.any() and (~promo_mask).any():
                        try:
                            promo_mean = float(data.loc[promo_mask, "quantidade"].mean())
                            normal_mean = float(data.loc[~promo_mask, "quantidade"].mean())
                        except (TypeError, ValueError):
                            promo_mean = None
                            normal_mean = None
                        if normal_mean not in (None, 0):
                            promo_hist_ratio = promo_mean / normal_mean

            if promo_drop and promo_hist_ratio is not None and promo_hist_ratio < 1:
                warnings.append(
                    {
                        "title": "Promoção reduz demanda",
                        "body": "A previsão com promoção ficou menor do que a previsão "
                        "de dia normal para o mesmo preço. Verifique o flag de promoção "
                        "ou possíveis inconsistências no histórico.",
                    }
                )

            if len(daily_model) < 20:
                warnings.append(
                    {
                        "title": "Amostra pequena",
                        "body": "Poucos dias válidos tornam o ajuste mais "
                        "sensível a ruído e podem gerar coeficientes instáveis.",
                    }
                )

            if model_info.get("price_learned") is False:
                warnings.append(
                    {
                        "title": "Pouca variação de preço",
                        "body": "O histórico não tem variação suficiente de preço "
                        "para estimar elasticidade com confiança. A recomendação "
                        "de preço foi ocultada.",
                    }
                )

            summary_notes = []

            promo_days = result.get("promo_days")
            promo_total_days = result.get("promo_total_days")
            promo_days_val = None
            total_days_val = None
            if promo_days is not None and promo_total_days:
                try:
                    promo_days_val = float(promo_days)
                    total_days_val = float(promo_total_days)
                except (TypeError, ValueError):
                    promo_days_val = None
                    total_days_val = None

            if trained_days > 0:
                line = f"Treino: {trained_days} dias usados no modelo."
                if promo_days_val is not None and total_days_val is not None:
                    normal_days_val = max(0.0, total_days_val - promo_days_val)

                    def _fmt_days(value):
                        if value is None:
                            return "—"
                        if abs(value - round(value)) < 0.05:
                            return str(int(round(value)))
                        return f"{value:.1f}"

                    line = (
                        f"{line} Promo: "
                        f"{_fmt_days(promo_days_val)} dias · "
                        f"Normal: {_fmt_days(normal_days_val)} dias."
                    )
                summary_notes.append(line)

            if rupture_days > 0:
                if filter_info.get("used_rupture_filter"):
                    summary_notes.append(
                        f"Ruptura: {rupture_days} dias com estoque zerado "
                        "foram excluídos do modelo."
                    )
                else:
                    summary_notes.append(
                        f"Ruptura: {rupture_days} dias com estoque zerado; "
                        "não foi possível excluir todos por falta de histórico."
                    )

            if summary_notes:
                summary.extend(summary_notes)

            result["summary"] = summary
            result["warnings"] = warnings
            if include_rupture or extreme_elasticity:
                result["risk"] = "Alto"

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "products": products,
            "stores": STORE_CODES,
            "result": result,
            "error": None,
            "selected": {
                "codigoint": codigoint,
                "start_date": start_date,
                "end_date": end_date,
                "stores": stores or [],
                "custom_price": custom_price,
            },
            "today": date.today().isoformat(),
            "selected_display": product_display(codigoint, products),
        },
    )


@app.get("/backtest", response_class=JSONResponse)
async def backtest(
    codigoint: str,
    start_date: str,
    end_date: str,
    stores: Optional[List[str]] = Query(None),
):
    selected_stores = None
    if stores:
        if "ALL" not in stores:
            selected_stores = stores
    promo_stores = _resolve_promo_stores(selected_stores)

    rows = fetch_sales(codigoint, start_date, end_date, selected_stores)
    daily = prepare_daily(rows)
    daily = fill_missing_days(daily, start_date, end_date)
    try:
        promo_ranges = fetch_promotions(
            codigoint, start_date, end_date, promo_stores
        )
    except Exception:
        promo_ranges = []
    daily = apply_promo_flags(daily, promo_ranges, promo_stores)
    regime_stats = compute_regime_stats(daily)
    daily_filtered, filter_info = filter_training_daily(
        daily, remove_promo=False, remove_rupture=True
    )
    daily_model = apply_qty_outlier_filter(
        daily_filtered, method=OUTLIER_METHOD, upper_quantile=OUTLIER_UPPER_Q
    )
    result, promo_interaction, _, _ = select_backtest_variant(daily_model)
    result.update(
        {
            "codigoint": codigoint,
            "start_date": start_date,
            "end_date": end_date,
            "stores": stores or ["ALL"],
            "filter_info": filter_info,
            "mode_stats": regime_stats,
            "promo_interaction": promo_interaction,
        }
    )
    return JSONResponse(result)


@app.get("/backtest/compare", response_class=JSONResponse)
async def backtest_compare(
    codigoint: str,
    end_date: Optional[str] = None,
    stores: Optional[List[str]] = Query(None),
):
    selected_stores = None
    if stores:
        if "ALL" not in stores:
            selected_stores = stores
    promo_stores = _resolve_promo_stores(selected_stores)

    if end_date is None:
        end_dt = date.today()
        end_date = end_dt.isoformat()
    else:
        try:
            end_dt = date.fromisoformat(end_date)
        except ValueError:
            return JSONResponse(
                {"ok": False, "message": "end_date inválido. Use YYYY-MM-DD."}
            )

    windows = [90, 180, 365]
    results = []
    for days in windows:
        start_dt = end_dt - timedelta(days=days - 1)
        start_date = start_dt.isoformat()
        rows = fetch_sales(codigoint, start_date, end_date, selected_stores)
        daily = prepare_daily(rows)
        daily = fill_missing_days(daily, start_date, end_date)
        try:
            promo_ranges = fetch_promotions(
                codigoint, start_date, end_date, promo_stores
            )
        except Exception:
            promo_ranges = []
        daily = apply_promo_flags(daily, promo_ranges, promo_stores)
        regime_stats = compute_regime_stats(daily)
        daily_filtered, filter_info = filter_training_daily(
            daily, remove_promo=False, remove_rupture=True
        )
        daily_model = apply_qty_outlier_filter(
            daily_filtered, method=OUTLIER_METHOD, upper_quantile=OUTLIER_UPPER_Q
        )
        result, promo_interaction, _, _ = select_backtest_variant(daily_model)
        result.update(
            {
                "window_days": days,
                "start_date": start_date,
                "end_date": end_date,
                "filter_info": filter_info,
                "mode_stats": regime_stats,
                "promo_interaction": promo_interaction,
            }
        )
        results.append(result)

    return JSONResponse(
        {
            "ok": True,
            "codigoint": codigoint,
            "stores": stores or ["ALL"],
            "windows": results,
        }
    )
