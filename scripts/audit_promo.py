import argparse
from datetime import date
import math
import os
import sys

import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.db import fetch_promotions, fetch_sales, fetch_sales_last_date
from app.main import STORE_CODES, apply_promo_flags, select_training_window
from app.ml import fill_missing_days, prepare_daily


def _parse_stores(raw: str | None):
    if not raw:
        return None
    items = [s.strip() for s in raw.split(",") if s.strip()]
    if not items:
        return None
    if any(item.upper() == "ALL" for item in items):
        return None
    return items


def _fmt_num(value, digits=2):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    return f"{value:.{digits}f}"


def _mean(series):
    if series is None or len(series) == 0:
        return None
    return float(series.mean())


def _median(series):
    if series is None or len(series) == 0:
        return None
    return float(series.median())


def _promo_stats(df):
    if df is None or df.empty or "promocao" not in df.columns:
        return {
            "total_days": 0,
            "promo_weighted_days": 0,
            "promo_any_days": 0,
            "promo_full_days": 0,
            "promo_share": None,
        }
    promo_vals = pd.to_numeric(df["promocao"], errors="coerce").fillna(0)
    promo_vals = promo_vals.clip(lower=0, upper=1)
    total_days = len(df)
    promo_weighted_days = float(promo_vals.sum())
    promo_any_days = int((promo_vals > 0).sum())
    promo_full_days = int((promo_vals >= 1 - 1e-9).sum())
    promo_share = (
        promo_weighted_days / float(total_days) if total_days > 0 else None
    )
    return {
        "total_days": total_days,
        "promo_weighted_days": promo_weighted_days,
        "promo_any_days": promo_any_days,
        "promo_full_days": promo_full_days,
        "promo_share": promo_share,
    }


def _split_promo(df):
    promo_vals = pd.to_numeric(df["promocao"], errors="coerce").fillna(0)
    promo_vals = promo_vals.clip(lower=0, upper=1)
    promo = df[promo_vals > 0]
    normal = df[promo_vals <= 0]
    return promo, normal


def _basic_effect(df, label):
    if df is None or df.empty:
        return {
            "label": label,
            "promo_qty_mean": None,
            "promo_qty_median": None,
            "normal_qty_mean": None,
            "normal_qty_median": None,
            "ratio_mean": None,
            "price_promo_mean": None,
            "price_normal_mean": None,
            "price_diff_pct": None,
        }
    promo, normal = _split_promo(df)
    promo_qty_mean = _mean(promo["quantidade"])
    promo_qty_median = _median(promo["quantidade"])
    normal_qty_mean = _mean(normal["quantidade"])
    normal_qty_median = _median(normal["quantidade"])
    ratio_mean = (
        promo_qty_mean / normal_qty_mean
        if promo_qty_mean is not None and normal_qty_mean not in (None, 0)
        else None
    )
    price_promo_mean = _mean(promo["preco_unit"])
    price_normal_mean = _mean(normal["preco_unit"])
    price_diff_pct = (
        (price_promo_mean - price_normal_mean) / price_normal_mean
        if price_promo_mean is not None
        and price_normal_mean not in (None, 0)
        else None
    )
    return {
        "label": label,
        "promo_qty_mean": promo_qty_mean,
        "promo_qty_median": promo_qty_median,
        "normal_qty_mean": normal_qty_mean,
        "normal_qty_median": normal_qty_median,
        "ratio_mean": ratio_mean,
        "price_promo_mean": price_promo_mean,
        "price_normal_mean": price_normal_mean,
        "price_diff_pct": price_diff_pct,
    }


def _print_effect(effect):
    print(f"\n{effect['label']}")
    print(
        "Qtd média (promo vs normal): "
        f"{_fmt_num(effect['promo_qty_mean'])} vs {_fmt_num(effect['normal_qty_mean'])} "
        f"(razão {_fmt_num(effect['ratio_mean'])})"
    )
    print(
        "Qtd mediana (promo vs normal): "
        f"{_fmt_num(effect['promo_qty_median'])} vs {_fmt_num(effect['normal_qty_median'])}"
    )
    print(
        "Preço médio (promo vs normal): "
        f"{_fmt_num(effect['price_promo_mean'])} vs {_fmt_num(effect['price_normal_mean'])} "
        f"(dif {_fmt_num(effect['price_diff_pct'] * 100 if effect['price_diff_pct'] is not None else None, 1)}%)"
    )


def _store_effects(
    codigoint,
    start_date,
    end_date,
    store_list,
    min_promo_days,
):
    rows = []
    for store in store_list:
        sales = fetch_sales(codigoint, start_date, end_date, [store])
        daily = prepare_daily(sales)
        daily = fill_missing_days(daily, start_date, end_date)
        try:
            promo_ranges = fetch_promotions(
                codigoint, start_date, end_date, [store]
            )
        except Exception:
            promo_ranges = []
        daily = apply_promo_flags(daily, promo_ranges, [store])
        if daily is None or daily.empty:
            continue
        promo_vals = pd.to_numeric(daily["promocao"], errors="coerce").fillna(0)
        promo_days_any = int((promo_vals > 0).sum())
        if promo_days_any < min_promo_days:
            continue
        promo, normal = _split_promo(daily)
        promo_qty_mean = _mean(promo["quantidade"])
        normal_qty_mean = _mean(normal["quantidade"])
        ratio = (
            promo_qty_mean / normal_qty_mean
            if promo_qty_mean is not None and normal_qty_mean not in (None, 0)
            else None
        )
        price_promo_mean = _mean(promo["preco_unit"])
        price_normal_mean = _mean(normal["preco_unit"])
        price_diff_pct = (
            (price_promo_mean - price_normal_mean) / price_normal_mean
            if price_promo_mean is not None
            and price_normal_mean not in (None, 0)
            else None
        )
        rows.append(
            {
                "store": store,
                "promo_days": promo_days_any,
                "ratio": ratio,
                "qty_promo": promo_qty_mean,
                "qty_normal": normal_qty_mean,
                "price_promo": price_promo_mean,
                "price_normal": price_normal_mean,
                "price_diff_pct": price_diff_pct,
            }
        )
    if not rows:
        return []
    df = pd.DataFrame(rows)
    df = df.sort_values(
        by=["ratio", "promo_days"],
        ascending=[True, False],
        na_position="last",
    )
    return df.to_dict(orient="records")


def main():
    parser = argparse.ArgumentParser(
        description="Audita histórico de promoção vs dias normais."
    )
    parser.add_argument("--codigoint", required=True)
    parser.add_argument("--stores", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--min-promo-days", type=int, default=6)
    args = parser.parse_args()

    selected_stores = _parse_stores(args.stores)
    store_list = selected_stores or STORE_CODES

    end_dt = date.today()
    if args.end:
        try:
            end_dt = date.fromisoformat(args.end)
        except ValueError:
            pass

    last_date = None
    try:
        last_date = fetch_sales_last_date(args.codigoint, selected_stores)
    except Exception:
        last_date = None
    if last_date:
        try:
            last_date = last_date.date()
        except AttributeError:
            pass
        if last_date < end_dt:
            end_dt = last_date

    best, _notes = select_training_window(
        args.codigoint, end_dt, selected_stores
    )
    start_date = best["start_date"]
    end_date = best["end_date"]
    daily = best["daily"]

    print("AUDITORIA PROMO")
    print(f"Produto: {args.codigoint}")
    print(
        f"Lojas: {'Todas' if selected_stores is None else ', '.join(selected_stores)}"
    )
    print(f"Janela usada: {best['days']} dias ({start_date} a {end_date})")

    stats = _promo_stats(daily)
    if stats["total_days"] > 0:
        print("\nCobertura de promo")
        print(
            f"Dias totais: {stats['total_days']} | "
            f"Promo (qualquer loja): {stats['promo_any_days']} | "
            f"Promo (todas as lojas): {stats['promo_full_days']} | "
            f"Promo ponderada: {_fmt_num(stats['promo_share'] * 100 if stats['promo_share'] is not None else None, 1)}%"
        )

    if daily is None or daily.empty:
        print("\nSem dados para a janela escolhida.")
        return

    _print_effect(_basic_effect(daily, "Todos os dias"))

    if "quantidade" in daily.columns:
        with_sales = daily[daily["quantidade"] > 0]
        _print_effect(_basic_effect(with_sales, "Somente dias com venda"))

    if "ruptura" in daily.columns:
        no_rupture = daily[daily["ruptura"] <= 0]
        _print_effect(_basic_effect(no_rupture, "Sem ruptura"))

    store_effects = _store_effects(
        args.codigoint,
        start_date,
        end_date,
        store_list,
        args.min_promo_days,
    )
    if store_effects:
        print("\nLojas com pior efeito (promo/normal)")
        for row in store_effects[:5]:
            print(
                f"{row['store']}: promo_days={row['promo_days']} | "
                f"ratio={_fmt_num(row['ratio'])} | "
                f"qty_promo={_fmt_num(row['qty_promo'])} vs qty_normal={_fmt_num(row['qty_normal'])} | "
                f"preco_diff={_fmt_num(row['price_diff_pct'] * 100 if row['price_diff_pct'] is not None else None, 1)}%"
            )


if __name__ == "__main__":
    main()
