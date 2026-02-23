import os
from contextlib import contextmanager

import mysql.connector

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

if load_dotenv:
    load_dotenv()

_DB_ENV_VARS = {
    "host": "DB_HOST",
    "user": "DB_USER",
    "password": "DB_PASSWORD",
    "database": "DB_NAME",
}


def _load_db_config():
    config = {}
    missing = []
    for key, env_name in _DB_ENV_VARS.items():
        value = os.getenv(env_name)
        if value is None or value == "":
            missing.append(env_name)
            continue
        config[key] = value
    if missing:
        missing_list = ", ".join(missing)
        raise RuntimeError(
            f"Missing required database env vars: {missing_list}"
        )
    return config


@contextmanager
def get_conn():
    conn = mysql.connector.connect(**_load_db_config())
    try:
        yield conn
    finally:
        conn.close()


def fetch_products():
    query = """
        SELECT codigo_produto, descricao_produto
        FROM produtos_nexello
        ORDER BY descricao_produto
    """
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
    return [{"codigoint": r[0], "descricao": r[1]} for r in rows]


def fetch_product_cost(codigoint):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT MAX(data_historico)
            FROM estoque_nexello
            WHERE codigo = %s
            """,
            (codigoint,),
        )
        max_date = cur.fetchone()
        if not max_date or not max_date[0]:
            return None
        cur.execute(
            """
            SELECT AVG(custo_medio)
            FROM estoque_nexello
            WHERE codigo = %s
              AND data_historico BETWEEN DATE_SUB(%s, INTERVAL 30 DAY) AND %s
              AND custo_medio > 0
            """,
            (codigoint, max_date[0], max_date[0]),
        )
        row = cur.fetchone()
    return row[0] if row else None


def _normalize_store_codes(stores):
    if not stores:
        return None
    normalized = []
    for store in stores:
        if store is None:
            continue
        raw = str(store).strip()
        if not raw:
            continue
        try:
            normalized.append(str(int(raw)))
        except ValueError:
            normalized.append(raw)
    return normalized or None


def _expand_store_codes_for_capture(stores):
    if not stores:
        return None
    tokens = []
    for store in stores:
        if store is None:
            continue
        raw = str(store).strip()
        if not raw:
            continue
        tokens.append(raw)
        if raw.isdigit():
            tokens.append(str(int(raw)))
            tokens.append(raw.zfill(3))
        else:
            tokens.append(raw.upper())
    seen = set()
    expanded = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        expanded.append(token)
    return expanded or None


def _store_code_tokens(value):
    if value is None:
        return set()
    raw = str(value).strip()
    if not raw:
        return set()
    tokens = {raw}
    if raw.isdigit():
        tokens.add(str(int(raw)))
        tokens.add(raw.zfill(3))
    else:
        tokens.add(raw.upper())
    return tokens


def fetch_sales(codigoint, start_date, end_date, stores):
    stores = _expand_store_codes_for_capture(stores)
    base = """
        SELECT
            c.SiglaLoja,
            c.DtMovimento,
            c.Quantidade,
            c.Custo,
            c.Venda,
            CASE
                WHEN e.quantidade_estoque IS NULL THEN NULL
                WHEN e.quantidade_estoque <= 0 THEN 1
                ELSE 0
            END AS ruptura
        FROM sgdados.sig_captura c
        LEFT JOIN nexello.estoque_nexello e
            ON e.codigo = c.CODIGOINT
           AND e.filial = c.SiglaLoja
           AND e.data_historico = c.DtMovimento
        WHERE c.CODIGOINT = %s
          AND c.DtMovimento BETWEEN %s AND %s
    """
    params = [codigoint, start_date, end_date]

    if stores:
        placeholders = ",".join(["%s"] * len(stores))
        base += f" AND c.SiglaLoja IN ({placeholders})"
        params.extend(stores)

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(base, params)
        rows = cur.fetchall()

    return rows


def fetch_sales_last_date(codigoint, stores):
    stores = _expand_store_codes_for_capture(stores)
    base = """
        SELECT MAX(c.DtMovimento)
        FROM sgdados.sig_captura c
        WHERE c.CODIGOINT = %s
    """
    params = [codigoint]

    if stores:
        placeholders = ",".join(["%s"] * len(stores))
        base += f" AND c.SiglaLoja IN ({placeholders})"
        params.extend(stores)

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(base, params)
        row = cur.fetchone()

    return row[0] if row else None


def _merge_date_ranges(ranges):
    cleaned = []
    for start, end in ranges:
        if start is None or end is None:
            continue
        cleaned.append((start, end))
    if not cleaned:
        return []
    cleaned.sort(key=lambda r: r[0])
    merged = [cleaned[0]]
    for start, end in cleaned[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def fetch_promotions(codigoint, start_date, end_date, stores):
    store_tokens_map = None
    if stores:
        store_tokens_map = {}
        for store in stores:
            raw = str(store).strip()
            if not raw:
                continue
            store_tokens_map[raw] = _store_code_tokens(raw)

    raw_code = str(codigoint).strip()
    codes = [raw_code]
    if raw_code.isdigit():
        trimmed = str(int(raw_code))
        if trimmed != raw_code:
            codes.append(trimmed)

    query = """
        SELECT pi.CodPromocao, p.DataInicio, p.DataFim, p.Lojas
        FROM sgdados.prc_promocaoitens pi
        JOIN sgdados.prc_promocoes p
            ON p.Codigo = pi.CodPromocao
        WHERE pi.CODIGOINT IN ({})
          AND p.DataFim >= %s
          AND p.DataInicio <= %s
    """
    placeholders = ",".join(["%s"] * len(codes))
    query = query.format(placeholders)
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(query, (*codes, start_date, end_date))
        rows = cur.fetchall()

    ranges = []
    ranges_by_store = {}
    if store_tokens_map:
        for store in store_tokens_map:
            ranges_by_store[store] = []
    for _, data_inicial, data_final, lojas in rows:
        if lojas is None:
            continue
        tokens = [
            t.strip()
            for t in str(lojas).replace(" ", "").split(",")
            if t.strip()
        ]
        if not tokens:
            continue
        if not store_tokens_map:
            ranges.append((data_inicial, data_final))
            continue

        token_sets = [_store_code_tokens(token) for token in tokens]
        if not token_sets:
            continue
        for store, store_tokens in store_tokens_map.items():
            matched = False
            for token_set in token_sets:
                if store_tokens & token_set:
                    matched = True
                    break
            if matched:
                ranges_by_store[store].append((data_inicial, data_final))

    if not store_tokens_map:
        return _merge_date_ranges(ranges)

    merged_by_store = {}
    for store, store_ranges in ranges_by_store.items():
        merged = _merge_date_ranges(store_ranges)
        if merged:
            merged_by_store[store] = merged
    return merged_by_store


def fetch_store_price(codigoint, stores):
    stores = _normalize_store_codes(stores)
    with get_conn() as conn:
        cur = conn.cursor()
        base = """
            SELECT AVG(valor_venda)
            FROM estoque_nexello
            WHERE codigo = %s
              AND DATE(data_historico) = CURDATE()
        """
        params = [codigoint]
        if stores:
            placeholders = ",".join(["%s"] * len(stores))
            base += f" AND filial IN ({placeholders})"
            params.extend(stores)
        cur.execute(base, params)
        row = cur.fetchone()

    return row[0] if row else None


def calc_margin(codigoint, mode, venda):
    if venda is None:
        return None
    try:
        query = "SELECT margem_calc(%s, %s, %s)"
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute(query, (codigoint, mode, venda))
            row = cur.fetchone()
        if row and row[0] is not None:
            return row[0]
    except Exception:
        pass

    cost = fetch_product_cost(codigoint)
    if cost is None:
        return None
    try:
        venda_val = float(venda)
        cost_val = float(cost)
    except (TypeError, ValueError):
        return None
    if venda_val <= 0:
        return None
    if str(mode).upper() == "P":
        return (venda_val - cost_val) / venda_val * 100
    if str(mode).upper() == "V":
        return venda_val - cost_val
    return None
