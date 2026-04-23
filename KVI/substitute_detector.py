"""
AGENT 3: SUBSTITUTE_DETECTOR
==============================
Находит товары-заменители внутри CATEGORY3 по месячной корреляции и цене,
а комплементы ищет отдельно внутри CATEGORY2.
"""

import json
import math
import os
import time
from collections import defaultdict

import duckdb
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

os.environ.setdefault(
    'MPLCONFIGDIR',
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output', 'kvi', '.mplconfig')
)

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(BASE)
DATA = os.path.join(WORKSPACE_ROOT, 'data')
OUT = os.path.join(WORKSPACE_ROOT, 'output', 'kvi')
os.makedirs(OUT, exist_ok=True)

HYPERPARAMETERS_PATH = os.path.join(BASE, 'hyperparameters.json')

DEFAULT_HYPERPARAMETERS = {
    'agent3': {
        'RANDOM_SEED': 42,
        'PRICE_SIMILARITY_THRESHOLD': 0.15,
        'MAX_ORDERS': 100_000,
        'MIN_ORDER_COUNT': 50,
        'MIN_COMPLEMENT_SUPPORT': 3,
        'SUBSTITUTE_CORR_THRESHOLD': -0.1,
        'CORR_THRESHOLD': 0.3,
        'LIFT_THRESHOLD': 1.5,
        'OUTPUT_CORR_FILTER': 0.1,
        'OUTPUT_LIFT_FILTER': 1.2,
    }
}


def _load_hyperparameters() -> dict:
    params = DEFAULT_HYPERPARAMETERS['agent3'].copy()
    if os.path.exists(HYPERPARAMETERS_PATH):
        with open(HYPERPARAMETERS_PATH, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            raw_agent3 = raw.get('agent3', raw)
            if isinstance(raw_agent3, dict):
                for key, value in raw_agent3.items():
                    if key in params:
                        params[key] = value
    return params


HP = _load_hyperparameters()

RANDOM_SEED = int(HP['RANDOM_SEED'])
PRICE_SIMILARITY_THRESHOLD = float(HP['PRICE_SIMILARITY_THRESHOLD'])
MAX_ORDERS = int(HP['MAX_ORDERS'])
MIN_ORDER_COUNT = int(HP['MIN_ORDER_COUNT'])
MIN_COMPLEMENT_SUPPORT = int(HP['MIN_COMPLEMENT_SUPPORT'])
SUBSTITUTE_CORR_THRESHOLD = float(HP['SUBSTITUTE_CORR_THRESHOLD'])
CORR_THRESHOLD = float(HP['CORR_THRESHOLD'])
LIFT_THRESHOLD = float(HP['LIFT_THRESHOLD'])
OUTPUT_CORR_FILTER = float(HP['OUTPUT_CORR_FILTER'])
OUTPUT_LIFT_FILTER = float(HP['OUTPUT_LIFT_FILTER'])

DEPENDENCY_TIMEOUT_SEC = 24 * 60 * 60
DEPENDENCY_POLL_SEC = 15

PAIR_COLUMNS = [
    'item_a',
    'item_b',
    'category1',
    'category2',
    'category3',
    'pair_scope',
    'co_purchases',
    'lift',
    'pearson_corr',
    'pair_type',
]

log = {
    'agent': 'SUBSTITUTE_DETECTOR',
    'status': 'SUCCESS',
    'rows_processed': 0,
    'warnings': [],
    'output_files': [],
    'hyperparameters': HP,
}


def _save_log() -> None:
    with open(os.path.join(OUT, 'agent3_log.json'), 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)


def _parse_ts_sql(column_name: str) -> str:
    return (
        f"COALESCE("
        f"TRY_STRPTIME(CAST({column_name} AS VARCHAR), '%Y-%m-%d %H:%M:%S'), "
        f"TRY_STRPTIME(CAST({column_name} AS VARCHAR), '%Y-%m-%d')"
        f")"
    )


def _binary_corr(co_purchases: int, order_count_a: int, order_count_b: int, total_orders: int) -> float:
    if total_orders <= 0 or order_count_a <= 0 or order_count_b <= 0:
        return 0.0

    denominator_term = (
        order_count_a * (total_orders - order_count_a) * order_count_b * (total_orders - order_count_b)
    )
    if denominator_term <= 0:
        return 0.0

    denominator = math.sqrt(float(denominator_term))
    if denominator <= 0:
        return 0.0

    numerator = float(co_purchases * total_orders - order_count_a * order_count_b)
    corr = numerator / denominator
    if corr > 1.0:
        return 1.0
    if corr < -1.0:
        return -1.0
    return float(corr)


def _standardize_series(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr

    centered = arr - float(arr.mean())
    norm = math.sqrt(float(np.dot(centered, centered)))
    if norm <= 0:
        return np.zeros_like(arr, dtype=float)

    return centered / norm


def _pearson_from_standardized_series(series_a: np.ndarray | None, series_b: np.ndarray | None) -> float:
    if series_a is None or series_b is None:
        return 0.0
    if series_a.shape != series_b.shape or series_a.size < 2:
        return 0.0

    corr = float(np.dot(series_a, series_b))
    if corr > 1.0:
        return 1.0
    if corr < -1.0:
        return -1.0
    return corr


def _is_complement_pair(corr: float, lift: float) -> bool:
    return corr > CORR_THRESHOLD and lift > LIFT_THRESHOLD


def _classify_substitute_pair(corr: float, price_a: float, price_b: float) -> str:
    if corr < SUBSTITUTE_CORR_THRESHOLD:
        if pd.notna(price_a) and pd.notna(price_b):
            avg_price = (price_a + price_b) / 2
            if avg_price > 0 and abs(price_a - price_b) / avg_price < PRICE_SIMILARITY_THRESHOLD:
                return 'DIRECT_SUBSTITUTE'
        return 'CANNIBALIZE'

    return 'INDEPENDENT'


def _sample_orders(con: duckdb.DuckDBPyConnection, total_valid_orders: int) -> int:
    max_orders = MAX_ORDERS
    if total_valid_orders <= max_orders:
        con.execute('CREATE TEMP TABLE sampled_orders AS SELECT ORDERID FROM valid_orders')
        return total_valid_orders

    sample_rate = max_orders / total_valid_orders
    log['warnings'].append(
        f'Sampling applied: {max_orders:,} max orders from {total_valid_orders:,} valid orders (ratio={sample_rate:.4f}).'
    )

    con.execute(
        f"""
        CREATE TEMP TABLE sampled_orders AS
        WITH ranked AS (
            SELECT
                ORDERID,
                order_month,
                ROW_NUMBER() OVER (
                    PARTITION BY order_month
                    ORDER BY HASH(CAST(ORDERID AS VARCHAR) || '-{RANDOM_SEED}')
                ) AS rn,
                COUNT(*) OVER (PARTITION BY order_month) AS month_size
            FROM valid_orders
        ),
        monthly_sample AS (
            SELECT ORDERID
            FROM ranked
            WHERE rn <= GREATEST(1, CAST(ROUND(month_size * {sample_rate}) AS BIGINT))
        )
        SELECT ORDERID
        FROM monthly_sample
        ORDER BY HASH(CAST(ORDERID AS VARCHAR) || '-{RANDOM_SEED}')
        LIMIT {MAX_ORDERS}
        """
    )

    return int(con.execute('SELECT COUNT(*) FROM sampled_orders').fetchone()[0])


def _pair_category3_label(category3_a: str, category3_b: str) -> str:
    if category3_a == category3_b:
        return category3_a
    return f'{category3_a} <> {category3_b}'


def _save_histograms(corr_values: list[float], lift_values: list[float]) -> list[str]:
    corr_path = os.path.join(OUT, 'agent3_corr_hist.png')
    lift_path = os.path.join(OUT, 'agent3_lift_hist.png')

    corr_threshold = CORR_THRESHOLD
    corr_filter = OUTPUT_CORR_FILTER
    lift_threshold = LIFT_THRESHOLD
    lift_filter = OUTPUT_LIFT_FILTER

    # Корреляции
    fig, ax = plt.subplots(figsize=(10, 6))
    if corr_values:
        ax.hist(corr_values, bins=np.linspace(-1, 1, 41), color='#4C78A8', alpha=0.85, edgecolor='white')
    else:
        ax.text(0.5, 0.5, 'No correlation data', ha='center', va='center', transform=ax.transAxes)
    ax.axvline(corr_filter, color='#7f8c8d', linestyle=':', linewidth=1.5, label=f'output filter = {corr_filter}')
    ax.axvline(corr_threshold, color='#c0392b', linestyle='--', linewidth=2, label=f'CORR_THRESHOLD = {corr_threshold}')
    ax.set_title('Distribution of Pearson correlations', fontsize=13)
    ax.set_xlabel('Pearson correlation')
    ax.set_ylabel('Pair count')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.25)
    fig.tight_layout()
    fig.savefig(corr_path, dpi=140, bbox_inches='tight')
    plt.close(fig)

    # Лифты: лог-шкала, потому что распределение очень скошено
    positive_lifts = [value for value in lift_values if value > 0]
    zero_lifts = len(lift_values) - len(positive_lifts)
    fig, ax = plt.subplots(figsize=(10, 6))
    if positive_lifts:
        min_positive = min(positive_lifts)
        max_positive = max(positive_lifts)
        if max_positive <= min_positive:
            max_positive = min_positive * 10
        bins = np.logspace(np.log10(min_positive), np.log10(max_positive), 40)
        ax.hist(positive_lifts, bins=bins, color='#F58518', alpha=0.85, edgecolor='white')
        ax.set_xscale('log')
    else:
        ax.text(0.5, 0.5, 'No positive lift data', ha='center', va='center', transform=ax.transAxes)
    ax.axvline(lift_filter, color='#7f8c8d', linestyle=':', linewidth=1.5, label=f'output filter = {lift_filter}')
    ax.axvline(lift_threshold, color='#c0392b', linestyle='--', linewidth=2, label=f'LIFT_THRESHOLD = {lift_threshold}')
    ax.set_title(f'Distribution of lift values (log scale; zero-lift pairs = {zero_lifts})', fontsize=13)
    ax.set_xlabel('Lift')
    ax.set_ylabel('Pair count')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.25)
    fig.tight_layout()
    fig.savefig(lift_path, dpi=140, bbox_inches='tight')
    plt.close(fig)

    return [corr_path, lift_path]


def _wait_for_file(
    path: str,
    label: str,
    timeout_sec: int = DEPENDENCY_TIMEOUT_SEC,
    min_mtime: float | None = None,
) -> None:
    start = time.time()
    last_notice = 0.0

    while True:
        exists = os.path.exists(path) and os.path.getsize(path) > 0
        fresh_enough = min_mtime is None or (exists and os.path.getmtime(path) >= min_mtime)
        if exists and fresh_enough:
            return

        elapsed = time.time() - start
        if elapsed >= timeout_sec:
            raise TimeoutError(f'Timed out waiting for {label} at {path}')

        if elapsed - last_notice >= 60:
            print(f'[AGENT 3] Waiting for {label}...', flush=True)
            last_notice = elapsed

        time.sleep(DEPENDENCY_POLL_SEC)


def main():
    print('[AGENT 3] SUBSTITUTE_DETECTOR starting...', flush=True)
    t0 = time.time()
    con = duckdb.connect()
    print(
        f"[AGENT 3] Hyperparameters: CORR_THRESHOLD={CORR_THRESHOLD}, "
        f"LIFT_THRESHOLD={LIFT_THRESHOLD}, "
        f"PRICE_SIMILARITY_THRESHOLD={PRICE_SIMILARITY_THRESHOLD}",
        flush=True,
    )

    flags_path = os.path.join(OUT, 'clean_flags.csv')
    bf_path = os.path.join(OUT, 'behavioral_features.csv')

    try:
        _wait_for_file(flags_path, 'clean_flags.csv')
        flags_mtime = os.path.getmtime(flags_path)
        _wait_for_file(bf_path, 'behavioral_features.csv', min_mtime=flags_mtime)

        bf = pd.read_csv(bf_path)
        bf['ITEMID'] = bf['ITEMID'].astype(str).str.strip()
        bf['CATEGORY2'] = bf['CATEGORY2'].fillna('').astype(str).str.strip()
        bf['CATEGORY3'] = bf['CATEGORY3'].fillna('').astype(str).str.strip()
        bf['avg_price'] = pd.to_numeric(bf['avg_price'], errors='coerce')

        eligible = bf[
            (bf['order_count'] > MIN_ORDER_COUNT)
            & (bf['CATEGORY3'] != '')
        ][['ITEMID', 'CATEGORY1', 'CATEGORY2', 'CATEGORY3', 'avg_price']].drop_duplicates('ITEMID').copy()

        if eligible.empty:
            log['status'] = 'WARNING'
            log['warnings'].append('No eligible items found (order_count > 50 and non-empty CATEGORY3).')
            hist_paths = _save_histograms([], [])
            pairs_df = pd.DataFrame(columns=PAIR_COLUMNS)
            pairs_path = os.path.join(OUT, 'item_pairs.csv')
            pairs_df.to_csv(pairs_path, index=False)
            sub_map_path = os.path.join(OUT, 'substitute_map.json')
            with open(sub_map_path, 'w', encoding='utf-8') as f:
                json.dump({}, f, indent=2, ensure_ascii=False)
            log['output_files'] = hist_paths + [pairs_path, sub_map_path]
            log['rows_processed'] = 0
            _save_log()
            return

        eligible = eligible.sort_values(['CATEGORY1', 'CATEGORY2', 'CATEGORY3', 'ITEMID']).reset_index(drop=True)
        price_map = eligible.set_index('ITEMID')['avg_price'].to_dict()
        category1_map = eligible.set_index('ITEMID')['CATEGORY1'].to_dict()
        print(f'[AGENT 3] Eligible items: {len(eligible):,}', flush=True)

        con.execute(
            f"""
            CREATE TEMP VIEW order_details_raw AS
            SELECT
                CAST(ORDERID AS VARCHAR) AS ORDERID,
                CAST(ORDERDETAILID AS VARCHAR) AS ORDERDETAILID,
                CAST(ITEMID AS VARCHAR) AS ITEMID
            FROM read_csv_auto('{DATA}/Order_Details.csv', header=true, delim=',')
            """
        )
        con.execute(
            f"""
            CREATE TEMP VIEW orders_raw AS
            SELECT
                CAST(ORDERID AS VARCHAR) AS ORDERID,
                CAST(DATE_ AS VARCHAR) AS DATE_
            FROM read_csv_auto('{DATA}/Orders.csv', header=true, delim=',')
            """
        )
        con.execute(
            f"""
            CREATE TEMP VIEW flags_raw AS
            SELECT
                CAST(ORDERDETAILID AS VARCHAR) AS ORDERDETAILID,
                CAST(is_valid AS INTEGER) AS is_valid
            FROM read_csv_auto('{flags_path}', header=true)
            """
        )

        con.execute(
            """
            CREATE TEMP VIEW valid_lines AS
            SELECT DISTINCT od.ORDERID, od.ITEMID
            FROM order_details_raw od
            JOIN flags_raw f USING (ORDERDETAILID)
            WHERE f.is_valid = 1
            """
        )

        con.execute(
            """
            CREATE TEMP TABLE valid_order_ids AS
            SELECT DISTINCT ORDERID
            FROM valid_lines
            """
        )

        con.execute(
            f"""
            CREATE TEMP TABLE valid_orders AS
            SELECT DISTINCT
                o.ORDERID,
                DATE_TRUNC('month', {_parse_ts_sql('o.DATE_')}) AS order_month
            FROM orders_raw o
            JOIN valid_order_ids v USING (ORDERID)
            WHERE {_parse_ts_sql('o.DATE_')} IS NOT NULL
            """
        )

        total_valid_orders = int(con.execute('SELECT COUNT(*) FROM valid_orders').fetchone()[0])
        if total_valid_orders == 0:
            log['status'] = 'WARNING'
            log['warnings'].append('No valid orders left after clean_flags filtering.')
            hist_paths = _save_histograms([], [])
            pairs_df = pd.DataFrame(columns=PAIR_COLUMNS)
            pairs_path = os.path.join(OUT, 'item_pairs.csv')
            pairs_df.to_csv(pairs_path, index=False)
            sub_map_path = os.path.join(OUT, 'substitute_map.json')
            with open(sub_map_path, 'w', encoding='utf-8') as f:
                json.dump({}, f, indent=2, ensure_ascii=False)
            log['output_files'] = hist_paths + [pairs_path, sub_map_path]
            log['rows_processed'] = 0
            _save_log()
            return

        all_months = pd.Index(
            con.execute('SELECT DISTINCT order_month FROM valid_orders ORDER BY order_month').df()['order_month']
        )

        sampled_orders_count = _sample_orders(con, total_valid_orders)
        print(f'[AGENT 3] Sampled orders: {sampled_orders_count:,}', flush=True)

        con.register('eligible_items', eligible[['ITEMID', 'CATEGORY1', 'CATEGORY2', 'CATEGORY3']])
        con.execute(
            """
            CREATE TEMP TABLE valid_eligible_lines AS
            SELECT DISTINCT vl.ORDERID, vl.ITEMID, ei.CATEGORY1, ei.CATEGORY2, ei.CATEGORY3
            FROM valid_lines vl
            JOIN sampled_orders so USING (ORDERID)
            JOIN eligible_items ei USING (ITEMID)
            """
        )

        valid_eligible_rows = int(con.execute('SELECT COUNT(*) FROM valid_eligible_lines').fetchone()[0])
        log['rows_processed'] = valid_eligible_rows
        print(f'[AGENT 3] Valid sampled order-item rows: {valid_eligible_rows:,}', flush=True)

        cat3_groups = eligible.groupby('CATEGORY3')['ITEMID'].apply(list).to_dict()
        cat2_groups = eligible.groupby('CATEGORY2')['ITEMID'].apply(list).to_dict()
        pairs_records = []
        sub_map = defaultdict(lambda: {'substitutes': set(), 'complements': set(), 'cannibals': set()})
        corr_values: list[float] = []
        lift_values: list[float] = []

        sample_orders = sampled_orders_count

        print(f'[AGENT 3] Analyzing {len(cat3_groups):,} CATEGORY3 groups for substitutes/cannibals...', flush=True)
        for cat3 in sorted(cat3_groups):
            cat_lines = con.execute(
                """
                SELECT vl.ORDERID, vl.ITEMID, vo.order_month, vl.CATEGORY1, vl.CATEGORY2, vl.CATEGORY3
                FROM valid_eligible_lines vl
                JOIN valid_orders vo USING (ORDERID)
                WHERE vl.CATEGORY3 = ?
                """,
                [cat3],
            ).df()

            if cat_lines.empty:
                continue

            cat_lines = cat_lines.drop_duplicates().sort_values(['ORDERID', 'ITEMID']).reset_index(drop=True)
            cat_lines['ITEMID'] = cat_lines['ITEMID'].astype(str).str.strip()
            item_levels = sorted(cat_lines['ITEMID'].unique().tolist())
            if len(item_levels) < 2:
                continue

            monthly_counts = (
                cat_lines.groupby(['order_month', 'ITEMID']).size().unstack(fill_value=0)
                .reindex(index=all_months, columns=item_levels, fill_value=0)
            )
            series_map = {
                item_id: _standardize_series(monthly_counts[item_id].to_numpy(dtype=float))
                for item_id in item_levels
            }
            order_codes, _ = pd.factorize(cat_lines['ORDERID'], sort=False)
            item_codes = pd.Categorical(cat_lines['ITEMID'], categories=item_levels).codes
            matrix = coo_matrix(
                (
                    np.ones(len(cat_lines), dtype=np.int8),
                    (order_codes, item_codes),
                ),
                shape=(len(np.unique(order_codes)), len(item_levels)),
            ).tocsr()

            item_counts = np.asarray(matrix.getnnz(axis=0)).ravel().astype(int)
            cooc = (matrix.T @ matrix).tocoo()
            cooc_lookup = {}
            for row_idx, col_idx, value in zip(cooc.row, cooc.col, cooc.data):
                if row_idx == col_idx:
                    continue
                key = (row_idx, col_idx) if row_idx < col_idx else (col_idx, row_idx)
                cooc_lookup[key] = int(value)

            category1 = str(cat_lines['CATEGORY1'].iloc[0]) if 'CATEGORY1' in cat_lines.columns else ''
            category2 = str(cat_lines['CATEGORY2'].iloc[0]) if 'CATEGORY2' in cat_lines.columns else ''
            pair_scope = f'WITHIN_CATEGORY3_n{len(item_levels)}'

            for (i, j), co_purchases in cooc_lookup.items():
                if co_purchases < MIN_COMPLEMENT_SUPPORT:
                    continue

                item_a = str(item_levels[i])
                item_b = str(item_levels[j])
                count_a = int(item_counts[i])
                count_b = int(item_counts[j])

                if count_a <= 0 or count_b <= 0:
                    continue

                price_a = price_map.get(item_a, np.nan)
                price_b = price_map.get(item_b, np.nan)
                corr = _pearson_from_standardized_series(series_map.get(item_a), series_map.get(item_b))
                if corr >= SUBSTITUTE_CORR_THRESHOLD:
                    continue

                pair_type = _classify_substitute_pair(corr, price_a, price_b)
                if pair_type not in ('DIRECT_SUBSTITUTE', 'CANNIBALIZE'):
                    continue

                lift = co_purchases * sample_orders / (count_a * count_b)
                corr_values.append(float(corr))
                lift_values.append(float(lift))

                pairs_records.append(
                    {
                        'item_a': item_a,
                        'item_b': item_b,
                        'category1': category1,
                        'category2': category2,
                        'category3': cat3,
                        'pair_scope': pair_scope,
                        'co_purchases': co_purchases,
                        'lift': round(float(lift), 4),
                        'pearson_corr': round(float(corr), 4),
                        'pair_type': pair_type,
                    }
                )

                if pair_type == 'DIRECT_SUBSTITUTE':
                    sub_map[item_a]['substitutes'].add(item_b)
                    sub_map[item_b]['substitutes'].add(item_a)
                else:
                    sub_map[item_a]['cannibals'].add(item_b)
                    sub_map[item_b]['cannibals'].add(item_a)

        print(f'[AGENT 3] Analyzing {len(cat2_groups):,} CATEGORY2 groups for complements...', flush=True)
        for cat2 in sorted(cat2_groups):
            cat_lines = con.execute(
                """
                SELECT vl.ORDERID, vl.ITEMID, vo.order_month, vl.CATEGORY1, vl.CATEGORY2, vl.CATEGORY3
                FROM valid_eligible_lines vl
                JOIN valid_orders vo USING (ORDERID)
                WHERE vl.CATEGORY2 = ?
                """,
                [cat2],
            ).df()

            if cat_lines.empty:
                continue

            cat_lines = cat_lines.drop_duplicates().sort_values(['ORDERID', 'ITEMID']).reset_index(drop=True)
            cat_lines['ITEMID'] = cat_lines['ITEMID'].astype(str).str.strip()
            item_levels = sorted(cat_lines['ITEMID'].unique().tolist())
            if len(item_levels) < 2:
                continue

            monthly_counts = (
                cat_lines.groupby(['order_month', 'ITEMID']).size().unstack(fill_value=0)
                .reindex(index=all_months, columns=item_levels, fill_value=0)
            )
            series_map = {
                item_id: _standardize_series(monthly_counts[item_id].to_numpy(dtype=float))
                for item_id in item_levels
            }
            order_codes, _ = pd.factorize(cat_lines['ORDERID'], sort=False)
            item_codes = pd.Categorical(cat_lines['ITEMID'], categories=item_levels).codes
            matrix = coo_matrix(
                (
                    np.ones(len(cat_lines), dtype=np.int8),
                    (order_codes, item_codes),
                ),
                shape=(len(np.unique(order_codes)), len(item_levels)),
            ).tocsr()

            item_counts = np.asarray(matrix.getnnz(axis=0)).ravel().astype(int)
            cooc = (matrix.T @ matrix).tocoo()
            cooc_lookup = {}
            for row_idx, col_idx, value in zip(cooc.row, cooc.col, cooc.data):
                if row_idx == col_idx or value <= 0:
                    continue
                key = (row_idx, col_idx) if row_idx < col_idx else (col_idx, row_idx)
                cooc_lookup[key] = int(value)

            if not cooc_lookup:
                continue

            meta = cat_lines.drop_duplicates('ITEMID').set_index('ITEMID')[['CATEGORY1', 'CATEGORY3']].to_dict('index')

            for (i, j), co_purchases in cooc_lookup.items():
                if co_purchases < MIN_COMPLEMENT_SUPPORT:
                    continue

                item_a = str(item_levels[i])
                item_b = str(item_levels[j])
                count_a = int(item_counts[i])
                count_b = int(item_counts[j])

                if count_a <= 0 or count_b <= 0:
                    continue

                lift = co_purchases * sample_orders / (count_a * count_b)
                if lift <= LIFT_THRESHOLD:
                    continue

                corr = _pearson_from_standardized_series(series_map.get(item_a), series_map.get(item_b))
                if corr <= CORR_THRESHOLD:
                    continue

                corr_values.append(float(corr))
                lift_values.append(float(lift))

                category3_a = str(meta.get(item_a, {}).get('CATEGORY3', ''))
                category3_b = str(meta.get(item_b, {}).get('CATEGORY3', ''))

                # CATEGORY2 loop keeps only cross-CATEGORY3 links to avoid duplicating WITHIN_CATEGORY3 pairs.
                if category3_a == category3_b:
                    continue

                category1_a = str(meta.get(item_a, {}).get('CATEGORY1', category1_map.get(item_a, '')))
                category1_b = str(meta.get(item_b, {}).get('CATEGORY1', category1_map.get(item_b, '')))
                pair_type = 'COMPLEMENT' if _is_complement_pair(corr, lift) else 'INDEPENDENT'
                category1 = category1_a if category1_a == category1_b else f'{category1_a} <> {category1_b}'
                pair_scope = f'WITHIN_CATEGORY2_CROSS_CATEGORY3_n{len(item_levels)}'

                pairs_records.append(
                    {
                        'item_a': item_a,
                        'item_b': item_b,
                        'category1': category1,
                        'category2': cat2,
                        'category3': _pair_category3_label(category3_a, category3_b),
                        'pair_scope': pair_scope,
                        'co_purchases': co_purchases,
                        'lift': round(float(lift), 4),
                        'pearson_corr': round(float(corr), 4),
                        'pair_type': pair_type,
                    }
                )

                if pair_type == 'COMPLEMENT':
                    sub_map[item_a]['complements'].add(item_b)
                    sub_map[item_b]['complements'].add(item_a)

        pairs_df = pd.DataFrame(pairs_records, columns=PAIR_COLUMNS)
        if not pairs_df.empty:
            pairs_df = pairs_df.sort_values(['pair_type', 'category2', 'category3', 'item_a', 'item_b']).reset_index(drop=True)
        else:
            log['status'] = 'WARNING'
            log['warnings'].append('No item pairs passed the output filters for substitutes or complements.')

        hist_paths = _save_histograms(corr_values, lift_values)

        pairs_path = os.path.join(OUT, 'item_pairs.csv')
        pairs_df.to_csv(pairs_path, index=False)

        sub_map_json = {
            item_id: {
                'substitutes': sorted(values['substitutes']),
                'complements': sorted(values['complements']),
                'cannibals': sorted(values['cannibals']),
            }
            for item_id, values in sorted(sub_map.items())
        }
        sub_map_path = os.path.join(OUT, 'substitute_map.json')
        with open(sub_map_path, 'w', encoding='utf-8') as f:
            json.dump(sub_map_json, f, indent=2, ensure_ascii=False)

        log['output_files'] = hist_paths + [pairs_path, sub_map_path]
        if log['status'] == 'SUCCESS' and log['warnings']:
            log['status'] = 'WARNING'

        print(f'[AGENT 3] Found {len(pairs_df):,} pairs', flush=True)
        print(f'[AGENT 3] Done in {round(time.time() - t0, 1)}s', flush=True)
        _save_log()

    except Exception as exc:
        log['status'] = 'FAILED'
        log['warnings'].append(f'{type(exc).__name__}: {exc}')
        _save_log()
        raise


if __name__ == '__main__':
    main()
