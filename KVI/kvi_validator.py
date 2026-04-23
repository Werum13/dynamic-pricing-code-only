"""
AGENT 1: DATA_VALIDATOR
========================
Проверяет качество данных перед анализом.

INPUT:
    - data/Orders.csv, data/Order_Details.csv, data/Categories_ENG.csv
    - data/LSTCSV* или data/elasticity.csv (если есть)

OUTPUT:
    - output/kvi/validation_report.json
    - output/kvi/clean_flags.csv
    - output/kvi/elasticity_by_itemid.csv

CRITICAL: если elasticity_coverage_pct < 50% — пайплайн останавливается.
"""

from pathlib import Path

import duckdb, json, os, sys, time

from elasticity_utils import standardize_elasticity_source

BASE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(BASE)
DATA = os.path.join(WORKSPACE_ROOT, 'data')
OUT  = os.path.join(WORKSPACE_ROOT, 'output', 'kvi')
os.makedirs(OUT, exist_ok=True)

log = {"agent": "DATA_VALIDATOR", "status": "SUCCESS",
       "rows_processed": 0, "warnings": [], "output_files": []}

def main():
    print("[AGENT 1] DATA_VALIDATOR starting...", flush=True)
    t0  = time.time()
    con = duckdb.connect()

    # Order_Details использует европейский формат десятичных (запятая внутри кавычек)
    con.execute(f"""
        CREATE VIEW order_details_raw AS
        SELECT ORDERID, ORDERDETAILID, ITEMID, ITEMCODE,
               TRY_CAST(REPLACE(AMOUNT,     ',', '.') AS DOUBLE) AS AMOUNT,
               TRY_CAST(REPLACE(UNITPRICE,  ',', '.') AS DOUBLE) AS UNITPRICE,
               TRY_CAST(REPLACE(TOTALPRICE, ',', '.') AS DOUBLE) AS TOTALPRICE
        FROM read_csv_auto('{DATA}/Order_Details.csv', header=true, delim=',', all_varchar=true)
    """)
    con.execute(f"""
        CREATE VIEW orders AS
        SELECT ORDERID, BRANCH_ID, DATE_, USERID, NAMESURNAME,
               TRY_CAST(REPLACE(TOTALBASKET, ',', '.') AS DOUBLE) AS TOTALBASKET
        FROM read_csv_auto('{DATA}/Orders.csv', header=true, delim=',', all_varchar=true)
    """)
    con.execute(f"""
        CREATE VIEW categories AS
        SELECT * FROM read_csv_auto('{DATA}/Categories_ENG.csv', header=true, delim=';')
    """)

    # 1. Временной диапазон и счётчики
    r = con.execute("""
        SELECT MIN(DATE_)::VARCHAR, MAX(DATE_)::VARCHAR, COUNT(*) FROM orders
    """).fetchone()
    min_date, max_date, total_orders = r[0], r[1], int(r[2])

    total_rows  = con.execute("SELECT COUNT(*) FROM order_details_raw").fetchone()[0]
    total_items = con.execute("SELECT COUNT(DISTINCT ITEMID) FROM order_details_raw").fetchone()[0]
    log["rows_processed"] = int(total_rows)
    print(f"[AGENT 1] Orders={total_orders:,} | Detail rows={total_rows:,} | Items={total_items:,}", flush=True)

    # 2. Стандартизация elasticity -> ITEMID
    elasticity_itemid_df, elasticity_meta = standardize_elasticity_source(
        Path(DATA),
        Path(DATA) / 'Categories_ENG.csv',
        Path(OUT),
    )

    if elasticity_itemid_df is not None:
        e_items = len(elasticity_itemid_df)
        cat_items = con.execute("SELECT COUNT(DISTINCT ITEMID) FROM categories").fetchone()[0]
        elasticity_coverage_pct = round(e_items / cat_items * 100, 2) if cat_items else 0.0

        print(
            f"[AGENT 1] Elasticity source: {elasticity_meta.get('source_path')} "
            f"({elasticity_meta.get('source_kind')}, {elasticity_meta.get('mapping_method')})",
            flush=True,
        )
        print(f"[AGENT 1] Elasticity coverage after ITEMCODE→ITEMID merge: {elasticity_coverage_pct}%", flush=True)

        if elasticity_meta.get('unmatched_rows', 0) > 0:
            log["warnings"].append(
                f"{elasticity_meta['unmatched_rows']:,} elasticity rows could not be mapped from item code to ITEMID."
            )

        if elasticity_meta.get('duplicate_itemids', 0) > 0:
            log["warnings"].append(
                f"{elasticity_meta['duplicate_itemids']:,} ITEMID values have multiple elasticity rows; first value kept."
            )

        if elasticity_coverage_pct < 50:
            msg = f"CRITICAL: elasticity_coverage_pct={elasticity_coverage_pct}% < 50%. Pipeline halted."
            log["status"] = "CRITICAL"
            log["warnings"].append(msg)
            print(f"[AGENT 1] {msg}", flush=True)
            _save_log(log); sys.exit(2)
    else:
        elasticity_coverage_pct = "N/A"
        log["warnings"].append("elasticity source not found — Agent 4 will use proxy elasticity based on price_cv.")
        print("[AGENT 1] WARNING: elasticity source not found.", flush=True)

    # 3. Пропуски / нулевые цены
    bad_price = con.execute("""
        SELECT COUNT(*) FROM order_details_raw WHERE UNITPRICE IS NULL OR UNITPRICE <= 0
    """).fetchone()[0]
    missing_price_pct = round(bad_price / total_rows * 100, 4) if total_rows else 0

    bad_basket = con.execute("""
        SELECT COUNT(*) FROM orders WHERE TOTALBASKET IS NULL OR TOTALBASKET <= 0
    """).fetchone()[0]
    if bad_basket > 0:
        log["warnings"].append(f"{bad_basket:,} orders have NULL/zero TOTALBASKET.")

    # 4. Дубликаты ORDERDETAILID
    dup_count = con.execute("""
        SELECT COUNT(*) FROM (
            SELECT ORDERDETAILID FROM order_details_raw
            GROUP BY ORDERDETAILID HAVING COUNT(*) > 1
        )
    """).fetchone()[0]
    if dup_count > 0:
        log["warnings"].append(f"{dup_count:,} duplicate ORDERDETAILID groups.")

    # 5. Аномальные цены (> mean + 3σ) per CATEGORY3
    print("[AGENT 1] Computing price anomalies...", flush=True)
    anomaly_pct_raw = con.execute("""
        WITH stats AS (
            SELECT c.CATEGORY3,
                   AVG(od.UNITPRICE)    AS mu,
                   STDDEV(od.UNITPRICE) AS sigma
            FROM order_details_raw od
            JOIN categories c USING (ITEMID)
            WHERE od.UNITPRICE > 0
            GROUP BY c.CATEGORY3
        )
        SELECT ROUND(100.0 *
            SUM(CASE WHEN od.UNITPRICE > s.mu + 3*s.sigma THEN 1 ELSE 0 END)
            / NULLIF(COUNT(*), 0), 4)
        FROM order_details_raw od
        JOIN categories c USING (ITEMID)
        JOIN stats s ON s.CATEGORY3 = c.CATEGORY3
        WHERE od.UNITPRICE > 0
    """).fetchone()[0]
    price_anomaly_pct = float(anomaly_pct_raw or 0)
    if price_anomaly_pct > 5:
        log["warnings"].append(f"High price anomaly rate: {price_anomaly_pct}%")

    # 6. clean_flags.csv
    print("[AGENT 1] Building clean_flags.csv...", flush=True)
    flags_path = os.path.join(OUT, 'clean_flags.csv')
    con.execute(f"""
        COPY (
            SELECT od.ORDERDETAILID,
                   CASE WHEN od.UNITPRICE  IS NULL OR od.UNITPRICE  <= 0 THEN 0
                        WHEN od.AMOUNT     IS NULL OR od.AMOUNT     <= 0 THEN 0
                        WHEN od.TOTALPRICE IS NULL OR od.TOTALPRICE <= 0 THEN 0
                        WHEN o.TOTALBASKET IS NULL OR o.TOTALBASKET <= 0 THEN 0
                        ELSE 1 END AS is_valid
            FROM order_details_raw od
            LEFT JOIN orders o USING (ORDERID)
        ) TO '{flags_path}' (HEADER, DELIMITER ',')
    """)
    valid_count = con.execute(f"""
        SELECT COUNT(*) FROM read_csv_auto('{flags_path}', header=true) WHERE is_valid = 1
    """).fetchone()[0]
    print(f"[AGENT 1] Valid rows: {valid_count:,} / {total_rows:,}", flush=True)

    # 7. validation_report.json
    report = {
        "date_range":                   {"min": str(min_date), "max": str(max_date)},
        "total_orders":                 total_orders,
        "total_items":                  int(total_items),
        "total_detail_rows":            int(total_rows),
        "valid_detail_rows":            int(valid_count),
        "elasticity_coverage_pct":      elasticity_coverage_pct,
        "elasticity_source":            elasticity_meta.get('source_path') if elasticity_itemid_df is not None else None,
        "elasticity_source_kind":       elasticity_meta.get('source_kind') if elasticity_itemid_df is not None else None,
        "elasticity_mapping_method":    elasticity_meta.get('mapping_method') if elasticity_itemid_df is not None else None,
        "elasticity_itemid_rows":       int(elasticity_meta.get('distinct_itemids', 0)) if elasticity_itemid_df is not None else 0,
        "elasticity_unmatched_rows":    int(elasticity_meta.get('unmatched_rows', 0)) if elasticity_itemid_df is not None else 0,
        "elasticity_duplicate_itemids": int(elasticity_meta.get('duplicate_itemids', 0)) if elasticity_itemid_df is not None else 0,
        "missing_price_rows_pct":       missing_price_pct,
        "price_anomaly_pct":            price_anomaly_pct,
        "duplicate_orderdetail_groups": int(dup_count),
        "zero_totalbasket_orders":      int(bad_basket),
        "elapsed_sec":                  round(time.time() - t0, 1),
        "warnings":                     log["warnings"]
    }
    report_path = os.path.join(OUT, 'validation_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    log["output_files"] = [flags_path, report_path]
    if elasticity_meta.get('item_level_path'):
        log["output_files"].append(elasticity_meta['item_level_path'])
    print(f"[AGENT 1] Done in {report['elapsed_sec']}s. Status: {log['status']}", flush=True)
    _save_log(log)
    return report

def _save_log(log):
    with open(os.path.join(OUT, 'agent1_log.json'), 'w') as f:
        json.dump(log, f, indent=2)

if __name__ == '__main__':
    main()
