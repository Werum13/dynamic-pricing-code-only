"""
AGENT 2: FEATURE_ENGINEER
===========================
Рассчитывает поведенческие метрики по каждому товару.

INPUT:
  - data/Order_Details.csv, data/Orders.csv, data/Categories_ENG.csv
    - output/kvi/clean_flags.csv  (от Agent 1)

OUTPUT:
    - output/kvi/behavioral_features.csv
    Колонки: ITEMID, ITEMNAME, CATEGORY1..4, BRAND,
             order_count, unique_buyers, penetration_rate,
             avg_price, price_cv, avg_basket_share,
             freq_per_buyer, avg_amount, repeat_buyer_rate,
             q1_orders, q2_orders, q3_orders, q4_orders
"""

import duckdb, json, os, time

BASE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(BASE)
DATA = os.path.join(WORKSPACE_ROOT, 'data')
OUT  = os.path.join(WORKSPACE_ROOT, 'output', 'kvi')
os.makedirs(OUT, exist_ok=True)

log = {"agent": "FEATURE_ENGINEER", "status": "SUCCESS",
       "rows_processed": 0, "warnings": [], "output_files": []}

def main():
    print("[AGENT 2] FEATURE_ENGINEER starting...", flush=True)
    t0  = time.time()
    con = duckdb.connect()

    flags_path = os.path.join(OUT, 'clean_flags.csv')
    if not os.path.exists(flags_path):
        raise FileNotFoundError("clean_flags.csv not found — run Agent 1 first.")

    # Регистрируем источники
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
    con.execute(f"CREATE VIEW categories AS SELECT * FROM read_csv_auto('{DATA}/Categories_ENG.csv',  header=true, delim=';')")
    con.execute(f"CREATE VIEW flags      AS SELECT * FROM read_csv_auto('{flags_path}',               header=true)")

    # Только валидные строки
    con.execute("""
        CREATE VIEW od AS
        SELECT od.*
        FROM order_details_raw od
        JOIN flags f USING (ORDERDETAILID)
        WHERE f.is_valid = 1
    """)

    # Количество уникальных покупателей (знаменатель для penetration_rate)
    total_buyers = con.execute("""
        SELECT COUNT(DISTINCT USERID) FROM orders
    """).fetchone()[0]
    total_orders = con.execute("SELECT COUNT(DISTINCT ORDERID) FROM od").fetchone()[0]
    print(f"[AGENT 2] Valid distinct orders={total_orders:,}, total buyers={total_buyers:,}", flush=True)

    # Квартальные продажи
    print("[AGENT 2] Computing quarterly counts...", flush=True)
    con.execute("""
        CREATE TEMP TABLE quarterly AS
        SELECT od.ITEMID,
               QUARTER(STRPTIME(o.DATE_, '%Y-%m-%d %H:%M:%S')) AS q,
               COUNT(DISTINCT od.ORDERID) AS cnt
        FROM od
        JOIN orders o USING (ORDERID)
        GROUP BY od.ITEMID, q
    """)

    # Покупатели с 2+ покупками на товар
    print("[AGENT 2] Computing repeat buyers...", flush=True)
    con.execute("""
        CREATE TEMP TABLE buyer_counts AS
        SELECT od.ITEMID, o.USERID, COUNT(DISTINCT od.ORDERID) AS purchases
        FROM od
        JOIN orders o USING (ORDERID)
        GROUP BY od.ITEMID, o.USERID
    """)

    # Основные метрики
    print("[AGENT 2] Aggregating behavioral features...", flush=True)
    out_path = os.path.join(OUT, 'behavioral_features.csv')
    con.execute(f"""
        COPY (
            WITH base AS (
                SELECT
                    od.ITEMID,
                    COUNT(DISTINCT od.ORDERID)                             AS order_count,
                    COUNT(DISTINCT o.USERID)                               AS unique_buyers,
                    AVG(od.UNITPRICE)                                      AS avg_price,
                    STDDEV(od.UNITPRICE) / NULLIF(AVG(od.UNITPRICE), 0)    AS price_cv,
                    AVG(od.TOTALPRICE / NULLIF(o.TOTALBASKET, 0))          AS avg_basket_share,
                    AVG(od.AMOUNT)                                         AS avg_amount
                FROM od
                JOIN orders o USING (ORDERID)
                GROUP BY od.ITEMID
            ),
            repeat_b AS (
                SELECT ITEMID,
                       SUM(CASE WHEN purchases >= 2 THEN 1 ELSE 0 END)
                       / NULLIF(COUNT(*), 0) AS repeat_buyer_rate
                FROM buyer_counts
                GROUP BY ITEMID
            ),
            q_pivot AS (
                SELECT ITEMID,
                       SUM(CASE WHEN q=1 THEN cnt ELSE 0 END) AS q1_orders,
                       SUM(CASE WHEN q=2 THEN cnt ELSE 0 END) AS q2_orders,
                       SUM(CASE WHEN q=3 THEN cnt ELSE 0 END) AS q3_orders,
                       SUM(CASE WHEN q=4 THEN cnt ELSE 0 END) AS q4_orders
                FROM quarterly
                GROUP BY ITEMID
            )
            SELECT
                b.ITEMID,
                c.ITEMNAME,
                c.CATEGORY1, c.CATEGORY2, c.CATEGORY3, c.CATEGORY4,
                c.BRAND,
                b.order_count,
                b.unique_buyers,
                b.unique_buyers / {total_buyers}.0  AS penetration_rate,
                b.avg_price,
                b.price_cv,
                b.avg_basket_share,
                b.order_count / NULLIF(b.unique_buyers, 0) AS freq_per_buyer,
                b.avg_amount,
                COALESCE(r.repeat_buyer_rate, 0)    AS repeat_buyer_rate,
                COALESCE(q.q1_orders, 0)            AS q1_orders,
                COALESCE(q.q2_orders, 0)            AS q2_orders,
                COALESCE(q.q3_orders, 0)            AS q3_orders,
                COALESCE(q.q4_orders, 0)            AS q4_orders
            FROM base b
            LEFT JOIN categories  c USING (ITEMID)
            LEFT JOIN repeat_b    r USING (ITEMID)
            LEFT JOIN q_pivot     q USING (ITEMID)
            ORDER BY b.order_count DESC
        ) TO '{out_path}' (HEADER, DELIMITER ',')
    """)

    row_count = con.execute(f"SELECT COUNT(*) FROM read_csv_auto('{out_path}', header=true)").fetchone()[0]
    log["rows_processed"] = int(row_count)
    log["output_files"]   = [out_path]
    print(f"[AGENT 2] Done: {row_count:,} items in {round(time.time()-t0,1)}s", flush=True)

    with open(os.path.join(OUT, 'agent2_log.json'), 'w') as f:
        json.dump(log, f, indent=2)

if __name__ == '__main__':
    main()
