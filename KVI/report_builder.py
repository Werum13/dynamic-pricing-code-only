"""
AGENT 5: REPORT_BUILDER
========================
Формирует итоговый аналитический отчёт.

INPUT:
    - output/kvi/validation_report.json
    - output/kvi/kvi_candidates.csv
    - output/kvi/kvi_scores_full.csv
    - output/kvi/item_pairs.csv

OUTPUT:
    - output/kvi/kvi_report.html
    - output/kvi/kvi_final_list.xlsx
"""

import pandas as pd
import numpy as np
import json
import os
import base64
import time

os.environ.setdefault(
    'MPLCONFIGDIR',
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output', 'kvi', '.mplconfig')
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from io import BytesIO

BASE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(BASE)
OUT  = os.path.join(WORKSPACE_ROOT, 'output', 'kvi')
os.makedirs(OUT, exist_ok=True)

log = {"agent": "REPORT_BUILDER", "status": "SUCCESS",
       "rows_processed": 0, "warnings": [], "output_files": []}

# ── Вспомогательная функция: сохранить plot как base64 PNG ────────────────────
def fig_to_b64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def main():
    print("[AGENT 5] REPORT_BUILDER starting...", flush=True)
    t0 = time.time()

    # Загрузка данных
    with open(os.path.join(OUT, 'validation_report.json'), encoding='utf-8') as f:
        vr = json.load(f)

    cand_df  = pd.read_csv(os.path.join(OUT, 'kvi_candidates.csv'),  dtype={'ITEMID': str})
    full_df  = pd.read_csv(os.path.join(OUT, 'kvi_scores_full.csv'), dtype={'ITEMID': str})
    pairs_df = pd.read_csv(os.path.join(OUT, 'item_pairs.csv'),       dtype={'item_a': str, 'item_b': str})

    cand_df = cand_df.merge(full_df[['ITEMID', 'CATEGORY1']], on='ITEMID', how='left')

    cand_df  = cand_df.sort_values('kvi_score_final', ascending=False)
    top20    = cand_df.head(20)

    # ── Scatter-plot ──────────────────────────────────────────────────────────
    full_df['is_kvi'] = full_df['ITEMID'].isin(cand_df['ITEMID'])
    full_df['abs_elasticity'] = full_df['elasticity'].abs()
    full_df['basket_size']    = (full_df['avg_basket_share'].fillna(0).clip(0, 1) * 2000 + 30)

    fig, ax = plt.subplots(figsize=(12, 8))
    for is_kvi, grp in full_df.groupby('is_kvi'):
        color = '#e74c3c' if is_kvi else '#95a5a6'
        label = 'KVI-товар' if is_kvi else 'Остальные'
        ax.scatter(grp['penetration_rate'], grp['abs_elasticity'],
                   s=grp['basket_size'], c=color, alpha=0.55, label=label, edgecolors='white', linewidths=0.3)

    # Аннотации топ-20
    for _, row in top20.iterrows():
        match = full_df[full_df['ITEMID'] == row['ITEMID']]
        if match.empty:
            continue
        r = match.iloc[0]
        name = str(r.get('ITEMNAME', row['ITEMID']))[:25]
        ax.annotate(name,
                    xy=(r['penetration_rate'], r['abs_elasticity']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=6.5, color='#2c3e50',
                    arrowprops=dict(arrowstyle='-', color='#bdc3c7', lw=0.5))

    ax.set_xlabel('Penetration Rate (доля уникальных покупателей)', fontsize=11)
    ax.set_ylabel('|Elasticity| (чувствительность к цене)', fontsize=11)
    ax.set_title('KVI-карта товаров\n(размер точки = доля в корзине)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)
    scatter_b64 = fig_to_b64(fig)
    plt.close(fig)

    # ── Сводная таблица по категориям ─────────────────────────────────────────
    cat_table = (cand_df.groupby(['CATEGORY1', 'CATEGORY2'])
                 .apply(lambda g: g[['ITEMNAME', 'kvi_score_final',
                                     'elasticity', 'penetration_rate']].head(3))
                 .reset_index())

    # ── Рекомендации для каждого кандидата ────────────────────────────────────
    def get_subs_compl(itemid):
        subs_rows  = pairs_df[(pairs_df['item_a']==itemid) | (pairs_df['item_b']==itemid)]
        subs  = subs_rows[subs_rows['pair_type']=='DIRECT_SUBSTITUTE']
        compl = subs_rows[subs_rows['pair_type']=='COMPLEMENT']
        cann  = subs_rows[subs_rows['pair_type']=='CANNIBALIZE']
        return (
            len(subs), len(compl), len(cann)
        )

    def recommend(row, n_subs, n_compl, n_cann):
        if n_subs >= 3:
            return "🔍 Мониторить цены конкурентов"
        if n_cann >= 2:
            return "⚠️ Осторожно с промо"
        return "⚓ Якорная позиция"

    cand_df = cand_df.copy()
    cand_df['n_subs']  = 0
    cand_df['n_compl'] = 0
    cand_df['n_cann']  = 0
    for idx, row in cand_df.iterrows():
        ns, nc, ncan = get_subs_compl(row['ITEMID'])
        cand_df.at[idx, 'n_subs']  = ns
        cand_df.at[idx, 'n_compl'] = nc
        cand_df.at[idx, 'n_cann']  = ncan
    cand_df['recommendation'] = cand_df.apply(
        lambda r: recommend(r, r['n_subs'], r['n_compl'], r['n_cann']), axis=1)

    # ── HTML ──────────────────────────────────────────────────────────────────
    warnings_html = ''.join(
        f'<li class="warn">⚠️ {w}</li>' for w in vr.get('warnings', [])
    ) or '<li>Предупреждений нет</li>'

    top20_rows = ''.join(
        f"""<tr>
            <td>{i+1}</td>
            <td>{r.get('ITEMNAME', r['ITEMID'])}</td>
            <td>{r.get('CATEGORY2','—')}</td>
            <td>{r.get('BRAND','—')}</td>
            <td>{r['kvi_score_final']:.3f}</td>
            <td>{r['elasticity']:.3f}</td>
            <td>{r['penetration_rate']:.4f}</td>
            <td>{r['recommendation']}</td>
        </tr>"""
        for i, (_, r) in enumerate(cand_df.head(20).iterrows())
    )

    # Сводная таблица CATEGORY1 → CATEGORY2 → KVI
    cat_html = ''
    for cat1, grp1 in cand_df.groupby('CATEGORY1'):
        cat_html += f'<h3>🏷️ {cat1}</h3><ul>'
        for cat2, grp2 in grp1.groupby('CATEGORY2'):
            items_str = ', '.join(
                f'<b>{r.get("ITEMNAME", r["ITEMID"])}</b> ({r["kvi_score_final"]:.3f})'
                for _, r in grp2.head(3).iterrows()
            )
            cat_html += f'<li><i>{cat2}</i>: {items_str}</li>'
        cat_html += '</ul>'

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>KVI-анализ — Отчёт</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; color: #2c3e50; background: #f8f9fa; }}
  h1   {{ color: #2980b9; border-bottom: 3px solid #2980b9; padding-bottom: 10px; }}
  h2   {{ color: #34495e; border-left: 5px solid #2980b9; padding-left: 12px; margin-top: 40px; }}
  h3   {{ color: #7f8c8d; }}
  table{{ border-collapse: collapse; width: 100%; font-size: 13px; background: white; }}
  th   {{ background: #2980b9; color: white; padding: 10px 8px; text-align: left; }}
  td   {{ padding: 8px; border-bottom: 1px solid #ecf0f1; }}
  tr:hover td {{ background: #eaf4fb; }}
  .kpi-box  {{ display:inline-block; background:white; border-radius:8px; padding:16px 24px;
               margin:8px; box-shadow:0 2px 6px rgba(0,0,0,.1); text-align:center; }}
  .kpi-val  {{ font-size:2em; font-weight:bold; color:#2980b9; }}
  .kpi-lbl  {{ font-size:.85em; color:#7f8c8d; }}
  ul.warn   {{ color: #e74c3c; }}
  li.warn   {{ margin: 4px 0; }}
  .glossary {{ background: white; padding: 20px; border-radius: 8px; font-size: 13px; }}
  .glossary dt {{ font-weight: bold; color: #2980b9; margin-top: 8px; }}
  .glossary dd {{ margin-left: 16px; color: #555; }}
  img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,.12); }}
</style>
</head>
<body>

<h1>📊 KVI-анализ розничной сети — Итоговый отчёт</h1>
<p style="color:#7f8c8d">Сгенерировано автоматически | Данные: {vr['date_range']['min']} — {vr['date_range']['max']}</p>

<h2>1. Ключевые показатели</h2>
<div>
  <div class="kpi-box"><div class="kpi-val">{vr['total_orders']:,}</div><div class="kpi-lbl">Заказов</div></div>
  <div class="kpi-box"><div class="kpi-val">{vr['total_items']:,}</div><div class="kpi-lbl">Товаров</div></div>
  <div class="kpi-box"><div class="kpi-val">{len(cand_df):,}</div><div class="kpi-lbl">KVI-кандидатов</div></div>
  <div class="kpi-box"><div class="kpi-val">{cand_df['CATEGORY2'].nunique()}</div><div class="kpi-lbl">Категорий (L2)</div></div>
  <div class="kpi-box"><div class="kpi-val">{vr['missing_price_rows_pct']}%</div><div class="kpi-lbl">Пропусков в цене</div></div>
</div>

<h2>2. Предупреждения по данным</h2>
<ul>{warnings_html}</ul>

<h2>3. KVI-карта (scatter plot)</h2>
<img src="data:image/png;base64,{scatter_b64}" alt="KVI scatter plot"/>

<h2>4. Топ-20 KVI-товаров</h2>
<table>
  <thead><tr>
    <th>#</th><th>Товар</th><th>Категория L2</th><th>Бренд</th>
    <th>KVI-score</th><th>Эластичность</th><th>Penetration</th><th>Рекомендация</th>
  </tr></thead>
  <tbody>{top20_rows}</tbody>
</table>

<h2>5. KVI по категориям</h2>
{cat_html}

<h2>6. Глоссарий</h2>
<div class="glossary">
<dl>
  <dt>KVI (Key Value Item)</dt>
  <dd>Товар, по цене которого покупатели оценивают ценовой уровень магазина.</dd>
  <dt>KVI-score</dt>
  <dd>Взвешенная сумма нормализованных метрик: эластичность (30%), проникновение (25%), частота (20%), доля в корзине (15%), вариация цены (5%), лояльность (5%).</dd>
  <dt>Elasticity (ценовая эластичность)</dt>
  <dd>Изменение объёма продаж при изменении цены на 1%. Отрицательное число: чем больше абсолютное значение, тем чувствительнее покупатели к цене.</dd>
  <dt>Penetration Rate</dt>
  <dd>Доля уникальных покупателей, купивших данный товар хотя бы раз.</dd>
  <dt>Basket Share</dt>
  <dd>Средняя доля стоимости товара в общей сумме чека.</dd>
  <dt>DIRECT_SUBSTITUTE</dt>
  <dd>Взаимозаменяемый товар из той же категории с похожей ценой (±15%).</dd>
  <dt>COMPLEMENT</dt>
  <dd>Товар-дополнитель — часто покупается вместе с KVI.</dd>
  <dt>Якорная позиция</dt>
  <dd>Товар, формирующий ценовое восприятие магазина. Цена должна быть конкурентной.</dd>
  <dt>Seasonal penalty</dt>
  <dd>Понижающий коэффициент для товаров с высокой сезонностью (CV квартальных продаж > 0.5).</dd>
</dl>
</div>

</body>
</html>"""

    html_path = os.path.join(OUT, 'kvi_report.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"[AGENT 5] HTML report saved: {html_path}", flush=True)

    # ── Excel ─────────────────────────────────────────────────────────────────
    xlsx_path = os.path.join(OUT, 'kvi_final_list.xlsx')
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        cand_df.drop(columns=['n_subs','n_compl','n_cann'], errors='ignore').to_excel(
            writer, sheet_name='KVI Candidates', index=False)
        full_df.to_excel(writer, sheet_name='All Scores', index=False)
        pd.DataFrame([vr]).T.reset_index().rename(
            columns={'index': 'Metric', 0: 'Value'}).to_excel(
            writer, sheet_name='Validation Report', index=False)
    print(f"[AGENT 5] Excel saved: {xlsx_path}", flush=True)

    log["rows_processed"] = len(cand_df)
    log["output_files"]   = [html_path, xlsx_path]
    print(f"[AGENT 5] Done in {round(time.time()-t0,1)}s", flush=True)

    with open(os.path.join(OUT, 'agent5_log.json'), 'w') as f:
        json.dump(log, f, indent=2)

if __name__ == '__main__':
    main()
