"""
ORCHESTRATOR: KVI RETAIL ANALYSIS PIPELINE
============================================
Запускает 5 агентов в правильном порядке:

  [Agent 1: DATA_VALIDATOR]
         │
         ├── CRITICAL → стоп
         │
         ▼
  [Agent 2: FEATURE_ENGINEER] ──── параллельно ────  [Agent 3: SUBSTITUTE_DETECTOR]
         └──────────────────────┬───────────────────────────────┘
                                ▼
                      [Agent 4: KVI_SCORER]
                                │
                                ▼
                      [Agent 5: REPORT_BUILDER]

Использование:
    python3 kvi_orchestrator.py

Параметры (необязательно, по умолчанию всё как есть):
    --skip-agent1   пропустить валидацию (если clean_flags.csv уже есть)
    --skip-agent3   пропустить детектор субститутов
"""

import subprocess
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(BASE)
OUT  = os.path.join(WORKSPACE_ROOT, 'output', 'kvi')
os.makedirs(OUT, exist_ok=True)

HYPERPARAMETERS_PATH = os.path.join(BASE, 'hyperparameters.json')

SCRIPT_TO_AGENT_LOG = {
    'kvi_validator.py': 'agent1_log.json',
    'feature_engineer.py': 'agent2_log.json',
    'substitute_detector.py': 'agent3_log.json',
    'kvi_scorer.py': 'agent4_log.json',
    'report_builder.py': 'agent5_log.json',
}

DEFAULT_AGENT3_HYPERPARAMETERS = {
    "RANDOM_SEED": 42,
    "PRICE_SIMILARITY_THRESHOLD": 0.15,
    "MAX_ORDERS": 100_000,
    "MIN_ORDER_COUNT": 50,
    "MIN_COMPLEMENT_SUPPORT": 3,
    "SUBSTITUTE_CORR_THRESHOLD": -0.1,
    "CORR_THRESHOLD": 0.3,
    "LIFT_THRESHOLD": 1.5,
    "OUTPUT_CORR_FILTER": 0.1,
    "OUTPUT_LIFT_FILTER": 1.2,
}


def _load_agent3_hyperparameters() -> dict:
    params = DEFAULT_AGENT3_HYPERPARAMETERS.copy()
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


AGENT3_HYPERPARAMETERS = _load_agent3_hyperparameters()

# ── Параметры запуска ─────────────────────────────────────────────────────────
RUN_CONFIG = {
    "random_seed":  42,
    "agent3_hyperparameters": AGENT3_HYPERPARAMETERS,
    "min_order_count_substitute": AGENT3_HYPERPARAMETERS["MIN_ORDER_COUNT"],
    "price_similarity_threshold": AGENT3_HYPERPARAMETERS["PRICE_SIMILARITY_THRESHOLD"],
    "max_orders_for_correlation": AGENT3_HYPERPARAMETERS["MAX_ORDERS"],
    "kvi_weights": {
        "elasticity":    0.30,
        "penetration":   0.25,
        "frequency":     0.20,
        "basket_share":  0.15,
        "price_cv":      0.05,
        "repeat_buyer":  0.05,
    }
}

with open(os.path.join(OUT, 'run_config.json'), 'w') as f:
    json.dump(RUN_CONFIG, f, indent=2)


def run_agent(script_name: str, label: str) -> dict:
    """Запускает агента как subprocess. Возвращает статус и лог."""
    print(f"\n{'='*60}", flush=True)
    print(f"▶  Запуск {label}", flush=True)
    print(f"{'='*60}", flush=True)

    script_path = os.path.join(BASE, script_name)
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=False,   # выводим stdout/stderr в консоль в реальном времени
        text=True
    )
    elapsed = round(time.time() - t0, 1)

    # Определяем агентский лог по имени скрипта
    agent_log_name = SCRIPT_TO_AGENT_LOG.get(script_name)
    agent_log_path = os.path.join(OUT, agent_log_name) if agent_log_name else None

    status_info = {
        "agent":      label,
        "returncode": result.returncode,
        "elapsed_sec": elapsed,
        "log_file":    agent_log_path if os.path.exists(agent_log_path) else None
    }

    if result.returncode == 0:
        print(f"✅ {label} завершён за {elapsed}s", flush=True)
        status_info["status"] = "SUCCESS"
    elif result.returncode == 2:
        print(f"🚨 {label} — CRITICAL WARNING (код 2)", flush=True)
        status_info["status"] = "CRITICAL"
    else:
        print(f"❌ {label} завершился с ошибкой (код {result.returncode})", flush=True)
        status_info["status"] = "FAILED"

    return status_info


def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║        KVI RETAIL ANALYSIS PIPELINE — ORCHESTRATOR       ║
╚══════════════════════════════════════════════════════════╝
""", flush=True)

    pipeline_log = []
    t_start = time.time()

    # ── AGENT 1: DATA_VALIDATOR ───────────────────────────────────────────────
    skip_agent1 = '--skip-agent1' in sys.argv
    if skip_agent1 and os.path.exists(os.path.join(OUT, 'clean_flags.csv')):
        print("⏭  Agent 1 пропущен (clean_flags.csv уже существует)", flush=True)
        a1_status = {"agent": "DATA_VALIDATOR", "status": "SKIPPED"}
    else:
        a1_status = run_agent('kvi_validator.py', 'Agent 1: DATA_VALIDATOR')
        pipeline_log.append(a1_status)

        if a1_status["status"] == "CRITICAL":
            print("\n🚨 ПАЙПЛАЙН ОСТАНОВЛЕН: критическая ошибка в Agent 1.", flush=True)
            print("   Проверьте output/kvi/agent1_log.json для деталей.", flush=True)
            _save_pipeline_log(pipeline_log, t_start)
            sys.exit(2)

        if a1_status["status"] == "FAILED":
            print("\n❌ ПАЙПЛАЙН ОСТАНОВЛЕН: Agent 1 упал с ошибкой.", flush=True)
            _save_pipeline_log(pipeline_log, t_start)
            sys.exit(1)

    # ── AGENTS 2 & 3: параллельно ────────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print("▶  Запуск Agent 2 и Agent 3 параллельно...", flush=True)
    print(f"{'='*60}", flush=True)

    skip_agent3 = '--skip-agent3' in sys.argv

    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = {
            ex.submit(run_agent, 'feature_engineer.py',  'Agent 2: FEATURE_ENGINEER'):  'a2',
        }
        if not skip_agent3:
            futures[ex.submit(run_agent, 'substitute_detector.py', 'Agent 3: SUBSTITUTE_DETECTOR')] = 'a3'

        results = {}
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as e:
                results[key] = {"agent": key, "status": "FAILED", "error": str(e)}
            pipeline_log.append(results.get(key, {}))

    # Если Agent 3 пропущен — создаём пустые заглушки
    if skip_agent3:
        import pandas as pd
        pd.DataFrame(columns=['item_a','item_b','category2','category3','pair_scope','co_purchases','lift','pearson_corr','pair_type']
                    ).to_csv(os.path.join(OUT, 'item_pairs.csv'), index=False)
        with open(os.path.join(OUT, 'substitute_map.json'), 'w') as f:
            json.dump({}, f)
        print("⏭  Agent 3 пропущен — созданы пустые item_pairs.csv и substitute_map.json", flush=True)

    for key, res in results.items():
        if res.get("status") == "FAILED":
            print(f"\n⚠️  {res['agent']} упал, но пайплайн продолжается (некритично).", flush=True)

    # Проверяем что Agent 2 успешен (критично для Agent 4)
    if results.get('a2', {}).get('status') not in ('SUCCESS',):
        print("\n❌ Agent 2 не завершился успешно. Agent 4 требует behavioral_features.csv.", flush=True)
        _save_pipeline_log(pipeline_log, t_start)
        sys.exit(1)

    # ── AGENT 4: KVI_SCORER ───────────────────────────────────────────────────
    a4_status = run_agent('kvi_scorer.py', 'Agent 4: KVI_SCORER')
    pipeline_log.append(a4_status)

    if a4_status["status"] == "FAILED":
        print("\n❌ ПАЙПЛАЙН ОСТАНОВЛЕН: Agent 4 упал.", flush=True)
        _save_pipeline_log(pipeline_log, t_start)
        sys.exit(1)

    # ── AGENT 5: REPORT_BUILDER ───────────────────────────────────────────────
    a5_status = run_agent('report_builder.py', 'Agent 5: REPORT_BUILDER')
    pipeline_log.append(a5_status)

    # ── Итоговый отчёт оркестратора ───────────────────────────────────────────
    total_elapsed = round(time.time() - t_start, 1)
    _save_pipeline_log(pipeline_log, t_start)

    print(f"""
╔══════════════════════════════════════════════════════════╗
║                 ПАЙПЛАЙН ЗАВЕРШЁН ✅                      ║
╚══════════════════════════════════════════════════════════╝

⏱  Общее время: {total_elapsed}s

📁 Выходные файлы:
    • output/kvi/validation_report.json   — отчёт о качестве данных
    • output/kvi/clean_flags.csv          — флаги валидных строк
    • output/kvi/behavioral_features.csv  — метрики по товарам
    • output/kvi/item_pairs.csv           — пары субститутов/дополнителей
    • output/kvi/substitute_map.json      — карта замен
    • output/kvi/kvi_scores_full.csv      — все KVI-баллы
    • output/kvi/kvi_candidates.csv       — список KVI-кандидатов
    • output/kvi/kvi_report.html          — аналитический отчёт (открыть в браузере)
    • output/kvi/kvi_final_list.xlsx      — Excel для ценового отдела
    • output/kvi/pipeline_log.json        — сводный лог всех агентов
""", flush=True)

    # Краткое резюме из kvi_candidates
    try:
        import pandas as pd
        cand = pd.read_csv(os.path.join(OUT, 'kvi_candidates.csv'), dtype={'ITEMID': str})
        n_kvi   = len(cand)
        n_cat2  = cand['CATEGORY2'].nunique()
        top5    = cand.sort_values('kvi_score_final', ascending=False).head(5)
        print(f"📊 Найдено KVI-товаров: {n_kvi} в {n_cat2} категориях (L2)\n")
        print("🏆 Топ-5 KVI:")
        for _, r in top5.iterrows():
            name = r.get('ITEMNAME', r['ITEMID'])
            print(f"   {name[:40]:<40} | score={r['kvi_score_final']:.3f} | {r.get('CATEGORY2','')}")
    except Exception:
        pass


def _save_pipeline_log(pipeline_log, t_start):
    log = {
        "pipeline_status": "COMPLETED",
        "total_elapsed_sec": round(time.time() - t_start, 1),
        "agents": pipeline_log
    }
    with open(os.path.join(OUT, 'pipeline_log.json'), 'w') as f:
        json.dump(log, f, indent=2)


if __name__ == '__main__':
    main()
