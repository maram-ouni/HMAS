# ─────────────────────────────────────────────────────────
# audit_trace.py — TRACE STRUCTURÉE PAR TENTATIVE (V2.1, changement #1)
#
# Le CSV/JSON produit par MASCoordinator ne garde qu'UNE ligne par tâche
# (la tentative finale/meilleure — cf. _build_result/_build_error_result).
# La trace interne multi-tentatives du cycle BDI n'existait auparavant que
# dans les print() de stdout, ce qui prive gt_analyzer.py (regret, Nash) de
# toute donnée exploitable : ces fonctions groupent par task_id et exigent
# plusieurs lignes par tâche, que le CSV ne fournit jamais.
#
# Ce module ajoute — en complément, jamais en remplacement — un JSON par
# TENTATIVE, un fichier par task_id. Zéro effet sur le score core (accepted/
# quality/U_V restent calculés et publiés exactement comme avant) : purement
# additif et auditable.
# ─────────────────────────────────────────────────────────

from __future__ import annotations

import json
import os
import sys
import threading
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR

_TRACE_DIR = os.path.join(RESULTS_DIR, "traces")


class AuditTraceRecorder:
    """
    Trace JSONL append-only, un fichier par task_id, une ligne par tentative.

    `path` suit le même principe d'isolation que WorldModel/StrategyLearner :
    None (défaut) = répertoire partagé `results/traces/`, un `path` dédié
    isole un run comparatif (voir mas_experiment.py --isolated).
    """

    def __init__(self, path: Optional[str] = None):
        self._dir = path or _TRACE_DIR
        self._lock = threading.Lock()
        os.makedirs(self._dir, exist_ok=True)

    def record_attempt(self, task_id: str, record: dict) -> None:
        record = dict(record)
        record.setdefault("task_id", task_id)
        record.setdefault("ts", time.time())
        line = json.dumps(record, ensure_ascii=False, default=str)
        path = os.path.join(self._dir, f"{task_id}.jsonl")
        with self._lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
