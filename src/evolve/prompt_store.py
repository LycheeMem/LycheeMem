"""Prompt 版本持久化存储（SQLite 后端）。

三张表：
- prompt_versions: 所有 prompt 的版本历史与状态
- prompt_metrics:  按版本聚合的效果指标时间序列
- prompt_failure_cases: 失败案例归档，供 Optimizer 参考

额外表：
- evolve_events: 自进化过程事件（optimize / promote / probation / rollback 等）
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# ── 数据模型 ─────────────────────────────────────────────────────

@dataclass
class PromptVersion:
    prompt_name: str
    version: int
    prompt_text: str
    status: str  # active / candidate / archived / rolled_back
    created_at: str = ""
    reason: str = ""
    parent_version: int = 0
    eval_score: float | None = None
    eval_detail: str = ""
    version_id: str = ""

    def __post_init__(self):
        if not self.version_id:
            self.version_id = uuid.uuid4().hex[:12]
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class PromptMetricSnapshot:
    prompt_name: str
    version: int
    metric_name: str
    metric_value: float
    sample_count: int = 0
    recorded_at: str = ""
    detail: str = ""

    def __post_init__(self):
        if not self.recorded_at:
            self.recorded_at = datetime.now(timezone.utc).isoformat()


@dataclass
class PromptFailureCase:
    prompt_name: str
    version: int
    case_type: str  # retrieval_miss / encoding_loss / synthesis_drop / ...
    input_summary: str = ""
    expected: str = ""
    actual: str = ""
    diagnosis: str = ""
    recorded_at: str = ""
    case_id: str = ""

    def __post_init__(self):
        if not self.case_id:
            self.case_id = uuid.uuid4().hex[:12]
        if not self.recorded_at:
            self.recorded_at = datetime.now(timezone.utc).isoformat()


@dataclass
class EvolveEvent:
    """一次自进化过程事件（用于可观测性与前端时间线展示）。"""

    event_type: str  # optimize_attempt / candidate_created / promote_probation / probation_pass / probation_fail / rollback_manual / ...
    prompt_name: str = ""
    from_version: int | None = None
    to_version: int | None = None
    summary: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    event_id: int | None = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


# ── SQLite 存储 ──────────────────────────────────────────────────

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS prompt_versions (
    version_id      TEXT PRIMARY KEY,
    prompt_name     TEXT NOT NULL,
    version         INTEGER NOT NULL,
    prompt_text     TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'active',
    created_at      TEXT NOT NULL,
    reason          TEXT DEFAULT '',
    parent_version  INTEGER DEFAULT 0,
    eval_score      REAL,
    eval_detail     TEXT DEFAULT ''
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_pv_name_version
    ON prompt_versions(prompt_name, version);
CREATE INDEX IF NOT EXISTS idx_pv_name_status
    ON prompt_versions(prompt_name, status);

CREATE TABLE IF NOT EXISTS prompt_metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt_name     TEXT NOT NULL,
    version         INTEGER NOT NULL,
    metric_name     TEXT NOT NULL,
    metric_value    REAL NOT NULL,
    sample_count    INTEGER DEFAULT 0,
    recorded_at     TEXT NOT NULL,
    detail          TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_pm_name_version
    ON prompt_metrics(prompt_name, version);

CREATE TABLE IF NOT EXISTS prompt_failure_cases (
    case_id         TEXT PRIMARY KEY,
    prompt_name     TEXT NOT NULL,
    version         INTEGER NOT NULL,
    case_type       TEXT NOT NULL,
    input_summary   TEXT DEFAULT '',
    expected        TEXT DEFAULT '',
    actual          TEXT DEFAULT '',
    diagnosis       TEXT DEFAULT '',
    recorded_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pfc_name_version
    ON prompt_failure_cases(prompt_name, version);

CREATE TABLE IF NOT EXISTS evolve_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type      TEXT NOT NULL,
    prompt_name     TEXT DEFAULT '',
    from_version    INTEGER,
    to_version      INTEGER,
    summary         TEXT DEFAULT '',
    payload_json    TEXT DEFAULT '',
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ev_created
    ON evolve_events(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ev_prompt
    ON evolve_events(prompt_name, created_at DESC);
"""


class PromptStore:
    """Prompt 版本与指标的 SQLite 持久化后端。

    线程安全：内部使用 threading.Lock 保护写操作。
    """

    def __init__(self, db_path: str = "data/prompt_evolve.db"):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    # ── 初始化 ──

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    # ── Prompt Version CRUD ──

    def save_version(self, pv: PromptVersion) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO prompt_versions
                   (version_id, prompt_name, version, prompt_text, status,
                    created_at, reason, parent_version, eval_score, eval_detail)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    pv.version_id, pv.prompt_name, pv.version, pv.prompt_text,
                    pv.status, pv.created_at, pv.reason, pv.parent_version,
                    pv.eval_score, pv.eval_detail,
                ),
            )

    def get_active_version(self, prompt_name: str) -> PromptVersion | None:
        """获取某 prompt 当前激活版本。"""
        with self._connect() as conn:
            row = conn.execute(
                """SELECT * FROM prompt_versions
                   WHERE prompt_name = ? AND status = 'active'
                   ORDER BY version DESC LIMIT 1""",
                (prompt_name,),
            ).fetchone()
        return self._row_to_version(row) if row else None

    def get_version(self, prompt_name: str, version: int) -> PromptVersion | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM prompt_versions WHERE prompt_name = ? AND version = ?",
                (prompt_name, version),
            ).fetchone()
        return self._row_to_version(row) if row else None

    def get_latest_version_number(self, prompt_name: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT MAX(version) AS mv FROM prompt_versions WHERE prompt_name = ?",
                (prompt_name,),
            ).fetchone()
        return int(row["mv"]) if row and row["mv"] is not None else -1

    def list_versions(self, prompt_name: str) -> list[PromptVersion]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM prompt_versions WHERE prompt_name = ? ORDER BY version",
                (prompt_name,),
            ).fetchall()
        return [self._row_to_version(r) for r in rows]

    def list_all_prompt_names(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT prompt_name FROM prompt_versions ORDER BY prompt_name"
            ).fetchall()
        return [r["prompt_name"] for r in rows]

    def deactivate_all(self, prompt_name: str) -> None:
        """将某 prompt 的所有 active 版本设为 archived。"""
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE prompt_versions SET status = 'archived' WHERE prompt_name = ? AND status = 'active'",
                (prompt_name,),
            )

    def promote(self, prompt_name: str, version: int) -> None:
        """提升某版本为 active，同时将旧 active 归档。"""
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE prompt_versions SET status = 'archived' WHERE prompt_name = ? AND status = 'active'",
                (prompt_name,),
            )
            conn.execute(
                "UPDATE prompt_versions SET status = 'active' WHERE prompt_name = ? AND version = ?",
                (prompt_name, version),
            )

    def rollback(self, prompt_name: str, version: int) -> None:
        """将某版本标记为 rolled_back。"""
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE prompt_versions SET status = 'rolled_back' WHERE prompt_name = ? AND version = ?",
                (prompt_name, version),
            )

    # ── Metrics ──

    def record_metric(self, metric: PromptMetricSnapshot) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO prompt_metrics
                   (prompt_name, version, metric_name, metric_value, sample_count, recorded_at, detail)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    metric.prompt_name, metric.version, metric.metric_name,
                    metric.metric_value, metric.sample_count, metric.recorded_at,
                    metric.detail,
                ),
            )

    def get_metrics(
        self,
        prompt_name: str,
        version: int | None = None,
        metric_name: str | None = None,
        limit: int = 100,
    ) -> list[PromptMetricSnapshot]:
        clauses = ["prompt_name = ?"]
        params: list[Any] = [prompt_name]
        if version is not None:
            clauses.append("version = ?")
            params.append(version)
        if metric_name:
            clauses.append("metric_name = ?")
            params.append(metric_name)
        params.append(limit)
        sql = (
            f"SELECT * FROM prompt_metrics WHERE {' AND '.join(clauses)} "
            f"ORDER BY recorded_at DESC LIMIT ?"
        )
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_metric(r) for r in rows]

    # ── Failure Cases ──

    def record_failure(self, fc: PromptFailureCase) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO prompt_failure_cases
                   (case_id, prompt_name, version, case_type,
                    input_summary, expected, actual, diagnosis, recorded_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    fc.case_id, fc.prompt_name, fc.version, fc.case_type,
                    fc.input_summary, fc.expected, fc.actual,
                    fc.diagnosis, fc.recorded_at,
                ),
            )

    def get_failures(
        self,
        prompt_name: str,
        version: int | None = None,
        case_type: str | None = None,
        limit: int = 50,
    ) -> list[PromptFailureCase]:
        clauses = ["prompt_name = ?"]
        params: list[Any] = [prompt_name]
        if version is not None:
            clauses.append("version = ?")
            params.append(version)
        if case_type:
            clauses.append("case_type = ?")
            params.append(case_type)
        params.append(limit)
        sql = (
            f"SELECT * FROM prompt_failure_cases WHERE {' AND '.join(clauses)} "
            f"ORDER BY recorded_at DESC LIMIT ?"
        )
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_failure(r) for r in rows]

    def count_failures(self, prompt_name: str, version: int | None = None) -> int:
        clauses = ["prompt_name = ?"]
        params: list[Any] = [prompt_name]
        if version is not None:
            clauses.append("version = ?")
            params.append(version)
        sql = f"SELECT COUNT(*) AS cnt FROM prompt_failure_cases WHERE {' AND '.join(clauses)}"
        with self._connect() as conn:
            row = conn.execute(sql, params).fetchone()
        return int(row["cnt"]) if row else 0

    # ── Evolve Events ──

    def record_event(self, event: EvolveEvent) -> int:
        """记录一条自进化过程事件，返回 event_id。"""
        payload_json = ""
        try:
            payload_json = json.dumps(event.payload or {}, ensure_ascii=False)
        except Exception:
            payload_json = ""

        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO evolve_events
                   (event_type, prompt_name, from_version, to_version, summary, payload_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    event.event_type,
                    event.prompt_name or "",
                    event.from_version,
                    event.to_version,
                    event.summary or "",
                    payload_json,
                    event.created_at,
                ),
            )
            return int(cur.lastrowid or 0)

    def list_events(
        self,
        *,
        limit: int = 100,
        prompt_name: str | None = None,
        event_type: str | None = None,
    ) -> list[EvolveEvent]:
        clauses: list[str] = []
        params: list[Any] = []
        if prompt_name:
            clauses.append("prompt_name = ?")
            params.append(prompt_name)
        if event_type:
            clauses.append("event_type = ?")
            params.append(event_type)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM evolve_events {where} ORDER BY id DESC LIMIT ?"
        params.append(int(limit))

        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()

        result: list[EvolveEvent] = []
        for r in rows:
            payload: dict[str, Any] = {}
            try:
                payload = json.loads(r["payload_json"] or "{}")
            except Exception:
                payload = {}
            result.append(
                EvolveEvent(
                    event_id=int(r["id"]),
                    event_type=str(r["event_type"] or ""),
                    prompt_name=str(r["prompt_name"] or ""),
                    from_version=int(r["from_version"]) if r["from_version"] is not None else None,
                    to_version=int(r["to_version"]) if r["to_version"] is not None else None,
                    summary=str(r["summary"] or ""),
                    payload=payload if isinstance(payload, dict) else {},
                    created_at=str(r["created_at"] or ""),
                )
            )
        return result

    def get_event(self, event_id: int) -> EvolveEvent | None:
        with self._connect() as conn:
            r = conn.execute(
                "SELECT * FROM evolve_events WHERE id = ?",
                (int(event_id),),
            ).fetchone()
        if not r:
            return None
        payload: dict[str, Any] = {}
        try:
            payload = json.loads(r["payload_json"] or "{}")
        except Exception:
            payload = {}
        return EvolveEvent(
            event_id=int(r["id"]),
            event_type=str(r["event_type"] or ""),
            prompt_name=str(r["prompt_name"] or ""),
            from_version=int(r["from_version"]) if r["from_version"] is not None else None,
            to_version=int(r["to_version"]) if r["to_version"] is not None else None,
            summary=str(r["summary"] or ""),
            payload=payload if isinstance(payload, dict) else {},
            created_at=str(r["created_at"] or ""),
        )

    # ── Row Converters ──

    @staticmethod
    def _row_to_version(row: sqlite3.Row) -> PromptVersion:
        return PromptVersion(
            version_id=row["version_id"],
            prompt_name=row["prompt_name"],
            version=row["version"],
            prompt_text=row["prompt_text"],
            status=row["status"],
            created_at=row["created_at"],
            reason=row["reason"] or "",
            parent_version=row["parent_version"] or 0,
            eval_score=row["eval_score"],
            eval_detail=row["eval_detail"] or "",
        )

    @staticmethod
    def _row_to_metric(row: sqlite3.Row) -> PromptMetricSnapshot:
        return PromptMetricSnapshot(
            prompt_name=row["prompt_name"],
            version=row["version"],
            metric_name=row["metric_name"],
            metric_value=row["metric_value"],
            sample_count=row["sample_count"] or 0,
            recorded_at=row["recorded_at"],
            detail=row["detail"] or "",
        )

    @staticmethod
    def _row_to_failure(row: sqlite3.Row) -> PromptFailureCase:
        return PromptFailureCase(
            case_id=row["case_id"],
            prompt_name=row["prompt_name"],
            version=row["version"],
            case_type=row["case_type"],
            input_summary=row["input_summary"] or "",
            expected=row["expected"] or "",
            actual=row["actual"] or "",
            diagnosis=row["diagnosis"] or "",
            recorded_at=row["recorded_at"],
        )
