from __future__ import annotations

import argparse
import json
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import quote

try:
    import requests
except ImportError:
    sys.exit("请先安装 requests：pip install requests")


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_PATH = (
    PROJECT_ROOT
    / "reference"
    / "locomo-benchmark-eval"
    / "data"
    / "locomo"
    / "locomo10.json"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "reference"
    / "locomo-benchmark-eval"
    / "results"
    / "lychee_eval_results.json"
)


class APIClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def get(self, path: str, *, timeout: int = 120) -> dict[str, Any]:
        return self._request("GET", path, timeout=timeout)

    def post(self, path: str, payload: dict[str, Any], *, timeout: int = 7200) -> dict[str, Any]:
        return self._request("POST", path, json=payload, timeout=timeout)

    def delete(self, path: str, *, timeout: int = 7200) -> dict[str, Any]:
        return self._request("DELETE", path, timeout=timeout)

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        resp = self.session.request(method, url, **kwargs)
        if not resp.ok:
            raise RuntimeError(f"{method} {path} failed [{resp.status_code}]: {resp.text[:1000]}")
        if not resp.text.strip():
            return {}
        return resp.json()


def parse_range_string(value: str | None) -> set[int] | None:
    raw = str(value or "").strip()
    if not raw or raw.lower() == "all":
        return None

    result: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(x.strip()) for x in part.split("-", 1)]
            result.update(range(start, end + 1))
        else:
            result.add(int(part))
    return result


def parse_category_set(value: str | None) -> set[int]:
    raw = str(value or "").strip()
    if not raw or raw.lower() in {"none", "null", "false", "off"}:
        return set()
    return {int(part.strip()) for part in raw.split(",") if part.strip()}


def session_sort_key(key: str) -> int:
    match = re.fullmatch(r"session_(\d+)", key)
    return int(match.group(1)) if match else 0


def format_locomo_turn(turn: dict[str, Any]) -> str:
    speaker = str(turn.get("speaker") or "Unknown").strip() or "Unknown"
    text = str(turn.get("text") or "").strip()
    caption = str(turn.get("blip_caption") or "").strip()
    image_query = str(turn.get("query") or "").strip()

    lines: list[str] = []
    if text:
        lines.append(f"{speaker}: {text}")
    elif caption or image_query:
        lines.append(f"{speaker}:")
    if caption:
        lines.append(f"[Image caption]: {caption}")
    if image_query:
        lines.append(f"[Image query]: {image_query}")
    return "\n".join(lines).strip()


def load_locomo_users(
    input_path: Path,
    *,
    sample: str | None,
    user_override: str | None,
    exclude_categories: set[int],
    qa_range: str | None,
    qa_limit: int | None,
) -> list[dict[str, Any]]:
    selected_samples = parse_range_string(sample)
    selected_qas = parse_range_string(qa_range)

    try:
        raw_data = json.loads(input_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"读取 LoCoMo 数据失败: {input_path} ({exc})") from exc

    if not isinstance(raw_data, list):
        raw_data = [raw_data]

    users: list[dict[str, Any]] = []
    for sample_idx, item in enumerate(raw_data):
        if selected_samples is not None and sample_idx not in selected_samples:
            continue
        if not isinstance(item, dict):
            continue

        user_id = str(user_override or f"locomo_user_{sample_idx}").strip()
        conversation = item.get("conversation", item)
        if not isinstance(conversation, dict):
            continue

        sessions: list[dict[str, Any]] = []
        session_keys = [
            key
            for key in conversation
            if re.fullmatch(r"session_\d+", str(key))
        ]
        for session_key in sorted(session_keys, key=session_sort_key):
            turns = conversation.get(session_key) or []
            if not isinstance(turns, list):
                continue
            session_date = str(conversation.get(f"{session_key}_date_time") or "").strip()
            formatted_turns = [
                {
                    "speaker": str(turn.get("speaker") or "Unknown").strip() or "Unknown",
                    "content": format_locomo_turn(turn),
                }
                for turn in turns
                if isinstance(turn, dict)
            ]
            formatted_turns = [turn for turn in formatted_turns if turn["content"]]
            if formatted_turns:
                sessions.append({
                    "session_key": session_key,
                    "session_date": session_date,
                    "turns": formatted_turns,
                })

        qas: list[dict[str, Any]] = []
        for qa_idx, qa in enumerate(item.get("qa") or []):
            if selected_qas is not None and qa_idx not in selected_qas:
                continue
            if not isinstance(qa, dict):
                continue
            category = qa.get("category")
            try:
                category_int = int(category)
            except (TypeError, ValueError):
                category_int = category
            if isinstance(category_int, int) and category_int in exclude_categories:
                continue
            qas.append({
                "qa_index": qa_idx,
                "question": str(qa.get("question") or "").strip(),
                "ground_truth": str(qa.get("answer") or "").strip(),
                "evidence": qa.get("evidence"),
                "category": category_int,
                "user_id": user_id,
                "sample_index": sample_idx,
            })
            if qa_limit is not None and len(qas) >= qa_limit:
                break

        if sessions or qas:
            users.append({
                "user_id": user_id,
                "sample_index": sample_idx,
                "sessions": sessions,
                "qas": [qa for qa in qas if qa["question"]],
            })

    return users


def result_key(item: dict[str, Any]) -> str:
    return f"{item.get('user_id', '')}\t{item.get('question', '')}"


def load_existing_results(output_path: Path) -> list[dict[str, Any]]:
    if not output_path.exists():
        return []
    data = json.loads(output_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RuntimeError(f"结果文件不是 JSON 数组: {output_path}")
    return [item for item in data if isinstance(item, dict)]


def write_results(output_path: Path, results: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def safe_session_part(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    return text.strip("._-") or "session"


def make_ingest_session_id(user_id: str, session_key: str) -> str:
    sid = f"{safe_session_part(user_id)}__{safe_session_part(session_key)}"
    return sid[:128]


def list_session_ids(client: APIClient) -> list[str]:
    session_ids: list[str] = []
    offset = 0
    limit = 200
    while True:
        data = client.get(f"/sessions?offset={offset}&limit={limit}")
        batch = data.get("sessions") or []
        for item in batch:
            if isinstance(item, dict):
                sid = str(item.get("session_id") or "").strip()
                if sid:
                    session_ids.append(sid)
        if len(batch) < limit:
            break
        offset += limit
    return session_ids


def delete_session(client: APIClient, session_id: str) -> None:
    client.delete(f"/memory/session/{quote(session_id, safe='')}")


def clear_memory(client: APIClient, *, user_id: str) -> None:
    client.delete("/memory/graph/clear")
    client.delete("/memory/skills/clear")
    prefix = f"{safe_session_part(user_id)}__"
    for session_id in list_session_ids(client):
        if session_id == user_id or session_id.startswith(prefix):
            delete_session(client, session_id)


def ingest_user(
    client: APIClient,
    *,
    user_id: str,
    sessions: list[dict[str, Any]],
    skip_clear: bool,
    skip_skills: bool,
    session_workers: int,
) -> dict[str, Any]:
    if not skip_clear:
        print(f"[{user_id}] clear memory")
        clear_memory(client, user_id=user_id)

    def ingest_one(idx: int, session: dict[str, Any]) -> dict[str, Any]:
        session_client = APIClient(client.base_url)
        turns = list(session.get("turns") or [])
        session_key = str(session.get("session_key") or f"session_{idx}")
        ingest_session_id = make_ingest_session_id(user_id, session_key)
        session_date = str(session.get("session_date") or "").strip()
        print(f"[{user_id}] ingest {session_key} -> {ingest_session_id}: {len(turns)} turns")

        for turn in turns:
            if isinstance(turn, dict):
                content = str(turn.get("content") or "").strip()
                speaker = str(turn.get("speaker") or "").strip() or None
            else:
                content = str(turn or "").strip()
                speaker = None
            session_client.post(
                "/memory/append-turn",
                {
                    "session_id": ingest_session_id,
                    "role": "user",
                    "speaker": speaker,
                    "content": content,
                    "created_at": session_date or None,
                },
            )

        consolidate_payload: dict[str, Any] = {
            "session_id": ingest_session_id,
            "background": False,
            "skip_novelty_check": True,
            "skip_skills": skip_skills,
        }
        if session_date:
            consolidate_payload["session_date"] = session_date
        result = session_client.post("/memory/consolidate", consolidate_payload)
        status = result.get("status", "unknown")
        records = result.get("entities_added", 0)
        composites = result.get("facts_added", 0)
        print(f"[{user_id}] consolidate {session_key}: {status}, records={records}, composites={composites}")
        return {
            "session_key": session_key,
            "session_id": ingest_session_id,
            "session_date": session_date,
            "turn_count": len(turns),
            "result": result,
        }

    if not sessions:
        return {
            "turns_appended": 0,
            "sessions_ingested": 0,
            "consolidate_results": [],
        }

    workers = max(1, min(int(session_workers or 1), len(sessions)))
    consolidate_results: list[dict[str, Any]] = []
    if workers == 1:
        for idx, session in enumerate(sessions, start=1):
            consolidate_results.append(ingest_one(idx, session))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(ingest_one, idx, session): idx
                for idx, session in enumerate(sessions, start=1)
            }
            for future in as_completed(future_map):
                consolidate_results.append(future.result())

    consolidate_results.sort(key=lambda item: session_sort_key(str(item.get("session_key") or "")))
    total_turns = sum(int(item.get("turn_count") or 0) for item in consolidate_results)

    return {
        "turns_appended": total_turns,
        "sessions_ingested": len(sessions),
        "consolidate_results": consolidate_results,
    }


def collect_texts_from_provenance(provenance: list[Any]) -> list[str]:
    texts: list[str] = []
    seen: set[str] = set()
    for item in provenance:
        if not isinstance(item, dict):
            continue
        text = str(
            item.get("display_text")
            or item.get("semantic_text")
            or item.get("text")
            or item.get("summary")
            or ""
        ).strip()
        if text and text not in seen:
            seen.add(text)
            texts.append(text)
    return texts


def collect_ids_from_provenance(provenance: list[Any]) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for item in provenance:
        if not isinstance(item, dict):
            continue
        record_id = str(item.get("record_id") or item.get("id") or "").strip()
        if record_id and record_id not in seen:
            seen.add(record_id)
            ids.append(record_id)
    return ids


def answer_question(
    client: APIClient,
    *,
    qa: dict[str, Any],
    top_k: int,
    include_skills: bool,
    reference_time: str | None,
) -> dict[str, Any]:
    user_id = str(qa["user_id"])
    question = str(qa["question"])

    search_payload: dict[str, Any] = {
        "query": question,
        "top_k": top_k,
        "include_graph": True,
        "include_skills": include_skills,
        "synthesize": True,
        "mode": "full",
        "response_level": "full",
    }
    if reference_time:
        search_payload["reference_time"] = reference_time
    search_result = client.post("/memory/smart-search", search_payload)

    reason_payload: dict[str, Any] = {
        "session_id": f"{user_id}__qa",
        "user_query": question,
        "background_context": search_result.get("background_context", ""),
        "skill_reuse_plan": search_result.get("skill_reuse_plan", []),
        "retrieved_skills": search_result.get("skill_results", []),
        "append_to_session": False,
    }
    if reference_time:
        reason_payload["reference_time"] = reference_time
    reason_result = client.post("/memory/reason", reason_payload)

    provenance = list(search_result.get("provenance") or [])
    retrieved_texts = collect_texts_from_provenance(provenance)
    if not retrieved_texts and search_result.get("background_context"):
        retrieved_texts = [str(search_result.get("background_context"))]

    return {
        "question": question,
        "answer": reason_result.get("response", ""),
        "ground_truth": qa.get("ground_truth", ""),
        "user_id": user_id,
        "sample_index": qa.get("sample_index"),
        "qa_index": qa.get("qa_index"),
        "category": qa.get("category"),
        "evidence": qa.get("evidence"),
        "retrieved_total": search_result.get("total", 0),
        "kept_count": search_result.get("kept_count", 0),
        "dropped_count": search_result.get("dropped_count", 0),
        "retrieved_context_text": retrieved_texts,
        "retrieved_context_record_ids": collect_ids_from_provenance(provenance),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LoCoMo QA evaluation for the current LycheeMem memory API.",
    )
    parser.add_argument("--base-url", default="http://localhost:8000", help="LycheeMem API 地址")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH, help="LoCoMo JSON 路径")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="输出 JSON 路径")
    parser.add_argument("--sample", default=None, help="样本索引，例如 0、0-2、0,3；默认全部")
    parser.add_argument("--user", default=None, help="覆盖 user_id，通常只在单样本调试时使用")
    parser.add_argument("--qa-range", default=None, help="每个样本内的 QA 索引范围，例如 0-9")
    parser.add_argument("--qa-limit", type=int, default=None, help="每个样本最多评测多少题")
    parser.add_argument("--top-k", type=int, default=20, help="检索 top_k，后端上限为 50")
    parser.add_argument("--exclude-category", default="5", help="排除的 LoCoMo category，默认 5；传 none 关闭")
    parser.add_argument("--reference-time", default="2023-12-31", help="QA 推理参考时间")
    parser.add_argument("--skip-ingest", action="store_true", help="跳过摄入，直接使用已有记忆评测")
    parser.add_argument("--qa-only", action="store_true", help="只执行 QA，跳过清空、摄入和固化；等价于 --skip-ingest")
    parser.add_argument("--ingest-only", action="store_true", help="只摄入并固化，不回答 QA")
    parser.add_argument("--skip-clear", action="store_true", help="摄入前不清空语义/技能记忆和该用户 session")
    parser.add_argument("--session-workers", type=int, default=4, help="同一用户下并行摄入/固化的 LoCoMo session 数")
    parser.add_argument("--qa-workers", type=int, default=4, help="同一用户下并行回答的 QA 题目数")
    parser.add_argument("--include-skills", action="store_true", help="QA 检索时包含技能库")
    parser.add_argument("--extract-skills", action="store_true", help="摄入固化时也抽取技能；默认跳过以节省评测时间")
    parser.add_argument("--overwrite", action="store_true", help="忽略已有输出，从头写结果")
    args = parser.parse_args()

    if args.qa_only:
        args.skip_ingest = True
    if args.ingest_only and args.skip_ingest:
        sys.exit("--ingest-only 不能和 --skip-ingest 同时使用")
    if not 1 <= args.top_k <= 50:
        sys.exit("--top-k 必须在 1 到 50 之间")
    if args.qa_limit is not None and args.qa_limit < 1:
        sys.exit("--qa-limit 必须大于 0")
    if args.session_workers < 1:
        sys.exit("--session-workers 必须大于 0")
    if args.qa_workers < 1:
        sys.exit("--qa-workers 必须大于 0")

    client = APIClient(args.base_url)
    health = client.get("/health")
    print(f"Health: {health.get('status')} version={health.get('version')}")

    users = load_locomo_users(
        args.input_path,
        sample=args.sample,
        user_override=args.user,
        exclude_categories=parse_category_set(args.exclude_category),
        qa_range=args.qa_range,
        qa_limit=args.qa_limit,
    )
    if not users:
        sys.exit("没有可评测的 LoCoMo 数据")

    existing = [] if args.overwrite else load_existing_results(args.output)
    completed_keys = {result_key(item) for item in existing}
    results = list(existing)
    results_lock = threading.Lock()

    qa_order: dict[str, int] = {}
    order = 0
    for user in users:
        for qa in user["qas"]:
            qa_order[result_key(qa)] = order
            order += 1

    started = time.perf_counter()
    for user in users:
        user_id = str(user["user_id"])
        sessions = list(user.get("sessions") or [])
        qas = list(user.get("qas") or [])
        print(f"\n=== {user_id} | sessions={len(sessions)} | qa={len(qas)} ===")

        if not args.skip_ingest:
            ingest_user(
                client,
                user_id=user_id,
                sessions=sessions,
                skip_clear=args.skip_clear,
                skip_skills=not args.extract_skills,
                session_workers=args.session_workers,
            )
        else:
            print(f"[{user_id}] skip ingest")

        if args.ingest_only:
            continue

        pending_qas: list[tuple[int, dict[str, Any]]] = []
        for idx, qa in enumerate(qas, start=1):
            key = result_key(qa)
            if key in completed_keys:
                print(f"[{user_id}] skip answered {idx}/{len(qas)}")
                continue
            pending_qas.append((idx, qa))

        def qa_job(idx: int, qa: dict[str, Any]) -> dict[str, Any]:
            question = str(qa.get("question") or "")
            print(f"[{user_id}] QA {idx}/{len(qas)}: {question[:100]}")
            try:
                return answer_question(
                    APIClient(args.base_url),
                    qa=qa,
                    top_k=args.top_k,
                    include_skills=args.include_skills,
                    reference_time=args.reference_time,
                )
            except Exception as exc:
                print(f"[{user_id}] ERROR: {exc}")
                return {
                    "question": question,
                    "answer": f"ERROR: {exc}",
                    "ground_truth": qa.get("ground_truth", ""),
                    "user_id": user_id,
                    "sample_index": qa.get("sample_index"),
                    "qa_index": qa.get("qa_index"),
                    "category": qa.get("category"),
                    "evidence": qa.get("evidence"),
                    "error": str(exc),
                }

        if pending_qas:
            qa_workers = max(1, min(args.qa_workers, len(pending_qas)))
            if qa_workers == 1:
                for idx, qa in pending_qas:
                    result = qa_job(idx, qa)
                    with results_lock:
                        results.append(result)
                        completed_keys.add(result_key(result))
                        results.sort(key=lambda item: qa_order.get(result_key(item), 10**9))
                        write_results(args.output, results)
            else:
                with ThreadPoolExecutor(max_workers=qa_workers) as executor:
                    future_map = {
                        executor.submit(qa_job, idx, qa): qa
                        for idx, qa in pending_qas
                    }
                    for future in as_completed(future_map):
                        result = future.result()
                        with results_lock:
                            results.append(result)
                            completed_keys.add(result_key(result))
                            results.sort(key=lambda item: qa_order.get(result_key(item), 10**9))
                            write_results(args.output, results)

    elapsed = time.perf_counter() - started
    if args.ingest_only:
        print(f"\nIngest finished in {elapsed:.1f}s")
    else:
        print(f"\nDone: {len(results)} results -> {args.output} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
