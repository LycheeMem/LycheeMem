"""
LycheeMem — 纯 API Pipeline 流程示例脚本
==========================================

演示如何仅通过 HTTP API 完成完整的 Pipeline 流程，无需直接调用 Python 代码：

    Step 1: POST /memory/search     → 从图谱和技能库检索相关记忆
    Step 2: POST /memory/synthesize → LLM-as-Judge 评分融合，生成 background_context
    Step 3: POST /memory/reason     → 基于上下文推理，生成最终回答（写入会话）
    Step 4: POST /memory/consolidate→ 萃取本轮对话，固化到图谱/技能库

前置条件：
    - 服务已启动：python main.py
    - 安装依赖：pip install requests

用法：
    python examples/api_pipeline_demo.py
    python examples/api_pipeline_demo.py --base-url http://localhost:8000
    python examples/api_pipeline_demo.py --multi-turn        # 演示多轮对话
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
import uuid
from typing import Any

try:
    import requests
except ImportError:
    sys.exit("请先安装 requests：pip install requests")

# ─────────────────────────────────────────────────────────────────────────────
# ANSI 颜色（终端输出用，不影响功能）
# ─────────────────────────────────────────────────────────────────────────────
BOLD  = "\033[1m"
CYAN  = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED   = "\033[31m"
DIM   = "\033[2m"
RESET = "\033[0m"


def _hdr(title: str) -> None:
    bar = "─" * 60
    print(f"\n{BOLD}{CYAN}{bar}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{bar}{RESET}")


def _ok(label: str, value: Any = "") -> None:
    print(f"  {GREEN}✔{RESET} {BOLD}{label}{RESET}", end="")
    if value != "":
        print(f": {value}", end="")
    print()


def _info(label: str, value: Any = "") -> None:
    print(f"  {DIM}·{RESET} {label}", end="")
    if value != "":
        print(f": {value}", end="")
    print()


def _warn(msg: str) -> None:
    print(f"  {YELLOW}⚠{RESET}  {msg}")


def _err(msg: str) -> None:
    print(f"  {RED}✘{RESET}  {BOLD}{msg}{RESET}")


def _json_preview(obj: Any, indent: int = 4, max_len: int = 400) -> str:
    """将对象序列化为 JSON，超长时截断。"""
    text = json.dumps(obj, ensure_ascii=False, indent=indent)
    if len(text) > max_len:
        text = text[:max_len] + f"\n{' ' * indent}... (truncated)"
    return text


# ─────────────────────────────────────────────────────────────────────────────
# HTTP 辅助
# ─────────────────────────────────────────────────────────────────────────────

class APIClient:
    """轻量 HTTP 客户端，统一管理 base_url。"""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def post(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url}{path}"
        resp = self.session.post(url, json=payload, timeout=120)
        if not resp.ok:
            raise RuntimeError(
                f"POST {path} failed [{resp.status_code}]: {resp.text[:500]}"
            )
        return resp.json()

    def get(self, path: str, params: dict | None = None) -> dict:
        url = f"{self.base_url}{path}"
        resp = self.session.get(url, params=params, timeout=30)
        if not resp.ok:
            raise RuntimeError(
                f"GET {path} failed [{resp.status_code}]: {resp.text[:500]}"
            )
        return resp.json()


# ─────────────────────────────────────────────────────────────────────────────
# 流程步骤
# ─────────────────────────────────────────────────────────────────────────────

def step_health(client: APIClient) -> bool:
    _hdr("Step 0 · Health Check")    
    try:
        data = client.get("/health")
        _ok("服务状态", data.get("status"))
        _ok("版本",     data.get("version"))
        return True
    except Exception as e:
        _err(f"无法连接到服务：{e}")
        _warn("请确认 `python main.py` 已启动，默认端口 8000")
        return False


def step_search(
    client: APIClient,
    user_query: str,
    top_k: int = 5,
) -> dict:
    """Step 1: POST /memory/search"""
    _hdr("Step 1 · 记忆检索 /memory/search")
    _info("查询", user_query)

    t0 = time.perf_counter()
    result = client.post("/memory/search", {
        "query": user_query,
        "top_k": top_k,
        "include_graph": True,
        "include_skills": True,
    })
    elapsed = time.perf_counter() - t0

    graph_cnt  = len(result.get("graph_results", []))
    skill_cnt  = len(result.get("skill_results", []))
    total      = result.get("total", graph_cnt + skill_cnt)

    _ok("完成", f"耗时 {elapsed:.2f}s")
    _info("图谱命中", f"{graph_cnt} 条")
    _info("技能命中", f"{skill_cnt} 条")
    _info("合计",     f"{total} 条")

    # 打印摘要（仅显示前 2 条）
    for i, item in enumerate(result.get("graph_results", [])[:2], 1):
        fact = str(item.get("fact_text") or item.get("summary") or "").strip()
        if fact:
            _info(f"  graph[{i}]", textwrap.shorten(fact, width=80))

    for i, skill in enumerate(result.get("skill_results", [])[:2], 1):
        intent = str(skill.get("intent") or "").strip()
        if intent:
            _info(f"  skill[{i}]", textwrap.shorten(intent, width=80))

    return result


def step_synthesize(
    client: APIClient,
    user_query: str,
    search_result: dict,
) -> dict:
    """Step 2: POST /memory/synthesize"""
    _hdr("Step 2 · 记忆合成 /memory/synthesize")

    t0 = time.perf_counter()
    result = client.post("/memory/synthesize", {
        "user_query": user_query,
        "graph_results": search_result.get("graph_results", []),
        "skill_results": search_result.get("skill_results", []),
    })
    elapsed = time.perf_counter() - t0

    ctx = result.get("background_context", "")
    _ok("完成", f"耗时 {elapsed:.2f}s")
    _info("保留片段",  result.get("kept_count", 0))
    _info("丢弃片段",  result.get("dropped_count", 0))
    _info("可复用技能", len(result.get("skill_reuse_plan", [])))

    if ctx:
        preview = textwrap.shorten(ctx, width=160, placeholder=" …")
        _info("background_context 预览", preview)
    else:
        _info("background_context", "（空，无相关记忆）")

    return result


def step_reason(
    client: APIClient,
    session_id: str,
    user_query: str,
    synthesize_result: dict,
    append_to_session: bool = True,
) -> dict:
    """Step 3: POST /memory/reason"""
    _hdr("Step 3 · 最终推理 /memory/reason")

    t0 = time.perf_counter()
    result = client.post("/memory/reason", {
        "session_id": session_id,
        "user_query": user_query,
        "background_context": synthesize_result.get("background_context", ""),
        "skill_reuse_plan":   synthesize_result.get("skill_reuse_plan", []),
        "retrieved_skills":   [],   # 已由 synthesize 处理过，无需重复传
        "append_to_session":  append_to_session,
    })
    elapsed = time.perf_counter() - t0

    response_text = result.get("response", "")
    _ok("完成", f"耗时 {elapsed:.2f}s")
    _info("session_id",    result.get("session_id"))
    _info("wm_token_usage", result.get("wm_token_usage", 0))
    _info("append_to_session", append_to_session)

    print(f"\n  {BOLD}━━ 最终回答 ━━{RESET}")
    for line in response_text.splitlines():
        print(f"  {line}")
    print()

    return result


def step_consolidate(
    client: APIClient,
    session_id: str,
    retrieved_context: str,
    background: bool = True,
) -> dict:
    """Step 4: POST /memory/consolidate

    background=True（默认）：立即返回，固化在服务端后台线程中异步执行。
    background=False：同步等待完成，适合调试（可能耗时数分钟）。
    """
    _hdr("Step 4 · 记忆固化 /memory/consolidate")
    _info("模式", "后台异步" if background else "同步等待")

    timeout = 30 if background else 600  # 后台模式只需等待触发确认
    t0 = time.perf_counter()
    url = f"{client.base_url}/memory/consolidate"
    resp = client.session.post(
        url,
        json={
            "session_id": session_id,
            "retrieved_context": retrieved_context,
            "background": background,
        },
        timeout=timeout,
    )
    if not resp.ok:
        raise RuntimeError(f"POST /memory/consolidate failed [{resp.status_code}]: {resp.text[:400]}")
    result = resp.json()
    elapsed = time.perf_counter() - t0

    status = result.get("status", "?")
    if status == "started":
        _ok("已触发（后台执行中）", f"耗时 {elapsed:.2f}s")
        _info("提示", "固化正在服务端后台运行")
        _info("完成后可查看", f"GET /memory/graph  /  GET /memory/skills")
    else:
        _ok("完成", f"耗时 {elapsed:.2f}s  status={status}")
        _info("entities_added",  result.get("entities_added", 0))
        _info("facts_added",     result.get("facts_added", 0))
        _info("skills_added",    result.get("skills_added", 0))
        _info("has_novelty",     result.get("has_novelty"))

        if result.get("skipped_reason"):
            _warn(f"跳过原因：{result['skipped_reason']}")

        steps = result.get("steps", [])
        if steps:
            print(f"\n  {DIM}固化子步骤：{RESET}")
            for s in steps:
                status_icon = GREEN + "✔" + RESET if s.get("status") == "done" else YELLOW + "·" + RESET
                detail = textwrap.shorten(str(s.get("detail", "")), width=80)
                print(f"    {status_icon} {s.get('name', '?')}: {detail}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 多轮对话演示
# ─────────────────────────────────────────────────────────────────────────────

MULTI_TURN_DIALOGUE = [
    "我叫 Alex，目前在做一个 Python 微服务项目，技术栈是 FastAPI + PostgreSQL。",
    "这个项目需要支持多租户，每个租户的数据需要严格隔离，有什么好的方案推荐吗？",
    "我们决定使用 Row-Level Security（RLS）方案，你能帮我梳理一下实施步骤吗？",
]


def run_multi_turn(client: APIClient, session_id: str) -> None:
    """演示多轮对话的完整 Pipeline——每轮均执行 search→synthesize→reason。"""
    _hdr("多轮对话演示（3 轮）")
    print(f"  session_id: {BOLD}{session_id}{RESET}\n")

    last_context = ""
    for turn_idx, query in enumerate(MULTI_TURN_DIALOGUE, start=1):
        print(f"\n{BOLD}【第 {turn_idx} 轮】{RESET} {query}")

        # Search
        search_res = step_search(client, query, top_k=3)

        # Synthesize
        synth_res = step_synthesize(client, query, search_res)
        last_context = synth_res.get("background_context", "")

        # Reason（写入会话）
        step_reason(client, session_id, query, synth_res, append_to_session=True)

    # 最后一轮结束后统一固化（后台异步）
    _hdr("多轮对话结束 · 统一固化")
    step_consolidate(client, session_id, last_context, background=True)


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

SINGLE_TURN_QUERY = (
    "我在做一个 Python 项目，每次部署到生产前需要自动运行 ruff 格式检查和 pytest。"
    "有什么好的 CI 流程推荐吗？"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LycheeMem 纯 API Pipeline 流程示例脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(__doc__ or ""),
    )
    parser.add_argument("--base-url",  default="http://localhost:8000", help="API 地址")
    parser.add_argument("--username",  default="demo_user", help="用于注册/登录的用户名")
    parser.add_argument("--password",  default="demo_password_123", help="密码")
    parser.add_argument("--session-id", default=None, help="会话 ID（不指定时自动生成）")
    parser.add_argument("--multi-turn", action="store_true", help="演示多轮对话模式")
    parser.add_argument("--query",     default=SINGLE_TURN_QUERY, help="单轮模式下使用的查询文本")
    parser.add_argument("--top-k",     type=int, default=5, help="检索返回条数")
    parser.add_argument("--no-consolidate", action="store_true", help="跳过固化步骤")
    parser.add_argument("--sync-consolidate", action="store_true",
                        help="同步等待固化完成（默认后台异步，此选项适合调试，超时约 600s）")
    args = parser.parse_args()

    session_id = args.session_id or f"demo-{uuid.uuid4().hex[:8]}"

    print(f"\n{BOLD}LycheeMem — 纯 API Pipeline 流程演示{RESET}")
    print(f"  Base URL  : {args.base_url}")
    print(f"  Session   : {session_id}")
    print(f"  模式      : {'多轮对话' if args.multi_turn else '单轮对话'}")

    client = APIClient(args.base_url)

    # Step 0: 健康检查
    if not step_health(client):
        sys.exit(1)

    # ── 多轮 or 单轮 ──────────────────────────────────────────────────────────
    if args.multi_turn:
        run_multi_turn(client, session_id)
    else:
        user_query = args.query

        # Step 1: 搜索
        search_res = step_search(client, user_query, top_k=args.top_k)

        # Step 2: 合成
        synth_res = step_synthesize(client, user_query, search_res)

        # Step 3: 推理
        reason_res = step_reason(
            client, session_id, user_query, synth_res, append_to_session=True
        )
        _ = reason_res  # 供后续扩展使用

        # Step 4: 固化
        if not args.no_consolidate:
            step_consolidate(
                client,
                session_id,
                retrieved_context=synth_res.get("background_context", ""),
                background=not args.sync_consolidate,
            )
        else:
            _hdr("Step 4 · 记忆固化（已跳过）")
            _warn("使用 --no-consolidate 跳过了固化步骤")

    # ── 总结 ──────────────────────────────────────────────────────────────────
    _hdr("演示完成")
    print(f"  {GREEN}{BOLD}Pipeline 完整流程执行成功！{RESET}")
    print(f"\n  会话 ID: {BOLD}{session_id}{RESET}")
    print(f"  查看会话记忆: GET /memory/session/{session_id}")
    print(f"  查看图谱:     GET /memory/graph")
    print(f"  查看技能库:   GET /memory/skills")
    print(f"  API 文档:     {args.base_url}/docs\n")


if __name__ == "__main__":
    main()
