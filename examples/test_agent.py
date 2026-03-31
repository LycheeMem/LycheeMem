from __future__ import annotations

import argparse
import json
import textwrap
import time
import uuid
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class EvalResult:
    score: int
    passed: list[str]
    failed: list[str]


class LycheeClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def set_token(self, token: str) -> None:
        self.session.headers["Authorization"] = f"Bearer {token}"

    def post(self, path: str, payload: dict[str, Any], timeout: int = 120) -> dict[str, Any]:
        resp = self.session.post(f"{self.base_url}{path}", json=payload, timeout=timeout)
        if not resp.ok:
            raise RuntimeError(f"POST {path} failed [{resp.status_code}]: {resp.text[:500]}")
        return resp.json()

    def get(self, path: str, params: dict[str, Any] | None = None, timeout: int = 30) -> dict[str, Any]:
        resp = self.session.get(f"{self.base_url}{path}", params=params, timeout=timeout)
        if not resp.ok:
            raise RuntimeError(f"GET {path} failed [{resp.status_code}]: {resp.text[:500]}")
        return resp.json()

    def register_or_login(self, username: str, password: str) -> None:
        try:
            data = self.post("/auth/register", {
                "username": username,
                "password": password,
                "display_name": username,
            })
            self.set_token(data["token"])
        except Exception:
            data = self.post("/auth/login", {
                "username": username,
                "password": password,
            })
            self.set_token(data["token"])


class ScenarioRunner:
    def __init__(self, client: LycheeClient, session_id: str):
        self.client = client
        self.session_id = session_id

    def append_turn(self, user_query: str) -> dict[str, Any]:
        """
        通过 /memory/reason 让 turn 写入会话。
        """
        return self.client.post("/memory/reason", {
            "session_id": self.session_id,
            "user_query": user_query,
            "background_context": "",
            "skill_reuse_plan": [],
            "append_to_session": True,
        })

    def consolidate(self, retrieved_context: str = "") -> dict[str, Any]:
        return self.client.post("/memory/consolidate", {
            "session_id": self.session_id,
            "retrieved_context": retrieved_context,
            "background": False,
        }, timeout=600)

    def search(self, query: str, top_k: int = 10) -> dict[str, Any]:
        return self.client.post("/memory/search", {
            "query": query,
            "top_k": top_k,
            "include_graph": True,
            "include_skills": True,
        })

    def synthesize(self, query: str, search_result: dict[str, Any]) -> dict[str, Any]:
        return self.client.post("/memory/synthesize", {
            "user_query": query,
            "graph_results": search_result.get("graph_results", []),
            "skill_results": search_result.get("skill_results", []),
        })

    def reason(self, query: str, synthesize_result: dict[str, Any]) -> dict[str, Any]:
        return self.client.post("/memory/reason", {
            "session_id": self.session_id,
            "user_query": query,
            "background_context": synthesize_result.get("background_context", ""),
            "skill_reuse_plan": synthesize_result.get("skill_reuse_plan", []),
            "append_to_session": True,
        })

    def seed_long_history(self) -> None:
        history = [
            "项目叫 Atlas API，技术栈是 FastAPI + PostgreSQL，使用 Helm 部署到 Kubernetes。",
            "Atlas API 的生产发布通常在欧洲区集群执行，区域是 eu-west-1。",
            "发布前必须先执行 `alembic upgrade --sql` 做 migration dry-run。",
            "发布完成后必须观察服务健康状态至少 5 分钟。",
            "如果发布失败，回滚方式必须使用 `helm rollback`。",
            "我们还有另一个项目叫 Billing Worker，它也跑在 Kubernetes 上，但流程不同。",
            "Billing Worker 的 staging namespace 是 billing-staging-eu。",
            "Atlas API 之前有一次失败，原因是把 namespace 错写成了 atlas-staging。",
            "Atlas API 上次失败的第二个原因是跳过了 migration dry-run。",
            "对于 Atlas API，之前记录的生产 namespace 是 atlas-prod-eu。",
            "Redis 调优和 Atlas API 发布是两件不同的事，不要混在一起。",
            "更正一下，Atlas API 现在真正的生产 namespace 不是 atlas-prod-eu，而是 atlas-prod-eu-2。",
            "这次要发布的镜像 tag 是 release-2026-03-31-hotfix。",
            "这次不要直接给我一串命令，先给 checklist，再给步骤。",
        ]
        for idx, turn in enumerate(history, start=1):
            print(f"[seed {idx:02d}] {turn}")
            self.append_turn(turn)

        print("\n[consolidate] synchronously consolidating history...")
        self.consolidate()

    def run_final_task(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        final_query = (
            "现在请你为 Atlas API 制定这次生产发布方案。"
            "先给 checklist，再给步骤，并明确说明如何避免重蹈上次失败。"
        )
        search_res = self.search(final_query, top_k=10)
        synth_res = self.synthesize(final_query, search_res)
        reason_res = self.reason(final_query, synth_res)
        return search_res, synth_res, reason_res


def evaluate(background_context: str, final_response: str) -> EvalResult:
    text = f"{background_context}\n{final_response}".lower()

    checks = {
        "使用更新后的生产 namespace": "atlas-prod-eu-2" in text,
        "提到镜像 tag": "release-2026-03-31-hotfix" in text,
        "提到 migration dry-run": ("alembic upgrade --sql" in text) or ("dry-run" in text),
        "提到健康检查 5 分钟": ("5 分钟" in text) or ("5分钟" in text) or ("5 minutes" in text),
        "提到 Helm 发布": ("helm" in text and "upgrade" in text),
        "提到 Helm 回滚": ("helm rollback" in text),
        "提到避免历史失败": ("上次失败" in text) or ("避免" in text) or ("不要再" in text),
        "遵守先 checklist 再步骤": ("checklist" in text and "步骤" in text) or ("检查清单" in text and "步骤" in text),
        "没有把旧 namespace 当成最终答案": not (
            "atlas-staging" in final_response.lower() and "atlas-prod-eu-2" not in final_response.lower()
        ),
        "没有混入 Billing Worker": "billing worker" not in final_response.lower(),
    }

    passed = [k for k, v in checks.items() if v]
    failed = [k for k, v in checks.items() if not v]
    return EvalResult(score=len(passed), passed=passed, failed=failed)


def print_report(session_id: str, search_res: dict[str, Any], synth_res: dict[str, Any], reason_res: dict[str, Any], eval_res: EvalResult) -> None:
    print("\n" + "=" * 100)
    print("LONG-HORIZON MEMORY TEST REPORT")
    print("=" * 100)
    print(f"session_id  : {session_id}")
    print(f"search_mode : {search_res.get('search_mode')}")
    print(f"score       : {eval_res.score}/10")

    print("\n[retrieval_plan]")
    print(json.dumps(search_res.get("retrieval_plan", {}), ensure_ascii=False, indent=2))

    print("\n[passed]")
    for item in eval_res.passed:
        print(f"  ✔ {item}")

    print("\n[failed]")
    for item in eval_res.failed:
        print(f"  ✘ {item}")

    print("\n[background_context preview]")
    print(textwrap.shorten(synth_res.get("background_context", ""), width=1200, placeholder=" ..."))

    print("\n[final_response]")
    print(reason_res.get("response", ""))


def main() -> None:
    parser = argparse.ArgumentParser(description="Long-horizon action-aware memory test for LycheeMem")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--username", default="lh_memory_test_user")
    parser.add_argument("--password", default="lh_memory_test_password_123")
    parser.add_argument("--session-id", default=f"lh-mem-test-{uuid.uuid4().hex[:8]}")
    args = parser.parse_args()

    client = LycheeClient(args.base_url)
    client.register_or_login(args.username, args.password)

    runner = ScenarioRunner(client, args.session_id)

    start = time.perf_counter()
    runner.seed_long_history()
    search_res, synth_res, reason_res = runner.run_final_task()
    eval_res = evaluate(
        synth_res.get("background_context", ""),
        reason_res.get("response", ""),
    )
    elapsed = time.perf_counter() - start

    print_report(args.session_id, search_res, synth_res, reason_res, eval_res)
    print(f"\nelapsed: {elapsed:.2f}s")

    # 8/10 以上视为通过
    raise SystemExit(0 if eval_res.score >= 8 else 1)


if __name__ == "__main__":
    main()