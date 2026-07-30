#!/usr/bin/env python
"""统一 token 计量旁路组件 —— per-baseline 进程隔离记账,根治共享 vLLM 全局计数器的混计。

为什么需要它:
  共享 vLLM 服务的 generation_tokens_total 是全局计数器(label 只有 model_name 常量),
  多个 baseline / enterprise scaling 并发打同一个池时,"前后读数求差"会把别人的 token
  算进当前 job。本组件让每个 baseline 进程只记自己发出的 token,天然隔离,不依赖全局
  计数器、不需要串行独占、不需要代理(代理在生成关键路径上,故障会让 baseline 复现失败)。

绝不影响 baseline 复现/图质量的硬约束:
  - 只做旁路累加(在我们自己写的 wrapper 里读 response.usage)或纯读取已有产物;
  - 从不改任何 LLM 请求/响应内容、不改 wrapper 的 return 值;
  - 所有累加/解析全程 try/except 吞异常,记账失败绝不抛(与 run_lightrag 的 _log_failed_chunk 同理);
  - 线程安全(LinearRAG QA 16 线程 / LightRAG 协程并发)。

输出: results/token_usage/{baseline}_{dataset}.json , 区分 build / qa 两阶段。
取数来源:
  - wrapper_usage  : 我们自己的 LLM wrapper 里累加 response.usage (LightRAG / LinearRAG)
  - graphrag_log   : 读 graphrag 自己写的 indexing-engine.log 的 Metrics 块 (graphrag 建图)
  - hipporag_sqlite: 扫 HippoRAG 自己的 llm_cache sqlite 的 metadata (含 token)
"""
import os
import re
import json
import time
import threading
import sqlite3

# 2026-07-07 迁移: 随本文件所在 unified/ 走; URAG_ROOT 可覆盖(隔离工作区用, 与 unified_config 同语义)
RESULTS = (
    os.environ.get("U_RESULTS_ROOT")
    or os.path.join(
        os.environ.get("URAG_ROOT") or os.path.dirname(os.path.abspath(__file__)),
        "results",
    )
)
TOKEN_DIR = f"{RESULTS}/token_usage"


class TokenMeter:
    """线程安全的 per-phase token 累加器。
    phase 默认 'qa'(多数 wrapper 只在 QA 阶段被调);LightRAG 建图也走 wrapper,用 set_phase 切换。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._phase = "qa"
        self._acc = {"build": self._zero(), "qa": self._zero()}

    @staticmethod
    def _zero():
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}

    def set_phase(self, phase):
        if phase in ("build", "qa"):
            with self._lock:
                self._phase = phase

    def add(self, usage, phase=None):
        """累加一个 openai response.usage(有 .prompt_tokens / .completion_tokens / .total_tokens)。
        容错:usage 为 None 或任何异常都安全吞掉,绝不抛。"""
        try:
            if usage is None:
                return
            p = int(getattr(usage, "prompt_tokens", 0) or 0)
            c = int(getattr(usage, "completion_tokens", 0) or 0)
            t = int(getattr(usage, "total_tokens", 0) or 0) or (p + c)
            ph = phase if phase in ("build", "qa") else self._phase
            with self._lock:
                a = self._acc[ph]
                a["prompt_tokens"] += p
                a["completion_tokens"] += c
                a["total_tokens"] += t
                a["calls"] += 1
        except Exception:
            pass

    def set_bucket(self, phase, prompt, completion, calls=None):
        """直接写某 phase 的累计量(用于 graphrag_log / hipporag_sqlite 一次性读出的总数)。"""
        try:
            if phase not in ("build", "qa"):
                return
            with self._lock:
                self._acc[phase] = {
                    "prompt_tokens": int(prompt or 0),
                    "completion_tokens": int(completion or 0),
                    "total_tokens": int((prompt or 0) + (completion or 0)),
                    "calls": int(calls) if calls is not None else -1,
                }
        except Exception:
            pass

    def dump(self, baseline, dataset, source, note=""):
        """原子落盘 token_usage/{baseline}_{dataset}.json。容错:失败只打印不抛。"""
        try:
            os.makedirs(TOKEN_DIR, exist_ok=True)
            with self._lock:
                build = dict(self._acc["build"])
                qa = dict(self._acc["qa"])
            rec = {
                "baseline": baseline, "dataset": dataset, "source": source,
                "build": build, "qa": qa,
                "build_qa_total_tokens": build["total_tokens"] + qa["total_tokens"],
                "note": note, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            path = f"{TOKEN_DIR}/{baseline}_{dataset}.json"
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(rec, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
            print(f"[token_meter] {baseline}/{dataset} build_tok={build['total_tokens']} "
                  f"qa_tok={qa['total_tokens']} -> {path}", flush=True)
        except Exception as e:
            print(f"[token_meter-error] dump failed: {e}", flush=True)


def parse_graphrag_indexing_log(log_path, chat_model="qwen3.6-27b"):
    """读 graphrag indexing-engine.log 最后一个 chat 模型 Metrics 块 → (prompt, completion, success_calls)。
    取'最后一个'是因为 FileHandler append 模式会累积多次建图的 flush,最后一个=本次。
    纯读取,任何异常都安全返回 (0,0,0)。"""
    try:
        raw = open(log_path, errors="ignore").read()
        marker = f"Metrics for openai/{chat_model}: "
        ends = [m.end() for m in re.finditer(re.escape(marker), raw)]
        if not ends:
            return 0, 0, 0
        start = raw.index("{", ends[-1])
        depth, block = 0, None
        for i in range(start, len(raw)):
            ch = raw[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    block = raw[start:i + 1]
                    break
        if block is None:
            return 0, 0, 0
        d = json.loads(block)
        return (int(d.get("prompt_tokens", 0) or 0),
                int(d.get("completion_tokens", 0) or 0),
                int(d.get("successful_response_count", 0) or 0))
    except Exception as e:
        print(f"[token_meter-error] parse graphrag log: {e}", flush=True)
        return 0, 0, 0


def scan_hipporag_sqlite(sqlite_path):
    """扫 HippoRAG llm_cache sqlite 的 metadata 列累加 → (prompt, completion, rows)。
    纯读取,任何异常都安全返回 (0,0,0)。metadata 每行形如 {prompt_tokens, completion_tokens, finish_reason}。"""
    try:
        if not os.path.exists(sqlite_path):
            return 0, 0, 0
        con = sqlite3.connect(sqlite_path)
        try:
            rows = con.execute("SELECT metadata FROM cache").fetchall()
        finally:
            con.close()
        pt = ct = n = 0
        for (md,) in rows:
            try:
                d = json.loads(md) if isinstance(md, str) else md
                pt += int(d.get("prompt_tokens", 0) or 0)
                ct += int(d.get("completion_tokens", 0) or 0)
                n += 1
            except Exception:
                pass
        return pt, ct, n
    except Exception as e:
        print(f"[token_meter-error] scan hipporag sqlite: {e}", flush=True)
        return 0, 0, 0
