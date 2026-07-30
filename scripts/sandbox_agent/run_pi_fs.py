#!/usr/bin/env python
"""pi harness × 文件系统 QA 跑批(端点: pi_fs)。

pi 是整装 coding agent(bash/read 等工具内置), 本脚本只做实验外壳:
  - 每题: 子进程 `pi -p` cwd=该档 flat 树, --tools read,bash 只读工具面(bash 理论可写,
    跑前对树 chmod a-w 物理兜底), 每题独立 session 目录;
  - 答案 = pi stdout; 计量 = session JSONL 逐次 usage 求和 + 工具调用计数;
  - retrieved(近似口径, 报告需注明): 从 session 的 bash 命令与 read 参数里正则抽树内文件路径,
    经 filemap_flat.json 映射回 chunk_ids —— pi 的自由 bash 无法像 QwenAgent read_doc 那样精确;
  - 越界审计: session 里出现树外绝对路径的命令会被记入 audit 字段(人工复核)。
LLM: --llm vllm（本地兼容服务）/ gateway（冒烟，GW_KEY 环境变量）。
跑: python3 run_pi_fs.py --dataset enterprise_N1144 [--llm gateway --limit_q 3 --workers 4]
"""
import os
import sys
import json
import re
import time
import argparse
import threading
import subprocess

W = os.environ.get("URAG_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PI = os.environ["PI_BIN"]

SYS_APPEND = ("You are answering research questions over an enterprise document corpus. The current "
              "directory contains the corpus as {source}/{doc}.md files (sources: slack, gmail, jira, "
              "confluence, google_drive, linear, github, fireflies, hubspot). Explore with your tools "
              "(grep/ls/cat via bash, read). Stay strictly inside the current directory. "
              "IMPORTANT - your context window is small (32k). Keep every tool output tiny: "
              "use `grep -l` / `grep -m3 -h` with narrow patterns, `head -40`, and read at most "
              "~60 lines of a file at a time. NEVER cat/read a whole file or grep without -m/-l. "
              "Answer ONLY from document contents; be concise and factual; cite the relative file "
              "paths you used. Print the final answer as plain text.")


def provider_args():
    if ARGS.llm == "gateway":
        return ["--provider", "gateway", "--model", os.environ.get("GW_MODEL", "deepseek-v4-flash")]
    return ["--provider", "vllm", "--model", "qwen3.6-27b"]


def parse_session(sdir, tree_root):
    """session JSONL -> usage/工具统计 + 读过的树内文件 + 越界审计。"""
    stat = {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "tool_calls": 0,
            "bash_cmds": 0}
    files, audit = set(), []
    fs = sorted((os.path.join(sdir, f) for f in os.listdir(sdir) if f.endswith(".jsonl")),
                key=os.path.getmtime) if os.path.isdir(sdir) else []
    path_re = re.compile(r"[\w./-]+\.md\b")
    for fp in fs:
        for l in open(fp, errors="ignore"):
            try:
                e = json.loads(l)
            except Exception:
                continue
            m = e.get("message", e)
            if not isinstance(m, dict):
                continue
            u = m.get("usage") or {}
            if u:
                stat["llm_calls"] += 1
                stat["prompt_tokens"] += int(u.get("input", u.get("input_tokens", 0)) or 0)
                stat["completion_tokens"] += int(u.get("output", u.get("output_tokens", 0)) or 0)
            if m.get("role") == "assistant":
                for c in (m.get("content") or []):
                    if not (isinstance(c, dict) and c.get("type") == "toolCall"):
                        continue
                    stat["tool_calls"] += 1
                    args = c.get("arguments") or {}
                    blob = json.dumps(args, ensure_ascii=False)
                    if c.get("name") == "bash":
                        stat["bash_cmds"] += 1
                    for p in path_re.findall(blob):
                        # pi 常用绝对路径访问树内文件, 取 flat/ 之后的相对段再映射
                        rp = p.split("/flat/", 1)[1] if "/flat/" in p else p.lstrip("./")
                        if os.path.exists(os.path.join(tree_root, rp)):
                            files.add(rp)
                    # 只审计真实泄漏面: 项目根 W 之内、但本档树之外的绝对路径(其他档语料/答案文件)
                    for ap in re.findall(r"(?<![\w./-])/[\w./-]+", blob):
                        real = os.path.realpath(ap)
                        if (real.startswith(os.path.realpath(W) + os.sep)
                                and not real.startswith(os.path.realpath(tree_root))
                                and os.path.exists(real)):
                            audit.append(blob[:200])
                            break
    return stat, sorted(files), audit


def run_one(q, tree_root, filemap, sess_root):
    sdir = os.path.join(sess_root, q["id"])
    os.makedirs(sdir, exist_ok=True)
    cmd = [PI, "-p", "--no-extensions", "--no-skills", "--no-context-files",
           "--tools", "read,bash", "--session-dir", sdir,
           "--append-system-prompt", SYS_APPEND, *provider_args(), q["question"]]
    t0 = time.time()
    try:
        # stdin 必须重定向: pi -p 会挂在未关闭的 stdin 管道上永远等待(实测卡300s+, /dev/null 后 2s 出答案)
        r = subprocess.run(cmd, cwd=tree_root, capture_output=True, text=True,
                           stdin=subprocess.DEVNULL,
                           timeout=ARGS.timeout, env={**os.environ, "NO_COLOR": "1",
                                                      "no_proxy": "localhost",
                                                      "NO_PROXY": "localhost"})
        answer, err = r.stdout.strip(), ("" if r.returncode == 0 else r.stderr[-300:])
    except subprocess.TimeoutExpired:
        answer, err = "", "timeout"
    stat, files, audit = parse_session(sdir, tree_root)
    chunks = sorted({c for f in files for c in filemap.get(f, [])})
    return {"answer": answer, "err": err, "retrieved": chunks, "read_files": files,
            "audit": audit, "total_sec": round(time.time() - t0, 2), **stat}


def main():
    ds = f"{W}/data/{ARGS.dataset}"
    tree_root = f"{W}/sandbox_agent/trees/{ARGS.dataset}/flat"
    filemap = json.load(open(f"{W}/sandbox_agent/trees/{ARGS.dataset}/filemap_flat.json"))
    questions = [json.loads(l) for l in open(f"{ds}/questions.jsonl")]
    if ARGS.sample_file:
        ids = set(json.load(open(ARGS.sample_file))["question_ids"])
        questions = [q for q in questions if q["id"] in ids]
    if ARGS.limit_q:
        questions = questions[:ARGS.limit_q]
    out_dir = f"{W}/results/pi_fs/{ARGS.dataset}"
    os.makedirs(out_dir, exist_ok=True)
    sess_root = f"{out_dir}/sessions"
    outp = f"{out_dir}/predictions.jsonl"
    done = {json.loads(l)["id"] for l in open(outp)} if os.path.exists(outp) else set()
    todo = [q for q in questions if q["id"] not in done]
    print(f"[pi_fs] {ARGS.dataset} llm={ARGS.llm} 待跑 {len(todo)} (已完成 {len(done)})", flush=True)

    fout, mout = open(outp, "a"), open(f"{out_dir}/meter.jsonl", "a")
    lock = threading.Lock()
    t0 = time.time(); cnt = [0]

    def work(q):
        r = run_one(q, tree_root, filemap, sess_root)
        with lock:
            fout.write(json.dumps({"id": q["id"], "predicted_answer": r["answer"],
                                   "retrieved_chunk_ids": r["retrieved"]}, ensure_ascii=False) + "\n")
            fout.flush()
            meta = {k: v for k, v in r.items() if k not in ("answer", "retrieved")}
            meta["id"] = q["id"]
            mout.write(json.dumps(meta, ensure_ascii=False) + "\n")
            mout.flush()
            cnt[0] += 1
            if r["audit"]:
                print(f"  [audit] {q['id']} 越界痕迹 {len(r['audit'])} 条", flush=True)
            if cnt[0] % 5 == 0 or cnt[0] == len(todo):
                el = time.time() - t0
                print(f"  {cnt[0]}/{len(todo)}  {el:.0f}s  {cnt[0]/max(el,1):.2f}q/s", flush=True)

    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(ARGS.workers) as ex:
        list(ex.map(work, todo))
    fout.close(); mout.close()
    print("[done]", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="enterprise_N1144")
    ap.add_argument("--llm", default="vllm", choices=["vllm", "gateway"])
    ap.add_argument("--sample_file", default="")
    ap.add_argument("--limit_q", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=600)
    ARGS = ap.parse_args()
    main()
