#!/usr/bin/env python
"""codex harness × 文件系统 QA 跑批(端点: codex_fs)。pi_fs 的姊妹脚本。

codex(OpenAI Codex CLI) 是整装 coding agent, 本脚本只做实验外壳:
  - 每题: 子进程 `codex exec` cwd=该档 flat 树, 只读探索(bash/exec), 独立 CODEX_HOME session;
  - 后端: 本地 vLLM qwen3.6-27b, 经 CODEX_SHIM_URL 修正 responses 协议
    (developer->system role + 畸形 arguments 修复 + <think> 剥离 + 抽取最终答案到 sidecar);
  - 答案/命令: 从 shim 写出的 sidecar({ANSWER_DIR}/{md5(question)}.txt/.cmds) 读取
    (codex 自身对 qwen 的 <think> 输出解析后存空, 故绕过其 stdout/session, 用 shim 抓真值);
  - retrieved(近似口径): 从 .cmds 里正则抽树内 .md 路径, 经 filemap 映射回 chunk_ids。

前置: codex_shim.py 已启动且带 CODEX_SHIM_ANSWER_DIR=本脚本的 ANSWER_DIR。
跑: CODEX_SHIM_ANSWER_DIR="<answer-dir>" python3 run_codex_fs.py --dataset enterprise_N1144 \
      --sample_file results/query_sample_150.json --workers 4
"""
import os
import sys
import json
import re
import time
import hashlib
import argparse
import threading
import subprocess

W = os.environ.get("URAG_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODEX = os.environ["CODEX_BIN"]
SHIM_URL = os.environ["CODEX_SHIM_URL"]
ANSWER_DIR = os.environ["CODEX_SHIM_ANSWER_DIR"]

SYS_APPEND = ("You are answering ONE research question over an enterprise document corpus. The current "
              "directory contains the corpus as {source}/{doc}.md files (sources: slack, gmail, jira, "
              "confluence, google_drive, linear, github, fireflies, hubspot). Explore with your shell "
              "(grep/ls/cat). Stay strictly inside the current directory. "
              "IMPORTANT - keep every command output tiny: use `grep -l` / `grep -m3 -h` with narrow "
              "patterns and `head -40`; never cat a whole large file. "
              "CRITICAL OUTPUT DISCIPLINE: Do NOT write narration or plans in a text message. "
              "Every step is EITHER a shell command OR your single FINAL answer. Do not say things "
              "like 'let me look at X' as a message - just run the command. When you have enough "
              "evidence, produce your FINAL answer as one plain-text message: answer the question "
              "directly and completely from document contents, then list the relative file paths you used. "
              "Your final message must fully stand alone as the answer.")

PATH_RE = re.compile(r"[\w./-]+\.md\b")


def _qkey(qid):
    # 与 codex_shim 的 [[QID:...]] 标记约定一致, 直接用题号做 sidecar 文件名。
    return qid


def parse_cmds(cmds_file, tree_root, filemap):
    """从 shim 写出的 .cmds(每行一个 function_call arguments JSON) 抽访问过的树内文件 -> chunk_ids。"""
    files, ncmd = set(), 0
    if not os.path.exists(cmds_file):
        return [], 0
    for line in open(cmds_file, errors="ignore"):
        ncmd += 1
        for p in PATH_RE.findall(line):
            rp = p.split("/flat/", 1)[1] if "/flat/" in p else p.lstrip("./")
            if rp in filemap:
                files.add(rp)
    chunks = sorted({c for f in files for c in filemap.get(f, [])})
    return chunks, ncmd


def run_one(q, tree_root, filemap, sess_root):
    qk = _qkey(q["id"])
    ans_file = os.path.join(ANSWER_DIR, qk + ".txt")
    cmds_file = os.path.join(ANSWER_DIR, qk + ".cmds")
    for f in (ans_file, cmds_file):
        try:
            os.remove(f)
        except OSError:
            pass
    home = os.path.join(sess_root, q["id"] + "_home")
    os.makedirs(home, exist_ok=True)
    prompt = f"[[QID:{q['id']}]]\n" + SYS_APPEND + "\n\n" + q["question"]
    cmd = [CODEX, "exec",
           "-c", "model_provider=vllm",
           "-c", "model=qwen3.6-27b",
           "-c", f'model_providers.vllm.name="local vllm qwen"',
           "-c", f'model_providers.vllm.base_url="{SHIM_URL}"',
           "-c", 'model_providers.vllm.wire_api="responses"',
           "-c", "model_providers.vllm.request_max_retries=2",
           "-c", 'preferred_auth_method="apikey"',
           "--skip-git-repo-check",
           "--dangerously-bypass-approvals-and-sandbox",
           "-C", tree_root,
           prompt]
    t0 = time.time()
    err = ""
    try:
        r = subprocess.run(cmd, cwd=tree_root, capture_output=True, text=True,
                           stdin=subprocess.DEVNULL, timeout=ARGS.timeout,
                           env={**os.environ, "NO_COLOR": "1", "CODEX_HOME": home,
                                "OPENAI_API_KEY": "dummy",
                                "no_proxy": "localhost",
                                "NO_PROXY": "localhost"})
        if r.returncode != 0:
            err = r.stderr[-300:]
    except subprocess.TimeoutExpired:
        err = "timeout"
    answer = ""
    if os.path.exists(ans_file):
        answer = open(ans_file, errors="ignore").read().strip()
    chunks, ncmd = parse_cmds(cmds_file, tree_root, filemap)
    return {"answer": answer, "err": err, "retrieved": chunks,
            "cmds": ncmd, "total_sec": round(time.time() - t0, 2)}


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
    out_dir = f"{W}/results/codex_fs/{ARGS.dataset}"
    os.makedirs(out_dir, exist_ok=True)
    sess_root = f"{out_dir}/sessions"
    os.makedirs(sess_root, exist_ok=True)
    os.makedirs(ANSWER_DIR, exist_ok=True)
    outp = f"{out_dir}/predictions.jsonl"
    done = {json.loads(l)["id"] for l in open(outp)} if os.path.exists(outp) else set()
    todo = [q for q in questions if q["id"] not in done]
    print(f"[codex_fs] {ARGS.dataset} 待跑 {len(todo)} (已完成 {len(done)}) shim={SHIM_URL}", flush=True)

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
    ap.add_argument("--sample_file", default="")
    ap.add_argument("--limit_q", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=900)
    ARGS = ap.parse_args()
    main()
