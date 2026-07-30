#!/usr/bin/env python
"""codex(responses API) <-> vLLM 之间的薄 sanitizing shim。

背景: codex CLI 只支持 wire_api="responses"。codex 会在 input[] 里发 role=="developer"
(OpenAI 新惯例, 放系统指令)和偶发 role=="tool", 而 vLLM 的 /v1/responses 实现只认
user/system(developer/tool 直接 400 "Unexpected message role"), 这是 codex "死活跑不通"的根因。

本 shim 唯一职责: 把 input[] 里 type=="message" 项的 role 归一化:
  developer -> system, tool -> (合并进后续或转 user), 其余(user/system/assistant)原样。
不碰 function_call / function_call_output 项(实测 vLLM 原生支持)。透明流式转发。

运行前设置 CODEX_SHIM_UPSTREAM 和 CODEX_SHIM_PORT。
"""
import json, os, sys, re
from aiohttp import web, ClientSession, ClientTimeout

UPSTREAM = os.environ["CODEX_SHIM_UPSTREAM"]
PORT = int(os.environ["CODEX_SHIM_PORT"])
LOGF = os.environ.get("CODEX_SHIM_LOG", "")
HOP = {"host", "content-length", "content-encoding", "transfer-encoding", "connection"}

ROLE_MAP = {"developer": "system"}


def _log(msg):
    if LOGF:
        try:
            with open(LOGF, "a") as f:
                f.write(msg + "\n")
        except Exception:
            pass


def _texts_of(content):
    """responses content: [{type:input_text|output_text|text, text:..}, ...] 或 str。抽纯文本。"""
    if isinstance(content, str):
        return [content]
    out = []
    if isinstance(content, list):
        for b in content:
            if isinstance(b, dict) and isinstance(b.get("text"), str):
                out.append(b["text"])
    return out


def _sanitize(body: bytes) -> bytes:
    """把 input[] 里 role in (developer,system) 的 message 文本抽出合并进 top-level instructions,
    并从 input 移除 —— 解决 vLLM 'System message must be at the beginning' 400。
    role==tool 的 message 转 user。function_call / function_call_output 原样保留。"""
    try:
        d = json.loads(body)
    except Exception:
        return body
    inp = d.get("input")
    if not isinstance(inp, list):
        return body
    _dec = json.JSONDecoder()
    sys_chunks, kept, seen_roles, fixed_args = [], [], set(), 0
    for item in inp:
        if isinstance(item, dict) and item.get("type") == "message":
            r = item.get("role")
            seen_roles.add(r)
            if r in ("developer", "system"):
                sys_chunks.extend(_texts_of(item.get("content")))
                continue  # 移出 input, 合并进 instructions
            if r == "tool":
                item["role"] = "user"
        # 修复 qwen 偶发生成的畸形 function_call.arguments(带 trailing 字符 -> vLLM 400):
        # 用 raw_decode 只取第一个合法 JSON object, 丢弃尾部垃圾, 再序列化回去。
        if isinstance(item, dict) and item.get("type") == "function_call":
            a = item.get("arguments")
            if isinstance(a, str) and a:
                try:
                    json.loads(a)
                except Exception:
                    try:
                        obj, _ = _dec.raw_decode(a.lstrip())
                        item["arguments"] = json.dumps(obj)
                        fixed_args += 1
                    except Exception:
                        item["arguments"] = "{}"
                        fixed_args += 1
        kept.append(item)
    if sys_chunks:
        merged = ([d["instructions"]] if isinstance(d.get("instructions"), str) and d["instructions"] else []) + sys_chunks
        d["instructions"] = "\n\n".join(x for x in merged if x)
    d["input"] = kept
    _log(f"[sanitize] roles_in={sorted(x for x in seen_roles if x)} moved_sys={len(sys_chunks)} kept={len(kept)} fixed_args={fixed_args}")
    return json.dumps(d).encode()


DUMP = os.environ.get("CODEX_SHIM_DUMP", "")
STRIP_THINK = os.environ.get("CODEX_SHIM_STRIP_THINK", "1") != "0"
ANSWER_DIR = os.environ.get("CODEX_SHIM_ANSWER_DIR", "")  # 设置则把每题最终答案写 {md5(question)}.txt

import hashlib


_QID_RE = re.compile(r"\[\[QID:([^\]]+)\]\]")


def _question_key(body_dict):
    """从请求 input[] 的 user 文本里抽 [[QID:xxx]] 标记做 key(run_codex_fs 注入, 稳过 md5 匹配)。
    无标记时回退到 md5(最后一个非环境 user 文本)。"""
    inp = body_dict.get("input")
    if not isinstance(inp, list):
        return None
    cand = None
    for item in inp:
        if isinstance(item, dict) and item.get("type") == "message" and item.get("role") == "user":
            txt = "\n".join(_texts_of(item.get("content")))
            m = _QID_RE.search(txt)
            if m:
                return m.group(1).strip()
            if txt.strip().startswith("<environment_context>"):
                continue
            cand = txt
    return hashlib.md5(cand.encode("utf-8")).hexdigest() if cand else None

_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def _strip_think(text: str) -> str:
    # 去成对 <think>..</think>; qwen 常省略开标签, 只剩 '推理...</think>正文' -> 砍到最后一个 </think>
    text = _THINK_RE.sub("", text)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.lstrip("\n")


def _clean_msg_item(item):
    """就地清洗一个 message 型 output item 的 content[].text 里的 <think>。"""
    if not isinstance(item, dict) or item.get("type") != "message":
        return
    for c in item.get("content", []) or []:
        if isinstance(c, dict) and isinstance(c.get("text"), str):
            c["text"] = _strip_think(c["text"])


def _rewrite_sse(raw: bytes) -> bytes:
    """把 output_text.delta 流里的 <think> 去掉: 按 item 累计全文, 清洗后塞回该 item 首个 delta,
    其余 delta 置空; 其它事件(function_call 等)原样。解决 codex 因 <think> 存空答案的问题。"""
    try:
        text = raw.decode("utf-8")
    except Exception:
        return raw
    blocks = text.split("\n\n")
    full = {}  # item_id -> concatenated delta
    for blk in blocks:
        for line in blk.splitlines():
            if line.startswith("data:"):
                try:
                    d = json.loads(line[5:].strip())
                except Exception:
                    continue
                if d.get("type") == "response.output_text.delta":
                    iid = d.get("item_id", "")
                    full[iid] = full.get(iid, "") + d.get("delta", "")
    if not full:
        return raw
    cleaned = {iid: _strip_think(t) for iid, t in full.items()}
    emitted = set()
    out_blocks = []
    for blk in blocks:
        newlines = []
        for line in blk.splitlines():
            if line.startswith("data:"):
                try:
                    d = json.loads(line[5:].strip())
                except Exception:
                    newlines.append(line); continue
                t = d.get("type")
                if t == "response.output_text.delta":
                    iid = d.get("item_id", "")
                    if iid not in emitted:
                        d["delta"] = cleaned.get(iid, ""); emitted.add(iid)
                    else:
                        d["delta"] = ""
                    newlines.append("data: " + json.dumps(d))
                elif t == "response.output_text.done":
                    d["text"] = cleaned.get(d.get("item_id", ""), _strip_think(d.get("text", "")))
                    newlines.append("data: " + json.dumps(d))
                elif t in ("response.output_item.done", "response.output_item.added"):
                    _clean_msg_item(d.get("item"))
                    newlines.append("data: " + json.dumps(d))
                elif t in ("response.completed", "response.incomplete"):
                    for it in (d.get("response", {}) or {}).get("output", []) or []:
                        _clean_msg_item(it)
                    newlines.append("data: " + json.dumps(d))
                else:
                    newlines.append(line)
            else:
                newlines.append(line)
        out_blocks.append("\n".join(newlines))
    return ("\n\n".join(out_blocks)).encode()


def _extract_commands(raw: bytes):
    """从(已清洗的)SSE 抓所有 function_call 的 arguments 文本(codex 的 exec 命令), 供 harness 解析访问过的文件。"""
    try:
        text = raw.decode("utf-8")
    except Exception:
        return []
    cmds = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        try:
            d = json.loads(line[5:].strip())
        except Exception:
            continue
        if d.get("type") in ("response.completed", "response.incomplete"):
            for it in (d.get("response", {}) or {}).get("output", []) or []:
                if isinstance(it, dict) and it.get("type") == "function_call":
                    a = it.get("arguments")
                    if isinstance(a, str):
                        cmds.append(a)
    return cmds


def _extract_final_answer(raw: bytes):
    """从(已清洗的)SSE 抓 response.completed 里最后一个 message 的纯文本。"""
    try:
        text = raw.decode("utf-8")
    except Exception:
        return None
    ans = None
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        try:
            d = json.loads(line[5:].strip())
        except Exception:
            continue
        if d.get("type") in ("response.completed", "response.incomplete"):
            for it in (d.get("response", {}) or {}).get("output", []) or []:
                if isinstance(it, dict) and it.get("type") == "message":
                    t = "".join(c.get("text", "") for c in it.get("content", []) if isinstance(c, dict))
                    if t.strip():
                        ans = t.strip()
    return ans


async def handle(request):
    path_qs = request.rel_url.path_qs
    body = await request.read()
    is_resp = request.rel_url.path.endswith("/responses") and body
    qkey = None
    if is_resp:
        try:
            qkey = _question_key(json.loads(body))
        except Exception:
            qkey = None
        body = _sanitize(body)
    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP}
    timeout = ClientTimeout(total=None, sock_read=3600)
    async with ClientSession(timeout=timeout) as sess:
        async with sess.request(request.method, UPSTREAM + path_qs,
                                data=body, headers=headers) as up:
            out_headers = {k: v for k, v in up.headers.items() if k.lower() not in HOP}
            ctype = up.headers.get("Content-Type", "")
            # SSE 且需去 think: 全量缓冲后重写(批处理场景延迟无所谓, 换取答案可提取)
            if is_resp and STRIP_THINK and "event-stream" in ctype:
                raw = await up.read()
                raw = _rewrite_sse(raw)
                if ANSWER_DIR and qkey:
                    try:
                        os.makedirs(ANSWER_DIR, exist_ok=True)
                        ans = _extract_final_answer(raw)
                        if ans:
                            with open(os.path.join(ANSWER_DIR, qkey + ".txt"), "w") as f:
                                f.write(ans)
                        cmds = _extract_commands(raw)
                        if cmds:
                            with open(os.path.join(ANSWER_DIR, qkey + ".cmds"), "a") as f:
                                for c in cmds:
                                    f.write(c.replace("\n", " ") + "\n")
                    except Exception:
                        pass
                if DUMP:
                    try:
                        with open(DUMP, "a") as f:
                            f.write("\n===REQ===\n" + body.decode("utf-8", "replace") + "\n===RESP===\n" + raw.decode("utf-8", "replace"))
                    except Exception:
                        pass
                resp = web.StreamResponse(status=up.status, headers=out_headers)
                await resp.prepare(request)
                await resp.write(raw)
                await resp.write_eof()
                return resp
            resp = web.StreamResponse(status=up.status, headers=out_headers)
            await resp.prepare(request)
            async for chunk in up.content.iter_any():
                await resp.write(chunk)
            await resp.write_eof()
            return resp


app = web.Application(client_max_size=1024 ** 3)
app.router.add_route("*", "/{tail:.*}", handle)

if __name__ == "__main__":
    print(f"[codex_shim] :{PORT} -> {UPSTREAM}  (developer->system role fix)", flush=True)
    web.run_app(app, host="localhost", port=PORT, access_log=None)
