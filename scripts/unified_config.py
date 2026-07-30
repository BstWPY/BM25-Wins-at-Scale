"""统一实验契约 —— single source of truth。

四个 baseline (graphrag / LightRAG / HippoRAG / LinearRAG) 的适配脚本全部 import
这里的常量，改一处即全局生效，杜绝"各 baseline 默认值不同"导致的不公平。
凡是"必须统一"的维度都在这；方法内核差异（PPR/community/dense/sentence-graph）不在此。
"""
import os as __os

# ===== ① Chunking（pack_corpus.py 用） =====
CHUNK_SIZE = 1200            # tokens / chunk
CHUNK_OVERLAP = 100          # token overlap
ENCODING = "o200k_base"      # tiktoken，全 baseline 同一 tokenizer

# ===== ② 检索预算 =====
TOP_K = 5                    # 喂给 reader 的 chunk/passage 数（能调 top-k 的 baseline 统一到此）
# graphrag 的 local/global 是固有范式（选 B），不强凹 top-k，改用 token budget 对齐：
GRAPHRAG_MAX_CONTEXT_TOKENS = 2000   # ≈ TOP_K * 平均chunk(~110~400tok) 量级的检索内容预算

# ===== ③ Embedding（Qwen3-Embedding-0.6B，全 baseline 统一） =====
EMBED_MODEL_PATH = __os.path.join(
    __os.environ.get("MODEL_ROOT", "models"), "Qwen3-Embedding-0.6B"
)
EMBED_SERVED_NAME = "qwen3-embed-0.6b"
EMBED_DIM = 1024
EMBED_ENDPOINT = __os.environ["U_EMBED_ENDPOINT"]
EMBED_NORMALIZE = True
# Qwen3-Embedding 官方用法：query 侧加检索 instruction，doc 侧不加
EMBED_QUERY_INSTRUCTION = (
    "Instruct: Given a question, retrieve passages that help answer it\nQuery: "
)

# ===== ④ 建图 LLM（Qwen3.6-27B-FP8，全 baseline 统一） =====
LLM_MODEL_PATH = __os.path.join(
    __os.environ.get("MODEL_ROOT", "models"), "Qwen3.6-27B-FP8"
)
LLM_SERVED_NAME = "qwen3.6-27b"
LLM_ENDPOINT = __os.environ["U_LLM_ENDPOINT"]
LLM_API_KEY = "EMPTY"                             # vLLM 不校验

# ===== ⑤ 生成参数（建图 & QA reader 统一，可复现） =====
GEN_TEMPERATURE = 0.0
GEN_TOP_P = 1.0
GEN_MAX_TOKENS = 8192   # 上限设高让模型自然 stop、绝不人为截断。注意：不占显存（KV cache 由 max_model_len×max_num_seqs 启动时固定，max_tokens 只是生成闸门）
SEED = 42
# Qwen3.6 关 thinking（建图/QA 必须统一带）：OpenAI client 调用时传 extra_body=LLM_EXTRA_BODY
LLM_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}

# ===== ⑥ QA reader =====
# 先用本地 Qwen3.6-27B（= 建图同模型，零成本可复现）；后续抽样与 gpt-4o-mini 对比
READER_ENDPOINT = LLM_ENDPOINT
READER_MODEL = LLM_SERVED_NAME
# 2026-06-15 公平性修复(审计#3): reader prompt 必须跨范式统一, 否则 acc/qa-token 混入 reader 差异。
# 所有检索范式(HippoRAG/BM25/NaiveRAG)共用这一个 zero-shot reader, 只让"检索范式"成为唯一变量。
# (HippoRAG 原本回退到 musique 的 CoT+few-shot+Wikipedia前缀模板, 不公平; 已统一到此。sandbox 是 agentic 另算。)
READER_SYSTEM_PROMPT = (
    "You are a retrieval-based QA assistant. Answer the QUESTION using ONLY the information in the "
    "CONTEXT. Do not use outside knowledge or invent facts. If the CONTEXT does not contain enough "
    "information to answer, say so explicitly. Answer in the same language as the QUESTION; be concise."
)
def reader_user_msg(context, question):   # 统一 user 消息格式(BM25/NaiveRAG/HippoRAG 同口径)
    return f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"

# ===== 路径（可移植：随脚本所在 unified/ 目录走；env URAG_ROOT 可覆盖）=====
import os as _os
_HERE = _os.environ.get("URAG_ROOT") or _os.path.dirname(_os.path.abspath(__file__))
DATA_ROOT = _os.environ.get("U_DATA_ROOT") or _os.path.join(_HERE, "data")
RESULTS_ROOT = _os.environ.get("U_RESULTS_ROOT") or _os.path.join(_HERE, "results")

# 先跑的小数据集
SMALL_DATASETS = ["2wikimultihopqa", "hotpotqa"]
