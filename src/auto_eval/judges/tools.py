"""裁判可用的工具：web_search / fetch_page / calculate / python_run。

- web_search：Tavily/SerpAPI/Bing 三选一联网搜索；
- fetch_page：抓取网页正文，深挖核实；
- calculate：安全求值算术表达式（AST 白名单），核查计算题；
- python_run：受限执行 Python（子进程+超时），核查编程/逻辑题（默认关，注意安全）。
"""
from __future__ import annotations

import ast
import operator
import os
import subprocess
import sys
import time
from functools import partial

import httpx

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "联网搜索以核实事实或查找权威答案。事实/时新信息核查首选。",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "搜索查询词"}},
            "required": ["query"],
        },
    },
}

FETCH_PAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "fetch_page",
        "description": "抓取指定网页正文，深入核实搜索结果中的具体细节。输入完整 URL。",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "要抓取的网页 URL"}},
            "required": ["url"],
        },
    },
}

CALCULATE_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "安全求值算术表达式（仅支持数字与 + - * / ** % // 和括号），用于核查计算题答案。",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "算术表达式，如 17*24"}},
            "required": ["expression"],
        },
    },
}

PYTHON_RUN_TOOL = {
    "type": "function",
    "function": {
        "name": "python_run",
        "description": "执行一段 Python 代码并返回 stdout，用于核查编程题/逻辑（如运行样例代码看输出）。仅必要时使用。",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "要执行的 Python 代码"}},
            "required": ["code"],
        },
    },
}

_KEY_ENV = {"tavily": "TAVILY_API_KEY", "serpapi": "SERPAPI_API_KEY", "bing": "BING_API_KEY"}


def _search_tavily(query: str, topk: int, key: str) -> list[str]:
    r = httpx.post(
        "https://api.tavily.com/search",
        json={"api_key": key, "query": query, "max_results": topk},
        timeout=20.0,
    )
    r.raise_for_status()
    data = r.json()
    return [f"{x.get('title', '')}：{x.get('content', '')}" for x in data.get("results", [])][:topk]


def _search_serpapi(query: str, topk: int, key: str) -> list[str]:
    r = httpx.get(
        "https://serpapi.com/search",
        params={"engine": "google", "q": query, "api_key": key},
        timeout=20.0,
    )
    r.raise_for_status()
    data = r.json()
    out: list[str] = []
    box = data.get("answer_box") or {}
    if box.get("answer"):
        out.append(f"答案盒：{box['answer']}")
    if box.get("snippet"):
        out.append(f"摘要：{box['snippet']}")
    for x in (data.get("organic_results") or [])[:topk]:
        out.append(f"{x.get('title', '')}：{x.get('snippet', '')}")
    return out[:topk]


def _search_bing(query: str, topk: int, key: str) -> list[str]:
    r = httpx.get(
        "https://api.bing.microsoft.com/v7.0/search",
        headers={"Ocp-Apim-Subscription-Key": key},
        params={"q": query, "count": topk},
        timeout=20.0,
    )
    r.raise_for_status()
    data = r.json()
    webs = (data.get("webPages") or {}).get("value", [])
    return [f"{x.get('name', '')}：{x.get('snippet', '')}" for x in webs][:topk]


_SEARCH_CACHE: dict[str, tuple[float, list[str]]] = {}
_SEARCH_TTL = 3600  # 秒，相同 query 在 TTL 内复用结果（去重，省 API 调用）


def web_search(query: str, provider: str | None = None, topk: int = 3) -> list[str]:
    if not provider:
        return []
    ck = f"{provider}:{query}:{topk}"
    now = time.monotonic()
    if ck in _SEARCH_CACHE and now - _SEARCH_CACHE[ck][0] < _SEARCH_TTL:
        return _SEARCH_CACHE[ck][1]
    key = os.environ.get(_KEY_ENV.get(provider, ""))
    res: list[str] = []
    if key:
        try:
            if provider == "tavily":
                res = _search_tavily(query, topk, key)
            elif provider == "serpapi":
                res = _search_serpapi(query, topk, key)
            elif provider == "bing":
                res = _search_bing(query, topk, key)
        except Exception:
            pass
    _SEARCH_CACHE[ck] = (now, res)
    return res


def fetch_page(url: str, max_chars: int = 2000) -> str:
    try:
        r = httpx.get(
            url, timeout=15.0, follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (auto-eval-agent)"},
        )
        r.raise_for_status()
        text = r.text
        import re
        text = re.sub(r"<script.*?</script>", " ", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_chars]
    except Exception as e:
        return f"(抓取失败: {e})"


_SAFE_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def calculate(expression: str) -> str:
    """安全求值算术表达式（AST 白名单，无名字/属性/调用），核查计算题。"""
    try:
        node = ast.parse(expression, mode="eval").body
    except SyntaxError as e:
        return f"(表达式语法错误: {e})"

    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _SAFE_OPS:
            return _SAFE_OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _SAFE_OPS:
            return _SAFE_OPS[type(n.op)](ev(n.operand))
        raise ValueError("不支持的表达式元素")

    try:
        return str(ev(node))
    except Exception as e:
        return f"(求值失败: {e})"


def python_run(code: str, timeout: int = 5) -> str:
    """受限执行 Python（独立子进程 + 超时 + 输出截断），核查编程/逻辑题。

    安全说明：在隔离子进程里跑裁判决定要跑的代码；仅用于可信评测环境，
    生产环境应进一步沙箱化（容器/无文件系统等）。
    """
    try:
        r = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=timeout,
        )
        out = (r.stdout or "").strip()
        err = (r.stderr or "").strip()
        if r.returncode != 0:
            return f"(退出码 {r.returncode}) {err[:300]}"
        return out[:500] or "(无输出)"
    except subprocess.TimeoutExpired:
        return "(执行超时)"
    except Exception as e:
        return f"(执行失败: {e})"


def answer_lookup(query: str) -> str:
    """可扩展钩子：接入自有知识库/权威源。默认无实现。"""
    return ""


def build_tools(
    web_search_enabled: bool,
    search_provider,
    search_topk: int,
    fetch_enabled: bool,
    calculate_enabled: bool = True,
    python_enabled: bool = False,
):
    """返回 (工具定义列表, 名→可调用函数映射)。web_search 已绑定 provider/topk。"""
    defs, fmap = [], {}
    if web_search_enabled:
        defs.append(WEB_SEARCH_TOOL)
        fmap["web_search"] = partial(web_search, provider=search_provider, topk=search_topk)
    if fetch_enabled:
        defs.append(FETCH_PAGE_TOOL)
        fmap["fetch_page"] = fetch_page
    if calculate_enabled:
        defs.append(CALCULATE_TOOL)
        fmap["calculate"] = calculate
    if python_enabled:
        defs.append(PYTHON_RUN_TOOL)
        fmap["python_run"] = python_run
    return defs, fmap
