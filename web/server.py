"""FastAPI Web Server for Text2SQL Agent.

启动方式:
    uv run python web/server.py

SSE 事件协议 (data 字段为 JSON 字符串):
    {"type": "node_start", "node": "intent_router", "label": "🔀 意图路由"}
    {"type": "node_end",   "node": "intent_router"}
    {"type": "token",      "content": "根据查询..."}
    {"type": "sql_step",   "step_id": "1", "label": "查询用户总数", "sql": "SELECT ..."}
    {"type": "sql_confirm","sql": "SELECT ...", "session_id": "xxx"}
    {"type": "error",      "message": "..."}
    {"type": "done"}
"""

import asyncio
import json
import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiosqlite

# Windows UTF-8 console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

# ── 添加项目根路径 ────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from agent.graph import config as agent_config, initialize_async_graph_for_web, _NODE_LABELS, _CHECKPOINT_DB
from agent import sql_confirm
from agent import sql_step_emitter
from langchain_core.messages import HumanMessage, AIMessage

logger = logging.getLogger(__name__)

# 会话元数据 DB（复用 LangGraph 的 checkpoints.db）
_SESSIONS_DB = str(_CHECKPOINT_DB)

async def _init_sessions_db():
    """确保 sessions / constraints 表存在。"""
    async with aiosqlite.connect(_SESSIONS_DB) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                thread_id  TEXT PRIMARY KEY,
                name       TEXT NOT NULL DEFAULT '新对话',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS constraints (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id  TEXT NOT NULL,
                content    TEXT NOT NULL,
                enabled    INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL
            )
        """)
        await db.commit()

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

async def _db_create_session(thread_id: str, name: str) -> dict:
    now = _now_iso()
    async with aiosqlite.connect(_SESSIONS_DB) as db:
        await db.execute(
            "INSERT OR IGNORE INTO sessions (thread_id, name, created_at, updated_at) VALUES (?,?,?,?)",
            (thread_id, name, now, now),
        )
        await db.commit()
    return {"thread_id": thread_id, "name": name, "created_at": now, "updated_at": now}

async def _db_list_sessions() -> list[dict]:
    async with aiosqlite.connect(_SESSIONS_DB) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT thread_id, name, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]

async def _db_delete_session(thread_id: str) -> None:
    """删除一个会话及其所有关联数据（checkpoints / constraints / memory / plans）。"""
    # 1. checkpoints.db — sessions, constraints, LangGraph checkpoints/writes
    async with aiosqlite.connect(_SESSIONS_DB) as db:
        await db.execute("DELETE FROM sessions WHERE thread_id = ?", (thread_id,))
        await db.execute("DELETE FROM constraints WHERE thread_id = ?", (thread_id,))
        # LangGraph SqliteSaver tables
        for tbl in ("checkpoints", "checkpoint_writes", "checkpoint_blobs"):
            try:
                await db.execute(f"DELETE FROM {tbl} WHERE thread_id = ?", (thread_id,))
            except Exception:
                pass
        await db.commit()

    # 2. memory_cards.db — separate DB file next to checkpoints.db
    _memory_db = str(_ROOT / "memory_cards.db")
    try:
        async with aiosqlite.connect(_memory_db) as db:
            await db.execute("DELETE FROM memory_cards WHERE thread_id = ?", (thread_id,))
            await db.execute("DELETE FROM memory_cursors WHERE thread_id = ?", (thread_id,))
            await db.commit()
    except Exception:
        pass

    # 3. report/sessions/ — plan JSON dirs that belong to this thread
    plans_dir = _ROOT / "report" / "sessions"
    if plans_dir.exists():
        import shutil
        for task_dir in plans_dir.iterdir():
            json_file = task_dir / "plan.json"
            if json_file.exists():
                try:
                    data = json.loads(json_file.read_text(encoding="utf-8"))
                    if data.get("thread_id", "") == thread_id:
                        shutil.rmtree(task_dir, ignore_errors=True)
                except Exception:
                    pass


async def _db_update_session(thread_id: str, name: str | None = None, touch: bool = False):
    sets, params = [], []
    if name is not None:
        sets.append("name = ?");  params.append(name)
    if touch or name is not None:
        sets.append("updated_at = ?"); params.append(_now_iso())
    if not sets:
        return
    params.append(thread_id)
    async with aiosqlite.connect(_SESSIONS_DB) as db:
        await db.execute(f"UPDATE sessions SET {', '.join(sets)} WHERE thread_id = ?", params)
        await db.commit()

# ── 服务器全局状态 ────────────────────────────────────────────────────────────
_web_graph = None  # 启动时初始化的异步图
_session_queues: dict[str, asyncio.Queue] = {}   # session_id → SSE 事件队列
_confirm_events: dict[str, asyncio.Event] = {}   # session_id → 等待用户确认的事件
_confirm_results: dict[str, tuple] = {}          # session_id → (action, reason)


# ── 请求/响应模型 ────────────────────────────────────────────────────────────

class ConfirmRequest(BaseModel):
    session_id: str
    action: str          # "execute" | "skip"
    reason: Optional[str] = ""


class NewSessionResponse(BaseModel):
    thread_id: str
    name: str
    created_at: str
    updated_at: str


class PatchSessionRequest(BaseModel):
    name: str


class AddConstraintRequest(BaseModel):
    content: str


class PatchConstraintRequest(BaseModel):
    enabled: bool


# ── 应用生命周期 ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务器启动时初始化异步图和会话 DB，关闭时清理资源。"""
    global _web_graph
    await _init_sessions_db()
    logger.info("Sessions DB ready")
    logger.info("Initializing web async graph...")
    _web_graph = await initialize_async_graph_for_web()
    logger.info("Web async graph ready")
    yield
    logger.info("Shutting down web server")


app = FastAPI(title="Text2SQL Agent Web UI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── SQL 确认 Hook 工厂 ───────────────────────────────────────────────────────

def _make_sql_confirm_hook(
    session_id: str,
    loop: asyncio.AbstractEventLoop,
    queue: asyncio.Queue,
) -> callable:
    """创建 Web 模式下的 SQL 确认 hook，从工作线程调用时阻塞等待用户确认。"""

    def hook(sql: str) -> tuple[str, str]:
        async def _async_confirm():
            event = asyncio.Event()
            _confirm_events[session_id] = event
            # 向 SSE 队列推送 sql_confirm 事件
            await queue.put(json.dumps({
                "type": "sql_confirm",
                "sql": sql,
                "session_id": session_id,
            }))
            await event.wait()  # 等待前端用户响应
            return _confirm_results.pop(session_id, ("execute", ""))

        # 从线程池跨越到 asyncio event loop
        future = asyncio.run_coroutine_threadsafe(_async_confirm(), loop)
        return future.result(timeout=300)  # 5 分钟超时

    return hook


# ── API 端点 ──────────────────────────────────────────────────────────────────

@app.get("/api/sessions")
async def list_sessions():
    """列出所有历史会话（按最近活跃倒序）。"""
    return {"sessions": await _db_list_sessions()}


@app.post("/api/sessions", response_model=NewSessionResponse)
async def create_session():
    """创建新会话，持久化到 DB，返回会话元数据。"""
    thread_id = str(uuid.uuid4())
    result = await _db_create_session(thread_id, "新对话")
    return result


@app.patch("/api/sessions/{thread_id}")
async def rename_session(thread_id: str, body: PatchSessionRequest):
    """重命名会话。"""
    await _db_update_session(thread_id, name=body.name)
    return {"ok": True}


@app.delete("/api/sessions/{thread_id}")
async def delete_session(thread_id: str):
    """删除会话及其所有历史数据（checkpoints / constraints / memory / plans）。"""
    await _db_delete_session(thread_id)
    return {"ok": True}


@app.get("/api/sessions/{thread_id}/history")
async def get_history(thread_id: str):
    """获取指定会话的历史消息。每轮对话只返回 SQL（如有）和最终 AI 回答。"""
    if _web_graph is None:
        raise HTTPException(status_code=503, detail="Graph not initialized")
    try:
        config_with_thread = {"configurable": {"thread_id": thread_id}}
        snapshot = await _web_graph.aget_state(config_with_thread)
        if not snapshot:
            return {"messages": []}

        raw = list(snapshot.values.get("messages", []))

        # 识别真正的用户提问：第一条，或紧跟无 tool_calls 的 AIMessage（上一轮的最终回答）
        real_user_indices = []
        for i, msg in enumerate(raw):
            if isinstance(msg, HumanMessage):
                if i == 0:
                    real_user_indices.append(i)
                else:
                    prev = raw[i - 1]
                    if (isinstance(prev, AIMessage)
                            and prev.content
                            and not getattr(prev, "tool_calls", None)):
                        real_user_indices.append(i)

        messages = []
        for k, ui in enumerate(real_user_indices):
            messages.append({"role": "user", "content": raw[ui].content})

            end = real_user_indices[k + 1] if k + 1 < len(real_user_indices) else len(raw)
            turn_msgs = raw[ui + 1:end]

            # 从该轮消息中提取 SQL：
            # 1) AIMessage with tool_calls → simple_query 的工具调用 SQL
            # 2) AIMessage content 以 __sql__: 开头 → complex/data_analysis 注入的 SQL 标记
            turn_sqls = []
            for m in turn_msgs:
                if isinstance(m, AIMessage):
                    if getattr(m, "tool_calls", None):
                        for tc in m.tool_calls:
                            args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                            q = args.get("query", "") or args.get("sql", "")
                            if q:
                                turn_sqls.append(q)
                    elif isinstance(m.content, str) and m.content.startswith("__sql__:"):
                        # Format: __sql__:{step_id}:{label}:{sql}
                        parts = m.content.split(":", 3)
                        if len(parts) == 4:
                            turn_sqls.append(parts[3])

            # 最后一条有内容且无 tool_calls 的 AIMessage 为最终回答
            last_ai = None
            for m in reversed(turn_msgs):
                if isinstance(m, AIMessage) and m.content and not getattr(m, "tool_calls", None):
                    last_ai = m.content
                    break

            if last_ai:
                messages.append({
                    "role": "assistant",
                    "content": last_ai,
                    "sqls": turn_sqls,  # list of SQL strings (may be empty for general_chat)
                })

        return {"messages": messages}
    except Exception as e:
        logger.error(f"Failed to get history: {e}")
        return {"messages": []}


@app.get("/api/sessions/{thread_id}/plans")
async def get_session_plans(thread_id: str):
    """获取属于指定会话的所有任务链路（SessionPlan）。"""
    plans_dir = _ROOT / "report" / "sessions"
    result = []
    if plans_dir.exists():
        for task_dir in sorted(plans_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            json_file = task_dir / "plan.json"
            if json_file.exists():
                try:
                    data = json.loads(json_file.read_text(encoding="utf-8"))
                    if data.get("thread_id", "") == thread_id:
                        result.append(data)
                except Exception:
                    continue
    return {"plans": result}


# ── Constraints CRUD ──────────────────────────────────────────────────────────

@app.get("/api/sessions/{thread_id}/constraints")
async def list_constraints(thread_id: str):
    """获取指定会话的所有禁令。"""
    async with aiosqlite.connect(_SESSIONS_DB) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, content, enabled, created_at FROM constraints WHERE thread_id = ? ORDER BY id",
            (thread_id,),
        ) as cur:
            rows = await cur.fetchall()
    return {"constraints": [dict(r) for r in rows]}


@app.post("/api/sessions/{thread_id}/constraints", status_code=201)
async def add_constraint(thread_id: str, body: AddConstraintRequest):
    """向指定会话添加一条禁令。"""
    content = body.content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="content cannot be empty")
    now = _now_iso()
    async with aiosqlite.connect(_SESSIONS_DB) as db:
        cur = await db.execute(
            "INSERT INTO constraints (thread_id, content, enabled, created_at) VALUES (?,?,1,?)",
            (thread_id, content, now),
        )
        await db.commit()
        new_id = cur.lastrowid
    return {"id": new_id, "thread_id": thread_id, "content": content, "enabled": 1, "created_at": now}


@app.patch("/api/sessions/{thread_id}/constraints/{constraint_id}")
async def toggle_constraint(thread_id: str, constraint_id: int, body: PatchConstraintRequest):
    """启用/禁用指定禁令。"""
    async with aiosqlite.connect(_SESSIONS_DB) as db:
        await db.execute(
            "UPDATE constraints SET enabled = ? WHERE id = ? AND thread_id = ?",
            (1 if body.enabled else 0, constraint_id, thread_id),
        )
        await db.commit()
    return {"ok": True}


@app.delete("/api/sessions/{thread_id}/constraints/{constraint_id}")
async def delete_constraint(thread_id: str, constraint_id: int):
    """删除指定禁令。"""
    async with aiosqlite.connect(_SESSIONS_DB) as db:
        await db.execute(
            "DELETE FROM constraints WHERE id = ? AND thread_id = ?",
            (constraint_id, thread_id),
        )
        await db.commit()
    return {"ok": True}


async def _load_enabled_constraints(thread_id: str) -> list[str]:
    """加载指定会话中启用的禁令列表。"""
    async with aiosqlite.connect(_SESSIONS_DB) as db:
        async with db.execute(
            "SELECT content FROM constraints WHERE thread_id = ? AND enabled = 1 ORDER BY id",
            (thread_id,),
        ) as cur:
            rows = await cur.fetchall()
    return [row[0] for row in rows]



@app.get("/api/chat/stream")
async def chat_stream(query: str, thread_id: str, session_id: str):
    """SSE 流式端点：执行查询并以 SSE 事件推送节点更新和 token。"""

    if _web_graph is None:
        raise HTTPException(status_code=503, detail="Graph not initialized")

    # Prompt 注入防御
    from agent.prompt_guard import PromptGuard
    guard_result = PromptGuard.check_input(query)
    if guard_result.flagged:
        raise HTTPException(status_code=400, detail=f"输入不合法：{guard_result.category} {guard_result.reason}")

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    _session_queues[session_id] = queue

    # 注册 SQL 确认 hook
    hook = _make_sql_confirm_hook(session_id, loop, queue)
    sql_confirm.register_web_hook(session_id, hook)

    # 注册 SQL 步骤事件 hook（复杂/数据分析技能每步 SQL 执行后触发）
    def _sql_step_hook(step_id: str, label: str, sql: str) -> None:
        asyncio.run_coroutine_threadsafe(
            queue.put(json.dumps({
                "type": "sql_step",
                "step_id": step_id,
                "label": label,
                "sql": sql,
            })),
            loop,
        )
    sql_step_emitter.register_hook(session_id, _sql_step_hook)

    config_with_thread = {"configurable": {"thread_id": thread_id}}
    constraints_list = await _load_enabled_constraints(thread_id)
    state = {
        "messages": [HumanMessage(content=query)],
        "constraints": constraints_list,
    }

    def _extract_text(output) -> str:
        """从各种 LangChain/LangGraph 输出结构中提取文本内容。"""
        if output is None:
            return ""
        # AIMessage / BaseMessage — .content 直接是字符串
        text = getattr(output, "content", None)
        if text and isinstance(text, str):
            return text
        # dict with "messages" key (节点返回值 / AddableValuesDict)
        if isinstance(output, dict):
            msgs = output.get("messages") or []
            for m in reversed(list(msgs)):
                t = getattr(m, "content", None)
                if t and isinstance(t, str):
                    return t
            # dict with direct "content" key
            text = output.get("content")
            if text and isinstance(text, str):
                return text
        return ""

    async def _run_graph_to_queue():
        """在后台任务中运行图，将事件转换后推入 SSE 队列。"""
        current_node = ""
        node_has_tokens = False      # 当前回答节点是否已收到流式 token
        answer_content_sent = False  # 是否已向前端发送过 full_response（防重复/防漏）
        # 只对这些节点的 LLM 输出发送给前端
        _ANSWER_NODES = {"format_answer", "general_chat"}
        try:
            sql_confirm.set_web_session(session_id)
            sql_step_emitter.set_session(session_id)

            async for event in _web_graph.astream_events(state, config=config_with_thread, version="v2"):
                kind = event.get("event", "")
                metadata = event.get("metadata", {})

                if kind == "on_chain_start" and "langgraph_node" in metadata:
                    node = metadata["langgraph_node"]
                    if node != current_node:
                        current_node = node
                        node_has_tokens = False
                        if node in _ANSWER_NODES:
                            answer_content_sent = False  # 重置，准备接收新回答
                        label = _NODE_LABELS.get(node, f"▸ {node}")
                        await queue.put(json.dumps({
                            "type": "node_start",
                            "node": node,
                            "label": label,
                        }))

                elif kind == "on_chat_model_stream":
                    # 只对最终回答节点转发 token
                    if current_node in _ANSWER_NODES:
                        chunk = event.get("data", {}).get("chunk")
                        content = getattr(chunk, "content", None)
                        if content and isinstance(content, str):
                            node_has_tokens = True
                            answer_content_sent = True
                            await queue.put(json.dumps({
                                "type": "token",
                                "content": content,
                            }))

                elif kind == "on_chat_model_end":
                    # 回答节点非流式输出时（ainvoke），从 on_chat_model_end 捕获完整内容
                    if current_node in _ANSWER_NODES and not answer_content_sent:
                        output = event.get("data", {}).get("output")
                        text = _extract_text(output)
                        if text:
                            answer_content_sent = True
                            await queue.put(json.dumps({
                                "type": "full_response",
                                "content": text,
                            }))

                elif kind == "on_chain_end" and "langgraph_node" in metadata:
                    node = metadata["langgraph_node"]
                    # 双重保险：如果 on_chat_model_end 没捕到，在节点结束时再试一次
                    if node in _ANSWER_NODES and not answer_content_sent:
                        output = event.get("data", {}).get("output")
                        text = _extract_text(output)
                        if text:
                            answer_content_sent = True
                            await queue.put(json.dumps({
                                "type": "full_response",
                                "content": text,
                            }))
                    await queue.put(json.dumps({
                        "type": "node_end",
                        "node": node,
                    }))

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Graph error: {e}", exc_info=True)
            await queue.put(json.dumps({"type": "error", "message": str(e)}))
        finally:
            await queue.put(None)  # 哨兵：通知 SSE 生成器流结束

    task = asyncio.create_task(_run_graph_to_queue())

    async def _event_generator():
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield {"data": item}
            yield {"data": json.dumps({"type": "done"})}
        except asyncio.CancelledError:
            # 客户端断开
            task.cancel()
        finally:
            sql_confirm.unregister_web_hook(session_id)
            sql_step_emitter.unregister_hook(session_id)
            _session_queues.pop(session_id, None)
            _confirm_events.pop(session_id, None)
            # 刷新会话的 updated_at，使其在列表中排到最前
            await _db_update_session(thread_id, touch=True)

    return EventSourceResponse(_event_generator())


@app.post("/api/chat/confirm")
async def confirm_sql(body: ConfirmRequest):
    """接收前端的 SQL 执行确认结果，唤醒挂起的图执行。"""
    session_id = body.session_id
    action = body.action if body.action in ("execute", "skip") else "execute"
    _confirm_results[session_id] = (action, body.reason or "")

    event = _confirm_events.pop(session_id, None)
    if event:
        event.set()
        return {"ok": True}
    return {"ok": False, "detail": "No pending confirmation for this session"}


# ── 静态文件（生产模式）────────────────────────────────────────────────────────

_DIST = Path(__file__).parent / "frontend" / "dist"

if _DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(_DIST / "assets")), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        """所有非 API 路由返回 index.html（SPA 路由）。"""
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")
        return FileResponse(str(_DIST / "index.html"))
else:
    @app.get("/", include_in_schema=False)
    async def root():
        return {"message": "Frontend not built. Run: cd web/frontend && npm install && npm run build"}


# ── 启动入口 ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    port = int(os.environ.get("WEB_PORT", "8000"))
    uvicorn.run("web.server:app", host="0.0.0.0", port=port, reload=False)
