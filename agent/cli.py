"""
text2sql CLI — Command-Line Interface for the SQL Agent

Usage
─────
  python -m agent.cli --help
  python -m agent.cli query "how many albums are there?"
  python -m agent.cli stream "show top 5 customers by revenue"
  python -m agent.cli chat
  python -m agent.cli schema
  python -m agent.cli config

Global flags that work on every sub-command
───────────────────────────────────────────
  --db URI                Database connection URI
                          (default: $DB_URI or sqlite:///Chinook.db)
  --provider PROVIDER     LLM provider: tongyi | openai | anthropic | …
                          (default: $LLM_PROVIDER or tongyi)
  --model MODEL           LLM model name  (default: $LLM_MODEL or qwen-plus)
  --api-key KEY           API key         (default: $DASHSCOPE_API_KEY / $LLM_API_KEY)
  --env FILE              Load a specific .env file before startup
  --thread-id ID          Reuse an existing session thread (memory continuity)
  --log-level LEVEL       DEBUG | INFO | WARNING | ERROR  (default: WARNING)
  --json                  Print final result as JSON (machine-readable)
  --no-cache              Disable LLM response cache for this run

Sub-commands
────────────
  query   QUESTION        Single-turn query, node-level streaming output
  stream  QUESTION        Single-turn query, token-level streaming output
  chat                    Interactive multi-turn chat session
  schema  [TABLE]         Show database schema (all tables or a single table)
  config                  Show resolved configuration values
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid

# ── Windows UTF-8 fix ────────────────────────────────────────────────────────
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# Argument parser
# ══════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="text2sql",
        description=(
            "Text-to-SQL Agent — convert natural language questions into SQL "
            "queries and execute them against your database.\n\n"
            "Three routing skills:\n"
            "  simple_query   — single-table or straightforward multi-table queries\n"
            "  complex_query  — multi-step Plan-Execute for complex analytical queries\n"
            "  data_analysis  — 8-step deep analysis: explore → plan → SQL → "
            "insights → report → export CSV/Excel"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Environment variables (override with flags above):\n"
            "  DB_URI                 Database connection string\n"
            "  LLM_PROVIDER           LLM backend (tongyi / openai / …)\n"
            "  LLM_MODEL              Model name\n"
            "  DASHSCOPE_API_KEY      API key for Tongyi\n"
            "  LLM_API_KEY            Generic API key\n"
            "  MILVUS_HOST / PORT     Milvus vector DB for dual-tower retrieval\n"
            "  RETRIEVAL_ENABLED      true/false — enable dual-tower schema pruning\n"
            "  LLM_CACHE_ENABLED      true/false — enable SQLite LLM response cache\n\n"
            "Examples:\n"
            "  text2sql query \"How many tracks does each album have?\"\n"
            "  text2sql stream \"Top 5 customers by invoice total\" --json\n"
            "  text2sql chat --db mysql+pymysql://user:pass@localhost/mydb\n"
            "  text2sql schema --db sqlite:///Chinook.db\n"
            "  text2sql schema Album\n"
            "  text2sql config\n"
        ),
    )

    # ── Global options ───────────────────────────────────────────────────────
    g = parser.add_argument_group("connection & model")
    g.add_argument(
        "--db",
        metavar="URI",
        default=None,
        help="Database URI, e.g. sqlite:///mydb.db or mysql+pymysql://u:p@host/db",
    )
    g.add_argument(
        "--provider",
        metavar="PROVIDER",
        default=None,
        help="LLM provider: tongyi | openai | anthropic | groq | …",
    )
    g.add_argument(
        "--model",
        metavar="MODEL",
        default=None,
        help="LLM model name, e.g. qwen-plus | gpt-4o | claude-3-5-sonnet",
    )
    g.add_argument(
        "--api-key",
        metavar="KEY",
        default=None,
        help="API key (overrides DASHSCOPE_API_KEY / LLM_API_KEY env vars)",
    )
    g.add_argument(
        "--env",
        metavar="FILE",
        default=None,
        help="Path to a .env file to load before startup",
    )

    o = parser.add_argument_group("output & session")
    o.add_argument(
        "--thread-id",
        metavar="ID",
        default=None,
        help="Session thread ID for multi-turn memory continuity",
    )
    o.add_argument(
        "--log-level",
        metavar="LEVEL",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: WARNING)",
    )
    o.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Print the final result as JSON (machine-readable)",
    )
    o.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable LLM response cache for this run",
    )

    # ── Sub-commands ─────────────────────────────────────────────────────────
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # query
    p_query = sub.add_parser(
        "query",
        help="Single-turn query with node-level streaming output",
        description=(
            "Execute a natural language question against the database.\n"
            "Uses node-level streaming: each graph node prints its output "
            "as it completes."
        ),
    )
    p_query.add_argument("question", help="Natural language question to answer")

    # stream
    p_stream = sub.add_parser(
        "stream",
        help="Single-turn query with token-level streaming output",
        description=(
            "Execute a natural language question with real-time token streaming.\n"
            "LLM tokens are printed as they are generated."
        ),
    )
    p_stream.add_argument("question", help="Natural language question to answer")

    # chat
    sub.add_parser(
        "chat",
        help="Interactive multi-turn chat session with memory",
        description=(
            "Start an interactive chat session.\n"
            "The agent remembers previous turns within the same session.\n\n"
            "Session commands:\n"
            "  stream   Toggle token-level streaming mode\n"
            "  new      Start a fresh session (clears memory)\n"
            "  quit     Exit"
        ),
    )

    # schema
    p_schema = sub.add_parser(
        "schema",
        help="Show database schema",
        description="Display table names and DDL from the configured database.",
    )
    p_schema.add_argument(
        "table",
        nargs="?",
        default=None,
        help="Show DDL for a specific table (omit to list all tables)",
    )

    # config
    sub.add_parser(
        "config",
        help="Show resolved configuration values",
        description=(
            "Print all resolved configuration values (env vars + defaults).\n"
            "API keys are masked for security."
        ),
    )

    return parser


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _apply_overrides(args: argparse.Namespace) -> None:
    """Push CLI flag overrides into environment variables before graph init."""
    if args.db:
        os.environ["DB_URI"] = args.db
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider
    if args.model:
        os.environ["LLM_MODEL"] = args.model
    if args.api_key:
        os.environ["DASHSCOPE_API_KEY"] = args.api_key
        os.environ["LLM_API_KEY"] = args.api_key
    if args.no_cache:
        os.environ["LLM_CACHE_ENABLED"] = "false"
    os.environ["LOG_LEVEL"] = args.log_level


def _load_graph(args: argparse.Namespace):
    """Lazy-import and initialise the agent graph (expensive — defer until needed)."""
    import logging
    logging.basicConfig(level=getattr(logging, args.log_level))

    if args.env:
        from dotenv import load_dotenv
        load_dotenv(args.env, override=True)

    _apply_overrides(args)

    # Import after env overrides so config picks them up
    from agent.graph import graph, config, run_query, run_query_streaming
    return graph, config, run_query, run_query_streaming


def _print_result(result: dict, output_json: bool) -> None:
    if output_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        if result.get("export_files"):
            print("\nExported files:")
            for f in result["export_files"]:
                print(f"  {f}")


def _get_thread_id(args: argparse.Namespace) -> str:
    return args.thread_id or str(uuid.uuid4())


# ══════════════════════════════════════════════════════════════════════════════
# Sub-command handlers
# ══════════════════════════════════════════════════════════════════════════════

def _cmd_query(args: argparse.Namespace) -> int:
    _, _, run_query, _ = _load_graph(args)
    thread_id = _get_thread_id(args)
    result = run_query(args.question, thread_id)
    _print_result(result, args.output_json)
    return 0


def _cmd_stream(args: argparse.Namespace) -> int:
    _, _, _, run_query_streaming = _load_graph(args)
    thread_id = _get_thread_id(args)
    result = run_query_streaming(args.question, thread_id)
    _print_result(result, args.output_json)
    return 0


def _cmd_chat(args: argparse.Namespace) -> int:
    _, config, run_query, run_query_streaming = _load_graph(args)
    thread_id = _get_thread_id(args)
    streaming_mode = False

    db_display = config.database.uri.split("@")[-1] if "@" in config.database.uri else config.database.uri
    print("=" * 55)
    print("SQL Agent - Interactive Chat (multi-turn memory)")
    print(f"  DB     : {db_display}")
    print(f"  Model  : {config.llm.provider}:{config.llm.model}")
    print(f"  Thread : {thread_id[:8]}…")
    print("=" * 55)
    print("  stream  — toggle token-level streaming")
    print("  new     — start a fresh session")
    print("  quit    — exit")
    print()

    while True:
        try:
            mode_tag = "[token]" if streaming_mode else "[node ]"
            question = input(f"{mode_tag} > ").strip()
            if not question:
                continue
            if question.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            if question.lower() == "new":
                thread_id = str(uuid.uuid4())
                print(f"New session: {thread_id[:8]}…\n")
                continue
            if question.lower() == "stream":
                streaming_mode = not streaming_mode
                print(f"Streaming: {'token-level' if streaming_mode else 'node-level'}\n")
                continue
            print()
            if streaming_mode:
                result = run_query_streaming(question, thread_id)
            else:
                result = run_query(question, thread_id)
            if args.output_json:
                print(json.dumps(result, ensure_ascii=False, indent=2))
        except KeyboardInterrupt:
            print("\nInterrupted.")
            break
        except Exception as e:
            print(f"Error: {e}\n")
    return 0


def _cmd_schema(args: argparse.Namespace) -> int:
    _apply_overrides(args)
    if args.env:
        from dotenv import load_dotenv
        load_dotenv(args.env, override=True)

    from agent.config import get_config
    from agent.database import SQLDatabaseManager

    cfg = get_config()
    db = SQLDatabaseManager(cfg.database, security_config=cfg.security,
                            schema_cache_config=cfg.schema_cache)

    if args.table:
        # Single table DDL
        table_info = db.get_table_schema([args.table])
        if args.output_json:
            print(json.dumps({"table": args.table, "ddl": table_info}, ensure_ascii=False, indent=2))
        else:
            print(f"\n-- {args.table} --")
            print(table_info)
    else:
        tables = db.get_table_names()
        if args.output_json:
            schema = {t: db.get_table_schema([t]) for t in tables}
            print(json.dumps({"tables": tables, "schema": schema}, ensure_ascii=False, indent=2))
        else:
            print(f"\nDatabase: {cfg.database.uri.split('@')[-1] if '@' in cfg.database.uri else cfg.database.uri}")
            print(f"Tables ({len(tables)}):")
            for t in tables:
                print(f"  - {t}")
            print("\nRun `text2sql schema TABLE` to see a table's DDL.")
    return 0


def _cmd_config(args: argparse.Namespace) -> int:
    _apply_overrides(args)
    if args.env:
        from dotenv import load_dotenv
        load_dotenv(args.env, override=True)

    from agent.config import get_config

    def _mask(value: str | None, show: int = 4) -> str:
        if not value:
            return "(not set)"
        return value[:show] + "…" + "*" * 8

    try:
        cfg = get_config()
    except Exception as e:
        print(f"Config error: {e}", file=sys.stderr)
        return 1

    data = {
        "database": {
            "uri":               cfg.database.uri.split("@")[-1] if "@" in cfg.database.uri else cfg.database.uri,
            "max_query_results": cfg.database.max_query_results,
            "timeout_seconds":   cfg.database.timeout_seconds,
        },
        "llm": {
            "provider":    cfg.llm.provider,
            "model":       cfg.llm.model,
            "api_key":     _mask(cfg.llm.api_key),
            "temperature": cfg.llm.temperature,
            "max_tokens":  cfg.llm.max_tokens,
        },
        "retrieval": {
            "enabled":             cfg.retrieval.enabled,
            "milvus_host":         cfg.retrieval.milvus_host,
            "milvus_port":         cfg.retrieval.milvus_port,
            "top_k_columns":       cfg.retrieval.top_k_columns,
            "max_candidate_tables":cfg.retrieval.max_candidate_tables,
            "score_threshold":     cfg.retrieval.score_threshold,
        },
        "cache": {
            "enabled":     cfg.cache.enabled,
            "backend":     cfg.cache.backend,
            "sqlite_path": cfg.cache.sqlite_path,
        },
        "security": {
            "allowed_statements":  cfg.security.allowed_statements,
            "max_rows":            cfg.security.max_rows,
            "max_query_length":    cfg.security.max_query_length,
            "table_allowlist":     cfg.security.table_allowlist,
            "table_denylist":      cfg.security.table_denylist,
            "enable_audit_log":    cfg.security.enable_audit_log,
        },
        "output": {
            "report_dir": cfg.output.report_dir,
            "chart_dir":  cfg.output.chart_dir,
        },
        "logging": {
            "level":     cfg.logging.level,
            "file_path": cfg.logging.file_path,
        },
    }

    if args.output_json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print("\n── text2sql configuration ──")
        for section, values in data.items():
            print(f"\n[{section}]")
            for k, v in values.items():
                print(f"  {k:<26} {v}")
    return 0


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

_HANDLERS = {
    "query":  _cmd_query,
    "stream": _cmd_stream,
    "chat":   _cmd_chat,
    "schema": _cmd_schema,
    "config": _cmd_config,
}


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    handler = _HANDLERS.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    try:
        sys.exit(handler(args))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
