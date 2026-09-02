from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import tomli_w

CLIENT_NAMES: tuple[str, ...] = ("claude-desktop", "claude-code", "opencode", "codex")

ENDPOINT_DEFAULTS: dict[str, str] = {
    "chat_base_url": "http://127.0.0.1:8101/v1",
    "chat_model": "Qwen3.5-0.8B",
    "embed_base_url": "http://127.0.0.1:8102/v1",
    "embed_model": "jina-embeddings-v5-text-small",
    "rerank_base_url": "http://127.0.0.1:8103/v1",
    "rerank_model": "jina-reranker-v3.5",
}

ENDPOINT_ENV_VARS: dict[str, str] = {
    "chat_base_url": "SCHOLAR_RAG_CHAT_BASE_URL",
    "chat_model": "SCHOLAR_RAG_CHAT_MODEL",
    "embed_base_url": "SCHOLAR_RAG_EMBED_BASE_URL",
    "embed_model": "SCHOLAR_RAG_EMBED_MODEL",
    "rerank_base_url": "SCHOLAR_RAG_RERANK_BASE_URL",
    "rerank_model": "SCHOLAR_RAG_RERANK_MODEL",
}

ROOT_KEYS: dict[str, str] = {
    "claude-desktop": "mcpServers",
    "claude-code": "mcpServers",
    "opencode": "mcp",
}

ENTRY_NAME = "scholar-rag-mcp"

USAGE = (
    "usage: scholar-rag-mcp [command] [options]\n"
    "\n"
    "commands:\n"
    "  (none)     launch the MCP stdio server\n"
    "  install    install the package and register configured MCP clients\n"
    "  uninstall  unregister configured MCP clients and remove the package\n"
)


class SetupError(Exception):
    pass


def resolve_endpoints(
    flags: dict[str, str | None],
    env: dict[str, str],
    yes: bool,
    tty: bool,
    prompt: Callable[[str], str] | None = None,
) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for key, default in ENDPOINT_DEFAULTS.items():
        flag = flags.get(key)
        if flag is not None:
            resolved[key] = flag
            continue
        env_name = ENDPOINT_ENV_VARS[key]
        if env_name in env:
            resolved[key] = env[env_name]
            continue
        if yes:
            resolved[key] = default
            continue
        resolved[key] = _resolve_interactive(default, tty, prompt)
    return resolved


def _resolve_interactive(
    default: str,
    tty: bool,
    prompt: Callable[[str], str] | None,
) -> str:
    if not tty:
        raise SetupError(
            f"missing endpoint value; pass the matching -- flag or use --yes to "
            f"accept the default {default!r}"
        )
    if prompt is None:
        raise SetupError("interactive endpoint resolution requires a prompt callable")
    answer = prompt(f"accept endpoint default {default!r} by pressing enter, or type a value")
    return answer if answer else default


def config_path(client: str) -> Path:
    if client == "claude-desktop":
        if sys.platform == "darwin":
            return (
                Path.home()
                / "Library"
                / "Application Support"
                / "Claude"
                / "claude_desktop_config.json"
            )
        if sys.platform == "win32":
            appdata = os.environ.get("APPDATA")
            base = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
            return base / "Claude" / "claude_desktop_config.json"
        return Path.home() / ".config" / "Claude" / "claude_desktop_config.json"
    if client == "claude-code":
        return Path.home() / ".claude.json"
    if client == "opencode":
        json_path = Path.home() / ".config" / "opencode" / "opencode.json"
        jsonc_path = json_path.with_suffix(".jsonc")
        if not json_path.is_file() and jsonc_path.is_file():
            return jsonc_path
        return json_path
    if client == "codex":
        return Path.home() / ".codex" / "config.toml"
    raise ValueError(f"unknown MCP client: {client}")


def detect_clients() -> list[str]:
    detected = []
    for client in CLIENT_NAMES:
        path = config_path(client)
        found = path.is_file()
        if client == "codex":
            found = found or (Path.home() / ".codex").is_dir()
        if found:
            detected.append(client)
    return detected


def build_entry(client: str, endpoints: dict[str, str]) -> dict[str, Any]:
    env = {ENDPOINT_ENV_VARS[key]: value for key, value in endpoints.items()}
    if client == "opencode":
        return {"type": "local", "command": ["scholar-rag-mcp"], "environment": env}
    if client in ("claude-desktop", "claude-code"):
        return {"command": "scholar-rag-mcp", "env": env}
    raise ValueError(f"unknown MCP client: {client}")


def _dump_json(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2) + "\n"


def merge_json_client(path: Path, client: str, endpoints: dict[str, str]) -> str:
    if client not in ROOT_KEYS:
        raise ValueError(f"unknown MCP client: {client}")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {
            ROOT_KEYS[client]: {ENTRY_NAME: build_entry(client, endpoints)}
        }
        path.write_text(_dump_json(data), encoding="utf-8")
        return "created"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "parse-failed"
    root = data.setdefault(ROOT_KEYS[client], {})
    root[ENTRY_NAME] = build_entry(client, endpoints)
    path.write_text(_dump_json(data), encoding="utf-8")
    return "written"


def remove_json_client(path: Path, client: str) -> str:
    if client not in ROOT_KEYS:
        raise ValueError(f"unknown MCP client: {client}")
    if not path.is_file():
        return "missing"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "parse-failed"
    root = data.get(ROOT_KEYS[client])
    if not isinstance(root, dict) or ENTRY_NAME not in root:
        return "absent"
    del root[ENTRY_NAME]
    path.write_text(_dump_json(data), encoding="utf-8")
    return "removed"


def _codex_entry(endpoints: dict[str, str]) -> dict[str, Any]:
    return {
        "command": ["scholar-rag-mcp"],
        "env": {ENDPOINT_ENV_VARS[key]: value for key, value in endpoints.items()},
    }


def _backup_toml(path: Path) -> None:
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    backup = path.with_name(f"{path.name}.bak.{timestamp}")
    backup.write_bytes(path.read_bytes())


def merge_codex(path: Path, endpoints: dict[str, str]) -> str:
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {"mcp_servers": {ENTRY_NAME: _codex_entry(endpoints)}}
        path.write_text(tomli_w.dumps(data), encoding="utf-8")
        return "created"
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError:
        return "parse-failed"
    _backup_toml(path)
    servers = data.setdefault("mcp_servers", {})
    servers[ENTRY_NAME] = _codex_entry(endpoints)
    path.write_text(tomli_w.dumps(data), encoding="utf-8")
    return "written"


def remove_codex(path: Path) -> str:
    if not path.is_file():
        return "missing"
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError:
        return "parse-failed"
    servers = data.get("mcp_servers")
    if not isinstance(servers, dict) or ENTRY_NAME not in servers:
        return "absent"
    _backup_toml(path)
    del servers[ENTRY_NAME]
    path.write_text(tomli_w.dumps(data), encoding="utf-8")
    return "removed"


def manual_snippet(client: str, endpoints: dict[str, str]) -> str:
    if client == "codex":
        return tomli_w.dumps({"mcp_servers": {ENTRY_NAME: _codex_entry(endpoints)}})
    return _dump_json({ROOT_KEYS[client]: {ENTRY_NAME: build_entry(client, endpoints)}})


def run_install(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="scholar-rag-mcp install",
        description="Install the package and register the scholar-rag-mcp MCP client.",
    )
    for key in ENDPOINT_DEFAULTS:
        parser.add_argument(f"--{key.replace('_', '-')}")
    parser.add_argument("--client", choices=CLIENT_NAMES, help="register only this client")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="accept the endpoint defaults without prompting",
    )
    args = parser.parse_args(argv)
    if shutil.which("uv") is None:
        print("uv not found; install uv then rerun this command")
        print("  curl -LsSf https://astral.sh/uv/install.sh | sh")
        print("  pip install uv")
        print("  winget install --id=astral-sh.uv -e")
        return 1
    result = subprocess.run(
        ["uv", "tool", "install", ENTRY_NAME], capture_output=True, text=True
    )
    combined = result.stdout + result.stderr
    if result.returncode != 0 and "already installed" not in combined:
        print(result.stderr)
        return 1
    if result.returncode != 0:
        print("scholar-rag-mcp is already installed; continuing")
    pkg_status = "installed" if result.returncode == 0 else "already-installed"
    flags: dict[str, str | None] = {key: getattr(args, key) for key in ENDPOINT_DEFAULTS}
    try:
        endpoints = resolve_endpoints(
            flags=flags,
            env=dict(os.environ),
            yes=args.yes,
            tty=sys.stdin.isatty(),
            prompt=input,
        )
    except SetupError as exc:
        print(exc)
        return 1
    clients = [args.client] if args.client is not None else detect_clients()
    if not clients:
        print("no MCP client configs detected; manual snippets below")
        for client in CLIENT_NAMES:
            print(manual_snippet(client, endpoints))
        return 0
    results: dict[str, tuple[str, str]] = {}
    for client in clients:
        path = config_path(client)
        if client == "codex":
            status = merge_codex(path, endpoints)
        else:
            status = merge_json_client(path, client, endpoints)
        if status == "parse-failed":
            print(f"{client} config could not be parsed; manual snippet below")
            print(manual_snippet(client, endpoints))
        results[client] = (str(path), status)
    print("install summary")
    print(f"  package: {pkg_status}")
    for client in clients:
        written_path, status = results[client]
        print(f"  {client}: {written_path} ({status})")
    for key in ENDPOINT_DEFAULTS:
        print(f"  {key}: {endpoints[key]}")
    print('  next: start the model services, see README "Model deployment"')
    return 0


def run_uninstall(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="scholar-rag-mcp uninstall",
        description="Unregister the scholar-rag-mcp MCP client and optionally delete its data.",
    )
    parser.add_argument("--client", choices=CLIENT_NAMES, help="unregister only this client")
    parser.add_argument("--purge", action="store_true", help="delete the local data directory")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="skip the purge confirmation prompt",
    )
    args = parser.parse_args(argv)
    clients = [args.client] if args.client is not None else detect_clients()
    print("uninstall summary")
    for client in clients:
        path = config_path(client)
        status = remove_codex(path) if client == "codex" else remove_json_client(path, client)
        print(f"  {client}: {status}")
        if status == "parse-failed":
            print(f"  manual removal: delete the {ENTRY_NAME} entry from {path}")
    if shutil.which("uv") is None:
        print("uv not found; if scholar-rag-mcp was installed via uv, remove it manually with:")
        print("  uv tool uninstall scholar-rag-mcp")
    else:
        result = subprocess.run(
            ["uv", "tool", "uninstall", ENTRY_NAME], capture_output=True, text=True
        )
        if result.returncode != 0:
            print("scholar-rag-mcp was likely not installed via uv; skipping")
        else:
            print("  package: removed")
    from scholar_rag.core.config import Settings

    data_dir = Settings.load().data_dir
    if not args.purge:
        print(f"  data: {data_dir} (kept)")
        return 0
    print(f"  data: {data_dir.absolute()} (purge requested)")
    if not data_dir.exists():
        print(f"  data directory does not exist: {data_dir}; nothing to purge")
        return 0
    if not args.yes:
        answer = input("delete the data directory? type 'yes' to confirm: ")
        if answer.strip() != "yes":
            print(f"  purge aborted; data dir kept: {data_dir}")
            return 0
    try:
        shutil.rmtree(data_dir)
    except OSError:
        print(f"  failed to remove data directory {data_dir}; remove it manually")
        return 1
    print(f"  data: {data_dir} (purged)")
    return 0


def setup_main(argv: list[str]) -> int:
    if argv and argv[0] == "install":
        return run_install(argv[1:])
    if argv and argv[0] == "uninstall":
        return run_uninstall(argv[1:])
    print(USAGE, end="")
    return 2
