import json
import re
import sys
import tomllib
from pathlib import Path

import pytest

import scholar_rag.cli.setup as setup_mod
from scholar_rag.cli.setup import (
    CLIENT_NAMES,
    ENDPOINT_DEFAULTS,
    ENDPOINT_ENV_VARS,
    ROOT_KEYS,
    SetupError,
    build_entry,
    config_path,
    detect_clients,
    manual_snippet,
    merge_codex,
    merge_json_client,
    remove_codex,
    remove_json_client,
    resolve_endpoints,
)


def _fake_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)


@pytest.mark.parametrize("platform", ["linux", "darwin", "win32"])
def test_config_path_claude_desktop(platform, tmp_path, monkeypatch):
    _fake_home(tmp_path, monkeypatch)
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    monkeypatch.setattr(sys, "platform", platform)
    if platform == "darwin":
        expected = (
            tmp_path / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
        )
    elif platform == "win32":
        expected = tmp_path / "appdata" / "Claude" / "claude_desktop_config.json"
    else:
        expected = tmp_path / ".config" / "Claude" / "claude_desktop_config.json"
    assert config_path("claude-desktop") == expected


def test_config_path_claude_desktop_win32_without_appdata(tmp_path, monkeypatch):
    _fake_home(tmp_path, monkeypatch)
    monkeypatch.delenv("APPDATA", raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    expected = tmp_path / "AppData" / "Roaming" / "Claude" / "claude_desktop_config.json"
    assert config_path("claude-desktop") == expected


def test_config_path_claude_code(tmp_path, monkeypatch):
    _fake_home(tmp_path, monkeypatch)
    assert config_path("claude-code") == tmp_path / ".claude.json"


def test_config_path_opencode_json(tmp_path, monkeypatch):
    _fake_home(tmp_path, monkeypatch)
    cfg_dir = tmp_path / ".config" / "opencode"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "opencode.json").write_text("{}")
    assert config_path("opencode") == cfg_dir / "opencode.json"


def test_config_path_opencode_falls_back_to_jsonc(tmp_path, monkeypatch):
    _fake_home(tmp_path, monkeypatch)
    cfg_dir = tmp_path / ".config" / "opencode"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "opencode.jsonc").write_text("{}")
    assert config_path("opencode") == cfg_dir / "opencode.jsonc"


def test_config_path_opencode_defaults_to_json(tmp_path, monkeypatch):
    _fake_home(tmp_path, monkeypatch)
    assert config_path("opencode") == tmp_path / ".config" / "opencode" / "opencode.json"


def test_config_path_codex(tmp_path, monkeypatch):
    _fake_home(tmp_path, monkeypatch)
    assert config_path("codex") == tmp_path / ".codex" / "config.toml"


def test_detect_clients_empty_without_configs(tmp_path, monkeypatch):
    _fake_home(tmp_path, monkeypatch)
    assert detect_clients() == []


@pytest.mark.parametrize("client", list(CLIENT_NAMES))
def test_detect_clients_finds_each_when_config_exists(client, tmp_path, monkeypatch):
    _fake_home(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "platform", "linux")
    config_path(client).parent.mkdir(parents=True, exist_ok=True)
    config_path(client).write_text("{}")
    assert detect_clients() == [client]


def test_detect_codex_when_only_dir_exists(tmp_path, monkeypatch):
    _fake_home(tmp_path, monkeypatch)
    (tmp_path / ".codex").mkdir()
    assert detect_clients() == ["codex"]


def test_detect_opencode_when_only_jsonc_exists(tmp_path, monkeypatch):
    _fake_home(tmp_path, monkeypatch)
    (tmp_path / ".config" / "opencode").mkdir(parents=True)
    (tmp_path / ".config" / "opencode" / "opencode.jsonc").write_text("{}")
    assert detect_clients() == ["opencode"]


def test_detect_clients_preserves_order(tmp_path, monkeypatch):
    _fake_home(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "platform", "linux")
    for client in CLIENT_NAMES:
        config_path(client).parent.mkdir(parents=True, exist_ok=True)
        config_path(client).write_text("{}")
    assert detect_clients() == list(CLIENT_NAMES)


def test_endpoint_defaults_exact():
    assert ENDPOINT_DEFAULTS == {
        "chat_base_url": "http://127.0.0.1:8101/v1",
        "chat_model": "Qwen3.5-0.8B",
        "embed_base_url": "http://127.0.0.1:8102/v1",
        "embed_model": "jina-embeddings-v5-text-small",
        "rerank_base_url": "http://127.0.0.1:8103/v1",
        "rerank_model": "jina-reranker-v3.5",
    }


def test_endpoint_env_vars_exact():
    assert ENDPOINT_ENV_VARS == {
        "chat_base_url": "SCHOLAR_RAG_CHAT_BASE_URL",
        "chat_model": "SCHOLAR_RAG_CHAT_MODEL",
        "embed_base_url": "SCHOLAR_RAG_EMBED_BASE_URL",
        "embed_model": "SCHOLAR_RAG_EMBED_MODEL",
        "rerank_base_url": "SCHOLAR_RAG_RERANK_BASE_URL",
        "rerank_model": "SCHOLAR_RAG_RERANK_MODEL",
    }


def test_resolve_returns_all_six_keys():
    result = resolve_endpoints(flags={}, env={}, yes=True, tty=False)
    assert set(result) == set(ENDPOINT_DEFAULTS)


@pytest.mark.parametrize("key", list(ENDPOINT_ENV_VARS))
def test_flag_beats_env_and_default(key):
    result = resolve_endpoints(
        flags={key: "flag-value"},
        env={ENDPOINT_ENV_VARS[key]: "env-value"},
        yes=True,
        tty=False,
    )
    assert result[key] == "flag-value"


@pytest.mark.parametrize("key", list(ENDPOINT_ENV_VARS))
def test_env_beats_default(key):
    result = resolve_endpoints(
        flags={},
        env={ENDPOINT_ENV_VARS[key]: "env-value"},
        yes=True,
        tty=False,
    )
    assert result[key] == "env-value"


@pytest.mark.parametrize("key", list(ENDPOINT_ENV_VARS))
def test_default_when_no_flag_or_env(key):
    result = resolve_endpoints(flags={}, env={}, yes=True, tty=False)
    assert result[key] == ENDPOINT_DEFAULTS[key]


def test_none_flag_falls_back_to_env():
    result = resolve_endpoints(
        flags={"chat_model": None},
        env={ENDPOINT_ENV_VARS["chat_model"]: "env-model"},
        yes=True,
        tty=False,
    )
    assert result["chat_model"] == "env-model"


def test_yes_skips_prompt_and_returns_resolved_map():
    def fake_prompt(text):
        raise AssertionError("prompt must not be called when yes=True")

    result = resolve_endpoints(
        flags={"chat_model": "flag-model"},
        env={ENDPOINT_ENV_VARS["chat_base_url"]: "http://env"},
        yes=True,
        tty=True,
        prompt=fake_prompt,
    )
    assert result["chat_model"] == "flag-model"
    assert result["chat_base_url"] == "http://env"
    assert result["embed_model"] == ENDPOINT_DEFAULTS["embed_model"]


def test_tty_prompts_only_keys_that_need_default():
    prompts = []

    def fake_prompt(text):
        prompts.append(text)
        return ""

    result = resolve_endpoints(
        flags={"chat_base_url": "http://flag"},
        env={ENDPOINT_ENV_VARS["chat_model"]: "env-model"},
        yes=False,
        tty=True,
        prompt=fake_prompt,
    )
    missing = [
        key for key in ENDPOINT_DEFAULTS if key not in ("chat_base_url", "chat_model")
    ]
    assert len(prompts) == len(missing)
    for key, prompt_text in zip(missing, prompts, strict=True):
        assert ENDPOINT_DEFAULTS[key] in prompt_text
        assert result[key] == ENDPOINT_DEFAULTS[key]
    assert result["chat_base_url"] == "http://flag"
    assert result["chat_model"] == "env-model"


def test_tty_adopts_typed_prompt_values():
    def fake_prompt(text):
        for key, default in ENDPOINT_DEFAULTS.items():
            if default in text:
                return f"typed-{key}"
        raise AssertionError(f"unexpected prompt text: {text}")

    result = resolve_endpoints(flags={}, env={}, yes=False, tty=True, prompt=fake_prompt)
    assert result == {key: f"typed-{key}" for key in ENDPOINT_DEFAULTS}


def test_non_tty_raises_when_any_key_needs_default():
    with pytest.raises(SetupError) as excinfo:
        resolve_endpoints(flags={}, env={}, yes=False, tty=False)
    message = str(excinfo.value)
    assert "flag" in message
    assert "--yes" in message


@pytest.mark.parametrize("missing", list(ENDPOINT_DEFAULTS))
def test_non_tty_raises_for_single_missing_key(missing):
    flags = {key: f"flag-{key}" for key in ENDPOINT_DEFAULTS if key != missing}
    with pytest.raises(SetupError) as excinfo:
        resolve_endpoints(flags=flags, env={}, yes=False, tty=False)
    message = str(excinfo.value)
    assert "flag" in message
    assert "--yes" in message


@pytest.mark.parametrize("source", ["flags", "env"])
def test_non_tty_all_covered_returns_without_prompt(source):
    def never_prompt(text):
        raise AssertionError("prompt must not be called in non-tty mode")

    if source == "flags":
        flags = {key: f"flag-{key}" for key in ENDPOINT_DEFAULTS}
        env = {}
        expected = {key: f"flag-{key}" for key in ENDPOINT_DEFAULTS}
    else:
        flags = {}
        env = {ENDPOINT_ENV_VARS[key]: f"env-{key}" for key in ENDPOINT_DEFAULTS}
        expected = {key: f"env-{key}" for key in ENDPOINT_DEFAULTS}
    result = resolve_endpoints(
        flags=flags, env=env, yes=False, tty=False, prompt=never_prompt
    )
    assert result == expected


def test_non_tty_all_covered_without_prompt_callable():
    env = {ENDPOINT_ENV_VARS[key]: f"env-{key}" for key in ENDPOINT_DEFAULTS}
    result = resolve_endpoints(flags={}, env=env, yes=False, tty=False)
    assert result == {key: f"env-{key}" for key in ENDPOINT_DEFAULTS}


def _sample_endpoints():
    return {key: f"value-{key}" for key in ENDPOINT_DEFAULTS}


def _expected_env(endpoints):
    return {ENDPOINT_ENV_VARS[key]: value for key, value in endpoints.items()}


def test_root_keys_exact():
    assert ROOT_KEYS == {
        "claude-desktop": "mcpServers",
        "claude-code": "mcpServers",
        "opencode": "mcp",
    }


@pytest.mark.parametrize("client", ["claude-desktop", "claude-code"])
def test_build_entry_claude_shape(client):
    endpoints = _sample_endpoints()
    assert build_entry(client, endpoints) == {
        "command": "scholar-rag-mcp",
        "env": _expected_env(endpoints),
    }


def test_build_entry_opencode_shape():
    endpoints = _sample_endpoints()
    assert build_entry("opencode", endpoints) == {
        "type": "local",
        "command": ["scholar-rag-mcp"],
        "environment": _expected_env(endpoints),
    }


def test_build_entry_unknown_client_raises():
    with pytest.raises(ValueError):
        build_entry("codex", _sample_endpoints())


def test_merge_preserves_other_entries(tmp_path):
    path = tmp_path / "claude.json"
    path.write_text(
        json.dumps(
            {
                "mcpServers": {"other-server": {"command": "foo"}},
                "projects": {"x": 1},
                "theme": "dark",
            }
        )
    )
    assert merge_json_client(path, "claude-desktop", _sample_endpoints()) == "written"
    data = json.loads(path.read_text())
    assert data["projects"] == {"x": 1}
    assert data["theme"] == "dark"
    assert data["mcpServers"]["other-server"] == {"command": "foo"}
    assert data["mcpServers"]["scholar-rag-mcp"] == build_entry(
        "claude-desktop", _sample_endpoints()
    )


def test_merge_idempotent_reinstall(tmp_path):
    path = tmp_path / "claude.json"
    assert merge_json_client(path, "claude-code", _sample_endpoints()) == "created"
    before = path.read_bytes()
    assert merge_json_client(path, "claude-code", _sample_endpoints()) == "written"
    assert path.read_bytes() == before


def test_merge_replaces_existing_endpoint_values(tmp_path):
    path = tmp_path / "claude.json"
    merge_json_client(path, "claude-desktop", _sample_endpoints())
    updated = {key: f"new-{key}" for key in ENDPOINT_DEFAULTS}
    assert merge_json_client(path, "claude-desktop", updated) == "written"
    data = json.loads(path.read_text())
    assert data["mcpServers"]["scholar-rag-mcp"] == build_entry("claude-desktop", updated)


def test_merge_creates_fresh_claude_file(tmp_path):
    path = tmp_path / "claude.json"
    assert merge_json_client(path, "claude-desktop", _sample_endpoints()) == "created"
    data = json.loads(path.read_text())
    assert data == {
        "mcpServers": {"scholar-rag-mcp": build_entry("claude-desktop", _sample_endpoints())}
    }


def test_merge_creates_fresh_opencode_file(tmp_path):
    path = tmp_path / "opencode.json"
    assert merge_json_client(path, "opencode", _sample_endpoints()) == "created"
    data = json.loads(path.read_text())
    assert data == {"mcp": {"scholar-rag-mcp": build_entry("opencode", _sample_endpoints())}}


def test_merge_creates_parent_directories(tmp_path):
    path = tmp_path / "nested" / "deep" / "claude.json"
    assert merge_json_client(path, "claude-code", _sample_endpoints()) == "created"
    assert path.is_file()


def test_merge_write_is_indent_two_with_trailing_newline(tmp_path):
    path = tmp_path / "claude.json"
    merge_json_client(path, "claude-code", _sample_endpoints())
    data = json.loads(path.read_text())
    assert path.read_text() == json.dumps(data, indent=2) + "\n"


def test_merge_parse_failed_leaves_file_byte_identical(tmp_path):
    path = tmp_path / "claude.json"
    payload = b"{ 'jsonc': true }"
    path.write_bytes(payload)
    assert merge_json_client(path, "claude-desktop", _sample_endpoints()) == "parse-failed"
    assert path.read_bytes() == payload


def test_merge_json_client_unknown_client_raises(tmp_path):
    with pytest.raises(ValueError):
        merge_json_client(tmp_path / "config.toml", "codex", _sample_endpoints())


def test_remove_removed_entry(tmp_path):
    path = tmp_path / "claude.json"
    merge_json_client(path, "claude-desktop", _sample_endpoints())
    assert remove_json_client(path, "claude-desktop") == "removed"
    data = json.loads(path.read_text())
    assert "scholar-rag-mcp" not in data["mcpServers"]


def test_remove_keeps_other_entries(tmp_path):
    path = tmp_path / "claude.json"
    path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "other": {"command": "x"},
                    "scholar-rag-mcp": build_entry("claude-desktop", _sample_endpoints()),
                },
                "top": 1,
            }
        )
    )
    assert remove_json_client(path, "claude-desktop") == "removed"
    data = json.loads(path.read_text())
    assert data["top"] == 1
    assert data["mcpServers"] == {"other": {"command": "x"}}


def test_remove_absent_when_no_entry(tmp_path):
    path = tmp_path / "claude.json"
    payload = {"mcpServers": {"other": {"command": "x"}}}
    path.write_text(json.dumps(payload))
    assert remove_json_client(path, "claude-desktop") == "absent"
    assert json.loads(path.read_text()) == payload


def test_remove_absent_when_root_key_missing(tmp_path):
    path = tmp_path / "claude.json"
    payload = {"theme": "dark"}
    path.write_text(json.dumps(payload))
    assert remove_json_client(path, "claude-code") == "absent"
    assert json.loads(path.read_text()) == payload


def test_remove_parse_failed_leaves_file_byte_identical(tmp_path):
    path = tmp_path / "claude.json"
    payload = b"{ bad"
    path.write_bytes(payload)
    assert remove_json_client(path, "claude-desktop") == "parse-failed"
    assert path.read_bytes() == payload


def test_remove_missing(tmp_path):
    path = tmp_path / "claude.json"
    assert remove_json_client(path, "claude-desktop") == "missing"
    assert not path.exists()


@pytest.mark.parametrize("client", ["claude-desktop", "claude-code", "opencode"])
def test_remove_roundtrip_each_client(client, tmp_path):
    path = tmp_path / f"{client}.json"
    merge_json_client(path, client, _sample_endpoints())
    assert remove_json_client(path, client) == "removed"


@pytest.mark.parametrize("client", ["claude-desktop", "claude-code", "opencode"])
def test_manual_snippet_parses_to_full_config(client):
    text = manual_snippet(client, _sample_endpoints())
    data = json.loads(text)
    assert data[ROOT_KEYS[client]]["scholar-rag-mcp"] == build_entry(
        client, _sample_endpoints()
    )


@pytest.mark.parametrize("client", ["claude-desktop", "claude-code", "opencode"])
def test_manual_snippet_contains_all_six_env_keys(client):
    text = manual_snippet(client, _sample_endpoints())
    entry = json.loads(text)[ROOT_KEYS[client]]["scholar-rag-mcp"]
    env = entry.get("env") or entry.get("environment")
    assert set(env) == set(ENDPOINT_ENV_VARS.values())


_ORIGINAL_TOML = 'model = "gpt-5"\n[mcp_servers.other]\ncommand = "other-mcp"\n'


def _codex_backups(tmp_path):
    return sorted(tmp_path.glob("config.toml.bak.*"))


def test_merge_codex_entry_roundtrip(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(_ORIGINAL_TOML)
    endpoints = _sample_endpoints()
    assert merge_codex(path, endpoints) == "written"
    data = tomllib.loads(path.read_text())
    entry = data["mcp_servers"]["scholar-rag-mcp"]
    assert entry["command"] == ["scholar-rag-mcp"]
    assert entry["env"] == _expected_env(endpoints)
    assert set(entry["env"]) == set(ENDPOINT_ENV_VARS.values())


def test_merge_codex_preserves_other_sections(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(_ORIGINAL_TOML)
    assert merge_codex(path, _sample_endpoints()) == "written"
    data = tomllib.loads(path.read_text())
    assert data["model"] == "gpt-5"
    assert data["mcp_servers"]["other"] == {"command": "other-mcp"}


def test_merge_codex_creates_backup_identical_to_original(tmp_path):
    path = tmp_path / "config.toml"
    original = b'a = 1\n[mcp_servers.other]\nb = "x"\n'
    path.write_bytes(original)
    assert merge_codex(path, _sample_endpoints()) == "written"
    backups = _codex_backups(tmp_path)
    assert len(backups) == 1
    timestamp = backups[0].name[len("config.toml.bak.") :]
    assert re.fullmatch(r"\d{14}", timestamp)
    assert backups[0].read_bytes() == original


def test_merge_codex_remerge_idempotent(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(_ORIGINAL_TOML)
    assert merge_codex(path, _sample_endpoints()) == "written"
    before = path.read_bytes()
    assert merge_codex(path, _sample_endpoints()) == "written"
    assert path.read_bytes() == before


def test_merge_codex_fresh_create_without_backup(tmp_path):
    path = tmp_path / "config.toml"
    assert merge_codex(path, _sample_endpoints()) == "created"
    data = tomllib.loads(path.read_text())
    assert data == {
        "mcp_servers": {
            "scholar-rag-mcp": {
                "command": ["scholar-rag-mcp"],
                "env": _expected_env(_sample_endpoints()),
            }
        }
    }
    assert _codex_backups(tmp_path) == []


def test_merge_codex_parse_failed_leaves_file_byte_identical(tmp_path):
    path = tmp_path / "config.toml"
    payload = b'the = [ not closed'
    path.write_bytes(payload)
    assert merge_codex(path, _sample_endpoints()) == "parse-failed"
    assert path.read_bytes() == payload
    assert _codex_backups(tmp_path) == []


def test_remove_codex_removed_preserves_other_sections(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(_ORIGINAL_TOML)
    merge_codex(path, _sample_endpoints())
    before = path.read_bytes()
    assert remove_codex(path) == "removed"
    data = tomllib.loads(path.read_text())
    assert data["model"] == "gpt-5"
    assert data["mcp_servers"] == {"other": {"command": "other-mcp"}}
    assert any(backup.read_bytes() == before for backup in _codex_backups(tmp_path))


def test_remove_codex_absent(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(_ORIGINAL_TOML)
    assert remove_codex(path) == "absent"
    assert path.read_text() == _ORIGINAL_TOML


def test_remove_codex_missing(tmp_path):
    path = tmp_path / "config.toml"
    assert remove_codex(path) == "missing"


def test_remove_codex_parse_failed_leaves_file_byte_identical(tmp_path):
    path = tmp_path / "config.toml"
    payload = b"not [ valid"
    path.write_bytes(payload)
    assert remove_codex(path) == "parse-failed"
    assert path.read_bytes() == payload
    assert _codex_backups(tmp_path) == []


def test_manual_snippet_codex_is_toml():
    text = setup_mod.manual_snippet("codex", _sample_endpoints())
    data = tomllib.loads(text)
    entry = data["mcp_servers"]["scholar-rag-mcp"]
    assert entry["command"] == ["scholar-rag-mcp"]
    assert entry["env"] == _expected_env(_sample_endpoints())


class _FakeCompleted:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _stub_uv(monkeypatch, *, returncode=0, stdout="", stderr="", uv="uv"):
    for key in ENDPOINT_ENV_VARS.values():
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(setup_mod.shutil, "which", lambda name: uv)
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return _FakeCompleted(returncode, stdout, stderr)

    monkeypatch.setattr(setup_mod.subprocess, "run", fake_run)
    return calls


def test_install_uv_missing_returns_1(tmp_path, monkeypatch, capsys):
    _fake_home(tmp_path, monkeypatch)
    _stub_uv(monkeypatch, uv=None)
    assert setup_mod.run_install(["--yes"]) == 1
    out = capsys.readouterr().out
    assert "uv not found" in out
    assert "curl" in out
    assert "pip install uv" in out
    assert "winget" in out
    assert not (tmp_path / ".claude.json").exists()
    assert not (tmp_path / ".config").exists()


def test_install_uv_failure_returns_1(tmp_path, monkeypatch, capsys):
    _fake_home(tmp_path, monkeypatch)
    _stub_uv(monkeypatch, returncode=1, stderr="fatal: could not resolve")
    assert setup_mod.run_install(["--yes"]) == 1
    assert "fatal: could not resolve" in capsys.readouterr().out
    assert not (tmp_path / ".claude.json").exists()


def test_install_uv_already_installed_continues(tmp_path, monkeypatch, capsys):
    _fake_home(tmp_path, monkeypatch)
    calls = _stub_uv(
        monkeypatch, returncode=4, stdout="", stderr="error: package already installed"
    )
    (tmp_path / ".claude.json").write_text(json.dumps({"top": 1}))
    assert setup_mod.run_install(["--yes"]) == 0
    data = json.loads((tmp_path / ".claude.json").read_text())
    assert data["top"] == 1
    assert data["mcpServers"]["scholar-rag-mcp"] == build_entry(
        "claude-code", ENDPOINT_DEFAULTS
    )
    assert "already installed" in capsys.readouterr().out
    assert calls == [["uv", "tool", "install", "scholar-rag-mcp"]]


def test_install_happy_path_merges_defaults(tmp_path, monkeypatch, capsys):
    _fake_home(tmp_path, monkeypatch)
    _stub_uv(monkeypatch)
    (tmp_path / ".claude.json").write_text(
        json.dumps({"mcpServers": {"other-server": {"command": "other"}}, "top": 1})
    )
    opencode_dir = tmp_path / ".config" / "opencode"
    opencode_dir.mkdir(parents=True)
    opencode_path = opencode_dir / "opencode.json"
    opencode_path.write_text(json.dumps({"mcp": {"other": {"type": "local"}}, "foo": "bar"}))
    assert setup_mod.run_install(["--yes"]) == 0
    claude = json.loads((tmp_path / ".claude.json").read_text())
    assert claude["top"] == 1
    assert claude["mcpServers"]["other-server"] == {"command": "other"}
    assert claude["mcpServers"]["scholar-rag-mcp"] == build_entry(
        "claude-code", ENDPOINT_DEFAULTS
    )
    opencode = json.loads(opencode_path.read_text())
    assert opencode["foo"] == "bar"
    assert opencode["mcp"]["other"] == {"type": "local"}
    assert opencode["mcp"]["scholar-rag-mcp"] == build_entry(
        "opencode", ENDPOINT_DEFAULTS
    )
    assert not (tmp_path / ".codex").exists()
    out = capsys.readouterr().out
    assert "install summary" in out
    assert "package: installed" in out
    assert "chat_base_url: http://127.0.0.1:8101/v1" in out
    assert "Model deployment" in out


def test_install_zero_clients_prints_snippets_and_returns_0(tmp_path, monkeypatch, capsys):
    _fake_home(tmp_path, monkeypatch)
    _stub_uv(monkeypatch)
    assert setup_mod.run_install(["--yes"]) == 0
    out = capsys.readouterr().out
    assert "no MCP client configs" in out
    for client in ("claude-desktop", "claude-code", "opencode"):
        assert manual_snippet(client, ENDPOINT_DEFAULTS) in out
    assert "[mcp_servers.scholar-rag-mcp]" in out


def test_install_parse_failed_prints_snippet_and_continues(tmp_path, monkeypatch, capsys):
    _fake_home(tmp_path, monkeypatch)
    _stub_uv(monkeypatch)
    bad = tmp_path / ".claude.json"
    payload = b"{ 'jsonc': true }"
    bad.write_bytes(payload)
    opencode_dir = tmp_path / ".config" / "opencode"
    opencode_dir.mkdir(parents=True)
    opencode_path = opencode_dir / "opencode.json"
    opencode_path.write_text(json.dumps({"mcp": {"other": {"type": "local"}}}))
    assert setup_mod.run_install(["--yes"]) == 0
    assert bad.read_bytes() == payload
    opencode = json.loads(opencode_path.read_text())
    assert opencode["mcp"]["other"] == {"type": "local"}
    assert "scholar-rag-mcp" in opencode["mcp"]
    out = capsys.readouterr().out
    assert "could not be parsed" in out
    assert manual_snippet("claude-code", ENDPOINT_DEFAULTS) in out


class _FakeStdin:
    def isatty(self):
        return False


def test_install_non_tty_without_yes_returns_1(tmp_path, monkeypatch, capsys):
    _fake_home(tmp_path, monkeypatch)
    _stub_uv(monkeypatch)
    monkeypatch.setattr(sys, "stdin", _FakeStdin())
    assert setup_mod.run_install([]) == 1
    out = capsys.readouterr().out
    assert "flag" in out
    assert "--yes" in out
    assert not (tmp_path / ".claude.json").exists()


def test_setup_main_unknown_subcommand_returns_2(capsys):
    assert setup_mod.setup_main(["frobnicate"]) == 2
    assert "uninstall" in capsys.readouterr().out


def test_main_dispatch_install_propagates_exit_code(tmp_path, monkeypatch, capsys):
    _fake_home(tmp_path, monkeypatch)
    _stub_uv(monkeypatch)
    (tmp_path / ".claude.json").write_text(json.dumps({"top": 1}))
    monkeypatch.setattr(sys, "argv", ["scholar-rag-mcp", "install", "--yes"])
    from scholar_rag.server.main import main

    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 0
    data = json.loads((tmp_path / ".claude.json").read_text())
    assert "scholar-rag-mcp" in data["mcpServers"]


def test_main_dispatch_help_returns_zero(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["scholar-rag-mcp", "--help"])
    from scholar_rag.server.main import main

    main()
    out = capsys.readouterr().out
    assert "install" in out
    assert "uninstall" in out


def test_main_dispatch_unknown_arg_exits_2(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["scholar-rag-mcp", "frobnicate"])
    from scholar_rag.server.main import main

    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 2
    assert "install" in capsys.readouterr().out


def _with_data_dir(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    monkeypatch.setenv("SCHOLAR_RAG_DATA_DIR", str(data_dir))
    return data_dir


def test_uninstall_removes_entries_and_keeps_others(tmp_path, monkeypatch, capsys):
    _fake_home(tmp_path, monkeypatch)
    _with_data_dir(tmp_path, monkeypatch)
    _stub_uv(monkeypatch, uv=None)
    claude = tmp_path / ".claude.json"
    claude.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "scholar-rag-mcp": build_entry("claude-code", _sample_endpoints()),
                    "other-server": {"command": "other"},
                },
                "top": 1,
            }
        )
    )
    opencode_dir = tmp_path / ".config" / "opencode"
    opencode_dir.mkdir(parents=True)
    opencode = opencode_dir / "opencode.json"
    opencode.write_text(json.dumps({"mcp": {"other": {"type": "local"}}}))
    assert setup_mod.run_uninstall(["--yes"]) == 0
    claude_data = json.loads(claude.read_text())
    assert claude_data["top"] == 1
    assert claude_data["mcpServers"] == {"other-server": {"command": "other"}}
    assert json.loads(opencode.read_text()) == {"mcp": {"other": {"type": "local"}}}
    out = capsys.readouterr().out
    assert "uninstall summary" in out
    assert "claude-code: removed" in out
    assert "opencode: absent" in out


def test_uninstall_codex_removed_with_backup(tmp_path, monkeypatch, capsys):
    _fake_home(tmp_path, monkeypatch)
    _with_data_dir(tmp_path, monkeypatch)
    _stub_uv(monkeypatch, uv=None)
    codex = tmp_path / ".codex" / "config.toml"
    codex.parent.mkdir(parents=True)
    codex.write_text(_ORIGINAL_TOML)
    merge_codex(codex, _sample_endpoints())
    before = codex.read_bytes()
    assert setup_mod.run_uninstall(["--yes"]) == 0
    data = tomllib.loads(codex.read_text())
    assert data["model"] == "gpt-5"
    assert data["mcp_servers"] == {"other": {"command": "other-mcp"}}
    backups = sorted((tmp_path / ".codex").glob("config.toml.bak.*"))
    assert any(backup.read_bytes() == before for backup in backups)
    assert "codex: removed" in capsys.readouterr().out


def test_uninstall_client_flag_targets_only_that_client(tmp_path, monkeypatch, capsys):
    _fake_home(tmp_path, monkeypatch)
    _with_data_dir(tmp_path, monkeypatch)
    _stub_uv(monkeypatch, uv=None)
    claude = tmp_path / ".claude.json"
    claude.write_text(
        json.dumps(
            {"mcpServers": {"scholar-rag-mcp": build_entry("claude-code", _sample_endpoints())}}
        )
    )
    opencode_dir = tmp_path / ".config" / "opencode"
    opencode_dir.mkdir(parents=True)
    opencode = opencode_dir / "opencode.json"
    opencode.write_text(
        json.dumps({"mcp": {"scholar-rag-mcp": build_entry("opencode", _sample_endpoints())}})
    )
    assert setup_mod.run_uninstall(["--client", "opencode", "--yes"]) == 0
    assert "scholar-rag-mcp" in json.loads(claude.read_text())["mcpServers"]
    assert "scholar-rag-mcp" not in json.loads(opencode.read_text())["mcp"]
    out = capsys.readouterr().out
    assert "opencode: removed" in out
    assert "claude-code" not in out


def test_uninstall_purge_declined_keeps_data_dir(tmp_path, monkeypatch, capsys):
    _fake_home(tmp_path, monkeypatch)
    data_dir = _with_data_dir(tmp_path, monkeypatch)
    _stub_uv(monkeypatch, uv=None)
    data_dir.mkdir()
    payload = data_dir / "index"
    payload.write_text("x")
    monkeypatch.setattr("builtins.input", lambda prompt: "no")
    assert setup_mod.run_uninstall(["--purge"]) == 0
    assert payload.read_text() == "x"
    assert "purge aborted" in capsys.readouterr().out


def test_uninstall_purge_confirmed_deletes_data_dir(tmp_path, monkeypatch, capsys):
    _fake_home(tmp_path, monkeypatch)
    data_dir = _with_data_dir(tmp_path, monkeypatch)
    _stub_uv(monkeypatch, uv=None)
    data_dir.mkdir()
    (data_dir / "index").write_text("x")
    monkeypatch.setattr("builtins.input", lambda prompt: "yes")
    assert setup_mod.run_uninstall(["--purge"]) == 0
    assert not data_dir.exists()
    assert "purged" in capsys.readouterr().out


def test_uninstall_purge_yes_skips_prompt(tmp_path, monkeypatch, capsys):
    _fake_home(tmp_path, monkeypatch)
    data_dir = _with_data_dir(tmp_path, monkeypatch)
    _stub_uv(monkeypatch, uv=None)
    data_dir.mkdir()
    (data_dir / "index").write_text("x")

    def fail_prompt(text):
        raise AssertionError("prompt must not be called with --yes")

    monkeypatch.setattr("builtins.input", fail_prompt)
    assert setup_mod.run_uninstall(["--purge", "--yes"]) == 0
    assert not data_dir.exists()
    assert str(data_dir.absolute()) in capsys.readouterr().out


def test_uninstall_purge_rmtree_failure_returns_1(tmp_path, monkeypatch, capsys):
    _fake_home(tmp_path, monkeypatch)
    data_dir = _with_data_dir(tmp_path, monkeypatch)
    _stub_uv(monkeypatch, uv=None)
    data_dir.mkdir()

    def fail_rmtree(path):
        raise OSError("denied")

    monkeypatch.setattr(setup_mod.shutil, "rmtree", fail_rmtree)
    assert setup_mod.run_uninstall(["--purge", "--yes"]) == 1
    out = capsys.readouterr().out
    assert str(data_dir.absolute()) in out
    assert "manually" in out


def test_uninstall_purge_missing_data_dir_is_not_error(tmp_path, monkeypatch, capsys):
    _fake_home(tmp_path, monkeypatch)
    _with_data_dir(tmp_path, monkeypatch)
    _stub_uv(monkeypatch, uv=None)
    assert setup_mod.run_uninstall(["--purge", "--yes"]) == 0
    assert "does not exist" in capsys.readouterr().out


def test_uninstall_uv_missing_returns_0_with_note(tmp_path, monkeypatch, capsys):
    _fake_home(tmp_path, monkeypatch)
    data_dir = _with_data_dir(tmp_path, monkeypatch)
    _stub_uv(monkeypatch, uv=None)
    assert setup_mod.run_uninstall(["--yes"]) == 0
    out = capsys.readouterr().out
    assert "uv not found" in out
    assert "uv tool uninstall scholar-rag-mcp" in out
    assert "(kept)" in out
    assert str(data_dir) in out


def test_uninstall_uv_failure_returns_0_with_note(tmp_path, monkeypatch, capsys):
    _fake_home(tmp_path, monkeypatch)
    _with_data_dir(tmp_path, monkeypatch)
    calls = _stub_uv(monkeypatch, returncode=1, stderr="tool not installed")
    assert setup_mod.run_uninstall(["--yes"]) == 0
    assert calls == [["uv", "tool", "uninstall", "scholar-rag-mcp"]]
    assert "likely not installed" in capsys.readouterr().out


def test_uninstall_uv_success_prints_removed(tmp_path, monkeypatch, capsys):
    _fake_home(tmp_path, monkeypatch)
    _with_data_dir(tmp_path, monkeypatch)
    calls = _stub_uv(monkeypatch)
    assert setup_mod.run_uninstall(["--yes"]) == 0
    assert calls == [["uv", "tool", "uninstall", "scholar-rag-mcp"]]
    assert "package: removed" in capsys.readouterr().out


def test_setup_main_uninstall_dispatch(tmp_path, monkeypatch, capsys):
    _fake_home(tmp_path, monkeypatch)
    _with_data_dir(tmp_path, monkeypatch)
    _stub_uv(monkeypatch, uv=None)
    claude = tmp_path / ".claude.json"
    claude.write_text(
        json.dumps({"mcpServers": {"scholar-rag-mcp": build_entry("claude-code", _sample_endpoints())}})
    )
    assert setup_mod.setup_main(["uninstall", "--yes"]) == 0
    assert "scholar-rag-mcp" not in json.loads(claude.read_text())["mcpServers"]
    assert "claude-code: removed" in capsys.readouterr().out
