from __future__ import annotations

from scholar_rag.services import mineru


def test_parse_api_backend_ensures_sidecar(tmp_path, monkeypatch):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    calls: list[object] = []

    class _Settings:
        mineru_backend = "api"
        mineru_api_url = "http://127.0.0.1:8010"

    monkeypatch.setattr(mineru.Settings, "load", staticmethod(lambda: _Settings()))
    monkeypatch.setattr(
        mineru, "_ensure_api_sidecar", lambda settings: calls.append(settings)
    )
    monkeypatch.setattr(mineru, "_parse_via_api", lambda path, url: "# md")
    assert mineru.parse(pdf) == "# md"
    assert len(calls) == 1


def test_parse_python_backend_skips_sidecar(tmp_path, monkeypatch):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    calls: list[object] = []

    class _Settings:
        mineru_backend = "python"
        mineru_api_url = "http://127.0.0.1:8010"

    monkeypatch.setattr(mineru.Settings, "load", staticmethod(lambda: _Settings()))
    monkeypatch.setattr(
        mineru, "_ensure_api_sidecar", lambda settings: calls.append(settings)
    )
    monkeypatch.setattr(mineru, "_parse_via_python", lambda path: "# md")
    assert mineru.parse(pdf) == "# md"
    assert calls == []


def test_ensure_api_sidecar_delegates(monkeypatch):
    seen: list[object] = []

    import scholar_rag.services.mineru_sidecar as sidecar

    monkeypatch.setattr(sidecar, "ensure_running", lambda settings: seen.append(settings))
    sentinel = object()
    mineru._ensure_api_sidecar(sentinel)
    assert seen == [sentinel]
