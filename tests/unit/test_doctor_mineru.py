from __future__ import annotations

import importlib.util
from pathlib import Path

_DOCTOR_PATH = Path(__file__).resolve().parents[2] / "scripts" / "doctor.py"
_spec = importlib.util.spec_from_file_location("doctor_under_test", _DOCTOR_PATH)
doctor = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(doctor)


class _Settings:
    def __init__(self, tmp_path, managed, url="http://127.0.0.1:8010"):
        self.data_dir = tmp_path
        self.mineru_backend = "api"
        self.mineru_managed = managed
        self.mineru_api_url = url
        self.mineru_bin = "mineru"


def test_doctor_managed_env_missing_reports_failure(tmp_path):
    report = doctor._Report()
    doctor._mineru_report(report, _Settings(tmp_path, managed=True))
    labels = [(ok, label) for ok, label, _note in report._items]
    assert (False, "mineru (managed env)") in labels


def _managed_binary(tmp_path):
    import sys as _sys

    if _sys.platform == "win32":
        return tmp_path / "mineru-env" / "Scripts" / "mineru-api.exe"
    return tmp_path / "mineru-env" / "bin" / "mineru-api"


def test_doctor_managed_env_present_reports_pass(tmp_path):
    binary = _managed_binary(tmp_path)
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"")
    report = doctor._Report()
    doctor._mineru_report(report, _Settings(tmp_path, managed=True))
    env_items = [item for item in report._items if item[1] == "mineru (managed env)"]
    assert env_items and env_items[0][0] is True


def test_doctor_managed_unreachable_is_not_failure(tmp_path, monkeypatch):
    binary = _managed_binary(tmp_path)
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"")
    import httpx

    def _boom(url, timeout=None):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(doctor.httpx, "get", _boom)
    report = doctor._Report()
    doctor._mineru_report(report, _Settings(tmp_path, managed=True))
    api_items = [item for item in report._items if item[1] == "mineru (api)"]
    assert api_items and api_items[0][0] is True
    assert "auto-start" in api_items[0][2]


def test_doctor_unmanaged_unreachable_is_failure(tmp_path, monkeypatch):
    import httpx

    def _boom(url, timeout=None):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(doctor.httpx, "get", _boom)
    report = doctor._Report()
    doctor._mineru_report(report, _Settings(tmp_path, managed=False))
    api_items = [item for item in report._items if item[1] == "mineru (api)"]
    assert api_items and api_items[0][0] is False
