import os
import re
import threading
import time

import pytest

from scholar_rag.core.config import Settings
from scholar_rag.core.errors import JobNotFoundError
from scholar_rag.core.jobs import JobManager


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("SCHOLAR_RAG_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))


@pytest.fixture
def manager(tmp_path):
    instance = JobManager(Settings(data_dir=tmp_path / "data"))
    yield instance
    instance.close()


def _wait(manager, job_id, statuses, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = manager.get(job_id)
        if record.status in statuses:
            return record
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not reach {statuses} (last: {manager.get(job_id).status})")


def test_submit_runs_handler_and_marks_succeeded(manager, tmp_path):
    entered = threading.Event()
    release = threading.Event()

    def handler(job_id, params):
        assert params == {"a": 1}
        entered.set()
        release.wait(5)
        manager.set_total(job_id, 3)
        manager.update_progress(job_id, 2, "doc2")
        return {"ok": True, "n": len(params)}

    manager.register_handler("work", handler)
    job_id = manager.submit("work", "kb", {"a": 1})
    assert job_id
    assert entered.wait(5)
    running = manager.get(job_id)
    assert running.status == "running"
    release.set()
    record = _wait(manager, job_id, {"succeeded"})
    assert record.status == "succeeded"
    assert record.kb == "kb"
    assert record.type == "work"
    assert record.total == 3
    assert record.done == 2
    assert record.current_doc == "doc2"
    assert record.error is None
    assert record.result == {"ok": True, "n": 1}
    assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", record.created_at)
    assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", record.updated_at)


def test_handler_failure_marks_failed_with_error(manager):
    def handler(job_id, params):
        raise ValueError("boom")

    manager.register_handler("work", handler)
    job_id = manager.submit("work", "kb", {})
    record = _wait(manager, job_id, {"failed"})
    assert record.status == "failed"
    assert record.error == "boom"
    assert record.result is None


def test_get_unknown_job_raises_job_not_found(manager):
    with pytest.raises(JobNotFoundError):
        manager.get("nope")


def test_job_without_registered_handler_fails(manager):
    job_id = manager.submit("mystery", "kb", {})
    record = _wait(manager, job_id, {"failed"})
    assert record.status == "failed"
    assert "no handler" in record.error


def test_recover_interrupted_marks_running_and_counts(manager):
    entered = threading.Event()
    release = threading.Event()

    def handler(job_id, params):
        entered.set()
        release.wait(5)
        return {}

    manager.register_handler("work", handler)
    job_id = manager.submit("work", "kb", {})
    assert entered.wait(5)
    assert manager.get(job_id).status == "running"
    assert manager.recover_interrupted() == 1


def test_recovered_job_stays_interrupted_after_worker_finishes(manager):
    entered = threading.Event()
    release = threading.Event()

    def handler(job_id, params):
        entered.set()
        release.wait(5)
        return {"done": True}

    manager.register_handler("work", handler)
    job_id = manager.submit("work", "kb", {})
    assert entered.wait(5)
    assert manager.recover_interrupted() == 1
    release.set()
    time.sleep(0.5)
    record = manager.get(job_id)
    assert record.status == "interrupted"
    assert record.result is None


def test_recover_interrupted_ignores_succeeded_jobs(manager):
    manager.register_handler("work", lambda job_id, params: {"ok": True})
    job_id = manager.submit("work", "kb", {})
    _wait(manager, job_id, {"succeeded"})
    assert manager.recover_interrupted() == 0


def test_concurrent_progress_updates_are_safe(manager):
    entered = threading.Event()
    release = threading.Event()
    errors: list[Exception] = []

    def worker(n):
        try:
            for i in range(20):
                manager.update_progress(job_id, i, f"doc-{n}-{i}")
        except Exception as exc:
            errors.append(exc)

    def handler(job_id_, params):
        entered.set()
        release.wait(5)
        threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        manager.update_progress(job_id_, 999, "final")
        return {"ok": True}

    manager.register_handler("work", handler)
    job_id = manager.submit("work", "kb", {})
    assert entered.wait(5)
    release.set()
    record = _wait(manager, job_id, {"succeeded"})
    assert errors == []
    assert record.done == 999
    assert record.current_doc == "final"


def test_delete_by_kb_removes_only_matching_rows(manager):
    manager.register_handler("work", lambda job_id, params: {"ok": True})
    j1 = manager.submit("work", "kb1", {})
    j2 = manager.submit("work", "kb2", {})
    _wait(manager, j1, {"succeeded"})
    _wait(manager, j2, {"succeeded"})
    assert manager.delete_by_kb("kb1") == 1
    with pytest.raises(JobNotFoundError):
        manager.get(j1)
    assert manager.get(j2).kb == "kb2"
