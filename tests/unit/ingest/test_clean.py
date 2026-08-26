from __future__ import annotations

from scholar_rag.ingest.clean import stage_clean


def test_removes_markdown_images() -> None:
    md = "Before ![](img/a.png) and ![alt text](b.jpg) after\n\nKeep this."
    out = stage_clean(md)
    assert "![](" not in out
    assert "b.jpg" not in out
    assert "Keep this." in out


def test_removes_figure_caption_lines() -> None:
    md = "Some text.\n\nFigure 1. Overview of the pipeline\n\nFigure S2 bar chart\n\nBody paragraph."
    out = stage_clean(md)
    assert "Figure 1." not in out
    assert "Figure S2" not in out
    assert "Body paragraph." in out
    assert "Some text." in out


def test_removes_html_tables() -> None:
    md = "before\n<table><tr><td>x</td></tr></table>\nafter"
    out = stage_clean(md)
    assert "<table>" not in out
    assert "before" in out
    assert "after" in out


def test_removes_pipe_tables() -> None:
    md = "intro\n| a | b |\n| --- | --- |\n| 1 | 2 |\nconclusion"
    out = stage_clean(md)
    assert "| a | b |" not in out
    assert "intro" in out
    assert "conclusion" in out


def test_removes_table_caption_lines() -> None:
    md = "Table 1. Measurements summary\n\nreal text"
    out = stage_clean(md)
    assert "Table 1." not in out
    assert "real text" in out


def test_collapses_excess_blank_lines() -> None:
    md = "a\n\n\n\n\nb\n\nc"
    out = stage_clean(md)
    assert "\n\n\n" not in out


def test_empty_input_returns_empty() -> None:
    assert stage_clean("") == ""
    assert stage_clean("   \n  \n") == ""


def test_output_has_single_trailing_newline() -> None:
    out = stage_clean("one\n\ntwo")
    assert out == "one\n\ntwo\n"
