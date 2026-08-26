from __future__ import annotations

import re


def stage_clean(markdown: str) -> str:
    if not markdown.strip():
        return ""
    text = re.sub(r"!\[.*?\]\(.*?\)\s*", "", markdown)
    text = re.sub(
        r"^(?:Figure|Fig\.)\s+S?\d+.*$",
        "",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    text = re.sub(r"<table[\s\S]*?</table>", "", text, flags=re.IGNORECASE)
    lines = text.split("\n")
    cleaned: list[str] = []
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if stripped.startswith("|"):
            while index < len(lines) and lines[index].strip().startswith("|"):
                index += 1
            continue
        cleaned.append(lines[index])
        index += 1
    text = "\n".join(cleaned)
    text = re.sub(
        r"^Table\s+S?\d+.*$",
        "",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"
