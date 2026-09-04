# v0.4.0 Release Notes

> 一键 MinerU sidecar + api 默认后端

## Breaking
- `mineru_backend` 默认值由 `python` 改为 `api`。依赖进程内 `import mineru`
  的用户需显式设置 `SCHOLAR_RAG_MINERU_BACKEND=python`

## New features
- `scholar-rag-mcp install --with-mineru`：自动建立隔离 Python 3.12 venv 并安装
  `mineru[core]==3.4.5`，写入托管配置（`mineru_managed=true`）
- 托管模式下首次解析自动拉起 `mineru-api`（`GET /health` 探测、最长等待 600s、
  日志 `mineru-api.log`、pid 文件 `mineru-api.pid`）；`uninstall` 自动停止
- `doctor` 输出托管环境存在性与"首次解析自动启动"提示

## Compatibility
- MinerU 要求 Python >=3.10,<3.14；托管 venv 固定 3.12，与主包 Python 版本无关
- 首次解析触发 MinerU 模型自动下载（中国大陆可设 `MINERU_MODEL_SOURCE=modelscope`）
- `python`/`cli` 后端保留；自管 `mineru-api`（非 localhost 或 `mineru_managed=false`）
  行为不变

## Upgrade
- 新用户：`uvx scholar-rag-mcp install --with-mineru`
- 既有用户：无需改动；如需托管执行 `scholar-rag-mcp install --with-mineru`