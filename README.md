# 哈利波特小说 RAG（FastAPI + FAISS + Next.js）

本项目仅面向《哈利波特》小说知识库的检索与问答。
唯一知识库文件：`data/book.pdf`（哈利波特小说 PDF）。

## 快速开始（Docker）

```bash
cp .env.example .env
# 编辑 .env：至少填 OPENAI_API_KEY + 账号密码

docker compose build
docker compose up -d
```

首次 ingest（抽取 → 分块 → 向量化 → FAISS 落盘）：
```bash
docker compose run --rm api python -m app.ingest
```

只验证切分（不跑 embeddings）：
```bash
docker compose run --rm --no-deps api python -m app.ingest --chunk-only
```

产物位置：
- `storage/chunks.jsonl`
- `storage/faiss/index.faiss`
- `storage/faiss/id_map.json`

## 访问入口
- Web UI：`http://localhost:3000`
- API：`http://localhost:8000/health`

## 评测

准备带 `gold_chunk_ids` 的问题集：
- `scripts/questions.jsonl`

运行：
```bash
python scripts/eval.py --questions scripts/questions.jsonl --report_dir storage/reports
```

## 目标与别名

可在 `storage/aliases.json` 配置别名映射（如“多个角色” → 实体列表）。
实体索引在 ingest 时自动构建：`storage/entity_index.json`。

## 账号登录

后端使用 session cookie 鉴权（`rag_session`），未登录无法访问 UI/API。  
账号密码在 `.env`：
- 研究员：`AUTH_USERNAME` / `AUTH_PASSWORD`
- 管理员：`AUTH_ADMIN_USERNAME` / `AUTH_ADMIN_PASSWORD`

