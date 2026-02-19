# RAG_web 常用命令速查

## 运行项目（推荐：Docker Compose）

### 首次启动
```bash
cp .env.example .env
# 编辑 .env：至少填 OPENAI_API_KEY + 账号密码

docker compose build
docker compose up -d
```

### 之后启动 / 停止 / 重启
```bash
docker compose up -d
docker compose down
docker compose restart
```

### 修改 `.env` 后如何生效
仅修改环境变量（例如检索门槛/模型名）通常不需要重建镜像；重启容器即可让 `env_file: .env` 重新加载：
```bash
docker compose up -d
```
如只想重启后端/前端其中一个：
```bash
docker compose restart api
docker compose restart web
```

如果你同时改了代码或 Dockerfile，想确保使用最新镜像再启动：
```bash
docker compose up -d --build
```
如需强制不使用缓存（更慢，但最“干净”）：
```bash
docker compose build --no-cache
docker compose up -d
```

### 查看状态与日志
```bash
docker compose ps
docker compose logs -f
docker compose logs -f api
docker compose logs -f web
```

### 访问地址
- Web UI：`http://localhost:3001`
- API：`http://localhost:8000`（健康检查：`/health`）

## 知识库 ingest（抽取->分块->向量化->FAISS 落盘）

### 完整 ingest（会写入 FAISS）
```bash
docker compose run --rm api python -m app.ingest
```

如遇到 embeddings 连接不稳/超时（常见于代理/网关），优先在 `.env` 调整：
```bash
INGEST_EMBED_BATCH_SIZE=16
INGEST_EMBED_TIMEOUT_SECONDS=60
INGEST_EMBED_MAX_RETRIES=6
INGEST_EMBED_BACKOFF_SECONDS=1
INGEST_EMBED_BACKOFF_MAX_SECONDS=15
OPENAI_EMBED_TIMEOUT_SECONDS=30
```

如你的 PDF 前面有大量目录/前言/封面等“非正文”内容，建议先设置正文起始页并做 chunk-only 验证：
```bash
INGEST_CONTENT_START_PAGE=38
INGEST_MIN_CHUNK_CHARS=120
docker compose run --rm --no-deps api python -m app.ingest --chunk-only
```

如中途 API Key 失效/网络断开：本项目会在 `storage/faiss/ingest_state.json` 记录进度，并且每个 batch 成功就会落盘 FAISS；恢复后直接重新运行同一条 ingest 命令即可继续。

如需强制从头开始（不续跑）：
```bash
docker compose run --rm api python -m app.ingest --no-resume
```

### 只做切分（不跑 embedding、不重建 FAISS）
```bash
docker compose run --rm api python -m app.ingest --chunk-only
# 等价：--no-embed
```

### 产物位置
- `storage/chunks.jsonl`
- `storage/faiss/index.faiss`
- `storage/faiss/id_map.json`

## 本地开发（不使用 Docker）

### 后端（FastAPI）
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt

# 让后端读到 .env（任选其一）
export $(cat .env | sed 's/#.*//g' | xargs)
# 或：手动在环境变量里设置 OPENAI_API_KEY 等

uvicorn app.main:app --reload --port 8000 --app-dir backend
```

### 前端（Next.js）
```bash
cd web
npm install
npm run dev
```

## 评测脚本
```bash
python scripts/eval.py --questions scripts/questions.jsonl --report_dir storage/reports
```

## 常见排障

## 性能优化（把回答压到 ~10s）

优先改 `.env` 这几项（越小越快）：
```bash
OPENAI_CHAT_MODEL=gpt-4.1-mini          # 换更快模型可显著降时延
OPENAI_CHAT_MAX_TOKENS=400              # 输出越短越快（复杂问题可提高到 900 左右）

RAG_MAX_CHUNKS=4                        # 发送给大模型的证据条数（复杂问题可提高到 8 左右）
RAG_EVIDENCE_CHUNK_MAX_CHARS=500        # 每条证据的最大字符数
RAG_EVIDENCE_TOTAL_MAX_CHARS=2500       # 所有证据合计上限（最关键）
```

并确保这些耗时开关是关闭状态（除非你确实需要它们）：
```bash
# RETRIEVAL_RERANK_ENABLED=true
# RETRIEVAL_EXPAND_ENABLED=true
# RETRIEVAL_HYBRID_ALPHA=0.7
```

### 如何确认瓶颈
看后端日志里的 timing（重建后生效）：
```bash
docker compose logs --tail 300 api | rg '"event": "timing"'
```
重点关注 `timing_ms.retrieve`：正常应是几十毫秒~几百毫秒；如果是秒级，通常是索引/数据未缓存或磁盘 I/O 过慢。

如果日志里 `timing_detail_ms.embed` 占了大头（比如 3000ms+），说明瓶颈在 embeddings 接口（外部网络/模型速度），优先：
- 把 `.env` 的 `OPENAI_EMBED_MODEL` 改成 `text-embedding-3-small`（更快）
- 需要时启用 `RETRIEVAL_FALLBACK_BM25_ON_EMBED_FAILURE=true`，避免 embeddings 偶发超时把整次请求拖到几十秒

### 开发期如何避免反复 `pip install`
默认 `docker compose up -d --build` 会触发镜像重建；如果缓存被清掉/基础镜像更新，就会再次跑 `pip install`。

本仓库提供了本地开发用的 `docker-compose.override.yml`（会自动生效）：
- 把 `./backend/app` 挂载到容器的 `/app/app`（改 Python 代码不需要 rebuild）
- `uvicorn` 启用 `--reload`（改代码自动重载）

常用命令：
```bash
# 不触发 build（最快）
docker compose up -d --no-build

# 只重启 API（一般不用，reload 会自动重载）
docker compose restart api

# 只有当你改了 backend/requirements.txt 才需要
docker compose build api
docker compose up -d --no-build api
```

### ingest 提示 “PDF 文本抽取结果过少”
- 通常是扫描版 PDF（图片）或文本不可抽取：换文本版 PDF 或先 OCR。
- 可在 `.env` 调整：`INGEST_EXTRACTOR` / `INGEST_MIN_NONEMPTY_PAGES` / `INGEST_MIN_TOTAL_CHARS`

## 本地 Rerank 服务（bge-reranker-base）

本项目支持把“重排（rerank）”从 OpenAI 切换为本地开源模型服务（独立容器），API 通过 HTTP 调用。

### 启动方式
1) `.env` 开启 rerank 并指向本地服务：
```bash
RETRIEVAL_RERANK_ENABLED=true
RETRIEVAL_RERANK_BASE_URL=http://rerank:9001
RETRIEVAL_RERANK_CANDIDATES=12
RETRIEVAL_RERANK_MAX_CHARS=400
```

2) 构建并启动：
```bash
docker compose up -d --build
```

重建并启动（推荐）：docker compose up -d --build --force-recreate

说明：
- 首次启动会从 Hugging Face 下载模型权重（需要网络）；缓存目录：`storage/hf_cache/`
- 内存紧张时建议保持 `RERANK_PRELOAD_MODEL=false`（默认懒加载：第一次请求才加载模型）

### 预下载/预热（可选）
- 预下载：启动后访问一次 `http://localhost:9001/health` 并发一条 rerank 请求即可触发下载与加载
- 预热：把 `.env` 的 `RERANK_PRELOAD_MODEL=true`，然后 `docker compose restart rerank`
