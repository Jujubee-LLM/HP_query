# 上线指南（RAG_web）

本文记录项目上线相关信息；以后所有“上线相关内容”都写在此文件。

## 最省事的上线方式
- 一台云服务器（VPS）跑 `docker compose`（API + Web + rerank 同机）。
- 域名 + DNS + HTTPS 反向代理对外提供访问。

## 需要的网站/服务（最小集合）
- 域名注册商：买域名。
- DNS：Cloudflare 免费 DNS。
- 云服务器：DigitalOcean / Linode 等 VPS。
- HTTPS 证书：Let’s Encrypt 免费证书。
- OpenAI API：按量计费。

## 免费域名/子域
- 没有真正“免费的顶级域名”（一般都需要购买）。
- 有免费二级域名（子域）：如 `*.vercel.app` / `*.netlify.app` / `*.pages.dev`。
- 这些适合前端/静态站；自定义域名仍需自购。
- 本项目用 session cookie，前端和 API 最好同一可注册域（例如 `app.example.com` + `api.example.com`）。
  若前端在 `vercel.app` 而 API 在别的域，可能需要改 cookie 策略或改鉴权方案。

## 费用大概
- 服务器（月付）：1GB/2GB/4GB 等不同档位；建议至少 4GB RAM 起步。
- 域名（年付）：按后缀不同，价格差异较大。
- DNS：免费。
- HTTPS 证书：免费。
- OpenAI：按 token 计费（模型不同价格不同）。

## 上线大概耗时
- 账号注册 + 服务器创建：30–60 分钟
- 域名 + DNS 生效：通常几分钟到 24 小时
- 部署 + 首次 ingest：约 1–3 小时

## 本项目上线步骤（单机 Docker Compose）
1) 购买域名、创建云服务器（推荐 4GB RAM 起）。
2) 在服务器安装 Docker + Docker Compose。
3) 把代码和数据放到服务器：  
   - 上传 `data/book.pdf`  
   - 复制 `.env.example` 为 `.env`，填写 `OPENAI_API_KEY`、`AUTH_*`、`SESSION_SECRET`  
   - `WEB_ORIGIN` 设为前端域名（如 `https://example.com`）  
   - `NEXT_PUBLIC_API_BASE_URL` 设为 API 域名（如 `https://api.example.com`）
4) 构建与启动（生产不要带 `docker-compose.override.yml`）：  
   - `docker compose -f docker-compose.yml build`  
   - `docker compose -f docker-compose.yml up -d`  
   - `docker compose -f docker-compose.yml run --rm api python -m app.ingest`
5) 配置反向代理 + HTTPS（推荐 Caddy，自动签 Let’s Encrypt）：  
   - 前端域名反代 `localhost:3001`  
   - API 域名反代 `localhost:8000`
6) DNS 设置：  
   - `example.com` 和 `api.example.com` 指向服务器 IP  
   - 域名注册商把 NS 改到 Cloudflare
7) 安全加固（建议）：  
   - 防火墙只放行 80/443  
   - 8000/9001 不直接暴露公网

## 上线后的迭代（推荐流程）
- 代码迭代（手动）：  
  1) `git pull`  
  2) `docker compose -f docker-compose.yml up -d --build`
- 数据/知识库更新：  
  1) 替换 `data/book.pdf`  
  2) 重新跑 `python -m app.ingest`
- 回滚：  
  - 用 git tag/commit 回到上一个版本并重建
- 建议：有测试环境先验证，再更新生产。

## 复用本地 ingest（不必重新跑）
- 可以把本地 `storage/` 拷到服务器直接用，不必重新 ingest。
- 建议一起拷贝：  
  - `storage/faiss/index.faiss`  
  - `storage/faiss/id_map.json`  
  - `storage/chunks.jsonl`  
  - `storage/entity_index.json`（若存在）  
  - `storage/aliases.json`（若有自定义别名）  
  - `storage/logs.jsonl`（可选）
- 必须保持和生成时一致的嵌入与分块配置：
  - `OPENAI_EMBED_MODEL` / `EMBED_*`  
  - `INGEST_*`（分块/抽取参数）  
  - `DATA_PDF_PATH` 对应的 PDF 内容
- 以下情况需要重新 ingest：  
  - 更换 PDF  
  - 更换嵌入模型/提供方  
  - 调整分块/抽取参数
- 仅更换 chat 模型或 rerank 设置，无需重新 ingest。
