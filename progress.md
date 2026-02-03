# 进度记录（Progress）

## 已完成
- 目标/字段抽取、分解检索、多查询融合、结构化抽取与“有证据不拒答”的链路改造。
- 列表型问题加入“判定标准 + 关键描写/情节 + 简短总结”的自然输出模板（通用化，不耦合单题）。
- 新增哈利波特专用 `README.md`。
- 新增并扩充题集 `scripts/questions.jsonl`（带 `gold_chunk_ids`）。
- 新增 A/B 测试脚本 `scripts/ab_test.py` 与题集扩充方案 `scripts/questions_plan.md`。
- 修复本地运行 `eval.py` 路径指向问题（本地默认路径）。
- 修复 `eval.py` 与 `retrieve()` 返回值解包不一致。

## 正在进行
- 暂无（当前按你指示暂停 A/B 测试推进）。

## 待后期完成
- A/B 测试完善：确认 expand / rerank 真正生效并输出占比统计。
- 扩展题集至更高难度（多段/多跳/列表型），以便观察 expand / rerank 的增益。
- 若需要：对 `ab_test.py` 增加“是否开启”核验与汇总展示。

## 当前卡点 / 风险
- A/B 三组结果高度相似：可能因 rerank 服务未健康或 expand 无有效生成；也可能因题集过于“单 chunk 命中”。需要后续验证与改进题集。

