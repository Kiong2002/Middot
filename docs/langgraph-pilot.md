# LangGraph / LangSmith 纵切试验

本试验只验证地点消歧的人机中断链路，不替换现有主 Agent，也不改变前端 SSE 协议。

## 安全边界

- `MIDDOT_AGENT_RUNTIME` 默认是 `legacy`，因此仅安装依赖不会切换流量。
- 候选查询、自动判断、等待用户、恢复校验、位置写入分别是独立节点。
- `interrupt` 节点没有外部副作用；恢复时不会重复查询候选或重复执行 AI 判断。
- 写入使用由 `thread_id + request_id + participant_id + candidate_id` 生成的稳定
  `operation_id`。接入真实数据库时，必须给该字段增加唯一约束并返回已有结果。
- 测试使用 `InMemorySaver`；生产不得使用它。正式接入前需实现持久 checkpointer。
- LangSmith 默认关闭且采样率为 0。即使启用，默认也只上传字符串长度与哈希，不上传
  用户原文、地点名和地址；Trace 失败不影响业务。

## 试验链路

1. `resolve_candidates`：高德候选查询（只读，可缓存）。
2. `auto_select`：AI 在候选中判断；即使只有一个候选，也只有置信度达到阈值才
   自动选择。
3. `request_choice`：不确定时暂停并向用户展示候选。
4. `resume`：同时校验当前 `interrupt_id` 与本次候选集合内的 `candidate_id`，旧卡片
   或伪造选项不能推进状态。
5. `commit_location`：使用幂等 `operation_id` 恰好一次写入。

## 运行测试

```bash
/root/.venvs/middot-langgraph-pilot/bin/python -m pytest -q tests/test_agent_runtime.py
```

测试覆盖默认回退、自动选择、等待后恢复、模拟重建、伪造候选拒绝、幂等写入与
Trace 脱敏。下一阶段才会把现有高德查询、AI selector、数据库写入和 SSE 事件适配到
这些接口，并通过 feature flag 做内部小流量验证。
