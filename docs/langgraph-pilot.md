# LangGraph / LangSmith 渐进迁移

第一阶段地点纵切已经接入现有地点工具、高德候选、AI 判断、数据库写入和前端 SSE
协议。第二阶段把主 Agent 的 planner、工具批次、确定性补偿、状态校验、等待和最终
回复也迁入显式 LangGraph 节点；15 个领域工具和前端事件协议保持不变。

## 安全边界

- `MIDDOT_AGENT_RUNTIME` 默认是 `legacy`，因此仅安装依赖不会切换流量。
- `MIDDOT_AGENT_ORCHESTRATOR` 独立控制主编排，默认也是 `legacy`。可以只回退主图，
  同时保留已经验证过的地点图。
- 候选查询、自动判断、等待用户、恢复校验、位置写入分别是独立节点。
- `interrupt` 节点没有外部副作用；恢复时不会重复查询候选或重复执行 AI 判断。
- 写入使用由 `thread_id + request_id + participant_id + candidate_id` 生成的稳定
  `operation_id`。接入真实数据库时，必须给该字段增加唯一约束并返回已有结果。
- 普通选择卡也写入 `agent_choice_interrupts`；进程重启后，服务端仍能按设备身份、
  一次性 token 和可见 label 校验并继续任务。
- 单元测试使用 `InMemorySaver`；应用接入使用独立 SQLite checkpoint，单机重启后可以
  恢复。未来多实例部署时必须换成 PostgresSaver。
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

## 主 Agent 图

1. `planner`：调用 DeepSeek，保存完整原始返回与工具调用。
2. `execute_tools`：执行现有工具、投影 `tool_call/tool_result/state_patch` SSE，并在
   单次 turn 内去重相同成功动作。
3. `deterministic_compensation`：交通方式变化后，模型漏调时确定性补算路线。
4. `verify`：检查搜索结果与路线状态是否闭环，问题返回 planner 修复。
5. `wait`：地点卡或普通选择卡出现后停止后续工具并稳定结束当前 SSE。
6. `finalize`：写对话历史、结束本地 trace，并发送最终 token/done。

每个 HTTP turn 使用 `agent:{conversation_id}:{trace_id}` 作为独立 thread_id。checkpoint
是编排状态的事实源；内存 session 只保留页面所需投影。单机使用 SQLite WAL，多实例
放量前必须换成 Postgres checkpointer。

## 运行测试

```bash
/root/.venvs/middot-langgraph-pilot/bin/python -m pytest -q tests/test_agent_runtime.py
```

测试覆盖默认回退、自动选择、等待后恢复、模拟重建、伪造候选拒绝、幂等写入、主图
工具循环、重复调用、自动补偿、调用上限与 Trace 脱敏。集成测试还覆盖地点和普通选择
在清空内存 session、重建 runtime 后继续恢复，并验证一次性 token 不能重复消费。

## 启用与回退

- 启用地点图：`MIDDOT_AGENT_RUNTIME=langgraph`
- 启用主编排：`MIDDOT_AGENT_ORCHESTRATOR=langgraph`
- 只回退主编排：`MIDDOT_AGENT_ORCHESTRATOR=legacy` 后重启应用
- 完全回退：两个开关都设为 `legacy` 后重启应用
- LangSmith 独立控制：`MIDDOT_LANGSMITH_TRACING=true` 且配置采样率和 API key；关闭
  LangSmith 不影响 LangGraph，也不影响本地 `agent_traces`。
