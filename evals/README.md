# Middot Agent 评测

运行：

```bash
python -m evals.run
python -m evals.run --json
```

`scenarios.jsonl` 是版本化的确定性编排评测集。每条场景给出用户目标、模型返回轨迹、
业务初态和可机器判定的期望结果；运行时使用真正的 LangGraph 主图，验证工具去重、
搜索闭环、等待/暂停、失败停止、路线重算、参与者槽位和记忆工具边界。

报告中的 `scenario_pass_rate` 是**编排场景通过率**，不是线上自然语言任务完成率。
模型返回由数据集回放，因此这组数字适合防止框架、工具循环和状态机回归。后续接入
真实模型评测时，应另报语义解析准确率、工具选择准确率和端到端任务完成率，不能与
这里的指标混写。

真实模型语义评测会调用 DeepSeek，并读取当前 `app_v2.py` 里的真实解析/复核 Prompt：

```bash
python -m evals.run_live_semantic --source app_v2.py
```

它不会导入应用，也不会连接业务数据库、Redis或高德，只评估自然语言到结构化会面
意图的准确率。由于会产生模型调用费用，不纳入每次 `pytest`。
