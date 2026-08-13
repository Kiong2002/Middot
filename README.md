<p align="center">
  <img src="docs/readme/hero.svg" width="100%" alt="Middot — 约在真正的中间">
</p>

<p align="center">
  <a href="https://middot.myowl.me/"><strong>立即体验</strong></a>
  &nbsp;·&nbsp;
  <a href="#本地运行">本地运行</a>
  &nbsp;·&nbsp;
  <a href="#小-mid-会记住什么">会面记忆</a>
</p>

<p align="center">
  <img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10+-2676E5?style=flat-square">
  <img alt="Flask" src="https://img.shields.io/badge/Flask-lightweight-111827?style=flat-square">
  <img alt="License MIT" src="https://img.shields.io/badge/License-MIT-12A594?style=flat-square">
</p>

## 约在真正的中间

**Middot** 是一个为多人见面而生的地点决策工具。

告诉小 Mid 大家从哪里出发、想吃什么或想做什么，它会兼顾每个人的路程，找到一个不只“地理居中”，也更适合真正见面的地方。

<p align="center">
  <img src="docs/readme/journey.svg" width="100%" alt="从出发地到公平会面点的三步流程">
</p>

## 不只是搜一个地点

| 公平地见面 | 和小 Mid 一起调整 | 越用越懂你的会面档案 |
|:---:|:---:|:---:|
| 同时比较每个人的真实路线，而不是只算地图直线中点 | 直接说“安静一点”“换成火锅”“阿杰从浙大出发” | 记住常约的人、常用地点和明确偏好，并保留可追溯来源 |

你还可以：

- 邀请朋友进入同一个房间，一起填写位置和调整方案
- 在地图上移动会面锚点，控制搜索范围
- 分别选择公交、驾车、骑行、步行或最快方式
- 保存并继续历史对话，随时删除不再需要的记录
- 在知识图谱里查看、确认、修改或忘掉一条关系

## 小 Mid 会记住什么

Middot 不会把每次搜索都当成事实。只有明确表达、证据足够或由你亲自确认的信息，才会进入会面档案；不确定和冲突的内容会留在“待确认”。

<p align="center">
  <img src="docs/readme/memory-graph.svg" width="100%" alt="可确认、可修改、可遗忘的会面记忆图谱">
</p>

这套记忆始终遵循三件事：

1. **看得见**：知道小 Mid 记住了什么，以及它来自哪次对话。
2. **改得动**：待确认关系可以确认或忽略，正式关系可以修改或删除。
3. **会过期**：地点和时效性信息不会被当作永远不变的事实。

## 本地运行

```bash
git clone https://github.com/Kiong2002/Middot.git
cd Middot
cp .env.example .env
bash start.sh
```

按 `.env.example` 填写地图与模型配置，然后打开 <http://localhost:5000>。

需要 Python 3.10 或更高版本。项目不依赖前端构建工具，克隆后即可运行。

## 现在正在做

- [x] 多人出发地与公平路线比较
- [x] 小 Mid 自然语言调整方案
- [x] 房间协作与共享会面状态
- [x] 历史对话与后台记忆整理
- [x] 可追溯、可确认的会面档案
- [x] 可搜索、可移动的记忆图谱
- [ ] 更自然的多人协作提醒
- [ ] 更丰富的会面复盘与收藏体验

## License

MIT
