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

<p align="center">
  <img src="docs/readme/fair-meeting.svg" width="100%" alt="比较我·国贸和朋友·清华的真实路线">
</p>

Middot 会分别计算每个人的交通时间，再在公平区域内寻找真正适合见面的地方。你也可以移动锚点、调整范围，或为每个人分别选择公交、驾车、骑行和步行。

## 朋友自己填，方案一起看

<p align="center">
  <img src="docs/readme/room-collab.svg" width="100%" alt="邀请朋友加入 Middot 房间并同步位置">
</p>

建立房间后，把口令发给朋友就行。谁填了位置、谁换了交通方式、会面点为什么变了，房间里的每个人都能看到。

## 和小 Mid 说人话就好

<p align="center">
  <img src="docs/readme/mid-chat.svg" width="100%" alt="通过小 Mid 对话调整会面方案">
</p>

不用把条件全部重填一遍。直接说“安静一点”“换成火锅”或“朋友改从清华出发”，小 Mid 会在当前方案上继续调整。

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
