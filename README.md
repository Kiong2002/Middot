# 中点 Middot

> 「我们在哪见？」—— 让 AI 帮你和朋友挑一个都不远、都开心的地方。

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000)](https://flask.palletsprojects.com/)
[![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek-9146FF)](https://platform.deepseek.com/)
[![高德地图](https://img.shields.io/badge/地图-高德%20JS%20API%202.0-E4392E)](https://console.amap.com/)
[![Live Demo](https://img.shields.io/badge/线上体验-meetmid.myowl.me-1F883D)](https://meetmid.myowl.me/)

线上体验：**[meetmid.myowl.me](https://meetmid.myowl.me/)**

---

## 这是什么

**中点 Middot** 是一个多人碰面地点决策工具。

你输入所有人的出发地和一句话需求（"找家安静点的火锅店"、"想吃日料，人均别太贵"），中点会：

1. 算出所有人的**地理中点**（可拖动锚点、拉半径），
2. 到高德地图上搜候选地点，
3. **分别为每个人**规划实时路线（公交/驾车/骑行/步行/最快），
4. 按"公平/评分/距离"三种口径排序，
5. 让**小 Mid**（内嵌 AI 助手）用自然语言改需求："再远一点"、"人均别超 150"、"再加一个朋友，她在望京"。

它不是"帮你搜餐厅"——它是**帮你做决定**。

---

## 核心亮点

- **多人 · 不止 A/B**：支持任意人数参与者（不再局限于两人对约）
- **锚点 + 半径**：不满意 AI 算的中点？拖到你想要的位置，设个"离所有人都别超 5km"
- **房间实时协作**：6 位房间号，朋友扫码进来，改锚点、改需求、改自己的位置——**3 秒同步给所有人**
- **小 Mid AI 助手**：DeepSeek + tool_calls，支持"加人 / 改位置 / 改锚点 / 换关键词 / 再搜一批"等结构化操作，前端每一次工具调用都有可撤销的活动日志
- **iOS Apple Maps 风视觉**：柔和圆角、玻璃质感、动效克制
- **收藏 & 历史**：常去的位置、以前找过的地方，一键找回
- **纯 HTML/CSS/JS 前端**：单文件 SPA，无构建工具，无 npm，无 React

---

## 快速开始

### 1. 拿两把 API Key

在项目根目录 `cp .env.example .env` 后填入：

```env
# DeepSeek: https://platform.deepseek.com/
DEEPSEEK_API_KEY=sk-...

# 高德 Web 服务 Key（后端 geocode / 路线 API）: https://console.amap.com/
AMAP_KEY=...

# 高德 Web 端 JS API Key（前端地图）——需要在控制台把 localhost/127.0.0.1 加到白名单
AMAP_JS_KEY=...
```

### 2. 启动

```bash
bash start.sh
```

首次运行会用 `uv` 建虚拟环境、装依赖、启动 `app_v2.py`。访问 <http://localhost:5000>。

手动启动：

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app_v2.py
```

---

## 使用指南

### 单人模式：先摆参与者，再搜

1. 侧边栏点「+ 参与者」加人，每人填地址或点地图选点
2. 设定锚点（可选：不设则用几何中点）
3. 输入需求，如「安静的咖啡厅，人均别超 60」
4. 「找餐厅」→ 出结果 → 点小圆点看每人到那的路线

### 多人模式：开房间

1. 顶栏点「房间」→「创建」→ 拿到 6 位房号（比如 `473521`）
2. 分享链接或口头告诉朋友
3. 朋友进来后填自己位置，你能实时看到，反之亦然
4. 谁改了锚点、关键词，其他人都会收到顶部横幅通知
5. 房主可以「锁定房间」防止陌生人加入

### 让小 Mid 帮你

点右下角 🧭 图标打开小 Mid 面板。它能：

- 「我在北大，Lisa 在人大，想吃火锅」→ 自动加两个参与者 + 填关键词
- 「换一个安静点的」→ 换关键词重搜
- 「再加一个朋友，她在望京」→ `add_participant`
- 「第 2 个人的位置改到国贸」→ `set_participant_location`
- 「锚点往北挪 2 公里」→ `shift_center`
- 每一次工具调用都会在 AI 面板顶部生成活动条目，可**撤销**

---

## 技术架构

### 后端流水线（`app_v2.py`）

```
用户查询 →  规划 Agent（提关键词 / 排序权重 / 评分阈值）
        →  高德 Search API（POI 候选）
        →  评分过滤 & 排序
        →  批量路线计算（并发 + QPS 限速）
        →  总结 Agent（生成 2-4 句推荐话术）
        →  SSE 流式返回 + 落盘 session 供后续路线重算
```

- **多 Agent 拆分**：把重的（长上下文、大数据）留给 Python，把轻的（自然语言理解）留给 LLM
- **Session 缓存**：切换出行方式不需要重搜——直接对已有 POI 重算路线
- **SSE 流式进度**：前端能看到「规划 → 搜索 → 路线 → 总结」逐步推进
- **rate limiter**：全局 QPS 限流，避免高德配额被打爆

### 小 Mid AI 助手（`/api/v2/assistant/stream`）

- **OpenAI 风格 tool_calls**：DeepSeek 支持工具调用，前端每收到一个 `tool_call` 立即渲染卡片
- **13 个工具**：`add_participant` / `set_participant_location` / `set_anchor` / `shift_center` / `set_query` / `set_radius` / `set_participant_prefer` / `search_now` / `remove_participant` / ...
- **服务端硬约束**：AI 只能改可白名单内的字段，重名/越界/参数错都会被拒
- **限流 + max iterations**：单轮最多 7 次工具调用，防止 AI 卡在死循环
- **前端 ActivityLog**：每次调用可撤销，被 AI 改的字段有 2s 高亮动画

### 房间协作（`app_v2.py` L2000+）

- **6 位纯数字房号**：避开常见电话号码前缀，24h 不复用
- **revision-based 增量拉取**：`GET /api/v2/rooms/<code>?since_rev=N`，无变化返回 `{unchanged: true}`
- **BEGIN IMMEDIATE 事务**：避免两个并发 update 拿到同一个 revision
- **3s 轮询 + 智能退避**：5 次无变化退到 8s；`document.hidden` 时暂停
- **权限分级**：房主能锁房 / 踢人 / 转让；成员能改自己位置 + 改锚点关键词
- **归属水印**：谁改了什么，顶部横幅显示 5s

### 前端（`static/index.html`）

- **单文件 SPA**：~6000 行 vanilla JS，无构建工具、无框架、无 npm
- **高德 JS API 2.0** + `HawkEye`（右上角小地图）+ `Driving/Walking/Riding/Transfer/Geolocation`
- **Lucide icons** + 自托管 marked/purify（Xiao Mid 消息 Markdown 渲染）
- **Apple Maps 视觉**：柔和圆角、玻璃质感面板、iOS 风 accordion
- **手机端专项**：三层抽屉（sidebar / results / assist），底部圆角、地图留 15vh 保底

---

## 项目结构

```
.
├── app_v2.py             # Flask 主入口 + 多 Agent 流水线 + AI 助手 + 房间 + 收藏
├── amap_client.py        # 高德 API 客户端（geocode / 路线 / POI 搜索）
├── requirements.txt
├── start.sh              # 一键启动
├── .env.example          # API Key 模板（复制成 .env）
├── static/
│   ├── index.html        # 单文件前端 SPA
│   └── vendor/           # 自托管 marked.min.js / purify.min.js
└── README.md
```

---

## 后端 API 一览

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/api/v2/search-stream` | 主流水线（SSE 流式） |
| `POST` | `/api/v2/routes` | 路线重算（用 session_id，不重搜） |
| `POST` | `/api/v2/assistant/stream` | 小 Mid 对话（SSE + tool_calls） |
| `GET`  | `/api/v2/session/<id>` | 查看 session 详情 |
| `POST` | `/api/v2/rooms` | 创建房间 |
| `POST` | `/api/v2/rooms/join` | 加入房间 |
| `GET`  | `/api/v2/rooms/<code>?since_rev=N` | 增量拉取房间状态 |
| `POST` | `/api/v2/rooms/<code>/update` | 更新房间（锚点/关键词/自己的位置等） |
| `POST` | `/api/v2/rooms/<code>/leave` | 离开房间 |
| `POST` | `/api/v2/rooms/<code>/lock` | 锁定/解锁房间（仅房主） |
| `POST` | `/api/v2/rooms/<code>/kick` | 踢人（仅房主） |
| `GET`/`POST`/`DELETE` | `/api/favorites` | 收藏（`kind=location \| poi`） |
| `POST` | `/api/geocode` | 地址 → 坐标 |
| `POST` | `/api/geocode-suggest` | 输入联想 |
| `POST` | `/api/nearby-search` | 附近 POI |
| `GET`  | `/api/config` | 前端配置（含 JS API Key） |

---

## 部署

生产环境用 rsync 直推 + `pkill -f app_v2.py` 重启：

```bash
rsync -avz app_v2.py amap_client.py root@<host>:/root/Meetmid-AMAP/
rsync -avz static/ root@<host>:/root/Meetmid-AMAP/static/
ssh root@<host> 'bash /root/restart_meetmid.sh'
```

`restart_meetmid.sh` 只是简单的 `pkill + nohup`，日志到 `server.log`。

---

## 依赖

```
flask==3.0.3
flask-cors==4.0.1
requests==2.31.0
openai>=2.0.0
httpx>=0.27.0
python-dotenv==1.0.1
```

Python **3.10+**（用到了 `str | None` 联合类型语法）。

---

## 常见问题

**地点搜索联想没结果？** 检查 `AMAP_KEY`，确认高德控制台的 QPS 配额没超。

**地图不显示？** 检查 `AMAP_JS_KEY`，确认高德控制台的域名白名单里加了你部署的域名（localhost 也要加）。

**部分路线显示"计算中"？** 高德路线 API 免费版 QPS 较低，后端已加限流；若仍慢可减少候选数量或错峰使用。

**公交路线时间对不上？** 高德公交 API 默认使用工作日中午 12:00 规划（避免末班车干扰）。可在「出发时间」处手动指定。

**中点 vs Meetmid 是同一个吗？** 是。项目内部代码仍用 `Meetmid-AMAP` 目录名（第一次 init 时叫这个），产品对外统一叫**中点 Middot**。

---

## Roadmap

- [x] 多人参与（不再局限 A/B）
- [x] 锚点 + 半径可视化
- [x] 房间实时协作（3s 轮询）
- [x] 小 Mid AI 助手（DeepSeek + tool_calls）
- [x] ActivityLog + 撤销
- [x] 位置收藏
- [ ] POI 收藏 & 我的历史
- [ ] 小 Mid 双模式（背后规划 + 结果抽屉，不改用户面板）
- [ ] 房间共享 AI（AI 建议全房间可见）

---

## License

MIT
