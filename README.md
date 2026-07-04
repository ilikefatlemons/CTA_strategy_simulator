# CTA 系统化交易平台 — 白皮书 / 项目计划书

> Status: **Planning phase — no code has been written yet.**
> This document is the single source of truth for scope, architecture, and build order.
> We will build step by step, phase by phase, only after each phase's design is agreed on.

---

## 1. 目标 (Objective)

为一篮子股票（跨波动率、跨板块）构建一个**规则统一、参数自适应**的 CTA 风格系统化交易策略，核心约束：

- **执行标准一致**：所有标的用同一套入场/出场/仓位规则，不因个股"性质"（波动率、板块）改变逻辑本身——差异只体现在**仓位大小**（vol-targeting），不体现在**规则**上。
- **规则可扩展**：入场信号（MACD金叉等）、止盈止损、再入场逻辑都要做成插件式模块，方便后续自己往里加规则，而不用改核心引擎。
- **可视化验证**：本地交互式K线图，左边显示K线+买卖点标记，右边显示策略净值曲线/回撤，用来直观验证信号和绩效是否符合预期。
- **直接用真实数据搭框架**（alpaca），跑通后再考虑接入更专业的数据源。

> MVP 原则：先做一个端到端能跑通的最小闭环（1个entry规则+1个exit规则+1个reentry规则+跨标的仓位分配+Sharpe+可视化），不追求功能完整，验证过闭环后再逐步加规则/指标。

---

## 2. 范围与非目标 (Scope / Non-goals)

**In scope (MVP):**
- 单一时间框架（5分钟bar）回测引擎
- alpaca 真实分钟级行情抓取（NVDA / KO / XOM / JPM 四标的，覆盖高/低波动率+不同板块）
- 可插拔规则框架：Entry(MACD金叉) / Exit(ATR TP-SL) / Re-entry(冷却期) / Position sizing(vol-inverse)
- 波动率倒数加权仓位分配 (vol-targeting)，**每日 rebalance**
- 绩效指标：**Sharpe only**（其余指标留到 MVP 验证通过后再加）
- 本地交互式可视化界面（K线+买卖点 左，净值曲线 右），用 `lightweight-charts-python`

**Out of scope (先不做，留interface):**
- 实盘/模拟盘下单对接（券商API）
- 多时间框架/跨周期信号融合
- 机器学习信号
- 分布式/多进程回测加速（先跑通单机版）

---

## 3. 系统架构 (Architecture)

采用**分层解耦**设计，保证"规则统一执行"这个硬约束在代码层面被强制，而不是靠约定：

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 5: Visualization (Plotly/Dash or lightweight-charts)  │
│  左: Candlestick + Buy/Sell markers   右: Equity curve/DD     │
└───────────────────────────▲───────────────────────────────────┘
                            │ reads
┌───────────────────────────┴───────────────────────────────────┐
│  Layer 4: Performance & Reporting                              │
│  Sharpe / Sortino / Calmar / MDD / Win rate / Profit factor    │
└───────────────────────────▲───────────────────────────────────┘
                            │ consumes trade log + equity series
┌───────────────────────────┴───────────────────────────────────┐
│  Layer 3: Backtest Engine (event-driven, bar-by-bar)           │
│  - 对每个标的独立跑规则，但规则实例是"同一份代码"                │
│  - 撮合逻辑、滑点/手续费假设、持仓状态机                          │
└───────────────────────────▲───────────────────────────────────┘
                            │ calls
┌───────────────────────────┴───────────────────────────────────┐
│  Layer 2: Strategy Rule Framework (可扩展核心)                 │
│  ├─ EntryRule (插件): MACD金叉, ... 自己加                      │
│  ├─ ExitRule (插件): TP/SL, trailing stop, time-stop...        │
│  ├─ ReentryRule: 出场后再入场条件                                │
│  └─ PositionSizer: vol-inverse weighting (标准CTA做法)          │
└───────────────────────────▲───────────────────────────────────┘
                            │ reads
┌───────────────────────────┴───────────────────────────────────┐
│  Layer 1: Data Layer                                            │
│  - 模拟分钟bar生成器 (GBM/跳跃扩散，可调波动率/板块相关性)         │
│  - 未来替换为真实CSV/数据库，接口不变                             │
└─────────────────────────────────────────────────────────────┘
```

**"执行标准一致"的关键设计点**：Layer 2 的规则类对所有标的是**同一个类实例/同一份参数配置**（除非显式声明某参数按波动率缩放，比如ATR倍数止损这种"规则内在需要相对化"的情况）。绝不会出现"某只股票走A逻辑，某只走B逻辑"这种分叉。

---

## 4. 核心模块设计

### 4.1 Data Layer — alpaca 真实数据
- MVP 标的篮子：**NVDA**(科技/高波动), **KO**(消费/低波动), **XOM**(能源), **JPM**(金融) — 覆盖不同板块+波动率，让 vol-targeting 的仓位差异真实可见
- 频率：**5分钟bar**（alpaca `1m` 只保留近7天历史，样本太少；`5m` 可回溯60天，足够跑出多笔交易和有意义的 Sharpe）
- 输出统一 schema（未来接真实数据源的契约）：`timestamp, symbol, sector, open, high, low, close, volume`
- 抓取后落地为本地 CSV（`data/raw/{symbol}_5m.csv`），下游所有层只读 CSV，不直接依赖 alpaca——方便以后换数据源或做校准
- 模拟数据生成器（GBM+板块相关性因子模型）作为**后续可选项**保留在设计里，用于压力测试/校准，不在 MVP 路径上

### 4.2 Strategy Rule Framework（可扩展规则框架）
用抽象基类 + 注册机制，形式大致如：

- `EntryRule` 基类 → 具体规则如 `MACDGoldenCross`、`RSIThreshold` 等都继承它，返回 long/short/none 信号
- `ExitRule` 基类 → `FixedTPSL`、`ATRTrailingStop`、`TimeBasedExit`
- `ReentryRule` 基类 → 出场后满足什么条件才允许再次入场（冷却期、反向信号确认等）
- `PositionSizer` 基类 → v1 实现 `InverseVolatilitySizer`（标准 vol-targeting：weight_i ∝ 1/σ_i，归一化后权重和=1；**每日 rebalance**，暂不设显式目标组合波动率标量，MVP先验证相对权重逻辑本身）

规则之间通过统一接口拼装成一个 `Strategy` 对象，加新规则 = 写一个新类 + 注册，不改引擎代码。

### 4.3 Backtest Engine
- Bar-by-bar 事件驱动（不是 vectorized，这样能正确处理"持仓状态机"和"再入场"逻辑）
- 每个标的独立维护仓位状态机：`FLAT → ENTERED → (TP/SL/EXIT) → COOLDOWN → FLAT`
- 统一撮合假设：滑点、手续费、下一bar开盘价成交（避免未来函数）

### 4.4 Performance & Reporting (MVP)
- 输出：trade log（每笔交易明细）+ equity curve（组合净值）+ per-symbol breakdown
- 指标：**Sharpe only**（Sortino/Calmar/MDD/Win rate等留作后续扩展项，见第2节非目标）
- 支持"整体组合" vs "单标的" 两级 Sharpe 对比，用来验证 vol-targeting 是否真的拉平了风险贡献

### 4.5 Visualization
- 技术选型：**`lightweight-charts-python`**（TradingView同款手感，轻量）
- 左图：K线 + 买卖点 marker（颜色区分 entry/exit/stop）+ 可缩放拖拽
- 右图：净值曲线 + 回撤阴影，与左图光标联动（hover 同步时间轴）

---

## 5. 建议的构建顺序 (Build Order)

不建议一上来就搭可视化——那是"看得见但立不住"的部分。按依赖关系从底层往上：

1. **Phase 0**：白皮书 + 目录结构 + 技术选型确认 ✅ 已完成
2. **Phase 1**：alpaca 数据抓取（NVDA/KO/XOM/JPM, 5m bar, 60天）→ 落地 CSV，确认 schema ✅ 已完成 (`src/data/fetch_alpaca.py` → `data/raw/{symbol}_5m.csv`, 各4680行)
3. **Phase 2（下一步）**：规则框架骨架（抽象基类 + MACD金叉入场 + ATR TP/SL出场 + 冷却期再入场）——先在单标的上跑通端到端
4. **Phase 3**：加入 InverseVolatilitySizer，扩展到四标的组合回测，每日 rebalance，验证"同规则不同仓位"
5. **Phase 4**：Sharpe 计算（组合级 + 单标的级对比）
6. **Phase 5**：交互式可视化（lightweight-charts-python：K线+买卖点 左，净值曲线 右）
7. **Phase 6+（MVP之后再说）**：Sortino/Calmar/MDD等更多指标、更多规则插件（RSI、布林带、trailing stop）、目标组合波动率标量、模拟数据生成器

每个 Phase 结束我们都会一起看结果、确认再进入下一步，不会连续写一堆代码再一次性甩给你看。

---

## 6. 技术选型 (Locked decisions)

- 可视化库：**`lightweight-charts-python`**
- 数据来源：**alpaca 真实数据**（NVDA/KO/XOM/JPM, 5m bar），模拟数据+因子模型相关性结构留作后续校准/压力测试用途
- 回测引擎支持**双向**（多空），CTA标准做法
- 仓位再平衡频率：**每日**（不是每根bar——避免追逐波动率估计本身的噪音，且更贴近真实交易成本假设）
- 目标组合波动率标量：MVP暂不设，先验证 InverseVolatilitySizer 的相对权重逻辑

---

## 7. 目录结构规划（尚未创建）

```
TradingApp/
├── README.md                  (本文件)
├── data/
│   └── simulated/              模拟行情输出
├── src/
│   ├── data/                   Layer 1
│   ├── rules/                  Layer 2 (entry/exit/reentry/sizing)
│   ├── engine/                 Layer 3
│   ├── performance/             Layer 4
│   └── viz/                    Layer 5
├── notebooks/                  探索性分析
└── tests/
```

---

## Next step

确认第 6 节的技术选型，然后从 **Phase 1（模拟数据生成器）** 开始动手写代码。
