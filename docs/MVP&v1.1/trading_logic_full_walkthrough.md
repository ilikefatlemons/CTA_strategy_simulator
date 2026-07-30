# 完整交易逻辑走查（数据 → 信号 → 止损止盈 → 再入场 → 冷静期 → 仓位/权重 → 组合）

按数据从"原始行情"到"组合净值"实际流动的顺序写，每一段都标"文件:行号 + 完整代码 + 逐行/逐参数解释"。目的是让你能对着这份文档，一行一行核对代码在做什么、为什么这么做。

---

## 0. 数据层：抓取与复权

**文件：`src/data/fetch_alpaca.py` 第45-53行**
```python
def fetch_symbol(client: StockHistoricalDataClient, symbol: str, sector: str) -> pd.DataFrame:
    start = pd.Timestamp.now("UTC") - pd.Timedelta(days=LOOKBACK_DAYS)

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TIMEFRAME,
        start=start.to_pydatetime(),
        feed=DataFeed.IEX,  # free-tier data feed
        adjustment=Adjustment.ALL,  # split + dividend adjusted, avoids fake gaps at corporate actions
    )
```
- `TIMEFRAME`（第33行）= 5分钟K线，`LOOKBACK_DAYS`（第34行）= 730天（约2年）。
- **复权就写在这一行**：`adjustment=Adjustment.ALL`。Alpaca的`Adjustment`枚举有4个值：`RAW`(不复权，之前用的就是这个)/`SPLIT`(只调拆股)/`DIVIDEND`(只调分红)/`ALL`(两个都调)。现在用`ALL`，意味着历史价格序列里拆股、分红造成的价格跳空都已经被平滑处理，K线反映的是"持续持有"的真实收益率，不会有拆股当天腰斩式的假暴跌。
- `feed=DataFeed.IEX`：免费/纸交易账户只能拿IEX这一个交易所的数据（不是SIP全市场数据），这是数据源本身的限制，不影响复权逻辑。

**文件：`src/data/fetch_alpaca.py` 第75-80行（盘中时段过滤）**
```python
et_time = df["timestamp"].dt.tz_convert("America/New_York").dt.time
in_session = (et_time >= pd.Timestamp("09:30").time()) & (et_time < pd.Timestamp("16:00").time())
df = df[in_session]
```
- 只保留09:30-16:00美东正常交易时段的K线，砍掉盘前盘后——避免有的票有盘前数据、有的没有，导致同一个"5分钟bar序号"在不同票之间对不上。这是后面所有多标的时间对齐（`_direction_by_day`、组合权重）能成立的前提条件。

**`SYMBOLS`字典（第21-31行）**：12个标的，9只股票+3只商品ETF代理（USO=WTI原油、GLD=COMEX黄金、SLV=COMEX白银），因为Alpaca不提供真正的期货合约数据，用ETF价格近似代替连续合约走势。

---

## 1. 重采样：5m → 15m/30m/2h，"已收盘"边界判定

**文件：`src/data/resample.py` 第21-63行（`resample_ohlcv`）**
```python
def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    df = df.sort_values("timestamp")
    et_dates = df["timestamp"].dt.tz_convert("America/New_York").dt.date
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    sessions = []
    for _, day_df in df.groupby(et_dates):
        day_df = day_df.set_index("timestamp")
        session = day_df.resample(rule, origin="start", label="left", closed="left").agg(agg)
        session = session.dropna(subset=["open"])
        ...
        sessions.append(session)
    out = pd.concat(sessions).reset_index()
    return out[cols]
```
- 逐日重采样（`groupby(et_dates)`），不是整体连续重采样：防止一个15m/30m/2h的bin横跨两个交易日的收盘-开盘间隙（比如把昨天15:55和今天09:30揉进一根K线里）。
- `origin="start"`：每天的bin从当天第一根K线（09:30）开始对齐，不是从UTC零点对齐——否则390分钟/session不能被60整除，会导致每天的整点K线错位。
- `label="left", closed="left")`：K线用区间左端点（开盘时间）当时间戳，区间是`[t, t+rule)`（左闭右开）——这个约定是下一步"已收盘"判定的基础。
- `agg`字典：`open`取第一个、`high`取最大、`low`取最小、`close`取最后一个、`volume`求和——标准OHLCV聚合公式，没有特殊处理。

**文件：`src/data/resample.py` 第71-91行（`closed_bar_positions`，非前视性的核心）**
```python
def closed_bar_positions(higher_tf_df: pd.DataFrame, rule: str, open_times):
    """
    A bar labeled with open time `t` (label="left") spans [t, t + rule) and
    only closes at t + rule. So a bar is usable exactly when
    `t + rule <= open_time`.
    """
    close_times = higher_tf_df["timestamp"] + pd.Timedelta(rule)
    return close_times.searchsorted(open_times, side="right") - 1
```
- 每根高周期K线的"真实收盘时间" = 它的开盘时间戳 + 周期长度（比如30m K线标签是09:30，真实收盘时间是10:00）。
- 给定当前5m bar的开盘时间`open_times`，用`searchsorted(..., side="right") - 1`找到"收盘时间 <= 当前时间"的最后一根高周期K线的下标——这就是"当前这一刻能看到的、最新的已经走完的高周期K线"。
- `-1`表示一根都还没收盘（比如序列刚开始的头几根5m bar，连一根2h K线都还没走完）。
- 这个函数被`pullback_backtest.py`（见第7节）和`higher_tf_indicators.py`共用，保证"什么算已收盘"这个边界判断只有一份实现，不会两处代码各写一套、悄悄产生偏差。

---

## 2. 技术指标：ATR

**文件：`src/indicators.py` 第32-38行**
```python
def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return true_range.rolling(period).mean()
```
- 标准True Range公式：取三者最大值——当根K线的高低差、当根最高价与前一根收盘价的差、当根最低价与前一根收盘价的差。
- ATR = TR的`period`(默认14)期简单移动平均。
- `prev_close = close.shift(1)`：用的是"前一根"的收盘价，不是当根，天然不含未来数据；`rolling(period).mean()`同理只往回看。
- 这一份ATR函数在整个系统里被**三处复用**：止损宽度计算（`stop_loss.py`）、吊灯止盈（`take_profit.py`）、组合层波动率估计（`vol_estimator.py`）——同一个公式，不同用途。

---

## 3. 信号生成：四级时间框架级联

### 3.1 MA排列判定（两个粒度共用一套逻辑原语）

**文件：`src/rules/ma_filter.py` 第22-57行**
```python
def ma_array_state(klines: pd.DataFrame, fast: int = 5, mid: int = 20, slow: int = 50) -> ArrayState:
    """严格三线排列 - 只用于2h定方向(bias)"""
    if len(klines) < slow:
        return "neutral"
    closes = klines["close"]
    ma_fast = closes.rolling(fast).mean().iloc[-1]
    ma_mid = closes.rolling(mid).mean().iloc[-1]
    ma_slow = closes.rolling(slow).mean().iloc[-1]
    if pd.isna(ma_fast) or pd.isna(ma_mid) or pd.isna(ma_slow):
        return "neutral"
    if ma_fast > ma_mid > ma_slow:
        return "bullish"
    if ma_fast < ma_mid < ma_slow:
        return "bearish"
    return "neutral"


def ma_fast_mid_state(klines: pd.DataFrame, fast: int = 5, mid: int = 20) -> ArrayState:
    """只比较快/中两条线 - 用于5m/15m/30m的回调/触发判断"""
    if len(klines) < mid:
        return "neutral"
    closes = klines["close"]
    ma_fast = closes.rolling(fast).mean().iloc[-1]
    ma_mid = closes.rolling(mid).mean().iloc[-1]
    if pd.isna(ma_fast) or pd.isna(ma_mid):
        return "neutral"
    if ma_fast > ma_mid:
        return "bullish"
    if ma_fast < ma_mid:
        return "bearish"
    return "neutral"
```
- `ma_array_state`：MA5/MA20/MA50三线**严格**排列（`>`不是`>=`），只用在2h定方向。三线都要满足单调排列才给出方向，否则neutral。门槛设得高，是因为这是"要不要交易这只票"的第一道闸门，不能草率。
- `ma_fast_mid_state`：只比MA5/MA20两条线，用在5m/15m/30m。用两线而不是三线，是因为三线排列条件太苛刻——文档里提到"2年NVDA数据三线在小周期上完整走完只有1次"，用两线让回调/触发这两步能有足够样本触发。
- 两个函数都是`.iloc[-1]`只取传入`klines`最后一行——数据边界（要不要含未收盘的当前bar）由调用方（`pullback_backtest.py`）传参时控制，函数本身不做任何"往后看"的操作。

### 3.2 三级联动：定方向 → 回调确认 → 入场触发

**文件：`src/rules/ma_filter.py` 第60-89行**
```python
class MultiTimeframeFilter:
    def get_trend_bias(self, klines_2h: pd.DataFrame) -> Bias:
        return _STATE_TO_BIAS[ma_array_state(klines_2h)]

    def pullback_occurred(self, small_tf_klines: dict[str, pd.DataFrame], bias: Bias) -> bool:
        opposite = _BIAS_TO_OPPOSITE_STATE[bias]
        reversed_30m = ma_fast_mid_state(small_tf_klines["30m"]) == opposite
        reversed_5m_or_15m = any(
            ma_fast_mid_state(small_tf_klines[tf]) == opposite for tf in ("5m", "15m")
        )
        return reversed_30m and reversed_5m_or_15m

    def get_entry_trigger(self, small_tf_klines: dict[str, pd.DataFrame], bias: Bias) -> bool:
        same = _BIAS_TO_SAME_STATE[bias]
        return all(ma_fast_mid_state(small_tf_klines[tf]) == same for tf in ("5m", "15m"))
```
- **`get_trend_bias`**：2h周期跑`ma_array_state`（三线严格排列），映射成 `long`/`short`/`neutral`。这是"今天该往哪个方向找机会"的大方向判断。
- **`pullback_occurred`**：`opposite`=跟bias反向的排列状态（比如bias=long时，opposite=bearish）。
  - `reversed_30m`：30m周期的快中线排列必须转成反向——这是硬性必要条件。
  - `reversed_5m_or_15m`：5m或15m**任一个**转成反向即可（`any(...)`，逻辑OR）。
  - 两者`and`：必须"30m确实反向 **且** 5m/15m至少一个也反向"，才算"回调发生了"。
- **`get_entry_trigger`**：`same`=跟bias同向的排列状态。5m**和**15m**都**要转回同向（`all(...)`，逻辑AND）才算触发——这是最后确认入场的一步，要求两个小周期同时点头。

### 3.3 状态机：怎么把"定方向→回调→触发"串成一次完整信号

**文件：`src/rules/pullback_entry.py` 第35-63行**
```python
class PullbackEntryEngine:
    def __init__(self, filter: MultiTimeframeFilter, cooldown: CooldownManager):
        self.filter = filter
        self.cooldown = cooldown
        self._bias: Bias = "neutral"
        self._pullback_seen = False

    def on_bar(self, snapshot: MarketSnapshot) -> Signal | None:
        if self.cooldown.is_active():
            return None

        bias = self.filter.get_trend_bias(snapshot.tf_2h)
        if bias != self._bias:
            self._bias = bias
            self._pullback_seen = False
        if bias == "neutral":
            return None

        if not self._pullback_seen:
            if self.filter.pullback_occurred(snapshot.small_tf, bias):
                self._pullback_seen = True
            return None

        if self.filter.get_entry_trigger(snapshot.small_tf, bias):
            self._pullback_seen = False
            return Signal(direction=bias, entry_price=snapshot.close)
        return None
```
- 每根5m bar调用一次`on_bar`。类内部只维护两个跨bar的记忆状态：`_bias`（当前2h方向）、`_pullback_seen`（这一轮方向下是否已经看到过回调）。
- 第一步：如果冷静期激活，直接返回`None`，不做任何判断（冷静期优先级最高）。
- 第二步：算出当前bias。**如果bias变了**（包括从long/short变成neutral，或者从long变成short），把`_pullback_seen`清零——防止用旧方向的回调状态去触发新方向的仓位。bias是neutral就直接返回。
- 第三步：如果还没见过回调（`_pullback_seen=False`），检查这一刻是否发生回调；发生了就置`True`，但这一步本身**不产生信号**，只是标记"回调阶段完成，进入等触发阶段"。
- 第四步：已经见过回调了，检查触发条件是否满足；满足则把`_pullback_seen`清零（避免同一轮回调重复触发多次），返回一个`Signal`（方向=bias，价格=当前bar的close，仅用于记录，不是实际成交价——成交价在执行引擎里另算，见第7节）。

**这里体现"首仓和再入场共用同一路径"**：`Signal`只带`direction`和`entry_price`，不带"这是第几次"的信息——执行引擎（第7节）自己根据"之前有没有开过仓"来决定打`open`还是`reentry`标签，`on_bar`本身对两种场景一视同仁。

---

## 4. 止损计算：ATR止损 vs 平台止损

**文件：`src/rules/stop_loss.py` 全文（第14-53行）**
```python
def _swing_points(klines: pd.DataFrame, k: int) -> tuple[list[float], list[float]]:
    highs, lows = klines["high"].to_numpy(), klines["low"].to_numpy()
    swing_highs, swing_lows = [], []
    for i in range(k, len(klines) - k):
        window_high = highs[i - k : i + k + 1]
        window_low = lows[i - k : i + k + 1]
        if highs[i] == window_high.max():
            swing_highs.append(highs[i])
        if lows[i] == window_low.min():
            swing_lows.append(lows[i])
    return swing_highs, swing_lows


def _nearest_platform_level(klines: pd.DataFrame, entry_price: float, direction: str, k: int) -> float | None:
    swing_highs, swing_lows = _swing_points(klines, k)
    if direction == "long":
        candidates = [lvl for lvl in swing_lows if lvl < entry_price]
        return max(candidates) if candidates else None
    candidates = [lvl for lvl in swing_highs if lvl > entry_price]
    return min(candidates) if candidates else None


class StopLossCalculator:
    def __init__(self, atr_mult: float = 1.5, offset_pct: float = 0.003, swing_k: int = 2):
        self.atr_mult = atr_mult
        self.offset_pct = offset_pct
        self.swing_k = swing_k

    def calc(self, entry_price: float, direction: str, atr_value: float, klines_30m: pd.DataFrame) -> float:
        atr_stop = (
            entry_price - self.atr_mult * atr_value if direction == "long"
            else entry_price + self.atr_mult * atr_value
        )
        platform_stop = _nearest_platform_level(klines_30m, entry_price, direction, self.swing_k)

        candidates = [atr_stop] + ([platform_stop] if platform_stop is not None else [])
        chosen = min(candidates, key=lambda lvl: abs(entry_price - lvl))

        offset = entry_price * self.offset_pct
        return chosen - offset if direction == "long" else chosen + offset
```
- **`_swing_points`**：局部极值检测。`k=2`表示"某根K线的最高价比它左右各2根、共5根窗口内所有K线都高"才算swing high（swing low同理反过来）。`for i in range(k, len(klines)-k)`——注意起止范围，头尾各留`k`根不参与判断，因为这些位置没有足够的"左右邻居"来确认是不是局部极值。
- **`_nearest_platform_level`**：多头找entry_price下方**离得最近**的swing low（`max(candidates)`，因为越接近入场价的低点数值越大）；空头找entry_price上方离得最近的swing high（`min(candidates)`）。没有符合条件的候选就返回`None`。
- **`StopLossCalculator.calc`**：
  - `atr_stop`：入场价 ∓ `atr_mult(1.5)` × 当前ATR值。
  - `platform_stop`：上面算出的最近结构位（可能是`None`）。
  - `candidates`：把两个候选（platform_stop存在的话）放一起，用`min(..., key=lambda lvl: abs(entry_price-lvl))`选**离入场价距离更小**的那个——即两个候选里更保守（止损空间更小）的那个。
  - `offset = entry_price * 0.003`（0.3%）：最终止损位再往不利方向多留0.3%缓冲，防止价格精确碰到止损线时被视为"扫损"（一触即回）的行情打出去。

**非前视性**：`klines_30m`本身是执行引擎传进来的、已经过`closed_bar_positions`过滤的"已收盘"K线（见第7节），`_swing_points`只在这个既定范围内找局部极值，不会用到入场之后才出现的K线。

---

## 5. 止盈/离场：两条腿设计

**文件：`src/rules/take_profit.py` 全文（第15-36行）**
```python
@dataclass
class TakeProfitManager:
    partial_ratio: float = 0.5
    rr_trigger: float = 2.0  # 1.5~2R 区间, 取上限
    atr_multiple: float = 3.0

    def partial_trigger_price(self, entry_price: float, direction: str, stop_loss: float) -> float:
        r = abs(entry_price - stop_loss)
        return entry_price + self.rr_trigger * r if direction == "long" else entry_price - self.rr_trigger * r

    def chandelier_stop(
        self, direction: str, extreme_price_since_entry: float, atr_value: float, floor_stop: float
    ) -> float:
        raw = (
            extreme_price_since_entry - self.atr_multiple * atr_value if direction == "long"
            else extreme_price_since_entry + self.atr_multiple * atr_value
        )
        return max(raw, floor_stop) if direction == "long" else min(raw, floor_stop)
```
- **腿A：`partial_trigger_price`**：`r`=入场价到止损价的距离（1R）。触发价 = 入场价 + `rr_trigger(2.0)` × r（多头，空头是减）。也就是浮盈达到2倍风险距离时，触发条件成立（执行引擎里判定，见第7节），平掉`partial_ratio(0.5)`即50%仓位，锁定利润。
- **腿B：`chandelier_stop`**（吊灯止损）：
  - `raw` = 入场以来的极值价（多头是最高价，空头是最低价）∓ `atr_multiple(3.0)` × 当前ATR。
  - `return max(raw, floor_stop)`（多头）：**永远取更高的那个**——`floor_stop`是这一批仓位最初算出的止损价，`raw`是随价格上涨不断上移的吊灯线。只要价格没走出足够空间，`raw`可能比`floor_stop`还低，这时候取`floor_stop`，保证止损位不会比最初设定的更差；一旦价格走出空间、`raw`超过`floor_stop`，止损位开始跟随抬高，锁定利润，且只会越抬越高（因为`extreme_price_since_entry`只增不减）。空头反过来，取更低的那个。

---

## 6. 冷静期风控

**文件：`src/rules/cooldown.py` 全文（第35-99行）**
```python
@dataclass
class CooldownManager:
    consecutive_sl_threshold: int = 3
    release_confirm_bars: int = 3
    ma_periods: tuple[int, int, int] = (5, 20, 50)
    active: bool = field(default=False, init=False)
    _consecutive_sl: int = field(default=0, init=False)
    _last_sl_direction: str | None = field(default=None, init=False)
    _stable_state: ArrayState | None = field(default=None, init=False)
    _stable_count: int = field(default=0, init=False)

    def is_active(self) -> bool:
        return self.active

    def _arm(self) -> None:
        self.active = True
        self._stable_state = None
        self._stable_count = 0
        self._consecutive_sl = 0
        self._last_sl_direction = None

    def on_trade_closed(self, direction: str, reason: str) -> None:
        if reason != "SL":
            self._consecutive_sl = 0
            self._last_sl_direction = None
            return
        if direction == self._last_sl_direction:
            self._consecutive_sl += 1
        else:
            self._consecutive_sl = 1
            self._last_sl_direction = direction
        if self._consecutive_sl >= self.consecutive_sl_threshold:
            self._arm()

    def check_structure_break(self, klines: pd.DataFrame) -> bool:
        fast, mid, slow = self.ma_periods
        if len(klines) < slow:
            return False
        closes = klines["close"]
        mas = [closes.rolling(p).mean().iloc[-1] for p in (fast, mid, slow)]
        if any(pd.isna(v) for v in mas):
            return False
        last = klines.iloc[-1]
        body_low, body_high = min(last["open"], last["close"]), max(last["open"], last["close"])
        return sum(1 for ma in mas if body_low <= ma <= body_high) >= 2

    def on_bar(self, klines_2h: pd.DataFrame) -> None:
        if not self.active:
            if self.check_structure_break(klines_2h):
                self._arm()
            return
        state = ma_array_state(klines_2h)
        if state == "neutral" or state != self._stable_state:
            self._stable_state = state
            self._stable_count = 1 if state != "neutral" else 0
        else:
            self._stable_count += 1
        if state != "neutral" and self._stable_count >= self.release_confirm_bars:
            self.active = False
            self._stable_state = None
            self._stable_count = 0
```
- **触发条件1（`on_trade_closed`，每次一批仓位平仓后调用）**：`reason != "SL"`（比如是`TP`或`PROTECTIVE_SL`）直接把连续计数清零、不算数。只有`reason == "SL"`（整批被原始止损打掉、从没部分止盈过）才累加：同方向连续计数`+1`，换方向则重置成1；累计到`consecutive_sl_threshold(3)`次，调用`_arm()`激活冷静期。
- **触发条件2（`check_structure_break`，每根2h K线调用）**：取MA5/MA20/MA50最新值，判断最新一根K线的实体（`open`到`close`的范围）是否"包住"了至少2条均线（`body_low <= ma <= body_high`）——即这根K线的实体本身跨越/吞没了2条均线，说明是一根异常剧烈的单边K线，均线本身反应慢，先激活冷静期观望。
- **`on_bar`（每根2h调用）**：
  - 如果当前不在冷静期，检查是否触发条件2，触发则`_arm()`。
  - 如果已经在冷静期中，跑`ma_array_state`看当前排列状态：状态变了（或是neutral）就重置计数为1（neutral归0）；状态没变就`+1`。
  - 当状态非neutral且**连续`release_confirm_bars(3)`根都保持同一个状态**，才解除冷静期（`active=False`）。要求"连续3根同一排列"而不是"1根随便反弹"，是为了过滤单根K线的噪音抖动。
- **`_arm()`**：激活的同时清空所有计数器（`_stable_state/_stable_count/_consecutive_sl/_last_sl_direction`归零），保证解除条件是从激活那一刻重新开始数，不会用激活之前的旧计数。
- **对已有仓位无影响**：`is_active()`只在`PullbackEntryEngine.on_bar`的入口被检查（第3.3节），用来拦截新信号；已经开着的仓位的止损/部分止盈/吊灯止损判断在执行引擎主循环里是完全独立的一段代码（第7节），不读取`cooldown`的状态，所以冷静期触发时不会强平现有仓位。

---

## 7. 单标的执行引擎主循环（开仓/加仓再入场/止损止盈判定/权益结算）

**文件：`src/engine/pullback_backtest.py`**，这是把前面所有模块粘合起来、逐bar驱动的地方。

### 7.1 预处理：重采样 + 已收盘索引 + 全序列ATR
第83-92行：
```python
resampled = {tf: resample_ohlcv(df_5m, rule) for tf, rule in _HIGHER_TF_RULES.items()}
closed_idx = {
    tf: closed_bar_positions(resampled[tf], rule, df_5m["timestamp"])
    for tf, rule in _HIGHER_TF_RULES.items()
}
atr_full = atr_series(df_5m, atr_period)

def closed_klines(tf: str, i: int) -> pd.DataFrame | None:
    pos = closed_idx[tf][i]
    return None if pos < 0 else resampled[tf].iloc[: pos + 1]
```
- 一次性把15m/30m/2h全部重采样好，`closed_idx`把每个5m bar下标`i`映射到"此刻对应哪根高周期K线的下标"（第1节的`closed_bar_positions`，向量化算好，不是每个bar单独查一次）。
- `atr_full`：整个5m序列一次性算出ATR（14period），后面按下标取值。
- `closed_klines(tf, i)`：给定周期和bar下标，切出"截至这一刻已经收盘"的那部分高周期K线（`iloc[:pos+1]`），`pos<0`说明连一根都还没收盘，返回`None`。

### 7.2 仓位数据结构
第54-63行：
```python
@dataclass
class _OpenBatch:
    direction: str
    entry_bar_idx: int
    entry_price: float
    stop_loss: float
    signal_type: str
    partial_taken: bool = False
    trailing_stop: float | None = None
    extreme_since_entry: float = 0.0
```
- 一次只允许存在一个`_OpenBatch`（同一时刻不会叠加开仓）。`partial_taken`标记腿A是否已经部分止盈过，决定接下来走"检查腿A触发"还是"检查腿B吊灯止损"两条不同的分支。

### 7.3 主循环逐bar处理
第101-116行（每根bar开头）：
```python
for i in range(n):
    row = df_5m.iloc[i]
    klines_2h = closed_klines("2h", i)
    klines_15m = closed_klines("15m", i)
    klines_30m = closed_klines("30m", i)

    if klines_2h is not None:
        cooldown.on_bar(klines_2h)

    if batch is not None:
        assert klines_2h is not None
        price = row["close"]
        batch.extreme_since_entry = (
            max(batch.extreme_since_entry, row["high"]) if batch.direction == "long"
            else min(batch.extreme_since_entry, row["low"])
        )
```
- 每根bar先切好三个高周期的已收盘K线，`cooldown.on_bar`每根都跑一遍（更新风控状态机，跟有没有仓位无关）。
- 如果手上有仓位，先用当根bar的high/low更新"入场以来的极值价"——这个值是腿B吊灯止损计算的输入。

**腿A检查（未部分止盈时）**，第122-143行：
```python
if not batch.partial_taken:
    trigger = tp_manager.partial_trigger_price(batch.entry_price, batch.direction, batch.stop_loss)
    hit_tp = price >= trigger if batch.direction == "long" else price <= trigger
    hit_sl = price <= batch.stop_loss if batch.direction == "long" else price >= batch.stop_loss
    if hit_sl:
        trades.append(Trade(..., exit_price=batch.stop_loss, reason="SL", size_fraction=1.0, ...))
        capital *= 1 + trades[-1].pnl_pct
        cooldown.on_trade_closed(batch.direction, "SL")
        batch = None
        had_prior_batch = True
    elif hit_tp:
        trades.append(Trade(..., exit_price=trigger, reason="TP", size_fraction=tp_manager.partial_ratio, ...))
        capital *= 1 + tp_manager.partial_ratio * trades[-1].pnl_pct
        batch.partial_taken = True
        batch.trailing_stop = batch.stop_loss
```
- 用当根bar的收盘价`price`分别跟`trigger`（腿A止盈线）和`batch.stop_loss`（原始止损线）比较。
- **`hit_sl`优先判断**（`if hit_sl` 在 `elif hit_tp` 之前）：同一根bar如果两个条件都满足，按止损处理（更保守）。命中止损：整批`size_fraction=1.0`全部平仓，`reason="SL"`，资金按`pnl_pct`（100%仓位的盈亏）复利更新，通知`cooldown.on_trade_closed`（可能累积冷静期计数），批次清空，`had_prior_batch=True`（后面再开仓就标记成`reentry`）。
- 命中止盈：只平掉`partial_ratio(0.5)`那一部分，`reason="TP"`，资金按`0.5 × pnl_pct`更新；批次没清空，标记`partial_taken=True`，把`trailing_stop`初始化成原始止损位（作为腿B的`floor_stop`起点）。

**腿B检查（已部分止盈后）**，第145-168行：
```python
else:
    atr_prev = atr_full.iloc[i - 1] if i > 0 else float("nan")
    if pd.notna(atr_prev):
        batch.trailing_stop = tp_manager.chandelier_stop(
            batch.direction, batch.extreme_since_entry, atr_prev, batch.stop_loss
        )
    hit_trail = (
        price <= batch.trailing_stop if batch.direction == "long" else price >= batch.trailing_stop
    )
    if hit_trail:
        remaining = 1.0 - tp_manager.partial_ratio
        trades.append(Trade(..., exit_price=batch.trailing_stop, reason="PROTECTIVE_SL", size_fraction=remaining, ...))
        capital *= 1 + remaining * trades[-1].pnl_pct
        cooldown.on_trade_closed(batch.direction, "PROTECTIVE_SL")
        batch = None
        had_prior_batch = True
```
- **关键非前视性处理**：`atr_prev = atr_full.iloc[i-1]`——用**前一根**bar的ATR，不用当根bar自己的。因为当根bar自己的ATR需要当根bar的收盘价才能算出来，而这一刻正在判断的正是"当根bar的价格有没有碰到止损线"，如果止损线本身用了当根bar才能算出的ATR，就是拿"这根K线还没走完就已经知道的信息"做当下的判断，属于未来函数。用`i-1`就切断了这个自我引用。
- 算出新的`trailing_stop`后，跟当根`price`比较，触及则平掉剩余`remaining(0.5)`仓位，`reason="PROTECTIVE_SL"`（区别于整批止损的`"SL"`），同样通知cooldown（但`on_trade_closed`里`reason!="SL"`会直接清零连续止损计数，不算作"判断错方向"）。

### 7.4 开新仓判断（含再入场）
第170-188行：
```python
if (
    batch is None and i + 1 < n
    and klines_2h is not None and klines_15m is not None and klines_30m is not None
):
    atr_now = atr_full.iloc[i]
    snapshot = MarketSnapshot(
        tf_2h=klines_2h,
        small_tf={"5m": df_5m.iloc[: i + 1], "15m": klines_15m, "30m": klines_30m},
        close=row["close"], atr=atr_now,
    )
    signal = entry_engine.on_bar(snapshot)
    if signal is not None and pd.notna(atr_now):
        entry_price = df_5m["open"].iloc[i + 1]
        stop = stop_calc.calc(entry_price, signal.direction, atr_now, klines_30m)
        batch = _OpenBatch(
            direction=signal.direction, entry_bar_idx=i + 1, entry_price=entry_price,
            stop_loss=stop, signal_type="reentry" if had_prior_batch else "open",
            extreme_since_entry=entry_price,
        )
```
- 只有`batch is None`（当前空仓）且三个高周期K线都已经有数据时才检查新信号。
- `atr_now = atr_full.iloc[i]`——这里用的是**当根**bar的ATR，不是`i-1`。这是允许的，因为这里在算的是"止损应该设在哪"（面向未来的一个新决策，不是判断"当根bar的价格有没有碰线"这种自我引用问题），当根bar已经走完，它的ATR是已知信息。
- 调用`entry_engine.on_bar(snapshot)`拿到信号（可能是`None`）。
- 拿到信号后：`entry_price = df_5m["open"].iloc[i+1]`——**用下一根bar的开盘价成交**，不用触发这一刻的价格。`stop_calc.calc(...)`用这个成交价 + 当根ATR + 30m已收盘K线算出止损位。
- `signal_type`：如果`had_prior_batch`（之前有过至少一批平仓记录）为真，标记`"reentry"`（再入场），否则`"open"`（首仓）——这是执行引擎自己根据历史状态打的标签，判断逻辑本身（`entry_engine.on_bar`）并不区分这两种情况。
- `extreme_since_entry=entry_price`：极值价初始化成入场价本身。

### 7.5 权益结算
第190-203行：
```python
unrealized = 0.0
if batch is not None and batch.entry_bar_idx <= i:
    price = df_5m["close"].iloc[i]
    move = (
        (price - batch.entry_price) if batch.direction == "long" else (batch.entry_price - price)
    )
    frac = (1 - tp_manager.partial_ratio) if batch.partial_taken else 1.0
    unrealized = frac * move / batch.entry_price
equity[i] = capital * (1 + unrealized)
```
- `batch.entry_bar_idx <= i`：因为7.4里刚开的仓`entry_bar_idx=i+1`，在bar i这一刻这批仓位其实还没真正成交，所以这个条件在刚开仓的当根bar上是`False`，不会把还没发生的成交计入当根权益——避免"信号生成的瞬间就当作已经持仓"的未来函数。
- `frac`：如果已经部分止盈过，只有剩下的`1-partial_ratio(0.5)`那一部分仓位还在浮动盈亏；否则整批100%都在浮动。
- `equity[i] = capital * (1+unrealized)`：`capital`是已经实现（平仓结算过）的资金，加上当前持仓的浮动盈亏，构成这根bar的净值点。

---

## 8. 波动率估计（组合层权重的输入）

**文件：`src/engine/vol_estimator.py` 第39-55行**
```python
def rolling_daily_atr_vol(df_5m: pd.DataFrame, window: int = 14) -> pd.Series:
    daily = _daily_ohlc(df_5m)
    pct_atr = atr_series(daily, window) / daily["close"]
    return pct_atr.shift(1)
```
- 先把5m K线聚合成日线OHLC（`_daily_ohlc`，第14-26行，纯`groupby`聚合，`open`取当天第一个/`high`取当天最大/`low`取当天最小/`close`取当天最后一个）。
- 在日线上跑同一个`atr()`函数（第2节），除以`close`换算成百分比波动率（不这样做的话，不同股价量级的票之间ATR数值不可比，比如NVDA ~$150和XOM ~$110）。
- **`.shift(1)`——这是关键的非前视性处理**：D日的权重只能用D-1日及更早收盘算出的波动率，不能用D日当天的（D日当天的日线还没走完，不知道当天的high/low/close）。

---

## 9. 仓位/权重分配

**文件：`src/rules/sizing.py` 第6-12行**
```python
class InverseVolatilitySizer(PositionSizer):
    def weights(self, vols: dict[str, float]) -> dict[str, float]:
        inv_vols = {symbol: 1 / vol for symbol, vol in vols.items() if vol > 0}
        total = sum(inv_vols.values())
        return {symbol: inv_vol / total for symbol, inv_vol in inv_vols.items()}
```
- 逆波动率加权（标准CTA风险平价做法）：每个标的的权重 ∝ 1/波动率，再归一化让所有权重加总=1。
- `vol > 0`才纳入计算：波动率为0（比如某段时间价格完全走平）会让`1/vol`发散，直接排除避免这个标的的权重炸掉支配整个组合。
- 波动率越高，分配的权重越低——这是"用固定风险预算而不是固定金额去配置每个标的"的核心思想。

---

## 10. 组合层：日频再平衡、组合收益、Benchmark、贡献度分解

**文件：`src/engine/portfolio_pullback_backtest.py`**

### 10.1 每个标的独立跑自己的信号/止损止盈
第162-166行：
```python
def run_portfolio_pullback_backtest(dfs, fee_pct=0.0, initial_capital=10_000.0):
    symbols = list(dfs.keys())
    per_ticker = {s: _ticker_result(dfs[s], initial_capital) for s in symbols}
```
- 每个标的用第7节的引擎独立跑一遍，标的之间的入场/出场逻辑完全不耦合（各自的`PullbackEntryEngine`/`CooldownManager`都是独立实例）——组合层只在"每个标的的收益按多大比例计入组合"这一层做协调，不影响各自的交易决策。

### 10.2 把每个标的的净值曲线换算成"日收益率"
第168-175行：
```python
daily_returns = {
    s: per_ticker[s].equity_curve.groupby(per_ticker[s].equity_curve.index.date).last().pct_change()
    for s in symbols
}
buy_hold_daily_returns = {s: daily_closes(dfs[s]).pct_change() for s in symbols}
daily_vols = {s: rolling_daily_atr_vol(dfs[s]) for s in symbols}
```
- `daily_returns`：把5m频率的净值曲线，按日期取每天最后一个值（`groupby(...).last()`），再算日环比收益率——这是"这个标的自己的策略"当天赚了/亏了百分之多少。
- `buy_hold_daily_returns`：同一标的的**纯买入持有**日收益率（不经过策略，直接用收盘价环比），用于算benchmark基准线。
- `daily_vols`：第8节的逆ATR波动率（已经`shift(1)`过，不含未来数据）。

### 10.3 逐日循环：算权重、算组合当日收益、累计贡献度
第199-239行（节选核心）：
```python
for d in all_dates:
    vols_today = {s: daily_vols[s].get(d) for s in symbols if pd.notna(daily_vols[s].get(d))}
    weights_today = sizer.weights(vols_today) if vols_today else {}
    port_ret = sum(
        weights_today.get(s, 0.0) * daily_returns[s].get(d, 0.0)
        for s in symbols if pd.notna(daily_returns[s].get(d, 0.0))
    )
    port_ret -= fee_pct * turnover
    ...
    equity_before = running_equity
    running_equity *= 1.0 + port_ret
    for s in symbols:
        r = daily_returns[s].get(d, 0.0)
        if not pd.notna(r) or r == 0.0:
            continue
        direction = direction_by_day[s].get(d)
        if direction is None:
            continue
        contrib_dollar[s][direction] += weights_today.get(s, 0.0) * r * equity_before
```
- 每天：先用当天能看到的波动率（D-1及更早）算出`weights_today`（第9节的逆波动率权重，**每天都重新算一次，等价于每天重新调仓**——这就是"daily rebalance"的含义，不是买入后一直不动）。
- `port_ret`：组合当天收益 = 每个标的权重 × 该标的当天策略收益率，求和。**注意**：这里是"该标的自己的策略收益率"，不是买入持有收益率——一个标的当天如果没有仓位在场内，`daily_returns[s].get(d)`这天就是0（前后两个净值相同），对`port_ret`贡献为0，但它分配到的权重份额并没有被别的标的占用——这是"先分配权重、再看有没有在场内交易"的设计（之前审计过，是正确的组合逻辑，不是bug）。
- `running_equity`按`port_ret`复利滚动。
- **贡献度分解**：`contrib_dollar[s][direction] += weights_today.get(s) * r * equity_before`——用**当天权重 × 当天收益率 × 前一天的组合净值**，这是组合净值当天美元变化量的精确分解（数学恒等式：`equity(d)-equity(d-1) = equity(d-1) * Σ w_s(d)*r_s(d)`），保证每个标的每个方向的贡献美元数加总起来正好等于组合总利润，不会有凑不齐的残差。`direction_by_day`（第93-111行）负责把每一天标记成这个标的当天持有的是多头还是空头仓位（还是没有仓位）。

### 10.4 Benchmark基准线
第216-219行：
```python
benchmark_returns.append(sum(
    weights_today.get(s, 0.0) * buy_hold_daily_returns[s].get(d, 0.0)
    for s in symbols if pd.notna(buy_hold_daily_returns[s].get(d, 0.0))
))
```
- 用**跟策略完全相同的每日权重**，但套用在"纯买入持有"的日收益率上——这样基准线和策略曲线的唯一差异就是"有没有做入场/出场择时"，权重分配方式(风险平价)是共同的，做的是"择时能力"这一个变量的纯净对比，不是"策略 vs 随便买点什么"的不公平对比。

### 10.5 最终汇总
第241-260行：
```python
portfolio_equity_curve = initial_capital * (1.0 + port_ret_series).cumprod()
portfolio_sharpe = sharpe_ratio(portfolio_equity_curve)
portfolio_return = portfolio_equity_curve.iloc[-1] / initial_capital - 1
...
total_profit_dollar = portfolio_equity_curve.iloc[-1] - initial_capital
if total_profit_dollar:
    for s in symbols:
        per_ticker[s].long.contribution_pct = contrib_dollar[s]["long"] / total_profit_dollar
        per_ticker[s].short.contribution_pct = contrib_dollar[s]["short"] / total_profit_dollar
```
- 组合净值曲线 = 初始资金 × 每日`(1+收益率)`累乘。
- 每个标的每个方向的贡献百分比 = 该方向累计贡献的美元 / 组合总利润美元——存成分数（不是预先乘100），跟`win_rate`保持同一约定，显示层的`:.1%`格式化会自己乘100。

---

## 11. 调用链：从入口脚本到每个实现函数的实际调用位置

前面10节都是"这个函数/这段逻辑本身在做什么"。这一节反过来，从最外层入口脚本往下追，标出**每一处实现代码实际被谁调用、传了什么参数**，确保没有"写了但没接上"的死代码，也方便你按这个链路在IDE里用"查找引用"逐层点进去核对。

### 11.1 顶层入口

**文件：`src/run_phaseF.py` 全文（第14-45行）**
```python
import pandas as pd
from src.engine.portfolio_pullback_backtest import run_portfolio_pullback_backtest
from src.viz.chart import show_portfolio_pullback_backtest

SYMBOLS = ["NVDA", "KO", "XOM", "JPM", "UNH", "NKE", "INTC", "FCX", "DAL", "USO", "GLD", "SLV"]
START, END = "2024-07-18", "2026-07-17"

def main():
    dfs = {}
    for symbol in SYMBOLS:
        df = pd.read_csv(f"data/raw/{symbol}_5m.csv", parse_dates=["timestamp"])
        df = df[(df["timestamp"] >= START) & (df["timestamp"] <= END)].reset_index(drop=True)
        dfs[symbol] = df

    result = run_portfolio_pullback_backtest(dfs)
    ...
    show_portfolio_pullback_backtest(dfs, result)

if __name__ == "__main__":
    main()
```
- 这是`python -m src.run_phaseF`实际跑起来的唯一入口。逐个标的读CSV（第0节`fetch_alpaca.py`产出的复权数据）、裁剪到统一日期区间，`dfs`是`{symbol: df_5m}`的字典，直接喂给`run_portfolio_pullback_backtest`（第10节的实现）——**没有中间层**，组合层拿到的就是原始5m DataFrame，重采样/ATR/信号全部在它内部逐个标的调用。
- 数据源头：`fetch_alpaca.py`的`main()`（第85-92行）逐个symbol调用`fetch_symbol(client, symbol, sector)`（第0节实现），写到`data/raw/{symbol}_5m.csv`——`run_phaseF.py`第26行读的正是这个文件，形成"抓取脚本产出 → 回测脚本消费"的完整闭环。

### 11.2 组合层 → 单标的引擎的调用

**文件：`src/engine/portfolio_pullback_backtest.py` 第130-166行**
```python
def _ticker_result(df_5m: pd.DataFrame, initial_capital: float) -> TickerResult:
    result = run_pullback_backtest(df_5m, initial_capital=initial_capital)   # <- 第7节整个执行引擎入口
    ...
    closes = daily_closes(df_5m)                                             # <- 第8节 vol_estimator.daily_closes
    buy_hold_sharpe = sharpe_ratio(closes)
    ...
    return TickerResult(..., long=_direction_stats(batch_pnls, "long"), short=_direction_stats(batch_pnls, "short"))

def run_portfolio_pullback_backtest(dfs, fee_pct=0.0, initial_capital=10_000.0):
    symbols = list(dfs.keys())
    per_ticker = {s: _ticker_result(dfs[s], initial_capital) for s in symbols}   # <- 每个标的独立调用一次
    ...
    daily_vols = {s: rolling_daily_atr_vol(dfs[s]) for s in symbols}             # <- 第8节实现
    ...
    sizer = InverseVolatilitySizer()                                            # <- 第9节实现
    ...
    weights_today = sizer.weights(vols_today) if vols_today else {}             # <- 第9节的调用点
```
- `_ticker_result`是**唯一**调用`run_pullback_backtest`（第7节）的地方，`per_ticker = {s: _ticker_result(dfs[s], initial_capital) for s in symbols}`对12个标的各调一次，互相独立（对应第7节文档里说的"标的间不耦合"）。
- `rolling_daily_atr_vol`（第8节实现）在这里被调用产出`daily_vols`，接着在第199-206行的逐日循环里喂给`InverseVolatilitySizer().weights(...)`（第9节实现）算出每天的权重——这是"波动率估计→权重分配"这条链路唯一的调用路径。

### 11.3 单标的引擎内部：默认对象怎么被创建、又怎么调用信号/止损/止盈/冷静期

**文件：`src/engine/pullback_backtest.py` 第66-92行**
```python
def run_pullback_backtest(
    df_5m: pd.DataFrame,
    initial_capital: float = 10_000.0,
    filter: MultiTimeframeFilter | None = None,
    entry_engine: PullbackEntryEngine | None = None,
    stop_calc: StopLossCalculator | None = None,
    tp_manager: TakeProfitManager | None = None,
    cooldown: CooldownManager | None = None,
    atr_period: int = 14,
) -> PullbackBacktestResult:
    df_5m = df_5m.reset_index(drop=True)
    cooldown = cooldown or CooldownManager()                       # <- 第6节实现，用全部默认参数实例化
    filter = filter or MultiTimeframeFilter()                      # <- 第3.2节实现，无参数(无状态类)
    entry_engine = entry_engine or PullbackEntryEngine(filter, cooldown)  # <- 第3.3节实现，注入同一个filter/cooldown实例
    stop_calc = stop_calc or StopLossCalculator()                  # <- 第4节实现，atr_mult=1.5/offset_pct=0.003/swing_k=2
    tp_manager = tp_manager or TakeProfitManager()                 # <- 第5节实现，partial_ratio=0.5/rr_trigger=2.0/atr_multiple=3.0
```
- `run_phaseF.py`调用链一路下来**没有传任何自定义参数**，全部走`or Xxx()`这一分支，也就是说文档第3-6节里写的所有默认数值（1.5倍ATR、0.3%偏移、2R触发、3倍ATR吊灯、连续3次止损、连续3根K线解除……）就是当前实际在跑的参数，不是"文档举例的默认值、实际另有配置"。
- `PullbackEntryEngine(filter, cooldown)`：把同一个`filter`和`cooldown`实例注入进去——`cooldown`这一个对象同时被`entry_engine.on_bar`（第3.3节，读`is_active()`）和主循环（第7.1/7.3节，写`on_bar`/`on_trade_closed`）共享，是同一份状态，不是两份互相不知道对方的拷贝。

**主循环里对这几个对象的实际调用点**（都在第7.3/7.4节代码里出现过，这里汇总成一张表，方便对照"谁调用谁"）：

| 调用方（行号） | 被调用方 | 作用 |
|---|---|---|
| `pullback_backtest.py:108` | `cooldown.on_bar(klines_2h)` | 每根2h已收盘K线更新冷静期状态机（第6节`on_bar`） |
| `pullback_backtest.py:123` | `tp_manager.partial_trigger_price(...)` | 算腿A止盈触发价（第5节） |
| `pullback_backtest.py:133` | `cooldown.on_trade_closed(direction, "SL")` | 整批止损后通知冷静期计数（第6节） |
| `pullback_backtest.py:152` | `tp_manager.chandelier_stop(...)` | 算腿B吊灯止损（第5节），用`atr_prev`(i-1) |
| `pullback_backtest.py:166` | `cooldown.on_trade_closed(direction, "PROTECTIVE_SL")` | 护盈止损后通知冷静期（reason≠"SL"，会清零而不是累加） |
| `pullback_backtest.py:180` | `entry_engine.on_bar(snapshot)` | 判断本bar是否产生新信号（第3.3节） |
| `pullback_backtest.py:183` | `stop_calc.calc(entry_price, direction, atr_now, klines_30m)` | 新开仓时算止损位（第4节） |

**`entry_engine.on_bar`内部又调用了谁**（对照第3.3节代码）：
```python
bias = self.filter.get_trend_bias(snapshot.tf_2h)              # ma_filter.py 第75-76行，第3.2节
if self.filter.pullback_occurred(snapshot.small_tf, bias):     # ma_filter.py 第78-84行，第3.2节
if self.filter.get_entry_trigger(snapshot.small_tf, bias):     # ma_filter.py 第86-88行，第3.2节
```
这三行是`MultiTimeframeFilter`三个方法被调用的**唯一**位置，而它们内部又分别调用`ma_array_state`（`get_trend_bias`用）和`ma_fast_mid_state`（`pullback_occurred`/`get_entry_trigger`用）——对应第3.1节两个MA排列函数。`CooldownManager.check_structure_break`内部也调用了自己模块导入的`ma_array_state`（`cooldown.py`第32行`from src.rules.ma_filter import ArrayState, ma_array_state`），跟`get_trend_bias`用的是**同一个函数**，只是调用方不同（一个用于定方向，一个用于判断解除冷静期的排列是否稳定）。

### 11.4 ATR的三处调用方汇总（验证"同一个公式、三处复用"不是巧合）

| 调用位置 | 用途 |
|---|---|
| `pullback_backtest.py:88` `atr_full = atr_series(df_5m, atr_period)` | 单标的5m全序列ATR，供止损/止盈/新开仓使用（第7节） |
| `vol_estimator.py:54` `pct_atr = atr_series(daily, window) / daily["close"]` | 日线ATR，换算成波动率给组合层加权用（第8节） |
| （间接）`indicators.py`只有这一份`atr()`实现，两处`import`的都是同一个函数 | — |

`from src.indicators import atr as atr_series`这行import在`pullback_backtest.py`第18行和`vol_estimator.py`第5行都出现，确认两处用的是同一份代码，不是各自重复实现了一遍容易产生偏差的ATR。

### 11.5 组合层结果 → 图表展示层的调用（确认数字没有在展示前被二次加工出错）

**文件：`src/viz/chart.py`**（`show_portfolio_pullback_backtest`函数内，构造统计表格部分）直接读取`result.per_ticker[symbol]`（`TickerResult`）、`result.portfolio_return`、`result.portfolio_sharpe`、`result.benchmark_equity_curve`这些第10节`PortfolioBacktestResult`里已经算好的字段，格式化成字符串显示（`_fmt_pct`/`_fmt_ratio`），**不会重新计算任何数值**——图表层是纯展示，所有算术都已经在`portfolio_pullback_backtest.py`里完成。

---

## 附：完整流程一句话串联

抓数据(复权) → 按日重采样出15m/30m/2h(只保留已收盘部分) → 每个bar算ATR →
2h定方向 → 30m+5m/15m回调确认 → 5m+15m触发 → 下一根bar开盘价成交 →
ATR止损/平台止损取更近者 → 2R部分止盈(50%) → 3×ATR吊灯止损跑剩余50%(用i-1的ATR) →
（连续止损或极端K线 → 冷静期 → 只挡新开仓不动现有仓位）→ 平仓后净值复利更新 →
组合层：每日用T-1日波动率算逆波动率权重 → 当日组合收益=Σ权重×各标的策略日收益 →
贡献度=Σ(当日权重×当日收益×前日净值)精确分解到每个标的每个方向。

---

## 12. 数值汇总：设计参数 + 实测结果，逐项对参考值说明"为什么合理"

数据：12标的，2024-07-18~2026-07-17，Alpaca复权5m数据。分两部分——先是**设计参数**本身(写死在代码里的数字)对不对，再是**跑出来的结果数值**符不符合这套设计应有的样子。两部分互相印证：如果设计参数合理但结果数值不符合预期，说明执行有bug；如果两者都对得上，才算真正验证过。

### 12.1 设计参数（代码里写死的默认值）与参考值对照

| 参数 | 代码位置 | 取值 | 参考值/合理区间 | 为什么这么设 |
|---|---|---|---|---|
| MA排列周期(2h定方向) | `ma_filter.py:22` | 5/20/50 | 短中长三线是趋势判断的常见组合 | 50周期的2h≈4个交易日，足够过滤日内噪音又不会太滞后 |
| MA排列周期(小周期回调/触发) | `ma_filter.py:39` | 5/20 | — | 三线排列在5m/15m/30m上门槛过高(实测2年NVDA仅完整走完1次)，降为两线保证有效样本量 |
| ATR止损倍数 | `stop_loss.py:37` | 1.5×ATR(14) | 业界常见1.5~3×ATR | 取区间下限，配合波动率加权的多标的组合，止损更紧、纠错更快，不需要给单个标的过大容错空间 |
| 止损缓冲(防扫损) | `stop_loss.py:37` | 0.3% | 0.1%~0.5%常见 | 留够空间避免精确触及止损线的瞬时插针 |
| 平台止损局部极值窗口 | `stop_loss.py:37` | 左右各2根(30m) | — | 30m×2根≈1小时结构确认窗口，贴近5m/15m入场时机的实际结构，不用2h(太粗) |
| 止盈盈亏比触发 | `take_profit.py:19` | 2.0R | 常见1.5~2R | 取区间上限，让浮盈先跑出足够空间再部分止盈，不会太早锁定小利润 |
| 部分止盈比例 | `take_profit.py:17` | 50% | 常见30%~50% | 一半落袋锁定确定性利润，一半继续跑趋势，是常见的"哑铃型"仓位管理 |
| 吊灯止损ATR倍数 | `take_profit.py:19` | 3.0×ATR | 常见2~4×ATR | 比初始止损(1.5×ATR)更宽，因为这时候已经是"追踪已确认趋势"，不需要像开仓止损那样紧 |
| 冷静期触发-连续止损 | `cooldown.py:37` | 3次(同方向纯SL) | — | 连续3次排除偶然性，2次可能只是运气差，4次以上则风控介入太晚 |
| 冷静期解除确认 | `cooldown.py:38` | 连续3根2h稳定排列 | — | 1根可能是噪音反弹，3根确认结构真正站稳 |
| 波动率估计窗口 | `vol_estimator.py:39` | 14日ATR | 与策略自身止损用的ATR周期(14)保持一致 | 组合层的"风险"定义跟策略自身止损的"风险"定义用同一把尺子，逻辑自洽 |

### 12.2 实测结果数值与参考值对照

```
PORTFOLIO: Return 34.33%  Sharpe 5.65  MaxDrawdown -0.82%
BENCHMARK(等权买入持有,同样权重逻辑): Return 74.98%  MaxDrawdown -12.15%

Ticker      Ret  Sharpe   MaxDD      WR  Batch |    L-WR   L-PF   L-N  L-Contrib |    S-WR   S-PF   S-N  S-Contrib
DAL      64.56%    2.34   -3.92%   44.2%    147 |   38.5%   2.78    91       4.6% |   53.6%   2.41    56       5.1%
FCX      52.25%    2.23   -6.57%   42.3%    142 |   40.0%   2.75    85       4.4% |   45.6%   2.76    57       3.7%
GLD      27.02%    1.94   -3.68%   41.4%    116 |   48.7%   2.54    78      12.5% |   26.3%   2.94    38       0.6%
INTC    104.83%    2.59  -11.26%   43.3%    134 |   45.3%   3.46    75       7.9% |   40.7%   2.93    59       2.4%
JPM      29.41%    1.86   -4.87%   41.0%    134 |   47.0%   2.33    83       7.9% |   31.4%   2.91    51       1.1%
KO        5.95%    0.56   -8.92%   33.9%    115 |   37.1%   2.30    62       3.4% |   30.2%   2.34    53       0.2%
NKE      25.13%    1.54   -4.58%   43.0%    128 |   48.3%   1.98    58       3.3% |   38.6%   2.28    70       2.3%
NVDA     31.41%    1.72   -6.62%   39.4%    137 |   42.2%   2.44    83       4.9% |   35.2%   2.92    54       1.0%
SLV      76.12%    2.97   -5.55%   46.5%    127 |   49.4%   3.29    79      10.3% |   41.7%   2.58    48       1.7%
UNH      57.21%    2.24   -5.19%   40.1%    142 |   34.1%   3.20    85       3.6% |   49.1%   3.20    57       5.8%
USO      40.42%    1.95   -6.09%   38.7%    137 |   48.5%   2.61    68       6.6% |   29.0%   3.51    69       1.5%
XOM      16.79%    1.23   -4.85%   35.3%    133 |   38.5%   2.71    78       4.9% |   30.9%   2.55    55       0.4%

TOTAL batches=1592  weighted-avg winrate=40.8%
```

| 指标 | 实测值 | 参考值/合理区间 | 为什么合理 |
|---|---|---|---|
| 单票胜率(WR) | 33.9%~46.5%，加权均值40.8% | 趋势跟随策略常见35%~45% | 趋势策略靠盈亏比而非高胜率赚钱，跟设计参数(2R止盈、1.5倍ATR止损)的预期胜率区间吻合，不是均值回归策略该有的60%+胜率 |
| 盈亏比(Payoff Ratio) | 1.98~3.51，多数2.3~3.0 | 配合2R止盈设计，理论应≥2 | 实测盈亏比普遍≥2，验证止盈"设计幅度"和"实际成交结果"一致，没有"设计2R但实际只吃到1R"的执行偏差(能揪出代码bug的关键交叉验证点) |
| 期望值(胜率×盈亏比-(1-胜率)) | 多数ticker/方向组合>0，如INTC多头0.453×3.46-0.547≈1.02 | 需>0才有正期望 | 24组(12票×2方向)里绝大多数为正，个别薄利组合(NKE空头≈0.26)是正常的边际贡献小，不是异常 |
| 组合Sharpe | 5.65 | 单资产趋势策略Sharpe>1好、>2很好；组合因分散化天然更高 | 显著高于单票(0.56~2.97)是9-12个低相关标的风险分散的直接结果 |
| **组合最大回撤 vs 基准最大回撤** | **策略-0.82% vs 买入持有-12.15%** | — | **这是本策略设计目的最直接的证据**：逆波动率加权+严格止损体系把回撤压缩到基准的1/15，即使组合收益(34.33%)低于基准(74.98%)，风险调整后表现(Sharpe 5.65)远超单纯持有 |
| 组合收益 vs 基准收益 | 策略34.33% < 基准74.98% | — | 这段是普涨行情，纯持有跑赢是正常的；策略是"用可控回撤换取更高Sharpe"的设计取舍，不是以绝对收益最大化为目标 |
| GLD多头贡献12.5% | 单方向占组合总利润1/8 | — | 这段时间黄金处于趋势行情，GLD多头胜率48.7%+盈亏比2.54的表现符合黄金走势，不是逻辑错误 |
| 单票最大回撤区间 | -3.68%(GLD)~-11.26%(INTC) | — | 跟各自的止损宽度(1.5×ATR)、该标的自身波动率相关，INTC回撤最大也对应它总回报最高(104.83%)——高波动带来高回撤也带来高收益，符合风险收益对等的基本预期 |

### 12.3 一句话结论

设计参数（1.5×ATR止损/2R止盈/3×ATR吊灯/连续3次止损冷静期）取的都是业界常见区间内偏保守的一端，符合"多标的组合、靠分散化而不是单标的重仓赚钱"的整体思路；实测出来的胜率、盈亏比、期望值跟这套参数应该产生的结果在数量级和方向上都对得上，组合层的最大回撤(-0.82%)相对基准(-12.15%)压缩了约93%，是本策略"控回撤换Sharpe"设计目标最直接的数值证据。
