1.1(timestamp open/close match frontend+backend)|1.2(?split/dividend cause candle geometry to distort)|1.3*(Misrecorded data--needle data)|1.4*(?long term trajectory risk--simply using tickers big today is biased)|1.6*(futures, contract splicing)
2.1(bar i future var)|2.4(check indicator data: does MA use future, does ATR use future--yes, trailing ATR use bar i close, increase trail SL)|2.6*(initialization leakage)
3.1(talked,intrabar path)|3.2(try to fill close on same bar)|3.4*(touch-implies-fill)|3.5*(transaction fee, spread & slippage)|3.6*(phantom leverage)
4.1(data snooping, use train-test pair)|4.4*(ticker# increase will use; multiticker does not imply reduced risk)|4.6
5.1*(latency)|5.3*(vendor, now just IEX)|5.4*(market impact for large capitals)|5.5*(short sale hard-to-borrow)

left loose: 2.5

Meeting:
split/dividend--how to delete（复权）--日线调整如何在分钟线上体现
future/contract splitting--how to create--中国商品期货交易方面相关信息--交易所规则，数据形式，交易时段，规则(order类型)

2.4--指标计算都是bar i封闭才计算

2，3可以并一起--一根k线上只能有一个操作（五分钟）

3.2用上一个bar的close来触发，成交在下一根bar open

min/max open跳空

portfolio return console: 
Batches(L/S), Up/Down days--use percentage
Avg Move Total delete
Contribution--assume single ticker portfolio (not entire port), calculate (L+S should add up to total gain/loss for a single ticker)

界面重新排版：
1.Return, Sharpe, B&H Return, B&H Sharpe, Contribution %
2.WinR%, WinR (L/S), Payoff, MaxDD
3.Batches (T/L/S), Up/Down days, Avg move.
Q: why up/down days are inconsistent among tickers

第一个：fix all
第二个：future/contract splitting--how to create--中国商品期货交易方面相关信息--交易所规则，数据形式，交易时段，规则(order类型)；下载一分钟级别期货，看主力连续合约拼接--用乘法方式，不用平移，是为了复利（平移保留价差不保证收益率）
第三个：回调如何定义，判断，变种，量化；怎么入场；从多头到空头过渡阶段的原则--用web research; 为什么不做突破；反手/延续如何判断；等价不是自洽，自洽放宽条件

1. ATR时间错配--day的1.5用了5m的1.5atr；2h最后一根是30m，MA会失真，要不要换到1h? 最终数据肯定是1m; MaxDD需要考虑日内回撤而不是单日计算；
2. FAK FOK 如何定义？开仓 平仓 平今--平今本质上就是day trade? &价差问题用前复权还是后复权？
3.