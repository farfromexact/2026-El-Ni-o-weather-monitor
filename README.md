# 全球天气驱动的白糖/棕榈油多头监控面板

这是一个 Streamlit 每日监控面板，用于跟踪全球主产区天气是否正在形成对白糖 `SR0` 和棕榈油 `P0` 的多头交易条件。

## 快速运行

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\streamlit run app.py
```

## 数据源

- 天气：[Open-Meteo Forecast API](https://open-meteo.com/en/docs) 和 [Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)。
- 行情：[AKShare 期货数据](https://akshare.akfamily.xyz/data/futures/futures.html) 的 `futures_main_sina` 主力连续合约接口，或 iFinD（首选 `cmd_history_quotation`，失败后回退 SDK 登录）。
- 合约代码：白糖 `SR0`，棕榈油 `P0`。

如果你希望直接使用 iFinD 登录方式，可在 `.streamlit/secrets.toml` 或环境变量里设置：
```
IFIND_REFRESH_TOKEN=...
IFIND_USERNAME=...
IFIND_PASSWORD=...
```

如果 AKShare 或行情源不可用，可以在面板里上传 CSV。字段至少包含 `date/open/high/low/close`，也支持中文字段 `日期/开盘价/最高价/最低价/收盘价`。

## 信号逻辑

面板不会只因为天气异常就提示建仓，而是分三层：

1. 天气分数：按主产区权重汇总未来 7-16 天降雨、最高温和水分差，并与历史同期中位数比较。
2. 价格确认分：检查收盘价是否站上 20/60/120 日均线、是否突破 20 日高点，以及成交量和持仓量是否放大。
3. 综合建仓提示：天气和价格同时达标才提示“开始试仓”。

默认建仓纪律：

- 天气分数达标、价格未达标：等待突破或回踩确认。
- 价格分数达标、天气未达标：技术偏强但不按天气主题追多。
- 天气和价格同时达标：只提示试仓，默认 20%-30% 计划仓位。
- 跌破 60 日均线或天气分数明显回落：降低或取消天气主题仓位。

## 风险说明

这个工具是研究与交易纪律面板，不构成投资建议。期货带杠杆，必须自行设置单笔风险、止损和总仓位上限。
