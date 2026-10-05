# SDS portfolio monitor

A Streamlit dashboard that tracks the SDS model portfolios against IHSG, using prices from Yahoo Finance.

## Run it

```
pip install -r requirements.txt
streamlit run app.py
```

It opens at http://localhost:8501. You need Python 3.10 or newer and an internet connection (for Yahoo Finance prices and the Google font).

## What's in the folder

```
app.py                   the dashboard
portfolios/              one CSV per portfolio; the file name becomes the portfolio name
.streamlit/config.toml   light and dark themes, font and colors (a hidden folder on Mac and Linux)
requirements.txt
```

## Portfolio files

Each file has the columns **Date, Symbol, Sector, Portfolio %**, like the four you started with.

- **Other layouts:** Indonesian headers (Tanggal, Kode, Sektor, Bobot), semicolon-separated files and decimal commas (7,20%) also work.
- **Symbols:** use IDX codes without the suffix. `.JK` is added for Yahoo automatically.
- **Dates:** 10/1/2026 and 01/10/2026 both read as 1 October 2026. When a date could be month/day or day/month, the app picks the reading that falls on a weekday and not in the future. The Data tab shows how each file was read.
- **Weights:**
  - between 99.5% and 100.5%: treated as rounding and scaled to 100%;
  - above 100.5%: scaled down, with a note in the Data tab (your three SDS_Foreign_40 files add up to 100.67–101.56%);
  - below 99.5%: the rest is held as cash.
- **Rebalancing:** add rows with the new date and new weights to the same file. Holdings dated d are bought at the close of d, and returns chain-link from one date to the next.
- **New portfolios:** drop a CSV into `portfolios/`, or upload one in the sidebar for the current session only.

## What each tab answers

- **Summary:** Who is ahead since the start, and by how much versus IHSG (the dashed line in the standings). Why the leader leads, what moved today, and an overall scorecard.
- **Performance:** Returns from 1 day to year-to-date, lead over IHSG, drawdowns and daily returns.
- **Attribution:** Which stocks and sectors drove each portfolio, positions with P&L in rupiah and lot sizes, and a head-to-head breakdown of why one portfolio beats another.
- **Holdings:** Concentration, sector weights, stock weights side by side, and how much the portfolios overlap.
- **Risk & backtest:** How today's weights would have behaved over 6 months to 3 years: growth, drawdowns, volatility, risk ratios, correlation and where the risk comes from.
- **Data:** Price coverage per stock, file checks, suspicious price moves and CSV downloads.

## How the numbers are calculated

- **Live returns** are buy-and-hold from the close on the start date ("Track from" in the sidebar), with weights drifting as prices move. They're total returns: Yahoo's dividend-adjusted closes. Prices shown in tables are actual closes.
- **Today** is the move from the previous close to the latest price, which is intraday while IDX is trading.
- **Contribution** is the target weight times the stock's return since entry. Contributions add up exactly to the portfolio's return. In the head-to-head view, the gap between two portfolios is the sum of (weight difference × stock return).
- **The backtest** applies today's target weights to daily returns over the chosen window, rebalanced daily. Treat it as a guide to risk, not a track record. Stocks that listed during the window (AADI, EMAS) are left out until they listed, and their weight is spread across the rest.
- **Ratios:**
  - Sharpe and Sortino subtract the risk-free rate, which defaults to the BI Rate of 5.75%.
  - Information ratio is the yearly excess return over IHSG divided by tracking error.
  - Capture ratios compare average returns on IHSG's up days and down days.
  - VaR and CVaR are historical, one day, at 95%.
- **The scorecard** ranks each portfolio on six measures and averages the ranks. One measure is the live return since start; the other five come from the backtest. Until there are about 20 live trading days, the live part is mostly noise.

## Troubleshooting

- **"Prices couldn't be loaded"**: check your connection, run `pip install -U yfinance`, then press **Refresh prices**. Yahoo sometimes limits requests; waiting a minute usually fixes it.
- **A stock shows "No prices" in the Data tab**: check that the code exists on finance.yahoo.com (for example BBCA.JK). Until it does, its weight is spread across the other holdings.
- **A strange jump in a chart**: the Data tab lists daily moves over 35%. That's beyond IDX price limits, and usually means an unadjusted split in Yahoo's data.
- **Plain font**: the app loads Plus Jakarta Sans from Google Fonts and falls back to the system font when offline.

## Changing defaults

The settings at the top of `app.py` set the defaults:

- `APP_TITLE`
- `DEFAULT_BENCHMARK`
- `DEFAULT_RISK_FREE`
- `DEFAULT_CAPITAL`
- `PRICE_TTL` (how long prices are cached)
- `PORTFOLIO_COLORS`
