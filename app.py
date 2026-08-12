import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta

# ==========================================
# 頁面配置 (優化手機端體驗)
# ==========================================
st.set_page_config(
    page_title="台股背離警報與分析",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("📈 台股背離掃描與多維分析")

# ==========================================
# 工具函數：獲取資料與指標計算
# ==========================================
@st.cache_data(ttl=3600)
def get_stock_data(ticker_symbol):
    try:
        df = yf.download(ticker_symbol, period="1y", interval="1d", progress=False)
        if df.empty or len(df) < 60:
            return None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # 1. RSI
        df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
        
        # 2. KD
        stoch = ta.momentum.StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3)
        df['K'] = stoch.stoch()
        df['D'] = stoch.stoch_signal()
        
        # 3. MACD
        macd = ta.trend.MACD(close=df['Close'])
        df['MACD_hist'] = macd.macd_diff()
        
        # 4. 成交量均線
        df['Vol_MA5'] = df['Volume'].rolling(5).mean()
        
        return df.dropna()
    except Exception as e:
        return None

def detect_divergence(df, lookback=30):
    """
    優化後的背離檢測算法：
    比對最近 5 天的極值與前 5~Lookback 天的極值
    """
    if df is None or len(df) < lookback:
        return {}

    recent = df.iloc[-5:]        # 最近 5 天
    past = df.iloc[-lookback:-5] # 更早之前 25 天

    results = {}

    # 1. RSI 背離
    rsi_bull = (recent['Low'].min() < past['Low'].min()) and (recent['RSI'].min() > past['RSI'].min())
    rsi_bear = (recent['High'].max() > past['High'].max()) and (recent['RSI'].max() < past['RSI'].max())
    results['RSI_Bullish'] = rsi_bull
    results['RSI_Bearish'] = rsi_bear

    # 2. KD 背離
    kd_bull = (recent['Low'].min() < past['Low'].min()) and (recent['K'].min() > past['K'].min())
    kd_bear = (recent['High'].max() > past['High'].max()) and (recent['K'].max() < past['K'].max())
    results['KD_Bullish'] = kd_bull
    results['KD_Bearish'] = kd_bear

    # 3. MACD 背離
    macd_bull = (recent['Low'].min() < past['Low'].min()) and (recent['MACD_hist'].min() > past['MACD_hist'].min())
    macd_bear = (recent['High'].max() > past['High'].max()) and (recent['MACD_hist'].max() < past['MACD_hist'].max())
    results['MACD_Bullish'] = macd_bull
    results['MACD_Bearish'] = macd_bear

    # 4. 價量背離 (價跌量縮 / 價漲量縮)
    # 價漲量縮：價格創近期新高，但5日均量小於過去均量
    vol_bear = (recent['High'].max() > past['High'].max()) and (recent['Vol_MA5'].iloc[-1] < past['Vol_MA5'].mean())
    # 價跌量縮：價格創近期新低，但5日均量顯著低於過去均量
    vol_bull = (recent['Low'].min() < past['Low'].min()) and (recent['Vol_MA5'].iloc[-1] < past['Vol_MA5'].mean())
    results['Price_Vol_Bearish'] = vol_bear
    results['Price_Vol_Bullish'] = vol_bull

    return results

# 擴充觀察股票池（增加中小型熱門股，容易出現背離訊號）
DEFAULT_STOCKS = {
    "2330.TW": "台積電", "2317.TW": "鴻海", "2454.TW": "聯發科", "2382.TW": "廣達",
    "2308.TW": "台達電", "2303.TW": "聯電", "2881.TW": "富邦金", "2882.TW": "國泰金",
    "3231.TW": "緯創", "2356.TW": "英業達", "2603.TW": "長榮", "2609.TW": "陽明",
    "2002.TW": "中鋼", "1301.TW": "台塑", "2379.TW": "瑞昱", "3034.TW": "聯詠"
}

# ==========================================
# 側邊欄：篩選與控制項
# ==========================================
st.sidebar.header("🔍 篩選條件設定")
selected_div_type = st.sidebar.selectbox(
    "選擇背離類型",
    [
        "RSI 底背離 (看多)", "RSI 頂背離 (看空)",
        "KD 底背離 (看多)", "KD 頂背離 (看空)",
        "MACD 底背離 (看多)", "MACD 頂背離 (看空)",
        "價量底背離 (價跌量縮)", "價量頂背離 (價漲量縮)"
    ]
)

div_map = {
    "RSI 底背離 (看多)": "RSI_Bullish",
    "RSI 頂背離 (看空)": "RSI_Bearish",
    "KD 底背離 (看多)": "KD_Bullish",
    "KD 頂背離 (看空)": "KD_Bearish",
    "MACD 底背離 (看多)": "MACD_Bullish",
    "MACD 頂背離 (看空)": "MACD_Bearish",
    "價量底背離 (價跌量縮)": "Price_Vol_Bullish",
    "價量頂背離 (價漲量縮)": "Price_Vol_Bearish",
}

target_key = div_map[selected_div_type]

# 執行掃描
matched_stocks = {}
with st.spinner("正在掃描市場背離訊號中..."):
    for symbol, name in DEFAULT_STOCKS.items():
        df = get_stock_data(symbol)
        if df is not None:
            divs = detect_divergence(df)
            if divs.get(target_key, False):
                matched_stocks[symbol] = name

# ==========================================
# 主畫面：顯示篩選結果與股票詳情
# ==========================================
st.subheader(f"🎯 符合【{selected_div_type}】的股票 ({len(matched_stocks)} 檔)")

if not matched_stocks:
    st.warning("⚠️ 目前追蹤清單中沒有符合該背離條件的股票。你可以切換側邊欄的其他背離類型試試看！")
else:
    selected_stock = st.selectbox(
        "請選擇要查看的股票：",
        options=list(matched_stocks.keys()),
        format_func=lambda x: f"{x.replace('.TW','')} {matched_stocks[x]}"
    )
    
    if selected_stock:
        df = get_stock_data(selected_stock)
        ticker_info = yf.Ticker(selected_stock)
        
        last_close = df['Close'].iloc[-1]
        prev_close = df['Close'].iloc[-2]
        change = last_close - prev_close
        pct_change = (change / prev_close) * 100
        
        c1, c2, c3 = st.columns(3)
        c1.metric("最新成交價", f"{last_close:.2f} 元", f"{change:+.2f} ({pct_change:+.2f}%)")
        c2.metric("成交量", f"{int(df['Volume'].iloc[-1]/1000):,} 張")
        c3.metric("52週最高/最低", f"{df['High'].max():.1f} / {df['Low'].min():.1f}")

        # Plotly 圖表
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'), row=1, col=1)
        colors = ['red' if c >= o else 'green' for c, o in zip(df['Close'], df['Open'])]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='成交量', marker_color=colors), row=2, col=1)
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 基本面", "🏦 籌碼面", "📰 消息面", "📐 趨勢與支撐壓力", "🔮 未來一週展望"])

        with tab1:
            try:
                info = ticker_info.info
                st.markdown(f"""
                - **產業類別**：{info.get('industry', 'N/A')}
                - **本益比 (PE)**：{info.get('trailingPE', 'N/A')}
                - **股價淨值比 (PB)**：{info.get('priceToBook', 'N/A')}
                - **殖利率**：{info.get('dividendYield', 0)*100:.2f}%
                """)
            except:
                st.write("暫無法獲取詳細基本面數據。")

        with tab2:
            vol_ma5 = df['Volume'].rolling(5).mean().iloc[-1]
            vol_ma20 = df['Volume'].rolling(20).mean().iloc[-1]
            st.write(f"- **5日均量**：{int(vol_ma5/1000):,} 張")
            st.write(f"- **20日均量**：{int(vol_ma20/1000):,} 張")

        with tab3:
            try:
                news = ticker_info.news
                if news:
                    for item in news[:3]:
                        st.markdown(f"- [{item.get('title')}]({item.get('link')})")
                else:
                    st.write("目前尚無最新新聞。")
            except:
                st.write("無法載入消息面資料。")

        with tab4:
            high_60 = df['High'].tail(60).max()
            low_60 = df['Low'].tail(60).min()
            ma20 = df['Close'].rolling(20).mean().iloc[-1]
            st.write(f"- **短期壓力位 (近60日高點)**：{high_60:.2f} 元")
            st.write(f"- **月線支撐 (20MA)**：{ma20:.2f} 元")
            st.write(f"- **強支撐位 (近60日低點)**：{low_60:.2f} 元")

        with tab5:
            st.markdown("##### 🤖 AI 綜合技術分析評估")
            is_bullish = "底背離" in selected_div_type
            if is_bullish:
                st.success(f"**預測方向：止跌反彈**\n出現【{selected_div_type}】，股價下探但指標未跟跌，短線具備築底反彈機會。")
            else:
                st.error(f"**預測方向：短線修正**\n出現【{selected_div_type}】，股價創高但指標未跟隨，推升力道減弱，防範回測 20MA ({ma20:.2f}元)。")
