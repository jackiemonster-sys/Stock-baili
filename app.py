import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta

# ==========================================
# 頁面配置
# ==========================================
st.set_page_config(
    page_title="台股背離掃描器",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 台股背離多重掃描與分析")

# ==========================================
# 資料獲取與技術指標計算
# ==========================================
@st.cache_data(ttl=3600)
def get_stock_data(ticker_symbol):
    try:
        df = yf.download(ticker_symbol, period="1y", interval="1d", progress=False)
        if df.empty or len(df) < 60:
            return None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # 1. RSI (14)
        df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
        
        # 2. KD (14, 3, 3)
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

def detect_all_divergences(df, window=20):
    """ 判斷近期的各類背離情況 """
    if df is None or len(df) < window + 5:
        return {}

    recent = df.iloc[-5:]         # 最近 5 天
    past = df.iloc[-window:-5]   # 前 15 天

    results = {}

    # RSI
    results['RSI 底背離'] = (recent['Low'].min() <= past['Low'].min()) and (recent['RSI'].min() > past['RSI'].min())
    results['RSI 頂背離'] = (recent['High'].max() >= past['High'].max()) and (recent['RSI'].max() < past['RSI'].max())

    # KD
    results['KD 底背離'] = (recent['Low'].min() <= past['Low'].min()) and (recent['K'].min() > past['K'].min())
    results['KD 頂背離'] = (recent['High'].max() >= past['High'].max()) and (recent['K'].max() < past['K'].max())

    # MACD
    results['MACD 底背離'] = (recent['Low'].min() <= past['Low'].min()) and (recent['MACD_hist'].min() > past['MACD_hist'].min())
    results['MACD 頂背離'] = (recent['High'].max() >= past['High'].max()) and (recent['MACD_hist'].max() < past['MACD_hist'].max())

    # 價量背離
    results['價量底背離 (價跌量縮)'] = (recent['Low'].min() <= past['Low'].min()) and (recent['Vol_MA5'].iloc[-1] < past['Vol_MA5'].mean())
    results['價量頂背離 (價漲量縮)'] = (recent['High'].max() >= past['High'].max()) and (recent['Vol_MA5'].iloc[-1] < past['Vol_MA5'].mean())

    return results

# 擴充預設掃描的熱門股票池
DEFAULT_STOCKS = {
    "2330.TW": "台積電", "2317.TW": "鴻海", "2454.TW": "聯發科", "2382.TW": "廣達",
    "2308.TW": "台達電", "2303.TW": "聯電", "2881.TW": "富邦金", "2882.TW": "國泰金",
    "3231.TW": "緯創", "2356.TW": "英業達", "2603.TW": "長榮", "2609.TW": "陽明",
    "2002.TW": "中鋼", "1301.TW": "台塑", "2379.TW": "瑞昱", "3034.TW": "聯詠",
    "2376.TW": "技嘉", "3017.TW": "奇鋐", "2345.TW": "智邦", "3661.TW": "世芯-KY"
}

# ==========================================
# 側邊欄：背離條件獨立/共同勾選
# ==========================================
st.sidebar.header("🔍 背離條件選擇")
st.sidebar.write("請選擇要篩選的背離類型（可多選）：")

# 讓使用者自由勾選要查看的背離指標
selected_conditions = []

col_a, col_b = st.sidebar.columns(2)

with col_a:
    st.markdown("**看多 (底背離)**")
    if st.checkbox("RSI 底背離", value=True): selected_conditions.append("RSI 底背離")
    if st.checkbox("KD 底背離"): selected_conditions.append("KD 底背離")
    if st.checkbox("MACD 底背離"): selected_conditions.append("MACD 底背離")
    if st.checkbox("價量底背離"): selected_conditions.append("價量底背離 (價跌量縮)")

with col_b:
    st.markdown("**看空 (頂背離)**")
    if st.checkbox("RSI 頂背離"): selected_conditions.append("RSI 頂背離")
    if st.checkbox("KD 頂背離"): selected_conditions.append("KD 頂背離")
    if st.checkbox("MACD 頂背離"): selected_conditions.append("MACD 頂背離")
    if st.checkbox("價量頂背離"): selected_conditions.append("價量頂背離 (價漲量縮)")

match_mode = st.sidebar.radio("多選時的篩選模式：", ["滿足「任一」勾選條件 (OR)", "必須「同時滿足」所有勾選條件 (AND)"])

# ==========================================
# 掃描邏輯與結果過濾
# ==========================================
matched_stocks = {}
stock_div_details = {}

if selected_conditions:
    with st.spinner("正在掃描市場資料中..."):
        for symbol, name in DEFAULT_STOCKS.items():
            df = get_stock_data(symbol)
            if df is not None:
                divs = detect_all_divergences(df)
                
                # 紀錄該股票有哪些指標發生背離
                triggered = [cond for cond in selected_conditions if divs.get(cond, False)]
                
                if match_mode == "滿足「任一」勾選條件 (OR)" and len(triggered) > 0:
                    matched_stocks[symbol] = name
                    stock_div_details[symbol] = triggered
                elif match_mode == "必須「同時滿足」所有勾選條件 (AND)" and len(triggered) == len(selected_conditions):
                    matched_stocks[symbol] = name
                    stock_div_details[symbol] = triggered

# ==========================================
# 主畫面顯示
# ==========================================
if not selected_conditions:
    st.info("👈 請在左側/上方選單中至少勾選一種背離條件進行掃描。")
elif not matched_stocks:
    st.warning("⚠️ 目前追蹤清單中沒有符合條件的股票。建議切換為「滿足任一條件 (OR)」或擴大觀察範圍。")
else:
    st.subheader(f"🎯 符合條件的股票 (共 {len(matched_stocks)} 檔)")
    
    selected_stock = st.selectbox(
        "請選擇要查看詳細分析的股票：",
        options=list(matched_stocks.keys()),
        format_func=lambda x: f"{x.replace('.TW','')} {matched_stocks[x]} (觸發: {', '.join(stock_div_details[x])})"
    )
    
    if selected_stock:
        df = get_stock_data(selected_stock)
        ticker_info = yf.Ticker(selected_stock)
        
        # 價格與漲跌
        last_close = df['Close'].iloc[-1]
        prev_close = df['Close'].iloc[-2]
        change = last_close - prev_close
        pct_change = (change / prev_close) * 100
        
        c1, c2, c3 = st.columns(3)
        c1.metric("最新成交價", f"{last_close:.2f} 元", f"{change:+.2f} ({pct_change:+.2f}%)")
        c2.metric("成交量", f"{int(df['Volume'].iloc[-1]/1000):,} 張")
        c3.metric("52週高 / 低", f"{df['High'].max():.1f} / {df['Low'].min():.1f}")

        # 顯示觸發背離標籤
        st.write("🚩 **這支股票目前觸發的背離類型：**")
        st.success(" / ".join(stock_div_details[selected_stock]))

        # 主圖表 (K線 + 指標)
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.25, 0.25])
        
        # K線
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'), row=1, col=1)
        
        # 成交量
        colors = ['red' if c >= o else 'green' for c, o in zip(df['Close'], df['Open'])]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='成交量', marker_color=colors), row=2, col=1)
        
        # RSI / KD
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='orange')), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['K'], name='K', line=dict(color='blue')), row=3, col=1)
        
        fig.update_layout(height=500, margin=dict(l=10, r=10, t=10, b=10), showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

        # 多維度分析頁籤
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
                st.write("暫無詳細基本面資料。")

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
                    st.write("目前無最新消息。")
            except:
                st.write("無法讀取新聞。")

        with tab4:
            high_60 = df['High'].tail(60).max()
            low_60 = df['Low'].tail(60).min()
            ma20 = df['Close'].rolling(20).mean().iloc[-1]
            st.write(f"- **壓力位 (近60日高點)**：{high_60:.2f} 元")
            st.write(f"- **月線 (20MA)**：{ma20:.2f} 元")
            st.write(f"- **支撐位 (近60日低點)**：{low_60:.2f} 元")

        with tab5:
            st.markdown("##### 🤖 未來一週走勢評估")
            has_bull = any("底背離" in d for d in stock_div_details[selected_stock])
            if has_bull:
                st.success(f"**短線看多 / 止跌訊號**\n觸發指標：{', '.join(stock_div_details[selected_stock])}。價格探底但指標未跟隨下探，有機會迎來反彈，目標看至 20MA ({ma20:.2f} 元)。")
            else:
                st.error(f"**短線看空 / 修正風險**\n觸發指標：{', '.join(stock_div_details[selected_stock])}。價格維持高位但指標強度遞減，注意拉回風險，初步支撐看 {ma20:.2f} 元。")
