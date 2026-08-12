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
    initial_sidebar_state="collapsed" # 手機上預設摺疊側邊欄
)

st.title("📈 台股背離掃描與多維分析")

# ==========================================
# 工具函數：獲取資料與指標計算
# ==========================================
@st.cache_data(ttl=3600)
def get_stock_data(ticker_symbol):
    """下載股票日 K 線資料"""
    try:
        df = yf.download(ticker_symbol, period="1y", interval="1d", progress=False)
        if df.empty or len(df) < 60:
            return None
        # 處理 yfinance 可能返回 MultiIndex 欄位的狀況
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # 計算技術指標
        # 1. RSI
        df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
        
        # 2. KD (Stochastic Oscillator)
        stoch = ta.momentum.StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3)
        df['K'] = stoch.stoch()
        df['D'] = stoch.stoch_signal()
        
        # 3. MACD
        macd = ta.trend.MACD(close=df['Close'])
        df['MACD_hist'] = macd.macd_diff()
        
        return df.dropna()
    except Exception as e:
        return None

def detect_divergence(df, window=10):
    """
    檢測背離：
    - 底背離 (Bullish Divergence): 價格創新低，但指標未創新低 (看多)
    - 頂背離 (Bearish Divergence): 價格創新高，但指標未創新高 (看空)
    """
    if df is None or len(df) < window * 2:
        return {}

    curr = df.iloc[-1]
    recent = df.iloc[-window:]
    prev = df.iloc[-2*window:-window]

    # 價格高低點
    price_low_recent = recent['Low'].min()
    price_low_prev = prev['Low'].min()
    price_high_recent = recent['High'].max()
    price_high_prev = prev['High'].max()

    results = {}

    # 1. 價量背離
    vol_recent_avg = recent['Volume'].mean()
    vol_prev_avg = prev['Volume'].mean()
    # 價漲量縮 (潛在頂背離/高點無量)
    results['Price_Vol_Bearish'] = (price_high_recent > price_high_prev) and (vol_recent_avg < vol_prev_avg)
    # 價跌量縮或跌破創低量卻未放大 (底背離徵兆)
    results['Price_Vol_Bullish'] = (price_low_recent < price_low_prev) and (vol_recent_avg < vol_prev_avg)

    # 2. RSI 背離
    rsi_low_recent = recent['RSI'].min()
    rsi_low_prev = prev['RSI'].min()
    rsi_high_recent = recent['RSI'].max()
    rsi_high_prev = prev['RSI'].max()
    results['RSI_Bullish'] = (price_low_recent < price_low_prev) and (rsi_low_recent > rsi_low_prev)
    results['RSI_Bearish'] = (price_high_recent > price_high_prev) and (rsi_high_recent < rsi_high_prev)

    # 3. KD 背離
    k_low_recent = recent['K'].min()
    k_low_prev = prev['K'].min()
    k_high_recent = recent['K'].max()
    k_high_prev = prev['K'].max()
    results['KD_Bullish'] = (price_low_recent < price_low_prev) and (k_low_recent > k_low_prev)
    results['KD_Bearish'] = (price_high_recent > price_high_prev) and (k_high_recent < k_high_prev)

    # 4. MACD 背離
    macd_low_recent = recent['MACD_hist'].min()
    macd_low_prev = prev['MACD_hist'].min()
    macd_high_recent = recent['MACD_hist'].max()
    macd_high_prev = prev['MACD_hist'].max()
    results['MACD_Bullish'] = (price_low_recent < price_low_prev) and (macd_low_recent > macd_low_prev)
    results['MACD_Bearish'] = (price_high_recent > price_high_prev) and (macd_high_recent < macd_high_prev)

    return results

# 預設自訂追蹤清單 (熱門台股)
DEFAULT_STOCKS = {
    "2330.TW": "台積電",
    "2317.TW": "鴻海",
    "2454.TW": "聯發科",
    "2382.TW": "廣達",
    "2308.TW": "台達電",
    "2303.TW": "聯電",
    "2881.TW": "富邦金",
    "2882.TW": "國泰金",
    "3231.TW": "緯創",
    "2356.TW": "英業達"
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

st.sidebar.markdown("---")
st.sidebar.caption("💡 提示：手機版可點擊左上角「>」開啟/關閉此選單。")

# 掃描股票庫
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

# 執行篩選
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
st.subheader(f"🎯 符合【{selected_div_type}】的股票")

if not matched_stocks:
    st.info("目前追蹤清單中沒有符合該背離條件的股票。")
else:
    # 下拉選單挑選符合的個股
    selected_stock = st.selectbox(
        "請選擇要查看的股票：",
        options=list(matched_stocks.keys()),
        format_func=lambda x: f"{x.replace('.TW','')} {matched_stocks[x]}"
    )
    
    if selected_stock:
        df = get_stock_data(selected_stock)
        ticker_info = yf.Ticker(selected_stock)
        
        # 1. 最近成交價與漲跌幅
        last_close = df['Close'].iloc[-1]
        prev_close = df['Close'].iloc[-2]
        change = last_close - prev_close
        pct_change = (change / prev_close) * 100
        
        c1, c2, c3 = st.columns(3)
        c1.metric("最新成交價", f"{last_close:.2f} 元", f"{change:+.2f} ({pct_change:+.2f}%)")
        c2.metric("成交量", f"{int(df['Volume'].iloc[-1]/1000):,} 張")
        c3.metric("52週最高/最低", f"{df['High'].max():.1f} / {df['Low'].min():.1f}")

        # 2. K線與背離圖表 (Plotly 互動式)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
        
        # K線
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'
        ), row=1, col=1)
        
        # 成交量
        colors = ['red' if c >= o else 'green' for c, o in zip(df['Close'], df['Open'])]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='成交量', marker_color=colors), row=2, col=1)
        
        fig.update_layout(height=450, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # 3. 分頁顯示：基本面、籌碼面、消息面、技術/支撐壓力、未來一週預測
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 基本面", "🏦 籌碼面", "📰 消息面", "📐 趨勢與支撐壓力", "🔮 未來一週展望"])

        with tab1:
            try:
                info = ticker_info.info
                st.markdown(f"""
                - **產業類別**：{info.get('industry', 'N/A')}
                - **本益比 (PE)**：{info.get('trailingPE', 'N/A')}
                - **股價淨值比 (PB)**：{info.get('priceToBook', 'N/A')}
                - **殖利率**：{info.get('dividendYield', 0)*100:.2f}%
                - **每股盈餘 (EPS)**：{info.get('trailingEps', 'N/A')} 元
                """)
            except:
                st.write("暫無法獲取詳細基本面數據。")

        with tab2:
            # 簡化計算籌碼（以近期成交量與均量比做參考）
            vol_ma5 = df['Volume'].rolling(5).mean().iloc[-1]
            vol_ma20 = df['Volume'].rolling(20).mean().iloc[-1]
            st.write(f"- **5日均量**：{int(vol_ma5/1000):,} 張")
            st.write(f"- **20日均量**：{int(vol_ma20/1000):,} 張")
            if vol_ma5 > vol_ma20:
                st.info("💡 短期成交量放大，資金關注度提升。")
            else:
                st.warning("💡 短期量能萎縮，觀望氣氛較濃。")

        with tab3:
            st.markdown("##### 相關個股新聞摘要")
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
            # 計算簡單支撐與壓力位 (以近 60 日最高最低與均線計算)
            high_60 = df['High'].tail(60).max()
            low_60 = df['Low'].tail(60).min()
            ma20 = df['Close'].rolling(20).mean().iloc[-1]
            ma60 = df['Close'].rolling(60).mean().iloc[-1]

            st.write(f"- **當前趨勢**：{'多頭排列' if last_close > ma20 > ma60 else '空頭/震盪整理'}")
            st.write(f"- **短期壓力位 (近60日高點)**：{high_60:.2f} 元")
            st.write(f"- **月線支撐 (20MA)**：{ma20:.2f} 元")
            st.write(f"- **強支撐位 (近60日低點)**：{low_60:.2f} 元")

        with tab5:
            # 未來一週發展預測綜合評估
            st.markdown("##### 🤖 AI 綜合技術分析評估")
            is_bullish = "底背離" in selected_div_type
            
            if is_bullish:
                st.success(f"""
                **預測方向：止跌反彈 / 短線看多**
                - **分析依據**：該股近期出現【{selected_div_type}】，顯示雖然價格拉回或下探，但下方動能（或指標）已有率先築底回升跡象。
                - **未來一週觀測點**：
                    1. 關注是否能帶量突破月線 ({ma20:.2f} 元) 壓力。
                    2. 若跌破近 60 日低點 ({low_60:.2f} 元) 則背離結構失效，需嚴格執行停損。
                """)
            else:
                st.error(f"""
                **預測方向：高點受阻 / 短線修正風險**
                - **分析依據**：該股近期出現【{selected_div_type}】，股價雖在相對高位，但指標動能或成交量未能同步創高，買盤推升力道衰竭。
                - **未來一週觀測點**：
                    1. 短線宜防範回測 20MA ({ma20:.2f} 元) 支撐。
                    2. 若突破高點 ({high_60:.2f} 元) 並補量，可化解背離疑慮，否則建議逢高適度減碼。
                """)
