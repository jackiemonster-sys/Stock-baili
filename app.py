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
    page_title="台股背離追蹤器 (半年區間版)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 台股背離多重掃描與半年歷史區間分析")

# ==========================================
# 資料獲取與技術指標計算
# ==========================================
@st.cache_data(ttl=3600)
def get_stock_data(ticker_symbol):
    try:
        df = yf.download(ticker_symbol, period="1y", interval="1d", progress=False)
        if df.empty or len(df) < 120:
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

def find_divergence_in_half_year(df, days_limit=120):
    """
    掃描過去半年 (約 120 個交易日) 內的背離事件
    回傳範例: { 'RSI 底背離': [{'start_date': '...', 'end_date': '...', 'start_price': 100, 'end_price': 95}], ... }
    """
    if df is None or len(df) < days_limit:
        return {}

    sub_df = df.tail(days_limit).copy()
    results = {}
    indicators = ['RSI', 'K', 'MACD_hist', 'Vol_MA5']
    
    div_names = {
        'RSI': ('RSI 底背離', 'RSI 頂背離'),
        'K': ('KD 底背離', 'KD 頂背離'),
        'MACD_hist': ('MACD 底背離', 'MACD 頂背離'),
        'Vol_MA5': ('價量底背離 (價跌量縮)', '價量頂背離 (價漲量縮)')
    }

    # 以每 15 個交易日為一滑動視窗，尋找局部高低點
    window = 15
    for ind in indicators:
        bull_key, bear_key = div_names[ind]
        results[bull_key] = []
        results[bear_key] = []

        # 遍歷歷史區間
        for i in range(len(sub_df) - window * 2):
            prev_chunk = sub_df.iloc[i : i + window]
            curr_chunk = sub_df.iloc[i + window : i + window * 2]

            # 價格高低點
            p_low1, p_low2 = prev_chunk['Low'].min(), curr_chunk['Low'].min()
            p_high1, p_high2 = prev_chunk['High'].max(), curr_chunk['High'].max()

            # 指標高低點
            i_low1, i_low2 = prev_chunk[ind].min(), curr_chunk[ind].min()
            i_high1, i_high2 = prev_chunk[ind].max(), curr_chunk[ind].max()

            # 1. 底背離：價格創新低，但指標/量能未創新低
            if p_low2 < p_low1 and i_low2 > i_low1:
                start_dt = prev_chunk['Low'].idxmin().strftime('%Y-%m-%d')
                end_dt = curr_chunk['Low'].idxmin().strftime('%Y-%m-%d')
                
                # 避免重複記錄距離過近的區間
                if not any(r['end_date'] == end_dt for r in results[bull_key]):
                    results[bull_key].append({
                        'start_date': start_dt,
                        'end_date': end_dt,
                        'start_price': float(p_low1),
                        'end_price': float(p_low2)
                    })

            # 2. 頂背離：價格創新高，但指標/量能未創新高
            if p_high2 > p_high1 and i_high2 < i_high1:
                start_dt = prev_chunk['High'].idxmax().strftime('%Y-%m-%d')
                end_dt = curr_chunk['High'].idxmax().strftime('%Y-%m-%d')
                
                if not any(r['end_date'] == end_dt for r in results[bear_key]):
                    results[bear_key].append({
                        'start_date': start_dt,
                        'end_date': end_dt,
                        'start_price': float(p_high1),
                        'end_price': float(p_high2)
                    })

    return results

# 熱門個股觀察清單
DEFAULT_STOCKS = {
    "2330.TW": "台積電", "2317.TW": "鴻海", "2454.TW": "聯發科", "2382.TW": "廣達",
    "2308.TW": "台達電", "2303.TW": "聯電", "2881.TW": "富邦金", "2882.TW": "國泰金",
    "3231.TW": "緯創", "2356.TW": "英業達", "2603.TW": "長榮", "2609.TW": "陽明",
    "2002.TW": "中鋼", "1301.TW": "台塑", "2379.TW": "瑞昱", "3034.TW": "聯詠",
    "2376.TW": "技嘉", "3017.TW": "奇鋐", "2345.TW": "智邦", "3661.TW": "世芯-KY"
}

# ==========================================
# 側邊欄：篩選控制項
# ==========================================
st.sidebar.header("🔍 背離條件選擇 (半年內)")

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

match_mode = st.sidebar.radio("篩選邏輯：", ["滿足「任一」勾選條件 (OR)", "必須「同時滿足」所有勾選條件 (AND)"])

# ==========================================
# 執行掃描與資料過濾
# ==========================================
matched_stocks = {}
stock_div_details = {}

if selected_conditions:
    with st.spinner("正在掃描近半年數據..."):
        for symbol, name in DEFAULT_STOCKS.items():
            df = get_stock_data(symbol)
            if df is not None:
                divs = find_divergence_in_half_year(df)
                
                # 收集這檔股票半年內觸發過哪些條件
                triggered_info = {}
                for cond in selected_conditions:
                    if len(divs.get(cond, [])) > 0:
                        triggered_info[cond] = divs[cond]

                if match_mode == "滿足「任一」勾選條件 (OR)" and len(triggered_info) > 0:
                    matched_stocks[symbol] = name
                    stock_div_details[symbol] = triggered_info
                elif match_mode == "必須「同時滿足」所有勾選條件 (AND)" and len(triggered_info) == len(selected_conditions):
                    matched_stocks[symbol] = name
                    stock_div_details[symbol] = triggered_info

# ==========================================
# 主畫面內容
# ==========================================
if not selected_conditions:
    st.info("👈 請在左側選單中至少勾選一種背離條件進行掃描。")
elif not matched_stocks:
    st.warning("⚠️ 近半年無符合條件的股票，可嘗試勾選「滿足任一條件 (OR)」或選取其他指標。")
else:
    st.subheader(f"🎯 半年內出現過目標背離的股票 (共 {len(matched_stocks)} 檔)")
    
    selected_stock = st.selectbox(
        "選擇股票以查看詳情與背離發生的日期區間：",
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
        c3.metric("52週高 / 低", f"{df['High'].max():.1f} / {df['Low'].min():.1f}")

        # 顯示近半年發生的背離區間詳細列表
        st.markdown("### 📌 近半年發生的背離時間區間")
        details = stock_div_details[selected_stock]
        
        for div_type, occurrences in details.items():
            st.markdown(f"**🔹 {div_type} （共出現 {len(occurrences)} 次）**")
            for idx, occ in enumerate(occurrences, 1):
                st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;• **區間 {idx}**：`{occ['start_date']}` ({occ['start_price']:.1f}元) ➔ `{occ['end_date']}` ({occ['end_price']:.1f}元)")

        # 圖表製作（包含歷史背離區間的垂直著色標示）
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.25, 0.25])
        
        # K 線與成交量
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'), row=1, col=1)
        colors = ['red' if c >= o else 'green' for c, o in zip(df['Close'], df['Open'])]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='成交量', marker_color=colors), row=2, col=1)
        
        # 指標
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='orange')), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['K'], name='K', line=dict(color='blue')), row=3, col=1)

        # 在 K 線圖上將背離區間標示出來 (透明半透明區塊)
        for div_type, occurrences in details.items():
            color = "rgba(0, 255, 0, 0.15)" if "底背離" in div_type else "rgba(255, 0, 0, 0.15)"
            for occ in occurrences:
                fig.add_vrect(
                    x0=occ['start_date'], x1=occ['end_date'],
                    fillcolor=color, opacity=0.5, layer="below", line_width=0,
                    row=1, col=1
                )

        fig.update_layout(height=500, margin=dict(l=10, r=10, t=10, b=10), showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

        # 頁籤說明
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
                st.write("暫無基本面資料。")

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
                    st.write("目前無新聞。")
            except:
                st.write("無法讀取新聞。")

        with tab4:
            high_60 = df['High'].tail(60).max()
            low_60 = df['Low'].tail(60).min()
            ma20 = df['Close'].rolling(20).mean().iloc[-1]
            st.write(f"- **60日壓力位**：{high_60:.2f} 元")
            st.write(f"- **月線 (20MA)**：{ma20:.2f} 元")
            st.write(f"- **60日支撐位**：{low_60:.2f} 元")

        with tab5:
            st.markdown("##### 🤖 未來一週走勢評估")
            has_bull = any("底背離" in d for d in details.keys())
            if has_bull:
                st.success(f"**歷史多頭背離效應**\n該股近半年出現過多頭背離，顯示下方買盤有支撐力道。短線可觀察 20MA ({ma20:.2f} 元) 是否站穩。")
            else:
                st.error(f"**歷史空頭背離效應**\n該股近半年出現過頂背離，在高位追高風險較大，防範拉回修正。")
