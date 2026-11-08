"""
退貨止血儀表板 (Streamlit)

依據：
- 上課用prompt/04報告呈現/退貨止血儀表板設計藍圖.md   （版面、組件、篩選器）
- 上課用prompt/04報告呈現/退貨止血儀表板_wireframe.html （F 型線框圖）
- reports/audit_returns_report.md                      （稽核校正後的口徑）
- AGENTS.md 資料集契約                                  （讀檔、去重、欄位）

執行方式（從任何目錄都可以）：
    streamlit run app.py

口徑說明（與稽核後報告一致）：
- 兩檔合併、去重（2010-12 重疊期）、剔除 UnitPrice <= 0 皆已在瘦身腳本完成
- 退貨列 = InvoiceNo 以 C 開頭，或 Quantity < 0
- 非實體項目 = StockCode 為 M / AMAZONFEE / BANK CHARGES / POST / D
- 客戶層級「必須包含只有退貨、沒有購買紀錄的客戶」（稽核發現的 bug 修正）
"""

import json
from math import erfc, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# 常數
# ---------------------------------------------------------------------------
# 部署版：資料已由 scripts/slim_online_retail_for_dashboard.py 預先去重、剔除 UnitPrice<=0，
# 並只保留 8 個欄位，存成 parquet。app.py 與 data/ 在同一個 repo 內。
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_FILE = DATA_DIR / "online_retail_slim.parquet"
META_FILE = DATA_DIR / "meta.json"

# 這五個代碼是平台費、人工調帳、郵資、折扣，不是實體商品
NON_PHYSICAL_CODES = ["M", "AMAZONFEE", "BANK CHARGES", "POST", "D"]

UK = "United Kingdom"
BIG_INVOICE_ALERT = 5_000       # 大單預警門檻（藍圖：單筆 > £5,000）
BIG_INVOICE_CRITICAL = 50_000   # 超大額全退門檻（藍圖：單張 > £50,000）
RETURN_RATE_WARN = 0.30         # 客戶退貨率警戒線（藍圖：30%）

# 莫蘭迪靜謐洞察色票
SLATE = "#5B7083"   # 霧霾岩石藍：基準
ROSE = "#C06C78"    # 乾燥玫瑰紅：警示
ROSE_DARK = "#A34E5A"
SAND = "#D4A373"    # 暖沙駝棕：次級關注
SAGE = "#7D9D86"    # 鼠尾草灰綠：健康
GREIGE = "#A5A59D"  # 亞麻暖灰：長尾
BG = "#F5F2EB"      # 羊皮紙米白
CARD = "#FAF8F5"
BORDER = "#DFDBD2"
TEXT = "#3D403D"


# ---------------------------------------------------------------------------
# 數據加載與清洗（有快取，只在第一次執行）
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="載入資料中…")
def load_data():
    """讀精簡 parquet → 標記退貨與金額。去重與價格過濾已在瘦身腳本完成。"""
    df = pd.read_parquet(DATA_FILE)
    meta_raw = json.loads(META_FILE.read_text(encoding="utf-8"))

    # 標記欄位
    df["IsReturn"] = df["InvoiceNo"].str.startswith("C") | (df["Quantity"] < 0)
    df["IsNonPhysical"] = df["StockCode"].isin(NON_PHYSICAL_CODES)
    # 正向銷售額：非退貨且數量為正
    df["SalesAmt"] = np.where(
        ~df["IsReturn"] & (df["Quantity"] > 0), df["Quantity"] * df["UnitPrice"], 0.0
    )
    # 退貨損失：退貨列的 |數量| × 單價
    df["ReturnLoss"] = np.where(df["IsReturn"], df["Quantity"].abs() * df["UnitPrice"], 0.0)
    df["Country"] = df["Country"].astype("category")

    meta = {
        "n_raw": meta_raw["n_raw"],
        "n_dedup": meta_raw["n_dedup"],
        "n_price": meta_raw["n_price"],
        "n_clean": len(df),
        "date_min": df["InvoiceDate"].min().date(),
        "date_max": df["InvoiceDate"].max().date(),
    }
    return df, meta


# ---------------------------------------------------------------------------
# 篩選器
# ---------------------------------------------------------------------------
def apply_base_filters(df, start, end, cust_attr):
    """日期 + 客群歸戶篩選（不含市場、不含帳務口徑）。"""
    mask = (df["InvoiceDate"] >= pd.Timestamp(start)) & (
        df["InvoiceDate"] < pd.Timestamp(end) + pd.Timedelta(days=1)
    )
    if cust_attr == "具名客戶":
        mask &= df["CustomerID"].notna()
    elif cust_attr == "未歸戶過客":
        mask &= df["CustomerID"].isna()
    return df[mask]


def apply_market(df, market):
    """市場篩選：全部 / 英國境內 / 海外跨境 / 指定國家。"""
    if market == "英國境內":
        return df[df["Country"] == UK]
    if market == "海外跨境":
        return df[df["Country"] != UK]
    if market != "全部":
        return df[df["Country"] == market]
    return df


def apply_scope(df, scope):
    """帳務口徑：僅實體商品 = 剔除五個非實體代碼。"""
    if scope == "僅實體商品":
        return df[~df["IsNonPhysical"]]
    return df


# ---------------------------------------------------------------------------
# 計算函數
# ---------------------------------------------------------------------------
def product_table(scoped):
    """各商品銷售額、退貨損失、累計佔比與 ABC 分級（帕雷托）。"""
    prod = scoped.groupby("StockCode").agg(
        Description=("Description", "first"),
        SalesAmt=("SalesAmt", "sum"),
        ReturnLoss=("ReturnLoss", "sum"),
    )
    prod = prod.sort_values("ReturnLoss", ascending=False)
    total = prod["ReturnLoss"].sum()
    prod["CumPct"] = prod["ReturnLoss"].cumsum() / total if total > 0 else 0.0
    # 累計 <=80% 為 A，<=95% 為 B，其餘為 C
    prod["Class"] = np.where(prod["CumPct"] <= 0.80, "A", np.where(prod["CumPct"] <= 0.95, "B", "C"))
    prod["ReturnRate"] = prod["ReturnLoss"] / prod["SalesAmt"].replace(0, np.nan)
    prod["Description"] = prod["Description"].fillna("")
    return prod


def customer_table(scoped):
    """客戶層級彙總。包含「只有退貨、沒有購買」的客戶（稽核修正）。"""
    cust = scoped[scoped["CustomerID"].notna()]
    cols = ["CustomerID", "GrossSpend", "ReturnLoss", "ReturnInvoices", "NetSpend", "ReturnRate", "Segment"]
    if cust.empty:
        return pd.DataFrame(columns=cols)

    # 直接對全部列 groupby，所以只退不買的客戶也會留下來
    table = cust.groupby("CustomerID").agg(
        GrossSpend=("SalesAmt", "sum"), ReturnLoss=("ReturnLoss", "sum")
    )
    return_invoices = cust[cust["IsReturn"]].groupby("CustomerID")["InvoiceNo"].nunique()
    table["ReturnInvoices"] = return_invoices.reindex(table.index).fillna(0).astype(int)
    table["NetSpend"] = table["GrossSpend"] - table["ReturnLoss"]
    table["ReturnRate"] = np.where(
        table["GrossSpend"] > 0, table["ReturnLoss"] / table["GrossSpend"], np.nan
    )
    table["Segment"] = np.select(
        [
            (table["NetSpend"] < 0) & (table["GrossSpend"] == 0),
            (table["NetSpend"] < 0) & (table["GrossSpend"] > 0),
            table["ReturnRate"] >= RETURN_RATE_WARN,
        ],
        ["負貢獻：純退無買", "負貢獻：買後退>買", "批發調節客（退貨率≥30%）"],
        default="黃金VIP／低退貨",
    )
    table = table.reset_index()
    table["CustomerID"] = table["CustomerID"].astype(int)
    return table[cols]


def country_table(scoped):
    """各國訂單退貨率（含退貨列的發票數 / 總發票數）與金額退貨率。"""
    invoices = scoped.groupby(["InvoiceNo", "Country"], observed=True).agg(
        HasReturn=("IsReturn", "any")
    ).reset_index()
    by_country = invoices.groupby("Country", observed=True).agg(
        Invoices=("InvoiceNo", "count"), ReturnInvoices=("HasReturn", "sum")
    )
    amounts = scoped.groupby("Country", observed=True).agg(
        SalesAmt=("SalesAmt", "sum"), ReturnLoss=("ReturnLoss", "sum")
    )
    table = by_country.join(amounts)
    table = table[table["Invoices"] > 0]
    table["InvoiceReturnRate"] = table["ReturnInvoices"] / table["Invoices"]
    table["AmountReturnRate"] = table["ReturnLoss"] / table["SalesAmt"].replace(0, np.nan)
    return table


def return_invoice_table(scoped):
    """每張退貨發票的退貨總額（用來找單張大額全退）。"""
    returns = scoped[scoped["IsReturn"]]
    if returns.empty:
        return pd.DataFrame(columns=["InvoiceNo", "ReturnLoss", "CustomerID", "Country", "InvoiceDate"])
    table = returns.groupby("InvoiceNo").agg(
        ReturnLoss=("ReturnLoss", "sum"),
        CustomerID=("CustomerID", "first"),
        Country=("Country", "first"),
        InvoiceDate=("InvoiceDate", "min"),
    )
    return table.reset_index().sort_values("ReturnLoss", ascending=False)


def two_proportion_z(x1, n1, x2, n2):
    """雙樣本比例 Z 檢定（x2/n2 對 x1/n1），回傳 (z, 雙尾 p)。"""
    p1, p2 = x1 / n1, x2 / n2
    pooled = (x1 + x2) / (n1 + n2)
    se = sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se
    return z, erfc(abs(z) / sqrt(2))


def money(value):
    return f"£{value:,.2f}"


def short_money(value):
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value >= 1_000_000:
        return f"{sign}£{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{sign}£{value / 1_000:.1f}k"
    return f"{sign}£{value:.0f}"


# ---------------------------------------------------------------------------
# 圖表與組件
# ---------------------------------------------------------------------------
def style_fig(fig, height):
    fig.update_layout(
        height=height,
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font=dict(color=TEXT),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=-0.25),
    )
    return fig


def create_kpi_cards(base, scoped, scope, cust_table, invoice_table):
    """③ 四張 KPI 卡片。"""
    total_return_scoped = scoped["ReturnLoss"].sum()
    sales_all = base["SalesAmt"].sum()
    non_physical_return = base.loc[base["IsNonPhysical"], "ReturnLoss"].sum()
    total_return_all = base["ReturnLoss"].sum()

    # 北極星指標：口徑內退貨 / 全部正向銷售（與稽核後報告相同）
    rate = total_return_scoped / sales_all if sales_all > 0 else 0
    strict_sales = scoped["SalesAmt"].sum()
    strict_rate = total_return_scoped / strict_sales if strict_sales > 0 else 0

    big = invoice_table[invoice_table["ReturnLoss"] > BIG_INVOICE_CRITICAL]
    negative = cust_table[cust_table["NetSpend"] < 0]

    c1, c2, c3, c4 = st.columns(4)
    label = "🎯 實體商品退貨率" if scope == "僅實體商品" else "🎯 帳面總退貨率"
    c1.metric(
        label,
        f"{rate:.2%}",
        delta=money(total_return_scoped),
        delta_color="off",
        help=f"分母為口徑外全部正向銷售額（與稽核後報告一致）。"
        f"若分母改為口徑內銷售額，則為 {strict_rate:.2%}。",
    )
    c2.metric(
        "🧾 非實體沖銷",
        money(non_physical_return),
        delta=f"佔總退貨 {non_physical_return / total_return_all:.2%}" if total_return_all > 0 else "-",
        delta_color="off",
        help="Manual / AMAZONFEE / BANK CHARGES / POST / D 五個代碼的退貨列合計。",
    )
    c3.metric(
        f"🚨 單張 > £{BIG_INVOICE_CRITICAL:,} 全退發票",
        f"{len(big)} 張",
        delta=f"合計 {money(big['ReturnLoss'].sum())}",
        delta_color="off",
    )
    c4.metric(
        "👤 淨貢獻為負客戶",
        f"{len(negative):,} 人",
        delta=(
            f"淨虧 {money(negative['NetSpend'].sum())}（佔總退貨 {abs(negative['NetSpend'].sum()) / total_return_all:.2%}）"
            if total_return_all > 0 and len(negative) > 0
            else "-"
        ),
        delta_color="off",
        help="客戶層級一律用全量帳務計算（含只退貨、沒有購買紀錄的客戶），不受「帳務口徑」影響。",
    )


def create_pareto_chart(prod):
    """③-1 商品退貨帕雷托圖：逐款排名的柱子 + 累計退貨 % 折線。

    不把商品合併成「其餘」大柱——合併柱用同一個刻度，會比單一商品高十倍以上，
    讓人誤以為「其餘」才是主因。這裡每根柱子都是一款商品，折線才是看集中度的主角。
    """
    total = prod["ReturnLoss"].sum()
    ranked = prod[prod["ReturnLoss"] > 0].copy()  # 沒有退貨損失的商品對圖沒有貢獻，只保留有退貨的
    if total <= 0 or ranked.empty:
        st.info("目前篩選條件下沒有退貨資料。")
        return

    ranked["Rank"] = np.arange(1, len(ranked) + 1)
    ranked["ClassName"] = ranked["Class"].map(
        {"A": "A類（累計前80%）", "B": "B類（80~95%）", "C": "C類（後5%）"}
    )
    n_a = int((ranked["Class"] == "A").sum())

    # 預設讓畫面剛好涵蓋 A 類界線再多一點；使用者可自行拉大／縮小
    default_n = int(min(len(ranked), max(30, n_a * 1.5)))
    show_n = st.slider("顯示前 N 款商品（依退貨損失由大到小）", 10, len(ranked), default_n, key="pareto_top_n")
    shown = ranked.head(show_n)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    bar_fig = px.bar(
        shown,
        x="Rank",
        y="ReturnLoss",
        color="ClassName",
        color_discrete_map={"A類（累計前80%）": ROSE, "B類（80~95%）": SAND, "C類（後5%）": GREIGE},
        hover_data={"Rank": True, "Description": True, "ReturnLoss": ":,.2f", "CumPct": ":.1%"},
    )
    for trace in bar_fig.data:
        trace.text = None
        fig.add_trace(trace, secondary_y=False)
    # 柱子很多時，標籤用商品代碼會擠成一團，所以 X 軸用「排名」，代碼放在懸停提示裡
    fig.add_trace(
        go.Scatter(x=shown["Rank"], y=shown["CumPct"], mode="lines", name="累計退貨 %",
                   line=dict(color=SLATE, width=2.5), customdata=shown[["Description"]],
                   hovertemplate="排名 %{x}<br>累計 %{y:.1%}<extra></extra>"),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(x=[1, show_n], y=[0.8, 0.8], mode="lines", name="80% 警戒線",
                   line=dict(color=ROSE, dash="dash")),
        secondary_y=True,
    )
    if n_a <= show_n:
        fig.add_vline(x=n_a + 0.5, line_dash="dot", line_color=ROSE,
                      annotation_text=f"A 類界線：前 {n_a:,} 款", annotation_position="top right")
    fig.update_yaxes(title_text="單款退貨損失 (£)", tickprefix="£", secondary_y=False)
    fig.update_yaxes(title_text="累計退貨佔比", tickformat=".0%", range=[0, 1.05], secondary_y=True)
    fig.update_xaxes(title_text="商品排名", type="linear")
    style_fig(fig, 430)
    st.plotly_chart(fig, width="stretch")

    summary = []
    for cls in ["A", "B", "C"]:
        group = prod[prod["Class"] == cls]
        summary.append(f"**{cls} 類** {len(group):,} 款（{len(group) / len(prod):.2%}）= {money(group['ReturnLoss'].sum())}（{group['ReturnLoss'].sum() / total:.2%}）")
    st.caption("　｜　".join(summary) + f"　｜　共 {len(prod):,} 款商品（其中 {len(ranked):,} 款有退貨），退貨總額 {money(total)}")


def create_product_drilldown(scoped, prod):
    """帕雷托下方的商品下鑽：看某商品的退貨國別與客戶。"""
    top_codes = prod.head(20)
    options = [f"{code}｜{row['Description'][:28]}" for code, row in top_codes.iterrows()]
    with st.expander("🔎 商品下鑽：選一個高退貨商品，看它的國別與客戶分佈"):
        if not options:
            st.write("沒有可下鑽的商品。")
            return
        choice = st.selectbox("商品（依退貨損失排序前 20）", options)
        code = choice.split("｜")[0]
        item_returns = scoped[(scoped["StockCode"] == code) & scoped["IsReturn"]]
        left, right = st.columns(2)
        by_country = (
            item_returns.groupby("Country", observed=True)["ReturnLoss"].sum().sort_values(ascending=False).head(8).reset_index()
        )
        fig = px.bar(by_country, x="ReturnLoss", y="Country", orientation="h", color_discrete_sequence=[SLATE],
                     title="退貨損失國別分佈")
        fig.update_layout(yaxis=dict(autorange="reversed"))
        left.plotly_chart(style_fig(fig, 280), width="stretch")
        by_cust = (
            item_returns.dropna(subset=["CustomerID"])
            .groupby("CustomerID")["ReturnLoss"].sum().sort_values(ascending=False).head(8).reset_index()
        )
        by_cust["CustomerID"] = by_cust["CustomerID"].astype(int)
        by_cust["ReturnLoss"] = by_cust["ReturnLoss"].map(money)
        right.write("退貨損失前 8 名客戶（僅具名客戶）")
        right.dataframe(by_cust, hide_index=True, width="stretch")


def create_customer_quadrant(cust_table):
    """③-2 客戶淨貢獻 vs 退貨率四象限散佈圖。"""
    if cust_table.empty:
        st.info("目前篩選條件下沒有具名客戶資料。")
        return
    data = cust_table.copy()
    # 只退不買的客戶退貨率無法定義，圖上統一放在 300% 位置；其餘超過 300% 也截在 300%
    data["RatePlot"] = data["ReturnRate"].fillna(3.0).clip(upper=3.0)
    # 淨貢獻跨度極大（負數到 +£57 萬），用「帶正負號的 log」壓縮；懸停仍顯示真實金額
    data["NetPlot"] = np.sign(data["NetSpend"]) * np.log10(1 + data["NetSpend"].abs())
    data["淨貢獻"] = data["NetSpend"].map(money)
    data["購買額"] = data["GrossSpend"].map(money)
    data["退貨額"] = data["ReturnLoss"].map(money)

    fig = px.scatter(
        data,
        x="RatePlot",
        y="NetPlot",
        color="Segment",
        color_discrete_map={
            "黃金VIP／低退貨": SAGE,
            "批發調節客（退貨率≥30%）": SAND,
            "負貢獻：買後退>買": ROSE,
            "負貢獻：純退無買": ROSE_DARK,
        },
        hover_data={"CustomerID": True, "購買額": True, "退貨額": True, "淨貢獻": True, "RatePlot": False, "NetPlot": False},
        opacity=0.75,
    )
    fig.add_vline(x=RETURN_RATE_WARN, line_dash="dash", line_color=SAND)
    fig.add_hline(y=0, line_dash="dash", line_color=ROSE)
    tick_values = [-100_000, -10_000, -1_000, 0, 1_000, 10_000, 100_000, 500_000]
    fig.update_yaxes(
        title="客戶淨貢獻（對數刻度，保留正負）",
        tickvals=[np.sign(v) * np.log10(1 + abs(v)) for v in tick_values],
        ticktext=[short_money(v) for v in tick_values],
    )
    fig.update_xaxes(title="個人金額退貨率（≥300% 與只退不買者置於 300%）", tickformat=".0%", range=[-0.05, 3.1])
    style_fig(fig, 430)
    st.plotly_chart(fig, width="stretch")
    st.caption(f"共 {len(data):,} 位具名客戶。橫虛線 = £0 損益平衡線，縱虛線 = 30% 退貨率警戒線。")


def create_country_chart(country_tab):
    """③-3 境內外雙指標橫向分組條形圖。"""
    if country_tab.empty or UK not in country_tab.index:
        st.info("目前篩選條件下沒有英國資料可作為基準，或沒有資料。")
        return None
    overseas = country_tab.drop(index=UK)
    rows = []
    uk = country_tab.loc[UK]
    rows.append(("英國（基準）", uk["InvoiceReturnRate"], uk["AmountReturnRate"], int(uk["Invoices"])))
    if len(overseas):
        o_inv = overseas["Invoices"].sum()
        o_ret = overseas["ReturnInvoices"].sum()
        o_sales = overseas["SalesAmt"].sum()
        o_loss = overseas["ReturnLoss"].sum()
        rows.append(("海外合計", o_ret / o_inv, o_loss / o_sales if o_sales else np.nan, int(o_inv)))
        for country, row in overseas.sort_values("Invoices", ascending=False).head(5).iterrows():
            rows.append((str(country), row["InvoiceReturnRate"], row["AmountReturnRate"], int(row["Invoices"])))
    chart_data = pd.DataFrame(rows, columns=["國家", "訂單退貨率", "金額退貨率", "總訂單數"])
    long = chart_data.melt(id_vars=["國家", "總訂單數"], var_name="指標", value_name="比率")
    fig = px.bar(
        long, x="比率", y="國家", color="指標", barmode="group", orientation="h",
        color_discrete_map={"訂單退貨率": SLATE, "金額退貨率": SAGE},
        hover_data={"總訂單數": True}, text=long["比率"].map(lambda v: f"{v:.2%}" if pd.notna(v) else ""),
        category_orders={"國家": chart_data["國家"].tolist()},
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickformat=".0%")
    fig.update_traces(textposition="outside")
    fig.add_vline(x=uk["InvoiceReturnRate"], line_dash="dash", line_color=GREIGE)
    style_fig(fig, 430)
    st.plotly_chart(fig, width="stretch")
    if len(overseas):
        z, p = two_proportion_z(
            uk["ReturnInvoices"], uk["Invoices"], overseas["ReturnInvoices"].sum(), overseas["Invoices"].sum()
        )
        st.caption(f"海外 vs 英國訂單退貨率：Z = {z:.2f}，p = {p:.2g}（雙樣本比例檢定）。灰虛線 = 英國訂單退貨率。")
    return chart_data


def create_action_panel(base, scope, cust_table, invoice_table):
    """③-4 高管行動追蹤清單（三個分頁）。"""
    tab_a, tab_b, tab_c = st.tabs(["A　負貢獻客戶複核", "B　A類實體商品", "C　B2B 大單預警"])

    with tab_a:
        negative = cust_table[cust_table["NetSpend"] < 0]
        pick = st.radio("子群", ["全部", "純退無買", "買後退>買"], horizontal=True)
        if pick == "純退無買":
            negative = negative[negative["GrossSpend"] == 0]
        elif pick == "買後退>買":
            negative = negative[negative["GrossSpend"] > 0]
        st.caption(f"{len(negative):,} 人，淨虧損合計 {money(negative['NetSpend'].sum())}")
        show = negative.sort_values("NetSpend")[["CustomerID", "GrossSpend", "ReturnLoss", "NetSpend", "ReturnInvoices", "Segment"]]
        st.dataframe(
            show.rename(columns={"GrossSpend": "購買額", "ReturnLoss": "退貨額", "NetSpend": "淨貢獻", "ReturnInvoices": "退貨單數", "Segment": "分類"}),
            hide_index=True, width="stretch", height=260,
        )
        st.download_button("匯出複核工單 CSV", show.to_csv(index=False).encode("utf-8-sig"), "negative_net_customers.csv", "text/csv")
        only_return = cust_table[(cust_table["NetSpend"] < 0) & (cust_table["GrossSpend"] == 0)]
        only_return_rows = base[base["CustomerID"].isin(only_return["CustomerID"]) & base["IsReturn"]]
        only_return_total = only_return_rows["ReturnLoss"].sum()
        if only_return_total > 0:
            share = only_return_rows.loc[only_return_rows["IsNonPhysical"], "ReturnLoss"].sum() / only_return_total
            st.caption(
                f"「純退無買」客戶的退貨額中，{share:.2%} 屬非實體代碼（主要是 Manual 人工調帳），"
                "較像帳務調整掛在客戶 ID 下，而非真實退貨；請先人工查核，再決定限制措施。"
            )

    with tab_b:
        # A 類以「全量帳面口徑」的帕雷托排名決定，再剔除非實體代碼
        prod_all = product_table(base)
        a_all = prod_all[prod_all["Class"] == "A"]
        a_physical = a_all[~a_all.index.isin(NON_PHYSICAL_CODES)]
        physical_sales = base["SalesAmt"].sum()
        physical_loss = base.loc[~base["IsNonPhysical"], "ReturnLoss"].sum()
        benchmark = physical_loss / physical_sales if physical_sales > 0 else 0
        excess = (a_physical["ReturnLoss"] - benchmark * a_physical["SalesAmt"]).clip(lower=0).sum()
        st.caption(
            f"全量口徑 A 類 {len(a_all):,} 款，其中實體商品 {len(a_physical):,} 款，退貨損失 {money(a_physical['ReturnLoss'].sum())}。"
        )
        st.caption(
            f"理論標竿（非預測）：若各品項退貨率降到實體商品退貨率 {benchmark:.2%}，超出部分合計 {money(excess)}。"
        )
        table = a_physical.reset_index()[["StockCode", "Description", "SalesAmt", "ReturnLoss", "ReturnRate"]]
        st.dataframe(
            table.rename(columns={"SalesAmt": "銷售額", "ReturnLoss": "退貨額", "ReturnRate": "退貨率"}),
            hide_index=True, width="stretch", height=260,
            column_config={"退貨率": st.column_config.NumberColumn(format="%.2f")},
        )

    with tab_c:
        big = invoice_table[invoice_table["ReturnLoss"] > BIG_INVOICE_ALERT].copy()
        st.caption(
            f"單張退貨發票 > £{BIG_INVOICE_ALERT:,}：{len(big):,} 張，其中 > £{BIG_INVOICE_CRITICAL:,}：{(big['ReturnLoss'] > BIG_INVOICE_CRITICAL).sum()} 張"
        )
        big["CustomerID"] = big["CustomerID"].map(lambda v: "" if pd.isna(v) else str(int(v)))
        big["InvoiceDate"] = big["InvoiceDate"].dt.date
        st.dataframe(
            big.rename(columns={"ReturnLoss": "退貨額", "CustomerID": "客戶", "Country": "國家", "InvoiceDate": "日期"}),
            hide_index=True, width="stretch", height=260,
        )
        st.caption("建議：單筆 > £5,000 的批發單採「樣品確認 + 30% 履約訂金」。")


# ---------------------------------------------------------------------------
# 主程式
# ---------------------------------------------------------------------------
def inject_css():
    st.markdown(
        f"""
        <style>
        .stApp {{ background-color: {BG}; }}
        [data-testid="stMetric"] {{
            background: {CARD}; border: 1px solid {BORDER}; border-radius: 12px; padding: 12px 16px;
        }}
        h1, h2, h3 {{ color: {TEXT}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="退貨止血儀表板", page_icon="📊", layout="wide")
    inject_css()
    st.title("📊 電商退貨歸因與止血儀表板")
    st.caption("Online Retail 兩年全量資料 · 總經理／業務副總／營運副總決策用")

    df, meta = load_data()

    # ① 頂部全局篩選列
    countries = sorted(df["Country"].cat.categories.tolist())
    with st.container(border=True):
        f1, f2, f3, f4 = st.columns([1.2, 1.4, 1, 1])
        scope = f1.radio("帳務口徑", ["僅實體商品", "全量帳面"], horizontal=True,
                         help="僅實體商品會剔除 Manual / AMAZONFEE / BANK CHARGES / POST / D 五個非實體代碼")
        date_value = f2.date_input("日期區間", value=(meta["date_min"], meta["date_max"]),
                                   min_value=meta["date_min"], max_value=meta["date_max"])
        market = f3.selectbox("市場", ["全部", "英國境內", "海外跨境"] + countries)
        cust_attr = f4.selectbox("客群歸戶", ["全部", "具名客戶", "未歸戶過客"])

    if isinstance(date_value, tuple) and len(date_value) == 2:
        start, end = date_value
    else:
        st.info("請選擇完整的起訖日期。")
        return

    base_no_market = apply_base_filters(df, start, end, cust_attr)
    base = apply_market(base_no_market, market)
    if base.empty:
        st.warning("目前篩選條件下沒有資料。")
        return
    scoped = apply_scope(base, scope)
    scoped_no_market = apply_scope(base_no_market, scope)

    # 客戶淨貢獻看的是客戶完整帳務，所以一律用全量帳面（base），不隨「帳務口徑」切換。
    # 否則切到「僅實體商品」時，只有非實體退貨（如 Manual 調帳）的客戶會憑空消失，數字會與報告對不上。
    cust_table = customer_table(base)
    invoice_table = return_invoice_table(scoped)
    prod = product_table(scoped)

    # ② KPI 卡片
    create_kpi_cards(base, scoped, scope, cust_table, invoice_table)

    # ③ 主體：左 60% / 右 40%
    left, right = st.columns([6, 4])
    with left:
        with st.container(border=True):
            st.subheader("商品退貨帕雷托 ABC")
            create_pareto_chart(prod)
            create_product_drilldown(scoped, prod)
        with st.container(border=True):
            st.subheader("客戶淨貢獻 vs 退貨率")
            create_customer_quadrant(cust_table)
    with right:
        with st.container(border=True):
            st.subheader("境內外雙指標")
            # 國別圖不套用「市場」篩選，才能看到各國對照
            create_country_chart(country_table(scoped_no_market))
        with st.container(border=True):
            st.subheader("高管行動追蹤")
            create_action_panel(base, scope, cust_table, invoice_table)

    # 頁尾
    st.divider()
    st.caption(
        f"資料口徑：合併 {meta['n_raw']:,} 列 → 去重剔除 {meta['n_dedup']:,} 列 → 剔除 UnitPrice≤0 共 {meta['n_price']:,} 列 → "
        f"有效 {meta['n_clean']:,} 列（保留 {meta['n_clean'] / meta['n_raw']:.2%}）。資料期間 {meta['date_min']} ~ {meta['date_max']}。"
        "退貨原因欄位不存在，運損／尺寸等成因需另行驗證。"
    )


if __name__ == "__main__":
    main()

