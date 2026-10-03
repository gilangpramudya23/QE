import io
import os

import pandas as pd
import plotly.express as px
import streamlit as st

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Credit Quality Dashboard", page_icon="📊", layout="wide")

MAX_TABLE_ROWS = 2000  # rows rendered in the records table (download always has everything)
DROP_COLS = ["Kontrak (migrate)", "Credit Analyst", "Expert 2", "Decision", "Decision.1"]
TARGET = "GOOD BAD"
DIMS = ["Area Credit 2021", "Branch", "LOB", "Is Override", "Score", "Kel DP", "Angke > 30"]
OPTIONAL_COLS = ["Amount Finance", "DP %"]  # used only for extra KPIs / AI context when present
COLORS = {"Good": "#2E9E6B", "Bad": "#E5484D"}


# ----------------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def list_sheets(raw: bytes, name: str) -> list:
    if name.lower().endswith(".csv"):
        return []
    return pd.ExcelFile(io.BytesIO(raw)).sheet_names


def clean_dim(s: pd.Series, col: str) -> pd.Series:
    """Make a slicer column clean text. 'Angke > 30' 5.0 -> '5' so it doesn't show decimals."""
    if col == "Angke > 30":
        num = pd.to_numeric(s, errors="coerce")
        if num.notna().sum() == s.notna().sum() and (num.dropna() % 1 == 0).all():
            s = num.astype("Int64")
    return s.astype("string").fillna("(blank)").str.strip().astype(str)


@st.cache_data(show_spinner="Reading data...")
def load_data(raw: bytes, name: str, sheet) -> pd.DataFrame:
    buf = io.BytesIO(raw)
    df = pd.read_csv(buf) if name.lower().endswith(".csv") else pd.read_excel(buf, sheet_name=sheet or 0)
    df.columns = df.columns.astype(str).str.strip()
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    if TARGET not in df.columns:
        return df  # validated by the caller (shows a clear message)
    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].astype(str).str.strip().str.title()
    df["is_bad"] = (df[TARGET] == "Bad").astype(int)
    for c in DIMS:
        if c in df.columns:
            df[c] = clean_dim(df[c], c)
    return df


def sort_vals(values):
    """Natural sort: numbers numerically (2 < 10), text alphabetically."""
    vals = list(values)
    try:
        return sorted(vals, key=float)
    except ValueError:
        return sorted(vals)


def summarize(d: pd.DataFrame, by) -> pd.DataFrame:
    g = d.groupby(by, observed=True).agg(Contracts=("is_bad", "size"), Bad=("is_bad", "sum")).reset_index()
    g["Good"] = g["Contracts"] - g["Bad"]
    g["Bad rate %"] = (g["Bad"] / g["Contracts"] * 100).round(2)
    return g


def top_categories(d: pd.DataFrame, dim: str, top_n: int):
    """Return the category order to plot (natural order if few, else top N by volume)."""
    counts = d[dim].value_counts()
    if len(counts) > 12:
        return list(counts.head(top_n).index)
    return sort_vals(counts.index)


# ----------------------------------------------------------------------------
# AI CONTEXT
# ----------------------------------------------------------------------------
def build_context(view, full, filters, dim, split, top_n) -> str:
    n, bad = len(view), int(view["is_bad"].sum())
    out = [
        "DATASET: auto-financing contracts. Target column 'GOOD BAD'; 'Bad' = bad contract, "
        "'Bad rate' = Bad / Contracts. 'Angke > 30' = number of installments overdue > 30 days.",
        f"ACTIVE FILTERS: {filters if filters else 'none (all data)'}",
        f"FILTERED VIEW: {n} contracts, {bad} bad, bad rate {bad / n * 100:.2f}% "
        f"(overall dataset: {len(full)} contracts, bad rate {full['is_bad'].mean() * 100:.2f}%)",
    ]
    if "Amount Finance" in view:
        out.append(f"Average Amount Finance in view: {view['Amount Finance'].mean():,.0f}")
    if "DP %" in view:
        out.append(f"Average DP % in view: {view['DP %'].mean():.2f}")
    out.append(f"\nCURRENT CHART: grouped by '{dim}'" + (f" split by '{split}'" if split else ""))
    keys = [dim, split] if split else [dim]
    out.append(summarize(view, keys).sort_values("Contracts", ascending=False).head(40).to_csv(index=False))
    out.append("OTHER BREAKDOWNS OF THE SAME FILTERED VIEW (top 10 by contracts):")
    for c in DIMS:
        if c in view and c != dim:
            out.append(f"[{c}]\n" + summarize(view, c).sort_values("Contracts", ascending=False).head(10).to_csv(index=False))
    return "\n".join(out)


SYSTEM_PROMPT = (
    "You are a senior credit-risk data analyst embedded in a dashboard. Interpret the results below for a "
    "business reader: lead with the key takeaway, then support it with specific numbers, then suggest "
    "next steps or questions worth checking. Use ONLY the numbers in the context; never invent data. "
    "Always mention sample size when a rate is based on few contracts (correlation is not causation). "
    "If the question needs data not in the context, say so and suggest which filter or grouping to select. "
    "Reply in the language the user writes in. Keep answers concise (under ~250 words unless asked).\n\n"
    "DASHBOARD CONTEXT (refreshed on every message):\n"
)


def get_secret_key() -> str:
    try:
        return st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        return ""


# ----------------------------------------------------------------------------
# SIDEBAR: data upload -> validation
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("📁 Data")
    up = st.file_uploader("Upload your Excel / CSV", type=["xlsx", "csv"])
    st.caption("🔒 Your file is processed in memory for this session only and is not saved. If you use the AI analyst, only aggregated results (counts and bad rates) are sent to OpenAI, never raw rows.")

if up is None:
    st.title("📊 Credit Quality Dashboard")
    st.info("👈 Upload your data file in the sidebar to start.")
    st.markdown("**The file must contain these columns (exact header names):**")
    st.code(", ".join([TARGET] + DIMS))
    st.caption(f"Optional (extra KPIs): {', '.join(OPTIONAL_COLS)}. "
               f"Columns {', '.join(DROP_COLS)} are removed automatically.")
    st.stop()

raw, fname = up.getvalue(), up.name
try:
    sheets = list_sheets(raw, fname)
    sheet = None
    if len(sheets) > 1:
        with st.sidebar:
            sheet = st.selectbox("Sheet", sheets)
    df = load_data(raw, fname, sheet)
except Exception as e:
    st.error(f"Could not read the file: {e}")
    st.stop()

missing = [c for c in DIMS + [TARGET] if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.caption(f"Columns found in your file: {list(df.columns)}")
    st.stop()
if df.empty:
    st.error(f"No rows with a value in '{TARGET}'.")
    st.stop()
unknown = sorted(set(df[TARGET]) - {"Good", "Bad"})
if unknown:
    st.warning(f"'{TARGET}' has unexpected values {unknown}; they are counted as Good.")
if df["is_bad"].sum() == 0:
    st.warning("There are no 'Bad' contracts in this data, so every bad rate will be 0%.")


# New file or sheet -> clear old filters and chat so stale selections can't break the page
sig = (fname, len(raw), sheet)
if st.session_state.get("data_sig") != sig:
    for c in DIMS:
        st.session_state.pop(f"f_{c}", None)
    st.session_state["messages"] = []
    st.session_state["data_sig"] = sig


def reset_filters():
    for c in DIMS:
        st.session_state[f"f_{c}"] = []


with st.sidebar:
    st.header("🎚️ Filters")
    st.caption("Leave empty = all values")
    mask = pd.Series(True, index=df.index)
    active = {}
    for c in DIMS:
        sel = st.multiselect(c, sort_vals(df[c].unique()), key=f"f_{c}", placeholder="All")
        if sel:
            mask &= df[c].isin(sel)
            active[c] = sel
    st.button("↺ Reset filters", on_click=reset_filters, use_container_width=True)

    st.header("🤖 AI settings")
    api_key = st.text_input("OpenAI API key", type="password", value=get_secret_key(), placeholder="sk-...")
    model = st.text_input("GPT model", value="gpt-4o-mini")
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state["messages"] = []

view = df[mask]

# ----------------------------------------------------------------------------
# HEADER + CHART CONTROLS
# ----------------------------------------------------------------------------
st.title("📊 Credit Quality Dashboard")
st.caption("Choose any column for the chart, filter in the sidebar, then ask the AI analyst at the bottom.")

if view.empty:
    st.warning("No data matches the current filters. Click **Reset filters** in the sidebar.")
    st.stop()

c1, c2, c3 = st.columns([2, 2, 1])
dim = c1.selectbox("📐 Group by (X-axis)", DIMS, index=0)
split_opts = ["(none)"] + [d for d in DIMS if d != dim]
split = c2.selectbox("🎨 Split by (optional)", split_opts)
split = None if split == "(none)" else split
top_n = c3.slider("Top N", 5, 40, 15, help="Used when a column has more than 12 categories (e.g. Branch)")

# ----------------------------------------------------------------------------
# KPI CARDS
# ----------------------------------------------------------------------------
n, bad = len(view), int(view["is_bad"].sum())
rate, base = bad / n * 100, df["is_bad"].mean() * 100
k1, k2, k3, k4 = st.columns(4)
k1.metric("Contracts", f"{n:,}", help=f"{n / len(df) * 100:.0f}% of all {len(df):,} contracts")
k2.metric("Good", f"{n - bad:,}")
k3.metric("Bad", f"{bad:,}")
k4.metric("Bad rate", f"{rate:.2f}%", delta=f"{rate - base:+.2f} pts vs overall", delta_color="inverse")
if n < 30:
    st.info("⚠️ Small sample (< 30 contracts): rates can swing a lot, interpret with care.")

# ----------------------------------------------------------------------------
# CHARTS
# ----------------------------------------------------------------------------
order = top_categories(view, dim, top_n)
plot_df = view[view[dim].isin(order)]
keys = [dim, split] if split else [dim]
g = summarize(plot_df, keys)

tab_chart, tab_table = st.tabs(["📈 Charts", "🗂️ Data & Tables"])

with tab_chart:
    left, right = st.columns(2)

    # 1) Contracts: Good vs Bad stacked (or by split)
    if split:
        fig1 = px.bar(g, x=dim, y="Contracts", color=split, barmode="stack",
                      category_orders={dim: order}, title=f"Contracts by {dim} (coloured by {split})")
    else:
        long = g.melt(id_vars=dim, value_vars=["Good", "Bad"], var_name=TARGET, value_name="Contracts")
        fig1 = px.bar(long, x=dim, y="Contracts", color=TARGET, barmode="stack", color_discrete_map=COLORS,
                      category_orders={dim: order, TARGET: ["Good", "Bad"]}, title=f"Contracts by {dim}: Good vs Bad")
    fig1.update_layout(xaxis_tickangle=-40, legend_title_text="")
    left.plotly_chart(fig1, use_container_width=True)

    # 2) Bad rate
    fig2 = px.bar(g, x=dim, y="Bad rate %", color=split, barmode="group", text="Bad rate %",
                  hover_data=["Contracts", "Bad"], category_orders={dim: order},
                  color_discrete_sequence=[COLORS["Bad"]] if not split else None,
                  title=f"Bad rate (%) by {dim}")
    fig2.update_layout(xaxis_tickangle=-40, legend_title_text="")
    right.plotly_chart(fig2, use_container_width=True)

    left2, right2 = st.columns(2)
    # 3) Donut
    tot = pd.DataFrame({TARGET: ["Good", "Bad"], "Contracts": [n - bad, bad]})
    fig3 = px.pie(tot, names=TARGET, values="Contracts", hole=0.55, color=TARGET,
                  color_discrete_map=COLORS, title="Good vs Bad (current filters)")
    left2.plotly_chart(fig3, use_container_width=True)

    # 4) Heatmap (needs a split) or volume share
    if split:
        piv = g.pivot(index=dim, columns=split, values="Bad rate %").reindex(order)
        fig4 = px.imshow(piv, text_auto=True, aspect="auto", color_continuous_scale="Reds",
                         title=f"Bad rate (%): {dim} × {split}")
    else:
        fig4 = px.pie(g.sort_values("Contracts", ascending=False), names=dim, values="Contracts",
                      title=f"Share of contracts by {dim}")
    right2.plotly_chart(fig4, use_container_width=True)

with tab_table:
    st.subheader(f"Summary by {' × '.join(keys)}")
    st.dataframe(g.sort_values("Contracts", ascending=False), use_container_width=True, hide_index=True)
    st.subheader("Filtered records")
    shown = view.drop(columns=["is_bad"])
    if len(shown) > MAX_TABLE_ROWS:
        st.caption(f"Showing the first {MAX_TABLE_ROWS:,} of {len(shown):,} rows. The download has all rows.")
    st.dataframe(shown.head(MAX_TABLE_ROWS), use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download filtered data (CSV)", shown.to_csv(index=False).encode("utf-8"),
                       "filtered_data.csv", "text/csv")

# ----------------------------------------------------------------------------
# AI ANALYST (chat with follow-ups)
# ----------------------------------------------------------------------------
st.divider()
st.subheader("🤖 AI Analyst")
st.caption("The AI sees the aggregated results of your current filters and chart selection "
           "(not the raw rows). Change a filter, then ask again.")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

quick = st.button("✨ Interpret the current dashboard")
typed = st.chat_input("Ask a follow-up, e.g. 'Which branch should we review first and why?'")
prompt = "Interpret the current dashboard results: key findings, risks, and what to check next." if quick else typed

if prompt:
    if not api_key:
        st.error("Enter your OpenAI API key in the sidebar first.")
        st.stop()
    from openai import OpenAI

    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    context = build_context(view, df, active, dim, split, top_n)
    msgs = [{"role": "system", "content": SYSTEM_PROMPT + context}] + st.session_state["messages"][-12:]

    with st.chat_message("assistant"):
        try:
            stream = OpenAI(api_key=api_key).chat.completions.create(model=model, messages=msgs, stream=True)

            def tokens():
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content

            reply = st.write_stream(tokens())
            st.session_state["messages"].append({"role": "assistant", "content": reply})
        except Exception as e:
            st.session_state["messages"].pop()  # drop the unanswered question
            st.error(f"OpenAI error: {e}")
