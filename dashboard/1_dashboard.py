"""
Streamlit dashboard: Sudden Traffic Impact on Company Performance.
One-page layout: company/keyword/feature inputs → KW timeseries, engineered features table,
EPS surprise distribution vs conditional, and summary stats.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
EVENT_TABLE_DIR = DATA_DIR / "event-table"
OUTLIER_THRESHOLD = 500

FEATURE_OPTIONS = [
    ("baseline_shift_z_e", "Baseline shift"),
    ("area_z_e", "Area (z)"),
    ("peak_z_e", "Peak (z)"),
]

# Slider ranges for the combination plot
FEATURE_SLIDER_RANGES = {
    "baseline_shift_z_e": (0.0, 1.2),
    "peak_z_e": (0.0, 20.0),
    "area_z_e": (0.0, 40.0),
}


def signed_log1p(x):
    """sign(x) * log(1 + |x|). Makes histograms readable."""
    return np.sign(x) * np.log1p(np.abs(x))


@st.cache_data
def load_timeseries():
    """Load and concat all normalized timeseries CSV parts."""
    ts_files = sorted(DATA_DIR.glob("normalized_timeseries_branded_*_part*.csv"))
    if not ts_files:
        return None
    dfs = [pd.read_csv(f) for f in ts_files]
    df = pd.concat(dfs, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df


@st.cache_data
def load_metadata():
    """Load metadata (keyword, company); filter status=success if present."""
    meta_files = sorted(DATA_DIR.glob("normalized_metadata_branded_*.csv"))
    if not meta_files:
        return None
    meta = pd.concat(
        [pd.read_csv(f, on_bad_lines="skip") for f in meta_files],
        ignore_index=True,
    )
    if "status" in meta.columns:
        meta = meta.loc[meta["status"] == "success", ["keyword", "company"]]
    else:
        meta = meta[["keyword", "company"]]
    meta = meta.drop_duplicates()
    return meta


@st.cache_data
def load_earnings():
    """Load earnings, preprocess like notebook: numeric surprisePercent, outlier filter, exclude first year."""
    path = DATA_DIR / "earnings_data.csv"
    if not path.exists():
        return None, None
    earnings = pd.read_csv(path)
    earnings["surprisePercent"] = pd.to_numeric(earnings["surprisePercent"], errors="coerce")
    earnings = earnings.dropna(subset=["surprisePercent"])
    earnings = earnings[
        (earnings["surprisePercent"] >= -OUTLIER_THRESHOLD)
        & (earnings["surprisePercent"] <= OUTLIER_THRESHOLD)
    ]
    earnings["date"] = pd.to_datetime(earnings["date"], utc=True)
    min_date = earnings["date"].min()
    cutoff = min_date + pd.DateOffset(years=1)
    earnings = earnings.loc[earnings["date"] >= cutoff].copy()
    y_all = earnings["surprisePercent"]
    return earnings, y_all


@st.cache_data
def load_quarter_events():
    """Load precomputed quarter_events from event-table."""
    path = EVENT_TABLE_DIR / "quarter_events.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    for col in ["t_start", "t_end", "quarter_end"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)
    for col in ["baseline_shift_z_e", "area_z_e", "peak_z_e"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["surprisePercent"] = pd.to_numeric(df["surprisePercent"], errors="coerce")
    return df


def get_companies_and_keywords(df_all, meta, quarter_events):
    """Companies that have quarter events; keyword list per company (from metadata + timeseries)."""
    # Company list: only companies that appear in quarter_events
    if quarter_events is None or quarter_events.empty or "company" not in quarter_events.columns:
        return [], {}
    companies = sorted(quarter_events["company"].unique().tolist())
    if df_all is None or meta is None:
        return companies, {c: [] for c in companies}
    keywords_in_ts = set(df_all["keyword"].unique())
    meta = meta[meta["keyword"].isin(keywords_in_ts)]
    company_to_keywords = {}
    for c in companies:
        kws = meta.loc[meta["company"] == c, "keyword"].unique().tolist()
        company_to_keywords[c] = sorted([k for k in kws if k in keywords_in_ts])
    return companies, company_to_keywords


# Page config and title 
st.set_page_config(page_title="Sudden Traffic Impact", layout="wide")
st.markdown("<h1 style='text-align: center;'>Sudden Traffic Impact on Company Performance</h1>", unsafe_allow_html=True)

st.markdown("<p style='text-align: center;'> Keyword timeseries data sourced from Google Trends. Events generated using STL decomposition residual. Quarter earnings data sourced from EODHD.</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'> Detailed methodology and event creation logic can be found in the <a href='https://github.com/eetujok/google-trends-for-eps-estimation/blob/main/project.ipynb'>project notebook</a>.</p>", unsafe_allow_html=True)

# Data loading
df_all = load_timeseries()
meta = load_metadata()
earnings, y_all = load_earnings()
quarter_events = load_quarter_events()

if earnings is None or y_all is None:
    st.error("earnings_data.csv not found in data/. Cannot show distributions.")
    st.stop()

if quarter_events is None or quarter_events.empty:
    st.warning(
        "Quarter events table not found. Run the pipeline to generate data/event-table/quarter_events.csv "
        "(e.g. from project root: run stl_decomposition pipeline and save to data/event-table/). "
        "Inputs and left panel may work with timeseries only; distribution panel will be limited."
    )

companies, company_to_keywords = get_companies_and_keywords(df_all, meta, quarter_events)
if not companies:
    st.warning("No companies with quarter events found. Run the pipeline to generate data/event-table/quarter_events.csv.")
    st.stop()

DEFAULT_COMPANY = "American Eagle Outfitters, Inc."
DEFAULT_KEYWORD = "american eagle"
idx_company = companies.index(DEFAULT_COMPANY) if DEFAULT_COMPANY in companies else 0
default_kws = company_to_keywords.get(companies[idx_company], [])
idx_keyword = default_kws.index(DEFAULT_KEYWORD) if DEFAULT_KEYWORD in default_kws else 0

st.subheader("Input")
c1, c2, c3, c4 = st.columns(4)

with c1:
    company = st.selectbox(
        "Company",
        options=companies,
        key="company",
        index=idx_company,
    )

with c2:
    keywords_for_company = company_to_keywords.get(company, [])
    # When company is the default company, keep default keyword index; else 0
    kw_index = keywords_for_company.index(DEFAULT_KEYWORD) if (company == DEFAULT_COMPANY and DEFAULT_KEYWORD in keywords_for_company) else 0
    keyword = st.selectbox(
        "Keyword",
        options=keywords_for_company or [""],
        key="keyword",
        index=kw_index,
    )
    if keyword == "" and keywords_for_company:
        keyword = keywords_for_company[0]

with c3:
    feature_options_display = [label for _, label in FEATURE_OPTIONS]
    feature_idx = st.selectbox(
        "Feature",
        options=range(len(FEATURE_OPTIONS)),
        format_func=lambda i: FEATURE_OPTIONS[i][1],
        key="feature",
    )
    feature_col = FEATURE_OPTIONS[feature_idx][0]

with c4:
    min_val, max_val = FEATURE_SLIDER_RANGES.get(feature_col, (0.0, 20.0))
    step = (max_val - min_val) / 100.0 if max_val > min_val else 0.01
    slider_value = st.slider(
        "Slider: Value",
        min_value=min_val,
        max_value=max_val,
        value=min_val + (max_val - min_val) * 0.3 if max_val > min_val else min_val,
        step=step,
        key="slider",
    )

st.markdown("---")

# Two columns: left (timeseries + table), right (distribution + summary)
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Keyword Time Series")
    if df_all is not None and keyword:
        ts_df = df_all.loc[df_all["keyword"] == keyword, ["date", "value"]].copy()
        ts_df = ts_df.sort_values("date")
        if not ts_df.empty:
            # Metrics for this keyword (for text box in plot)
            if quarter_events is not None and not quarter_events.empty and "keyword" in quarter_events.columns:
                qe_kw = quarter_events[quarter_events["keyword"] == keyword]
                n_spikes = qe_kw.drop_duplicates(subset=["company", "quarter_end"]).shape[0]
            else:
                n_spikes = 0
            fig, ax = plt.subplots(figsize=(8, 3))
            # Event spans in green, from quarter_events for this keyword
            if quarter_events is not None and not quarter_events.empty and "t_start" in quarter_events.columns and "t_end" in quarter_events.columns:
                qe_kw = quarter_events[quarter_events["keyword"] == keyword][["t_start", "t_end"]].drop_duplicates()
                for _, row in qe_kw.iterrows():
                    a, b = row["t_start"], row["t_end"]
                    if pd.notna(a) and pd.notna(b):
                        ax.axvspan(a, b, alpha=0.18, color="green")
            ax.plot(ts_df["date"], ts_df["value"], color="steelblue", linewidth=1)
            ax.set_xlabel("Date")
            ax.set_ylabel("Value")
            ax.set_title(f"Keyword: {keyword}")
            # Text box top right: Highest %, N. Spikes
            textstr = "\n".join([
                f"N. Spike Events: {n_spikes:,}",
            ])
            props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
            ax.text(0.98, 0.95, textstr, transform=ax.transAxes, fontsize=10, verticalalignment="top", horizontalalignment="right", bbox=props)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.markdown("""
            <p style='text-align: center;'>
            Spike events are marked in green.
            </p>
            """, unsafe_allow_html=True)

        else:
            st.info("No timeseries data for this keyword.")
    else:
        st.info("Select a company and keyword to see the timeseries.")
    st.subheader("Features of spike events")
    if quarter_events is not None and not quarter_events.empty and keyword:
        cols_show = ["t_start", "t_end", "baseline_shift_z_e", "area_z_e", "peak_z_e"]
        if "company" in quarter_events.columns:
            cols_show.append("company")
        if "quarter_end" in quarter_events.columns:
            cols_show.append("quarter_end")
        if "surprisePercent" in quarter_events.columns:
            cols_show.append("surprisePercent")
        cols_show = [c for c in cols_show if c in quarter_events.columns]
        table_df = quarter_events[quarter_events["keyword"] == keyword][cols_show].copy()
        renamed_table_df = table_df.rename(columns={
            "t_start": "Event Start",
            "t_end": "Event End",
            "baseline_shift_z_e": "Baseline shift",
            "area_z_e": "Area",
            "peak_z_e": "Event Peak",
            "quarter_end": "Quarter End",
            "surprisePercent": "EPS Surprise",
        })
        if not renamed_table_df.empty:
            st.dataframe(renamed_table_df, use_container_width=True)
        else:
            st.write("No events for this keyword.")
    else:
        st.info("Select a keyword and ensure quarter events are loaded to see the table.")


with col_right:
    st.subheader("EPS Surprise distribution: All quarters vs. quarters with spike events")
    if quarter_events is None or quarter_events.empty or feature_col not in quarter_events.columns:
        st.info("Load quarter_events and feature data to show distributions.")
    else:
        config_df = quarter_events.copy()
        config_df[feature_col] = pd.to_numeric(config_df[feature_col], errors="coerce")
        mask = config_df[feature_col] > slider_value
        sub = config_df.loc[mask].drop_duplicates(subset=["company", "quarter_end"])
        y_cond = sub["surprisePercent"].dropna()
        y_cond = y_cond[(y_cond >= -OUTLIER_THRESHOLD) & (y_cond <= OUTLIER_THRESHOLD)]
        y_all_f = y_all[(y_all >= -OUTLIER_THRESHOLD) & (y_all <= OUTLIER_THRESHOLD)]

        n_all = len(y_all_f)
        n_spikes = len(y_cond)

        if n_spikes == 0:
            st.warning(f"No quarters with {FEATURE_OPTIONS[feature_idx][1]} > {slider_value}.")
        else:
            fig, ax = plt.subplots(figsize=(8, 4))
            y_all_log = signed_log1p(y_all_f)
            y_cond_log = signed_log1p(y_cond)
            bins = np.linspace(
                min(y_all_log.min(), y_cond_log.min()),
                max(y_all_log.max(), y_cond_log.max()),
                40,
            )
            ax.hist(
                y_all_log,
                bins=bins,
                alpha=0.5,
                color="gray",
                label=f"All quarters, n = {n_all:,}",
                density=True,
            )
            ax.hist(
                y_cond_log,
                bins=bins,
                alpha=0.6,
                color="C0",
                label=f"Quarters w/ spikes ({feature_col} > {slider_value:.3g}), n = {n_spikes:,}",
                density=True,
            )
            ax.axvline(0, color="black", linestyle="--", alpha=0.7)
            ax.set_xlabel("EPS surprise (%) — signed log scale")
            ax.set_ylabel("Density")
            ax.set_title("EPS surprise: All quarters vs. quarters with spike events")
            tick_vals = np.array([-100, -10, -1, 0, 1, 10, 100, 1000, 1e4])
            ax.set_xticks(signed_log1p(tick_vals))
            ax.set_xticklabels([f"{t:.0f}" if abs(t) >= 1 else f"{t:.1f}" for t in tick_vals])
            ax.legend(loc="lower right")
            textstr = "\n".join([
                f"N. All Quarters: {n_all:,}",
                f"N. Quarters w/ Spikes: {n_spikes:,}",
            ])
            props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
            ax.text(0.02, 0.97, textstr, transform=ax.transAxes, fontsize=10, verticalalignment="top", bbox=props)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Explanation text
        st.subheader("Explanation")
        if n_spikes > 0:
            p_all_gt0 = float((y_all_f > 0).mean())
            p_all_lt0 = float((y_all_f < 0).mean())
            p_cond_gt0 = float((y_cond > 0).mean())
            p_cond_lt0 = float((y_cond < 0).mean())
            mean_all = float(y_all_f.mean())
            mean_cond = float(y_cond.mean())
            delta_mean = mean_cond - mean_all
            st.write(f"Probability of positive surprise increase {p_all_gt0*100:.1f}% → {p_cond_gt0*100:.1f}% in quarters with keyword traffic spikes")
            st.write(f"Probability of negative surprise decrease {p_all_lt0*100:.1f}% → {p_cond_lt0*100:.1f}% in quarters with keyword traffic spikes")
            st.write(f"Spike quarters show {delta_mean:+.1f}% higher average EPS surprise compared to all quarters.")
        else:
            st.write("No spike quarters in scope; summary not available.")
