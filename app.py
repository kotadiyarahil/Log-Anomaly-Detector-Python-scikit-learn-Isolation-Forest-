from __future__ import annotations

import datetime as dt
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# -----------------------------
# App configuration
# -----------------------------
st.set_page_config(
    page_title="Network Log Anomaly Detector",
    page_icon="🔐",
    layout="wide",
)


# -----------------------------
# Styling (dark dashboard theme)
# -----------------------------
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
            color: #e5e7eb;
        }
        h1, h2, h3, h4, h5, h6, p, label, span, div {
            color: #e5e7eb;
        }
        .metric-card {
            background-color: #111827;
            border-radius: 12px;
            padding: 14px;
            border: 1px solid #374151;
            box-shadow: 0 0 0 1px rgba(255,255,255,0.02) inset;
        }
        .metric-label {
            font-size: 0.9rem;
            color: #cbd5e1;
            margin-bottom: 4px;
        }
        .metric-value {
            font-size: 1.65rem;
            font-weight: 700;
            line-height: 1.1;
        }
        .metric-sub {
            font-size: 0.8rem;
            color: #94a3b8;
            margin-top: 6px;
        }
        .border-normal {
            border-left: 4px solid #2ecc71 !important;
        }
        .border-anomaly {
            border-left: 4px solid #e74c3c !important;
        }
        .border-info {
            border-left: 4px solid #38bdf8 !important;
        }
        .border-warning {
            border-left: 4px solid #f59e0b !important;
        }
        .section-card {
            background-color: rgba(17, 24, 39, 0.75);
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 14px;
        }
        .risk-green { color: #22c55e; font-weight: 700; }
        .risk-yellow { color: #f59e0b; font-weight: 700; }
        .risk-red { color: #ef4444; font-weight: 700; }
        .footer {
            text-align: center;
            margin-top: 2rem;
            color: #94a3b8;
            font-size: 0.9rem;
            padding-top: 1rem;
            border-top: 1px solid #334155;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


BASE_PATH = Path(r"D:\Projects\log-anomaly-detector")
NORMAL_COLOR = "#2ecc71"
ANOMALY_COLOR = "#e74c3c"


def safe_encode(le, values):
    classes = list(le.classes_)
    return [classes.index(v) if v in classes else 0 for v in values]


@st.cache_resource(show_spinner=False)
def load_artifacts():
    with open(BASE_PATH / "anomaly_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(BASE_PATH / "le_ip.pkl", "rb") as f:
        le_ip = pickle.load(f)
    with open(BASE_PATH / "le_action.pkl", "rb") as f:
        le_action = pickle.load(f)
    with open(BASE_PATH / "le_user.pkl", "rb") as f:
        le_user = pickle.load(f)
    with open(BASE_PATH / "features.pkl", "rb") as f:
        features = pickle.load(f)
    return model, le_ip, le_action, le_user, features


def preprocess_and_score(df: pd.DataFrame, threshold: float):
    model, le_ip, le_action, le_user, features = load_artifacts()

    # Ensure required columns exist
    required_cols = [
        "timestamp",
        "ip_address",
        "user_id",
        "action",
        "bytes_transferred",
        "response_time_ms",
        "failed_attempts",
        "hour",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    work_df = df.copy()
    work_df["timestamp"] = pd.to_datetime(work_df["timestamp"], errors="coerce")
    work_df["hour"] = work_df["hour"].fillna(work_df["timestamp"].dt.hour).fillna(0).astype(int)

    work_df["ip_encoded"] = safe_encode(le_ip, work_df["ip_address"].astype(str))
    work_df["action_encoded"] = safe_encode(le_action, work_df["action"].astype(str))
    work_df["user_encoded"] = safe_encode(le_user, work_df["user_id"].astype(str))

    for col in features:
        if col not in work_df.columns:
            work_df[col] = 0

    X = work_df[features]
    # Higher anomaly_score means more suspicious
    anomaly_score = -model.decision_function(X)
    work_df["anomaly_score"] = anomaly_score

    # Dynamic threshold based on slider contamination level.
    # Example: threshold=0.10 means top 10% highest scores are anomalies.
    score_cutoff = float(np.quantile(anomaly_score, 1 - threshold))
    work_df["predicted_anomaly"] = (work_df["anomaly_score"] >= score_cutoff).astype(int)

    return work_df


def metric_card(title: str, value: str, subtext: str, border_class: str):
    st.markdown(
        f"""
        <div class="metric-card {border_class}">
            <div class="metric-label">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_risk_label(threat_pct: float):
    if threat_pct < 10:
        return "GREEN", "risk-green"
    if threat_pct < 25:
        return "YELLOW", "risk-yellow"
    return "RED", "risk-red"


# -----------------------------
# Header
# -----------------------------
st.title("🔐 Network Log Anomaly Detector")
st.caption("AI-powered cybersecurity dashboard")
st.info(
    "This dashboard uses a trained Isolation Forest model to identify suspicious network "
    "activity from log data. Upload a CSV or run on the sample dataset to generate risk insights."
)


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Detection Controls")
source_choice = st.sidebar.radio(
    "Data Source",
    ("Use Sample Dataset", "Upload CSV"),
)

uploaded_file = None
if source_choice == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

contamination_threshold = st.sidebar.slider(
    "Contamination Threshold",
    min_value=0.05,
    max_value=0.30,
    value=0.10,
    step=0.01,
    help="Higher values flag more events as anomalies.",
)

run_detection = st.sidebar.button("Run Detection", type="primary")


if run_detection:
    try:
        with st.spinner("Processing logs and detecting anomalies..."):
            progress = st.progress(0, text="Initializing...")
            time.sleep(0.2)
            progress.progress(15, text="Loading data...")

            if source_choice == "Use Sample Dataset":
                data_df = pd.read_csv(BASE_PATH / "network_logs.csv")
            else:
                if uploaded_file is None:
                    st.warning("Please upload a CSV file to continue.")
                    st.stop()
                data_df = pd.read_csv(uploaded_file)

            time.sleep(0.2)
            progress.progress(45, text="Preprocessing features...")

            scored_df = preprocess_and_score(data_df, contamination_threshold)

            time.sleep(0.2)
            progress.progress(75, text="Generating dashboard visuals...")

            total_logs = int(len(scored_df))
            anomaly_count = int(scored_df["predicted_anomaly"].sum())
            normal_count = int(total_logs - anomaly_count)
            threat_pct = (anomaly_count / total_logs * 100) if total_logs else 0.0

            anomalies_df = scored_df[scored_df["predicted_anomaly"] == 1].copy()

            progress.progress(100, text="Completed.")
            time.sleep(0.2)
            progress.empty()

        # Top metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            metric_card("Total Log Entries", f"{total_logs:,}", "Events analyzed", "border-info")
        with col2:
            metric_card("Normal Events", f"{normal_count:,}", "Benign activity", "border-normal")
        with col3:
            metric_card("Anomalies Detected", f"{anomaly_count:,}", "Potential threats", "border-anomaly")
        with col4:
            delta_text = f"{threat_pct - (contamination_threshold * 100):+.1f}% vs threshold"
            metric_card("Threat Level %", f"{threat_pct:.2f}%", delta_text, "border-warning")

        st.markdown("### Detection Analytics")
        chart_col1, chart_col2 = st.columns(2)
        chart_col3, chart_col4 = st.columns(2)

        # Chart 1: Pie - normal vs anomaly
        pie_df = pd.DataFrame(
            {
                "Category": ["Normal", "Anomaly"],
                "Count": [normal_count, anomaly_count],
            }
        )
        fig_pie = px.pie(
            pie_df,
            names="Category",
            values="Count",
            color="Category",
            color_discrete_map={"Normal": NORMAL_COLOR, "Anomaly": ANOMALY_COLOR},
            title="Normal vs Anomaly Distribution",
            hole=0.35,
        )
        fig_pie.update_layout(template="plotly_dark")

        # Chart 2: Activity by hour grouped
        hour_df = (
            scored_df.groupby(["hour", "predicted_anomaly"]).size().reset_index(name="count")
        )
        hour_df["Type"] = hour_df["predicted_anomaly"].map({0: "Normal", 1: "Anomaly"})
        fig_hour = px.bar(
            hour_df,
            x="hour",
            y="count",
            color="Type",
            barmode="group",
            title="Activity by Hour (Normal vs Anomaly)",
            color_discrete_map={"Normal": NORMAL_COLOR, "Anomaly": ANOMALY_COLOR},
        )
        fig_hour.update_layout(template="plotly_dark", xaxis_title="Hour", yaxis_title="Event Count")

        # Chart 3: Top 10 suspicious IPs
        top_ips = (
            anomalies_df["ip_address"]
            .value_counts()
            .head(10)
            .sort_values(ascending=True)
            .reset_index()
        )
        top_ips.columns = ["ip_address", "count"]
        fig_ips = px.bar(
            top_ips,
            x="count",
            y="ip_address",
            orientation="h",
            title="Top 10 Suspicious IP Addresses",
            color_discrete_sequence=[ANOMALY_COLOR],
        )
        fig_ips.update_layout(template="plotly_dark", xaxis_title="Anomaly Count", yaxis_title="IP Address")

        # Chart 4: Top suspicious actions
        top_actions = (
            anomalies_df["action"]
            .value_counts()
            .head(10)
            .sort_values(ascending=True)
            .reset_index()
        )
        top_actions.columns = ["action", "count"]
        fig_actions = px.bar(
            top_actions,
            x="count",
            y="action",
            orientation="h",
            title="Top Suspicious Actions",
            color_discrete_sequence=[ANOMALY_COLOR],
        )
        fig_actions.update_layout(template="plotly_dark", xaxis_title="Anomaly Count", yaxis_title="Action")

        with chart_col1:
            st.plotly_chart(fig_pie, use_container_width=True)
        with chart_col2:
            st.plotly_chart(fig_hour, use_container_width=True)
        with chart_col3:
            st.plotly_chart(fig_ips, use_container_width=True)
        with chart_col4:
            st.plotly_chart(fig_actions, use_container_width=True)

        # Anomaly table
        st.markdown("### Top 20 Anomalous Log Entries")
        table_cols = [
            "timestamp",
            "ip_address",
            "user_id",
            "action",
            "bytes_transferred",
            "response_time_ms",
            "failed_attempts",
        ]
        anomaly_table = (
            anomalies_df[table_cols]
            .sort_values("failed_attempts", ascending=False)
            .head(20)
        )

        def red_highlight(_):
            return ["background-color: rgba(231, 76, 60, 0.25); color: #f8fafc"] * len(table_cols)

        st.dataframe(
            anomaly_table.style.apply(red_highlight, axis=1),
            use_container_width=True,
            hide_index=True,
        )

        # Threat summary section
        st.markdown("### Threat Summary")
        s_col1, s_col2, s_col3 = st.columns(3)

        most_ip = anomalies_df["ip_address"].mode().iloc[0] if not anomalies_df.empty else "N/A"
        common_attack = anomalies_df["action"].mode().iloc[0] if not anomalies_df.empty else "N/A"
        peak_hour = (
            int(anomalies_df["hour"].mode().iloc[0]) if not anomalies_df.empty else "N/A"
        )
        unique_suspicious_ips = int(anomalies_df["ip_address"].nunique()) if not anomalies_df.empty else 0
        risk_label, risk_class = get_risk_label(threat_pct)

        with s_col1:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("#### Key Findings")
            st.write(f"- Most suspicious IP: `{most_ip}`")
            st.write(f"- Most common attack type: `{common_attack}`")
            st.write(f"- Peak attack hour: `{peak_hour}`")
            st.markdown("</div>", unsafe_allow_html=True)

        with s_col2:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("#### Risk Assessment")
            st.markdown(
                f'Threat status: <span class="{risk_class}">{risk_label}</span>',
                unsafe_allow_html=True,
            )
            st.write(f"- Total anomalies: `{anomaly_count}`")
            st.write(f"- Unique suspicious IPs: `{unique_suspicious_ips}`")
            st.write(f"- Threat level: `{threat_pct:.2f}%`")
            st.markdown("</div>", unsafe_allow_html=True)

        with s_col3:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("#### Recommendations")
            st.write(f"- Block top suspicious IP: `{most_ip}`")
            st.write("- Enable 2FA for all privileged accounts")
            st.write("- Monitor after-hours activity patterns")
            st.write("- Review repeated failed login attempts")
            st.markdown("</div>", unsafe_allow_html=True)

        if threat_pct < 10:
            st.balloons()
            st.success("Threat level is low. System appears stable and safe.")

        st.caption(f"Last scan: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as exc:
        st.error(f"Detection failed: {exc}")
else:
    st.markdown(
        """
        <div class="section-card">
            Use the sidebar controls to select a data source, adjust threshold,
            and click <b>Run Detection</b> to start analysis.
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# Footer
# -----------------------------
st.markdown(
    """
    <div class="footer">
        Network Log Anomaly Detector | Cybersecurity + Data Science Portfolio Project
        | Sudeep | Biratnagar, Nepal | 2025
    </div>
    """,
    unsafe_allow_html=True,
)
