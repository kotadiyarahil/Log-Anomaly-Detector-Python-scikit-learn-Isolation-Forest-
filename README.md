# 🔐 Network Log Anomaly Detector

An AI-powered cybersecurity dashboard that detects suspicious 
network activity from log files using Machine Learning.

🚀 **Live App:** (https://log-anomaly-detector-7md7kwf7seys8whfa48xxt.streamlit.app/)

---

## 📌 Project Overview

Security teams deal with thousands of log entries daily — manually 
reviewing them is impossible. This tool uses the **Isolation Forest** 
ML algorithm to automatically detect anomalous behavior such as:

- 🔴 Brute force attacks
- 🔴 Data exfiltration attempts
- 🔴 Privilege escalation
- 🔴 Port scanning
- 🔴 Suspicious after-hours activity

---

## 🎯 Results

| Metric | Value |
|--------|-------|
| Algorithm | Isolation Forest |
| Dataset Size | 1,000 log entries |
| Normal Events | 900 |
| Anomalies | 100 |
| Anomaly Detection Rate | ~90% |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Scikit-learn | Isolation Forest ML model |
| Pandas & NumPy | Data processing |
| Plotly | Interactive charts |
| Streamlit | Web dashboard deployment |
| Jupyter Notebook | Model development |

---

## 📂 Project Structure

```
log-anomaly-detector/
├── app.py                    # Streamlit dashboard
├── generate_logs.ipynb       # Log data generation
├── model.ipynb               # ML model training
├── anomaly_model.pkl         # Trained Isolation Forest
├── le_ip.pkl                 # IP address encoder
├── le_action.pkl             # Action encoder
├── le_user.pkl               # User ID encoder
├── features.pkl              # Feature column names
├── network_logs.csv          # Generated log dataset
└── requirements.txt          # Dependencies
```

---

## 🔍 Key Features

- 📊 **Real-time Dashboard** — interactive charts and metrics
- 🌐 **IP Analysis** — identifies top suspicious IP addresses
- 🕐 **Time Analysis** — detects unusual after-hours activity
- ⚡ **Action Detection** — flags brute force and data theft attempts
- 💡 **Threat Summary** — risk level + actionable recommendations
- 📁 **File Upload** — analyze your own log files

---

## 📊 Dashboard Sections

### 1. Top Metrics
- Total log entries
- Normal vs anomalous events
- Overall threat level percentage

### 2. Visual Charts
- Pie chart: Normal vs Anomaly distribution
- Bar chart: Activity by hour of day
- Top 10 suspicious IP addresses
- Most common attack types

### 3. Anomaly Log Table
- Detailed view of all detected threats
- Sorted by failed login attempts

### 4. Threat Summary
- Key findings
- Risk assessment (Low/Medium/Critical)
- Security recommendations

---

## 🚀 How to Run Locally

```bash
git clone https://github.com/PeakCoder-Here/log-anomaly-detector.git
cd log-anomaly-detector
pip install -r requirements.txt
python -m streamlit run app.py
```

---

## 🔗 Related Projects

- [Customer Churn Predictor](https://github.com/PeakCoder-Here/customer-churn-predictor) — ML model to predict customer churn

---

## 👤 Author

**Sudeep** — CS Student | Biratnagar, Nepal
🌐 Live App: [log-anomaly-detector.streamlit.app](#)
📁 GitHub: [@PeakCoder-Here](https://github.com/PeakCoder-Here)
