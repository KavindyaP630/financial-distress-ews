
import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Early Warning System — Capital Goods Sector CSE",
    page_icon="⚠",
    layout="wide"
)

@st.cache_resource
def load_models():
    with open("lr_model.pkl", "rb") as f:
        lr = pickle.load(f)
    with open("rf_model.pkl", "rb") as f:
        rf = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        sc = pickle.load(f)
    return lr, rf, sc

lr_model, rf_model, scaler = load_models()

st.markdown("""
    <div style="background-color:#1F3864; padding:20px;
                border-radius:8px; margin-bottom:20px">
        <h1 style="color:white; margin:0; font-size:26px">
            ⚠ Early Warning System — Corporate Financial Distress
        </h1>
        <p style="color:#D6E4F0; margin:6px 0 0 0; font-size:14px">
            Capital Goods Sector | Colombo Stock Exchange |
            Powered by Logistic Regression and Random Forest
        </p>
    </div>
""", unsafe_allow_html=True)

st.sidebar.markdown("## About This Tool")
st.sidebar.markdown("""
This early warning system predicts the probability of financial distress
for Capital Goods sector companies listed on the **Colombo Stock Exchange**.

**How to use:**
1. Enter the company name and financial year
2. Input the five financial ratios from the annual report
3. Enter the GDP growth rate for that year
4. Click **Assess Distress Risk**
5. Review the risk classification and outputs

**Models used:**
- Logistic Regression (primary signal — higher recall)
- Random Forest (confirmation signal — higher precision)

**Risk tiers:**
- 🟢 Low Risk: probability < 30%
- 🟡 Moderate Risk: probability 30%–60%
- 🔴 High Risk: probability > 60%
""")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Research:**
Early Warning System Based on Financial Ratios
to Predict Corporate Financial Distress —
Capital Goods Sector, Sri Lanka
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### Sri Lanka GDP Growth Reference")
gdp_ref = pd.DataFrame({
    "Year": [2018, 2019, 2020, 2021, 2022, 2023, 2024],
    "GDP Growth (%)": [3.3, 2.3, -3.5, 3.3, -7.8, -2.3, 5.0]
})
st.sidebar.dataframe(gdp_ref, hide_index=True, use_container_width=True)
st.sidebar.caption("Source: Central Bank of Sri Lanka. "
                   "Verify latest values at cbsl.gov.lk")

st.markdown("### Company Information")
col_info1, col_info2 = st.columns(2)
with col_info1:
    company_name = st.text_input(
        "Company Name (for reference only)",
        placeholder="e.g. ACL Cables PLC"
    )
with col_info2:
    assessment_year = st.selectbox(
        "Financial Year of Annual Report (reference only — does not affect prediction)",
        options=[2025, 2024, 2023, 2022, 2021, 2020, 2019, 2018, "Other"]
    )

st.markdown("---")
st.markdown("### Financial Ratio Inputs")
st.markdown(
    "*Enter values directly from the company annual report. "
    "Hover over each label for calculation guidance.*"
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Liquidity & Leverage")

    wc_ta = st.number_input(
        "Working Capital / Total Assets  (WC_TA)",
        min_value=-2.0, max_value=2.0, value=0.10,
        step=0.01, format="%.4f",
        help="Formula: (Current Assets − Current Liabilities) / Total Assets. "
             "Negative values indicate current liabilities exceed current assets — "
             "a liquidity warning signal."
    )
    e_tl = st.number_input(
        "Equity / Total Liabilities  (E_TL)",
        min_value=-1.0, max_value=10.0, value=1.00,
        step=0.01, format="%.4f",
        help="Formula: Total Equity / Total Liabilities. "
             "Values above 1.0 mean equity exceeds all liabilities. "
             "Values below 0.5 indicate high leverage."
    )
    cl_ta = st.number_input(
        "Current Liabilities / Total Assets  (CL_TA)",
        min_value=0.0, max_value=1.5, value=0.35,
        step=0.01, format="%.4f",
        help="Formula: Current Liabilities / Total Assets. "
             "Higher values indicate greater short-term financial burden."
    )

with col2:
    st.markdown("#### Profitability & Cash Flow")

    re_ta = st.number_input(
        "Retained Earnings / Total Assets  (RE_TA)",
        min_value=-2.0, max_value=2.0, value=0.25,
        step=0.01, format="%.4f",
        help="Formula: Retained Earnings / Total Assets. "
             "Negative values indicate accumulated losses — "
             "a long-term financial deterioration signal."
    )
    ocf_td = st.number_input(
        "Operating Cash Flow / Total Debt  (OCF_TD)",
        min_value=-5.0, max_value=20.0, value=0.30,
        step=0.01, format="%.4f",
        help="Formula: Net Cash from Operating Activities / Total Debt. "
             "Negative values mean operations are consuming cash — "
             "a serious distress signal."
    )

st.markdown("---")
st.markdown("### Macroeconomic Context")
st.markdown(
    "*GDP growth rate for the financial year being assessed. "
    "Refer to the sidebar for Sri Lanka historical values.*"
)

gdp_growth = st.number_input(
    "GDP Growth Rate (%)  —  GDP_Growth",
    min_value=-15.0, max_value=15.0, value=3.0,
    step=0.1, format="%.1f",
    help="Annual GDP growth rate (%) for Sri Lanka for the year being assessed. "
         "Negative values indicate economic contraction. "
         "Source: Central Bank of Sri Lanka Annual Report (cbsl.gov.lk). "
         "2022 value: −7.8%  |  2023 value: −2.3%  |  2024 value: 5.0%"
)

st.markdown("---")

assess = st.button(
    "⚡ Assess Distress Risk",
    use_container_width=True,
    type="primary"
)

if assess:
    input_data   = np.array([[wc_ta, re_ta, e_tl, ocf_td, cl_ta, gdp_growth]])
    input_scaled = scaler.transform(input_data)

    lr_prob = lr_model.predict_proba(input_scaled)[0][1]
    rf_prob = rf_model.predict_proba(input_scaled)[0][1]
    lr_pred = int(lr_model.predict(input_scaled)[0])
    rf_pred = int(rf_model.predict(input_scaled)[0])

    if lr_prob < 0.30:
        tier       = "🟢 LOW RISK"
        tier_color = "#27AE60"
        tier_bg    = "#EAFAF1"
    elif lr_prob < 0.60:
        tier       = "🟡 MODERATE RISK"
        tier_color = "#D68910"
        tier_bg    = "#FEFDE7"
    else:
        tier       = "🔴 HIGH RISK"
        tier_color = "#E74C3C"
        tier_bg    = "#FDEDEC"

    high_confidence  = (lr_pred == 1 and rf_pred == 1)
    company_display  = company_name if company_name else "Company"

    st.markdown(f"""
        <div style="background-color:{tier_bg}; border-left:6px solid {tier_color};
                    padding:20px; border-radius:8px; margin-bottom:20px">
            <h2 style="color:{tier_color}; margin:0">{tier}</h2>
            <p style="color:#555; margin:6px 0 0 0; font-size:15px">
                {company_display} | Financial Year {assessment_year}
            </p>
        </div>
    """, unsafe_allow_html=True)

    if high_confidence:
        st.error(
            "⚠ HIGH CONFIDENCE DISTRESS SIGNAL — Both Logistic Regression and "
            "Random Forest independently predict financial distress. This "
            "dual-model agreement significantly reduces the likelihood of a "
            "false alarm."
        )
    elif lr_pred == 1 and rf_pred == 0:
        st.warning(
            "Logistic Regression signals distress but Random Forest does not "
            "confirm. Monitor closely — LR has higher recall but also more "
            "false alarms."
        )
    elif lr_pred == 0 and rf_pred == 1:
        st.warning(
            "Random Forest signals distress but Logistic Regression does not "
            "confirm. RF has higher precision — this signal warrants "
            "investigation."
        )
    else:
        st.success(
            "No distress signal from either model. Financial ratios and "
            "macroeconomic context are within acceptable ranges for this sector."
        )

    st.markdown("### Model Outputs")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Logistic Regression Probability", f"{lr_prob:.1%}",
                  help="Primary signal. Higher recall — better at catching "
                       "all distressed firms.")
    with m2:
        st.metric("Random Forest Probability", f"{rf_prob:.1%}",
                  help="Confirmation signal. Higher precision — more "
                       "confident when it flags distress.")
    with m3:
        st.metric("Combined Average Probability", f"{(lr_prob+rf_prob)/2:.1%}",
                  help="Simple average of both model outputs.")

    st.markdown("### Distress Probability Gauge")

    def colour_bar(p):
        return "#27AE60" if p < 0.30 else "#F39C12" if p < 0.60 else "#E74C3C"

    st.markdown(f"""
        <div style="margin-bottom:12px">
            <div style="display:flex; justify-content:space-between;
                        margin-bottom:4px">
                <span style="font-size:13px; font-weight:bold">
                    Logistic Regression</span>
                <span style="font-size:13px">{lr_prob:.1%}</span>
            </div>
            <div style="background:#E0E0E0; border-radius:8px; height:22px">
                <div style="background:{colour_bar(lr_prob)};
                            width:{lr_prob*100:.1f}%;
                            height:22px; border-radius:8px"></div>
            </div>
        </div>
        <div style="margin-bottom:12px">
            <div style="display:flex; justify-content:space-between;
                        margin-bottom:4px">
                <span style="font-size:13px; font-weight:bold">
                    Random Forest</span>
                <span style="font-size:13px">{rf_prob:.1%}</span>
            </div>
            <div style="background:#E0E0E0; border-radius:8px; height:22px">
                <div style="background:{colour_bar(rf_prob)};
                            width:{rf_prob*100:.1f}%;
                            height:22px; border-radius:8px"></div>
            </div>
        </div>
        <div style="display:flex; gap:20px; margin-top:8px; font-size:12px">
            <span>🟢 Low Risk: &lt; 30%</span>
            <span>🟡 Moderate Risk: 30%–60%</span>
            <span>🔴 High Risk: &gt; 60%</span>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### Ratio & Macro Analysis — Benchmarks vs Input Values")

    benchmarks = {
        "WC_TA": {"value": wc_ta, "healthy_min": 0.0, "healthy_max": 0.7,
                  "label": "Working Capital / Total Assets",
                  "healthy_label": "≥ 0.0 (positive working capital)"},
        "RE_TA": {"value": re_ta, "healthy_min": 0.0, "healthy_max": 0.7,
                  "label": "Retained Earnings / Total Assets",
                  "healthy_label": "≥ 0.0 (positive accumulated earnings)"},
        "E_TL":  {"value": e_tl,  "healthy_min": 0.5, "healthy_max": 10.0,
                  "label": "Equity / Total Liabilities",
                  "healthy_label": "≥ 0.5 (equity at least half of liabilities)"},
        "OCF_TD":{"value": ocf_td,"healthy_min": 0.0, "healthy_max": 5.0,
                  "label": "Operating Cash Flow / Total Debt",
                  "healthy_label": "≥ 0.0 (positive operating cash flow)"},
        "CL_TA": {"value": cl_ta, "healthy_min": 0.0, "healthy_max": 0.5,
                  "label": "Current Liabilities / Total Assets",
                  "healthy_label": "≤ 0.5 (manageable short-term burden)"},
        "GDP_Growth":{"value": gdp_growth,"healthy_min": 0.0,"healthy_max":15.0,
                  "label": "GDP Growth Rate (%)",
                  "healthy_label": "≥ 0.0 (positive economic growth)"},
    }

    ratio_data = []
    for key, b in benchmarks.items():
        v = b["value"]
        is_healthy = b["healthy_min"] <= v <= b["healthy_max"]
        ratio_data.append({
            "Input": b["label"],
            "Value": round(v, 4),
            "Healthy Range": b["healthy_label"],
            "Status": "✅ Healthy" if is_healthy else "⚠ Concern"
        })

    ratio_df = pd.DataFrame(ratio_data)

    def highlight_status(row):
        if "Concern" in row["Status"]:
            return ["","","",
                    "background-color:#FDEDEC;color:#E74C3C;font-weight:bold"]
        return ["","","",
                "background-color:#EAFAF1;color:#27AE60;font-weight:bold"]

    st.dataframe(ratio_df.style.apply(highlight_status, axis=1),
                 use_container_width=True, hide_index=True)

    st.markdown("### Why These Predictors Matter")

    importance_data = pd.DataFrame({
        "Predictor": ["WC_TA", "E_TL", "RE_TA", "OCF_TD",
                      "CL_TA", "GDP_Growth"],
        "Importance (%)": [35.68, 24.41, 18.10, 11.94, 8.59, 1.28]
    }).sort_values("Importance (%)", ascending=True)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3.5))
    colours = ["#E74C3C" if r == "GDP_Growth" else
               "#1F3864" if r == "WC_TA" else "#2E5B9A"
               for r in importance_data["Predictor"]]
    bars = ax.barh(importance_data["Predictor"],
                   importance_data["Importance (%)"],
                   color=colours, edgecolor="white", height=0.5)
    for bar, val in zip(bars, importance_data["Importance (%)"]):
        ax.text(bar.get_width() + 0.2,
                bar.get_y() + bar.get_height()/2,
                f"{val:.2f}%", va="center", fontsize=9)
    ax.set_xlabel("Feature Importance (%)")
    ax.set_title("Random Forest Feature Importances", fontweight="bold", pad=10)
    ax.set_xlim(0, 44)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("### Interpretation Guide")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        **Logistic Regression (Primary Signal)**
        - Distressed class recall: **90.00%**
        - AUC: 0.9286
        - Best for: catching all potentially distressed firms
        - Use when: early warning sensitivity is the priority
        """)
    with col_b:
        st.markdown("""
        **Random Forest (Confirmation Signal)**
        - Distressed class precision: **88.89%**
        - AUC: 0.9829
        - Best for: high-confidence distress confirmation
        - Use when: acting on a signal has significant cost
        """)

    st.markdown("---")
    st.caption(
        "⚠ Disclaimer: This tool is for research and analytical purposes only. "
        "Predictions are based on a model trained on 15 CSE Capital Goods firms "
        "over 2018–2024. Results should not be used as the sole basis for "
        "investment or credit decisions."
    )

else:
    st.info(
        "👆 Enter the five financial ratios and GDP growth rate above, "
        "then click **Assess Distress Risk** to generate the assessment."
    )

    st.markdown("### How to Calculate Each Input")
    guide_data = pd.DataFrame({
        "Input": ["WC_TA", "RE_TA", "E_TL", "OCF_TD",
                  "CL_TA", "GDP_Growth"],
        "Full Name": [
            "Working Capital / Total Assets",
            "Retained Earnings / Total Assets",
            "Equity / Total Liabilities",
            "Operating Cash Flow / Total Debt",
            "Current Liabilities / Total Assets",
            "GDP Growth Rate (%)"
        ],
        "Formula": [
            "(Current Assets − Current Liabilities) / Total Assets",
            "Retained Earnings / Total Assets",
            "Total Equity / Total Liabilities",
            "Net Cash from Operations / Total Borrowings",
            "Current Liabilities / Total Assets",
            "Annual % change in GDP — from CBSL Annual Report"
        ],
        "Source": [
            "Balance Sheet",
            "Balance Sheet — Equity section",
            "Balance Sheet",
            "Cash Flow Statement + Balance Sheet",
            "Balance Sheet",
            "Central Bank of Sri Lanka (cbsl.gov.lk)"
        ]
    })
    st.dataframe(guide_data, use_container_width=True, hide_index=True)
