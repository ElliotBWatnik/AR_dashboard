{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import streamlit as st\
import pandas as pd\
import numpy as np\
import plotly.express as px\
import google.generativeai as genai\
\
st.set_page_config(page_title="Acceptance Rate Dashboard", layout="wide")\
st.title("Acceptance Rate Performance Dashboard")\
\
# --- 1. FILE UPLOAD & PARSING ---\
uploaded_file = st.file_uploader("Upload your dataset (CSV or TSV)", type=['csv', 'tsv', 'txt'])\
\
if uploaded_file is not None:\
    # Try multiple encodings to handle Excel/System exports (fixes the 0xff error)\
    encodings = ['utf-8', 'utf-16', 'latin1']\
    df = None\
    \
    for enc in encodings:\
        try:\
            uploaded_file.seek(0)\
            df = pd.read_csv(uploaded_file, sep='\\t', encoding=enc)\
            if len(df.columns) < 5:\
                uploaded_file.seek(0)\
                df = pd.read_csv(uploaded_file, encoding=enc)\
            break\
        except UnicodeDecodeError:\
            continue\
            \
    if df is None:\
        st.error("Could not read the file. Unsupported encoding.")\
        st.stop()\
        \
    # Data Preparation\
    try:\
        df['Order Date'] = pd.to_datetime(df['Order Date'])\
        df['Month'] = df['Order Date'].dt.to_period('M').astype(str)\
        df['Acceptance Orders'] = df['Success Order Count'] + df['Recovery Order Count']\
    except Exception as e:\
        st.error(f"Error preparing data: \{e\}. Please ensure 'Order Date', 'Success Order Count', and 'Recovery Order Count' exist.")\
        st.stop()\
\
    # --- 2. CASCADING FILTERS ---\
    st.sidebar.header("Filters")\
    metric_choice = st.sidebar.selectbox("Select Metric for Trend Chart", ['Acceptance Rate', 'Success Rate', 'Recovery Rate'])\
    \
    sources = ['All'] + list(df['data_source'].dropna().unique())\
    sel_source = st.sidebar.selectbox("Data Source", sources)\
    if sel_source != 'All': df = df[df['data_source'] == sel_source]\
        \
    countries = ['All'] + list(df['Country'].dropna().unique())\
    sel_country = st.sidebar.selectbox("Country", countries)\
    if sel_country != 'All': df = df[df['Country'] == sel_country]\
        \
    cc_apm = ['All'] + list(df['CC VS APM'].dropna().unique())\
    sel_cc = st.sidebar.selectbox("CC VS APM", cc_apm)\
    if sel_cc != 'All': df = df[df['CC VS APM'] == sel_cc]\
        \
    pms = ['All'] + list(df['First Payment Method'].dropna().unique())\
    sel_pm = st.sidebar.selectbox("First Payment Method", pms)\
    if sel_pm != 'All': df = df[df['First Payment Method'] == sel_pm]\
\
    # --- 3. TREND CHART ---\
    st.subheader(f"Monthly \{metric_choice\} Trend")\
    trend_df = df.groupby('Month')[['Order Count', 'Success Order Count', 'Recovery Order Count', 'Acceptance Orders']].sum().reset_index()\
    \
    trend_df['Success Rate'] = np.where(trend_df['Order Count'] > 0, trend_df['Success Order Count'] / trend_df['Order Count'], 0)\
    trend_df['Recovery Rate'] = np.where(trend_df['Order Count'] > 0, trend_df['Recovery Order Count'] / trend_df['Order Count'], 0)\
    trend_df['Acceptance Rate'] = np.where(trend_df['Order Count'] > 0, trend_df['Acceptance Orders'] / trend_df['Order Count'], 0)\
    \
    fig = px.line(trend_df, x='Month', y=metric_choice, markers=True)\
    fig.update_layout(yaxis_tickformat='.2%')\
    st.plotly_chart(fig, use_container_width=True)\
\
    # --- 4. ATTRIBUTION TABLE ---\
    st.subheader("Performance vs Last Month (Mix & Yield Attribution)")\
    months = sorted(df['Month'].unique())\
    \
    if len(months) >= 2:\
        curr_month = months[-1]\
        prev_month = months[-2]\
        st.write(f"**Comparing \{curr_month\} vs \{prev_month\}**")\
        \
        agg_cols = ['Order Count', 'Success Order Count', 'Recovery Order Count', 'Acceptance Orders']\
        df_curr = df[df['Month'] == curr_month].groupby(['data_source', 'Country', 'First Payment Method'])[agg_cols].sum().reset_index()\
        df_prev = df[df['Month'] == prev_month].groupby(['data_source', 'Country', 'First Payment Method'])[agg_cols].sum().reset_index()\
        \
        merged = pd.merge(df_curr, df_prev, on=['data_source', 'Country', 'First Payment Method'], suffixes=('_curr', '_prev'), how='outer').fillna(0)\
        \
        merged['AR_curr'] = np.where(merged['Order Count_curr'] > 0, merged['Acceptance Orders_curr'] / merged['Order Count_curr'], 0)\
        merged['AR_prev'] = np.where(merged['Order Count_prev'] > 0, merged['Acceptance Orders_prev'] / merged['Order Count_prev'], 0)\
        merged['MoM_Delta'] = merged['AR_curr'] - merged['AR_prev']\
        \
        merged['Country_Orders_curr'] = merged.groupby(['data_source', 'Country'])['Order Count_curr'].transform('sum')\
        merged['Country_Orders_prev'] = merged.groupby(['data_source', 'Country'])['Order Count_prev'].transform('sum')\
        merged['W_sub_curr'] = np.where(merged['Country_Orders_curr'] > 0, merged['Order Count_curr'] / merged['Country_Orders_curr'], 0)\
        merged['W_sub_prev'] = np.where(merged['Country_Orders_prev'] > 0, merged['Order Count_prev'] / merged['Country_Orders_prev'], 0)\
        \
        merged['GT_Orders_curr'] = merged.groupby('data_source')['Order Count_curr'].transform('sum')\
        merged['GT_Orders_prev'] = merged.groupby('data_source')['Order Count_prev'].transform('sum')\
        merged['W_gt_curr'] = np.where(merged['GT_Orders_curr'] > 0, merged['Order Count_curr'] / merged['GT_Orders_curr'], 0)\
        merged['W_gt_prev'] = np.where(merged['GT_Orders_prev'] > 0, merged['Order Count_prev'] / merged['GT_Orders_prev'], 0)\
        \
        merged['Subtotal_Mix_Impact'] = (merged['W_sub_curr'] - merged['W_sub_prev']) * merged['AR_prev']\
        merged['Subtotal_Rate_Impact'] = merged['W_sub_curr'] * merged['MoM_Delta']\
        merged['GT_Mix_Impact'] = (merged['W_gt_curr'] - merged['W_gt_prev']) * merged['AR_prev']\
        merged['GT_Rate_Impact'] = merged['W_gt_curr'] * merged['MoM_Delta']\
        \
        display_df = merged[['data_source', 'Country', 'First Payment Method', 'AR_curr', 'MoM_Delta', \
                             'Subtotal_Mix_Impact', 'Subtotal_Rate_Impact', 'GT_Mix_Impact', 'GT_Rate_Impact']].copy()\
        \
        display_df.rename(columns=\{\
            'data_source': 'Entity', 'AR_curr': 'Latest AR', 'MoM_Delta': 'MoM Delta',\
            'Subtotal_Mix_Impact': 'Mix Impact (Country)', 'Subtotal_Rate_Impact': 'Rate Impact (Country)',\
            'GT_Mix_Impact': 'Mix Impact (Entity)', 'GT_Rate_Impact': 'Rate Impact (Entity)'\
        \}, inplace=True)\
        \
        display_df = display_df.sort_values(by=['Entity', 'Country', 'First Payment Method'])\
        format_dict = \{col: "\{:.2%\}" for col in display_df.columns if 'AR' in col or 'Delta' in col or 'Impact' in col\}\
        st.dataframe(display_df.style.format(format_dict), height=400, use_container_width=True)\
        \
        # --- 5. GEMINI AI INTEGRATION ---\
        st.divider()\
        st.subheader("\uc0\u55358 \u56598  AI Performance Insights")\
        api_key = st.text_input("Enter Gemini API Key to generate insights:", type="password")\
        \
        if st.button("Generate AI Insights"):\
            if not api_key:\
                st.warning("Please enter your Gemini API key.")\
            else:\
                with st.spinner("Gemini is analyzing the performance drivers..."):\
                    try:\
                        genai.configure(api_key=api_key)\
                        model = genai.GenerativeModel('gemini-1.5-flash')\
                        \
                        merged['GT_Total_Impact'] = merged['GT_Mix_Impact'] + merged['GT_Rate_Impact']\
                        top_pos = merged.sort_values(by='GT_Total_Impact', ascending=False).head(3)\
                        top_neg = merged.sort_values(by='GT_Total_Impact').head(3)\
                        \
                        global_ar_curr = merged['Acceptance Orders_curr'].sum() / merged['Order Count_curr'].sum() if merged['Order Count_curr'].sum() else 0\
                        global_ar_prev = merged['Acceptance Orders_prev'].sum() / merged['Order Count_prev'].sum() if merged['Order Count_prev'].sum() else 0\
                        global_delta = global_ar_curr - global_ar_prev\
                        \
                        prompt = f"""\
                        You are a payments performance analyst. Review the following Month-over-Month payment acceptance rate data.\
                        \
                        Context:\
                        - Global Acceptance rate shifted by \{global_delta:.4%\}.\
                        \
                        Top 3 Positive Drivers (helped the global rate):\
                        \{top_pos[['Country', 'First Payment Method', 'GT_Mix_Impact', 'GT_Rate_Impact']].to_string()\}\
                        \
                        Top 3 Negative Drivers (dragged the global rate down):\
                        \{top_neg[['Country', 'First Payment Method', 'GT_Mix_Impact', 'GT_Rate_Impact']].to_string()\}\
                        \
                        Task:\
                        Write a concise, professional executive summary for leadership. \
                        1. Provide a brief 2-sentence overview.\
                        2. Highlight the main positive and negative drivers, explaining if it was due to Mix (volume share shifting) or Rate (actual approval rate dropping/rising).\
                        3. Provide 3 actionable recommendations for the payments team to investigate.\
                        Keep the tone analytical and avoid fluff.\
                        """\
                        response = model.generate_content(prompt)\
                        st.success("Analysis Complete!")\
                        st.markdown(response.text)\
                    except Exception as e:\
                        st.error(f"An error occurred with the AI generation: \{e\}")\
    else:\
        st.info("Please upload data with at least two distinct months to view MoM attribution.")\
else:\
    st.info("Awaiting file upload...")}