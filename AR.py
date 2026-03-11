import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import google.generativeai as genai

st.set_page_config(page_title="Acceptance Rate Dashboard", layout="wide")
st.title("Acceptance Rate Performance Dashboard")

# --- 1. FILE UPLOAD & PARSING ---
uploaded_file = st.file_uploader("Upload your dataset (CSV or TSV)", type=['csv', 'tsv', 'txt'])

if uploaded_file is not None:
    # Try multiple encodings to handle Excel/System exports (fixes the 0xff error)
    encodings = ['utf-8', 'utf-16', 'latin1']
    df = None
    
    for enc in encodings:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep='\t', encoding=enc)
            if len(df.columns) < 5:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
            
    if df is None:
        st.error("Could not read the file. Unsupported encoding.")
        st.stop()
        
    # Data Preparation
    try:
        df['Order Date'] = pd.to_datetime(df['Order Date'])
        df['Month'] = df['Order Date'].dt.to_period('M').astype(str)
        df['Acceptance Orders'] = df['Success Order Count'] + df['Recovery Order Count']
    except Exception as e:
        st.error(f"Error preparing data: {e}. Please ensure 'Order Date', 'Success Order Count', and 'Recovery Order Count' exist.")
        st.stop()

    # --- 2. CASCADING FILTERS ---
    st.sidebar.header("Filters")
    metric_choice = st.sidebar.selectbox("Select Metric for Trend Chart", ['Acceptance Rate', 'Success Rate', 'Recovery Rate'])
    
    sources = ['All'] + list(df['data_source'].dropna().unique())
    sel_source = st.sidebar.selectbox("Data Source", sources)
    if sel_source != 'All': df = df[df['data_source'] == sel_source]
        
    countries = ['All'] + list(df['Country'].dropna().unique())
    sel_country = st.sidebar.selectbox("Country", countries)
    if sel_country != 'All': df = df[df['Country'] == sel_country]
        
    cc_apm = ['All'] + list(df['CC VS APM'].dropna().unique())
    sel_cc = st.sidebar.selectbox("CC VS APM", cc_apm)
    if sel_cc != 'All': df = df[df['CC VS APM'] == sel_cc]
        
    pms = ['All'] + list(df['First Payment Method'].dropna().unique())
    sel_pm = st.sidebar.selectbox("First Payment Method", pms)
    if sel_pm != 'All': df = df[df['First Payment Method'] == sel_pm]

    # --- 3. TREND CHART ---
    st.subheader(f"Monthly {metric_choice} Trend")
    trend_df = df.groupby('Month')[['Order Count', 'Success Order Count', 'Recovery Order Count', 'Acceptance Orders']].sum().reset_index()
    
    trend_df['Success Rate'] = np.where(trend_df['Order Count'] > 0, trend_df['Success Order Count'] / trend_df['Order Count'], 0)
    trend_df['Recovery Rate'] = np.where(trend_df['Order Count'] > 0, trend_df['Recovery Order Count'] / trend_df['Order Count'], 0)
    trend_df['Acceptance Rate'] = np.where(trend_df['Order Count'] > 0, trend_df['Acceptance Orders'] / trend_df['Order Count'], 0)
    
    fig = px.line(trend_df, x='Month', y=metric_choice, markers=True)
    fig.update_layout(yaxis_tickformat='.2%')
    st.plotly_chart(fig, use_container_width=True)

    # --- 4. ATTRIBUTION TABLE ---
    st.subheader("Performance vs Last Month (Mix & Yield Attribution)")
    months = sorted(df['Month'].unique())
    
    if len(months) >= 2:
        curr_month = months[-1]
        prev_month = months[-2]
        st.write(f"**Comparing {curr_month} vs {prev_month}**")
        
        agg_cols = ['Order Count', 'Success Order Count', 'Recovery Order Count', 'Acceptance Orders']
        df_curr = df[df['Month'] == curr_month].groupby(['data_source', 'Country', 'First Payment Method'])[agg_cols].sum().reset_index()
        df_prev = df[df['Month'] == prev_month].groupby(['data_source', 'Country', 'First Payment Method'])[agg_cols].sum().reset_index()
        
        merged = pd.merge(df_curr, df_prev, on=['data_source', 'Country', 'First Payment Method'], suffixes=('_curr', '_prev'), how='outer').fillna(0)
        
        merged['AR_curr'] = np.where(merged['Order Count_curr'] > 0, merged['Acceptance Orders_curr'] / merged['Order Count_curr'], 0)
        merged['AR_prev'] = np.where(merged['Order Count_prev'] > 0, merged['Acceptance Orders_prev'] / merged['Order Count_prev'], 0)
        merged['MoM_Delta'] = merged['AR_curr'] - merged['AR_prev']
        
        merged['Country_Orders_curr'] = merged.groupby(['data_source', 'Country'])['Order Count_curr'].transform('sum')
        merged['Country_Orders_prev'] = merged.groupby(['data_source', 'Country'])['Order Count_prev'].transform('sum')
        merged['W_sub_curr'] = np.where(merged['Country_Orders_curr'] > 0, merged['Order Count_curr'] / merged['Country_Orders_curr'], 0)
        merged['W_sub_prev'] = np.where(merged['Country_Orders_prev'] > 0, merged['Order Count_prev'] / merged['Country_Orders_prev'], 0)
        
        merged['GT_Orders_curr'] = merged.groupby('data_source')['Order Count_curr'].transform('sum')
        merged['GT_Orders_prev'] = merged.groupby('data_source')['Order Count_prev'].transform('sum')
        merged['W_gt_curr'] = np.where(merged['GT_Orders_curr'] > 0, merged['Order Count_curr'] / merged['GT_Orders_curr'], 0)
        merged['W_gt_prev'] = np.where(merged['GT
