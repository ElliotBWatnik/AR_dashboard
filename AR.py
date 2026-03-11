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
    # Try multiple encodings to handle Excel/System exports
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
    trend_df = df.groupby
