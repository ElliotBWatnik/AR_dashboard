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
    # Bulletproof file reader
    df = None
    encodings_to_try = ['utf-8', 'utf-16', 'latin1']
    
    for enc in encodings_to_try:
        try:
            uploaded_file.seek(0)
            temp_df = pd.read_csv(uploaded_file, sep='\t', encoding=enc)
            # If it didn't split into columns properly, try comma separation
            if len(temp_df.columns) < 5:
                uploaded_file.seek(0)
                temp_df = pd.read_csv(uploaded_file, encoding=enc)
            
            if len(temp_df.columns) >= 5:
                df = temp_df
                break  # We successfully read it!
        except Exception:
            continue  # Catch ALL errors and try the next encoding
            
    if df is None:
        st.error("🚨 Critical Error: Could not read the file. Please ensure it is a standard CSV or TSV.")
        st.stop()
        
    # --- 1.5 DEBUG PREVIEW ---
    with st.expander("🔍 Click here to view raw data preview & columns"):
        st.write(f"**Detected Columns:** {df.columns.tolist()}")
        st.dataframe(df.head())

    # --- 2. DATA PREPARATION ---
    try:
        # Standardize missing values
        df = df.fillna(0)
        
        # Ensure dates and numbers are the right format
        df['Order Date'] = pd.to_datetime(df['Order Date'])
        df['Month'] = df['Order Date'].dt.to_period('M').astype(str)
        
        # Convert necessary columns to numeric just in case they loaded as text
        for col in ['Success Order Count', 'Recovery Order Count', 'Order Count']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
        df['Acceptance Orders'] = df['Success Order Count'] + df['Recovery Order Count']
        
    except KeyError as e:
        st.error(f"🚨 Missing Column Error: Your dataset is missing the column {e}. Please check the exact spelling in your file.")
        st.stop()
    except Exception as e:
        st.error(f"🚨 Data Preparation Error: {e}")
        st.stop()

    # --- 3. CASCADING FILTERS ---
    st.sidebar.header("Filters")
    metric_choice = st.sidebar.selectbox("Select Metric for Trend Chart", ['Acceptance Rate', 'Success Rate', 'Recovery Rate'])
    
    # Safely create filters (handles cases where columns might be missing or entirely NaN)
    try:
        if 'data_source' in df.columns:
            sources = ['All'] + list(df['data_source'].astype(str).unique())
            sel_source = st.sidebar.selectbox("Data Source", sources)
            if sel_source != 'All': df = df[df['data_source'] == sel_source]
            
        if 'Country' in df.columns:
            countries = ['All'] + list(df['Country'].astype(str).unique())
            sel_country = st.sidebar.selectbox("Country", countries)
            if sel_country != 'All': df = df[df['Country'] == sel_country]
            
        if 'CC VS APM' in df.columns:
            cc_apm = ['All'] + list(df['CC VS APM'].astype(str).unique())
            sel_cc = st.sidebar.selectbox("CC VS APM", cc_apm)
            if sel_cc != 'All': df = df[df['CC VS APM'] == sel_cc]
            
        if 'First Payment Method' in df.columns:
            pms = ['All'] + list(df['First Payment Method'].astype(str).unique())
            sel_pm = st.sidebar.selectbox("First Payment Method", pms)
            if sel_pm != 'All': df = df[df['First Payment Method'] == sel_pm]
    except Exception as e:
        st.warning(f"Filter error: {e}")

    # --- 4. TREND CHART ---
    st.subheader(f"Monthly {metric_choice} Trend")
    trend_df = df.groupby('Month')[['Order Count', 'Success Order Count', 'Recovery Order Count', 'Acceptance Orders']].sum().reset_index()
    
    trend_df['Success Rate'] = np.where(trend_df['Order Count'] > 0, trend_df['Success Order Count'] / trend_df['Order Count'], 0)
    trend_df['Recovery Rate'] = np.where(trend_df['Order Count'] > 0, trend_df['Recovery Order Count'] / trend_df['Order Count'], 0)
    trend_df['Acceptance Rate'] = np.where(trend_df['Order Count'] > 0, trend_df['Acceptance Orders'] / trend_df['Order Count'], 0)
    
    fig = px.line(trend_df, x='Month', y=metric_choice, markers=True)
    fig.update_layout(yaxis_tickformat='.2%')
    st.plotly_chart(fig, use_container_width=True)

    # --- 5. ATTRIBUTION TABLE ---
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
        merged['W_gt_prev'] = np.where(merged['GT_Orders_prev'] > 0, merged['Order Count_prev'] / merged['GT_Orders_prev'], 0)
        
        merged['Subtotal_Mix_Impact'] = (merged['W_sub_curr'] - merged['W_sub_prev']) * merged['AR_prev']
        merged['Subtotal_Rate_Impact'] = merged['W_sub_curr'] * merged['MoM_Delta']
        merged['GT_Mix_Impact'] = (merged['W_gt_curr'] - merged['W_gt_prev']) * merged['AR_prev']
        merged['GT_Rate_Impact'] = merged['W_gt_curr'] * merged['MoM_Delta']
        
        display_df = merged[['data_source', 'Country', 'First Payment Method', 'AR_curr', 'MoM_Delta', 
                             'Subtotal_Mix_Impact', 'Subtotal_Rate_Impact', 'GT_Mix_Impact', 'GT_Rate_Impact']].copy()
        
        display_df.rename(columns={
            'data_source': 'Entity', 'AR_curr': 'Latest AR', 'MoM_Delta': 'MoM Delta',
            'Subtotal_Mix_Impact': 'Mix Impact (Country)', 'Subtotal_Rate_Impact': 'Rate Impact (Country)',
            'GT_Mix_Impact': 'Mix Impact (Entity)', 'GT_Rate_Impact': 'Rate Impact (Entity)'
        }, inplace=True)
        
        display_df = display_df.sort_values(by=['Entity', 'Country', 'First Payment Method'])
        format_dict = {col: "{:.2%}" for col in display_df.columns if 'AR' in col or 'Delta' in col or 'Impact' in col}
        st.dataframe(display_df.style.format(format_dict), height=400, use_container_width=True)
        
       # --- 6. GEMINI AI INTEGRATION ---
        st.divider()
        st.subheader("🤖 AI Performance Insights")
        api_key = st.text_input("Enter Gemini API Key to generate insights:", type="password")

        
        if st.button("Generate AI Insights"):
            if not api_key:
                st.warning("Please enter your Gemini API key.")
            else:
                with st.spinner("Gemini is analyzing the performance drivers..."):
                    try:
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel('gemini-pro-latest')
                        
                        merged['GT_Total_Impact'] = merged['GT_Mix_Impact'] + merged['GT_Rate_Impact']
                        top_pos = merged.sort_values(by='GT_Total_Impact', ascending=False).head(3)
                        top_neg = merged.sort_values(by='GT_Total_Impact').head(3)
                        
                        global_ar_curr = merged['Acceptance Orders_curr'].sum() / merged['Order Count_curr'].sum() if merged['Order Count_curr'].sum() else 0
                        global_ar_prev = merged['Acceptance Orders_prev'].sum() / merged['Order Count_prev'].sum() if merged['Order Count_prev'].sum() else 0
                        global_delta = global_ar_curr - global_ar_prev
                        
                        prompt = f"""
                        You are a payments performance analyst. Review the following Month-over-Month payment acceptance rate data.
                        
                        Context:
                        - Global Acceptance rate shifted by {global_delta:.4%}.
                        
                        Top 3 Positive Drivers (helped the global rate):
                        {top_pos[['Country', 'First Payment Method', 'GT_Mix_Impact', 'GT_Rate_Impact']].to_string()}
                        
                        Top 3 Negative Drivers (dragged the global rate down):
                        {top_neg[['Country', 'First Payment Method', 'GT_Mix_Impact', 'GT_Rate_Impact']].to_string()}
                        
                        Task:
                        Write a concise, professional executive summary for leadership. 
                        1. Provide a brief 2-sentence overview.
                        2. Highlight the main positive and negative drivers, explaining if it was due to Mix (volume share shifting) or Rate (actual approval rate dropping/rising).
                        3. Provide 3 actionable recommendations for the payments team to investigate.
                        Keep the tone analytical and avoid fluff.
                        """
                        response = model.generate_content(prompt)
                        st.success("Analysis Complete!")
                        st.markdown(response.text)
                    except Exception as e:
                        st.error(f"An error occurred with the AI generation: {e}")
    else:
        st.info("Please upload data with at least two distinct months to view MoM attribution.")
