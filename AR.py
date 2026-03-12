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
            if len(temp_df.columns) < 5:
                uploaded_file.seek(0)
                temp_df = pd.read_csv(uploaded_file, encoding=enc)
            
            if len(temp_df.columns) >= 5:
                df = temp_df
                break
        except Exception:
            continue
            
    if df is None:
        st.error("🚨 Critical Error: Could not read the file. Please ensure it is a standard CSV or TSV.")
        st.stop()

    # --- 2. DATA PREPARATION ---
    try:
        df = df.fillna(0)
        df['Order Date'] = pd.to_datetime(df['Order Date'])
        df['Month'] = df['Order Date'].dt.to_period('M').astype(str)
        
        for col in ['Success Order Count', 'Recovery Order Count', 'Order Count']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
        df['Acceptance Orders'] = df['Success Order Count'] + df['Recovery Order Count']
    except Exception as e:
        st.error(f"🚨 Data Preparation Error: {e}")
        st.stop()

    # --- 3. CASCADING FILTERS ---
    st.sidebar.header("Filters")
    metric_choice = st.sidebar.selectbox("Select Metric for Trend Chart", ['Acceptance Rate', 'Success Rate', 'Recovery Rate'])
    
    try:
        if 'data_source' in df.columns:
            sources = ['All'] + list(df['data_source'].astype(str).unique())
            sel_source = st.sidebar.selectbox("Data Source", sources)
            if sel_source != 'All': df = df[df['data_source'] == sel_source]
            
        # --- FIX: FREEZE THE UNFILTERED ENTITY DATA HERE ---
        # This locks in the true global volume before we start slicing by country/PM
        entity_df = df.copy()
            
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
        
        # Country Denominators (Dynamic based on filters)
        merged['Country_Orders_curr'] = merged.groupby(['data_source', 'Country'])['Order Count_curr'].transform('sum')
        merged['Country_Orders_prev'] = merged.groupby(['data_source', 'Country'])['Order Count_prev'].transform('sum')
        merged['W_sub_curr'] = np.where(merged['Country_Orders_curr'] > 0, merged['Order Count_curr'] / merged['Country_Orders_curr'], 0)
        merged['W_sub_prev'] = np.where(merged['Country_Orders_prev'] > 0, merged['Order Count_prev'] / merged['Country_Orders_prev'], 0)
        
        # Entity Denominators (Locked into the unfiltered entity_df!)
        entity_totals_curr = entity_df[entity_df['Month'] == curr_month].groupby('data_source')['Order Count'].sum().reset_index(name='GT_Orders_curr')
        entity_totals_prev = entity_df[entity_df['Month'] == prev_month].groupby('data_source')['Order Count'].sum().reset_index(name='GT_Orders_prev')
        
        merged = pd.merge(merged, entity_totals_curr, on='data_source', how='left').fillna(0)
        merged = pd.merge(merged, entity_totals_prev, on='data_source', how='left').fillna(0)
        
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
        
        # --- DYNAMIC COLOR HIGHLIGHTING ---
        format_dict = {col: "{:.2%}" for col in display_df.columns if 'AR' in col or 'Delta' in col or 'Impact' in col}
        
        country_cols = [col for col in display_df.columns if '(Country)' in col or 'Delta' in col]
        entity_cols = [col for col in display_df.columns if '(Entity)' in col]

        def get_highlighter(threshold, max_scale):
            def highlight(val):
                if not isinstance(val, (int, float)) or pd.isna(val):
                    return ''
                if val >= threshold:
                    intensity = min(val / max_scale, 1) 
                    return f'background-color: rgba(39, 174, 96, {0.15 + (0.4 * intensity)});'
                elif val <= -threshold:
                    intensity = min(abs(val) / max_scale, 1)
                    return f'background-color: rgba(231, 76, 60, {0.15 + (0.4 * intensity)});'
                return ''
            return highlight

        styled_df = display_df.style.format(format_dict)
        try:
            styled_df = styled_df.map(get_highlighter(0.001, 0.05), subset=country_cols)
            styled_df = styled_df.map(get_highlighter(0.0001, 0.01), subset=entity_cols)
        except AttributeError:
            styled_df = styled_df.applymap(get_highlighter(0.001, 0.05), subset=country_cols)
            styled_df = styled_df.applymap(get_highlighter(0.0001, 0.01), subset=entity_cols)
            
        st.dataframe(styled_df, height=400, use_container_width=True)

        # --- 6. DATA-DRIVEN VARIANCE KPI BOXES ---
        st.divider()
        
        total_orders_curr = merged['Order Count_curr'].sum()
        total_orders_prev = merged['Order Count_prev'].sum()
        
        if total_orders_curr > 0 and total_orders_prev > 0:
            total_ar_curr = merged['Acceptance Orders_curr'].sum() / total_orders_curr
            total_ar_prev = merged['Acceptance Orders_prev'].sum() / total_orders_prev
            total_delta = total_ar_curr - total_ar_prev
            
            merged['W_total_curr'] = merged['Order Count_curr'] / total_orders_curr
            merged['W_total_prev'] = merged['Order Count_prev'] / total_orders_prev
            
            total_mix_change = ((merged['W_total_curr'] - merged['W_total_prev']) * merged['AR_prev']).sum()
            total_perf_change = (merged['W_total_curr'] * (merged['AR_curr'] - merged['AR_prev'])).sum()
            
            direction = "increased ⬆️" if total_delta >= 0 else "decreased ⬇️"
            st.subheader(f"📊 Variance Analysis: AR {direction} by {abs(total_delta):.2%} MoM")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Performance Change (Filtered View)", value=f"{total_perf_change:+.2%}", help="Impact entirely due to AR going up/down")
            with col2:
                st.metric(label="Mix Change (Filtered View)", value=f"{total_mix_change:+.2%}", help="Impact entirely due to volume shifting")
# --- 7. GEMINI AI TEXT SUMMARY ---
        st.divider()
        st.subheader("🤖 Detailed AI Performance Analysis")
        api_key = st.text_input("Enter Gemini API Key to generate insights:", type="password")
        
        if st.button("Generate AI Analysis", key="ai_detailed_summary"):
            if not api_key:
                st.warning("Please enter your Gemini API key.")
            else:
                with st.spinner("Gemini is searching for insights and anomalies..."):
                    try:
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        
                        # 1. Pre-calculate the Country-level summary for the AI
                        country_summary = merged.groupby('Country').agg({
                            'Order Count_curr': 'sum',
                            'Order Count_prev': 'sum',
                            'Acceptance Orders_curr': 'sum',
                            'Acceptance Orders_prev': 'sum',
                            'Subtotal_Rate_Impact': 'sum',
                            'Subtotal_Mix_Impact': 'sum'
                        }).reset_index()
                        
                        country_summary['AR_curr'] = np.where(country_summary['Order Count_curr'] > 0, country_summary['Acceptance Orders_curr'] / country_summary['Order Count_curr'], 0)
                        country_summary['AR_prev'] = np.where(country_summary['Order Count_prev'] > 0, country_summary['Acceptance Orders_prev'] / country_summary['Order Count_prev'], 0)
                        country_summary['MoM_Delta'] = country_summary['AR_curr'] - country_summary['AR_prev']
                        country_summary = country_summary.sort_values(by='MoM_Delta', ascending=False)
                        
                        # Format Country data to % strings
                        cs_prompt = country_summary.copy()
                        for col in ['AR_curr', 'MoM_Delta', 'Subtotal_Rate_Impact', 'Subtotal_Mix_Impact']:
                            cs_prompt[col] = cs_prompt[col].apply(lambda x: f"{x:.4%}")
                        
                        # 2. Get the extreme PM drivers for Entity level
                        merged['GT_Total_Impact'] = merged['GT_Mix_Impact'] + merged['GT_Rate_Impact']
                        top_drivers = merged.sort_values(by='GT_Total_Impact', ascending=False)
                        
                        # Format Driver data to % strings
                        td_prompt = top_drivers.copy()
                        for col in ['AR_prev', 'AR_curr', 'MoM_Delta', 'GT_Rate_Impact', 'GT_Mix_Impact']:
                            td_prompt[col] = td_prompt[col].apply(lambda x: f"{x:.4%}")
                        
                        # 3. Expand the data window so the AI can find curious insights (Top 15 instead of Top 5)
                        top_pos_prompt = td_prompt[['Country', 'First Payment Method', 'AR_prev', 'AR_curr', 'MoM_Delta', 'GT_Rate_Impact', 'GT_Mix_Impact']].head(15)
                        top_neg_prompt = td_prompt[['Country', 'First Payment Method', 'AR_prev', 'AR_curr', 'MoM_Delta', 'GT_Rate_Impact', 'GT_Mix_Impact']].tail(15)

                        prompt = f"""
                        You are a Senior Payments Data Scientist providing an insightful, executive-level analysis of Month-over-Month (MoM) Acceptance Rate (AR) shifts. 
                        
                        Macro Data (Global Entity):
                        - Total AR Change: {total_delta:.4%} (Rate Impact: {total_perf_change:.4%}, Mix Impact: {total_mix_change:.4%})
                        
                        Country-Level Data:
                        {cs_prompt[['Country', 'AR_curr', 'MoM_Delta', 'Subtotal_Rate_Impact', 'Subtotal_Mix_Impact']].to_string(index=False)}
                        
                        Top Payment Method Drivers (Global Impact - Positive):
                        {top_pos_prompt.to_string(index=False)}
                        
                        Top Payment Method Drivers (Global Impact - Negative):
                        {top_neg_prompt.to_string(index=False)}
                        
                        Task: Write an insightful, analytical executive summary. Do not just regurgitate the largest numbers—connect the dots, find anomalies, and explain the "so what." Follow this EXACT structure:
                        
                        ### 🌍 Country Performance & Anomalies
                        - **Country Summary**: Markdown table (Country | Latest AR | MoM Delta | Rate Impact | Mix Impact). Order by MoM Delta (Highest to Lowest). 
                          CRITICAL HTML RULE: Wrap ONLY the numerical percentages in HTML color tags (`<span style="color:green">` for positive, `<span style="color:red">` for negative). NEVER color text/names. Include the % sign inside the tag.
                        - **Insightful Country Trends**: In 3-4 sentences, go beyond the obvious. Point out interesting dynamics, such as a country where a severe Rate drop was completely offset by a positive Mix shift, or a country that is severely dragging down the global average due to a localized issue. 
                        
                        ### 🏢 Global Entity Drivers
                        - **The Heavyweights**: In 3-4 sentences, identify the specific payment methods and countries that are truly moving the global needle. Look for curious findings—for example, did a payment method have a massive positive volume shift (Mix Impact) even though its actual approval rate dropped? Or did a highly successful payment method suddenly lose all its volume? Be specific.
                        
                        ### 🎯 Strategic Recommendations
                        Provide exactly 3 distinct, highly specific action items for the payments team. Base these on the most concerning anomalies, the largest negative rate drivers, or massive unexplained volume shifts observed in the data.
                        """
                        
                        # Generate and display the response
                        response = model.generate_content(prompt)
                        st.success("Analysis Complete!")
                        st.markdown(response.text, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"An AI error occurred: {e}")
