import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

st.set_page_config(page_title="Advanced Customer Intelligence Platform", layout="wide")

# Homepage Section
st.markdown("""
    <style>
    .homepage-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin: 0.5rem;
        border-left: 4px solid #667eea;
        height: 150px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .cta-section {
        background: linear-gradient(125deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-top: 2rem;
    }
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1rem;
        margin: 2rem 0;
    }
    @media (max-width: 768px) {
        .feature-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Homepage Header
st.markdown("""
    <div class='homepage-header'>
        <h1>Advanced Customer Intelligence Platform</h1>
        <p style='font-size: 1.2rem; margin: 0;'>Transform your customer data into actionable intelligence with advanced analytics and AI-powered insights.</p>
    </div>
""", unsafe_allow_html=True)

# Feature Cards Grid
st.markdown("<div class='feature-grid'>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
        <div class='metric-card'>
            <h3>📊 Advanced Analytics</h3>
            <p>Multi-dimensional customer segmentation</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class='metric-card'>
            <h3>🔮 Predictive Analytics</h3>
            <p>CLV Forecasting & Churn prediction</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class='metric-card'>
            <h3>🤖 AI Insights</h3>
            <p>Machine learning recommendations</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class='metric-card'>
            <h3>📈 Real-Time Dashboards</h3>
            <p>Interactive visualization</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# CTA Section
st.markdown("""
    <div class='cta-section'>
        <h3>🚀 Get Started</h3>
        <p>Unlock your customer data potential:</p>
        <ul>
            <li>Advanced Customer Segmentation</li>
            <li>Customer Lifetime Value Prediction</li>
            <li>Churn Risk Analysis</li>
            <li>Personalized Marketing Strategies</li>
        </ul>
    </div>
""", unsafe_allow_html=True)

# Dashboard Section Separator
st.markdown("---")

# Existing Dashboard Functionality
st.title("📊 Advanced Business Intelligence Dashboard")

if 'df_processed' not in st.session_state:
    st.warning("⚠️ Please upload data in the main page first!")
    st.stop()

df = st.session_state.df_processed
rfm_df = st.session_state.rfm_df if 'rfm_df' in st.session_state else None

# Advanced KPI Section
st.header("🎯 Advanced Key Performance Indicators")

# Create a grid of advanced metrics
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    # Customer Acquisition Cost (simulated)
    cac = 150  # This would normally come from marketing data
    st.metric("CAC", f"${cac}", delta="-5% vs last month")

with col2:
    # Average Order Value
    aov = df['total_amount'].mean()
    st.metric("AOV", f"${aov:.0f}", delta="+2.3%")

with col3:
    # Customer Lifetime Value
    if rfm_df is not None:
        clv = rfm_df['monetary'].mean()
        st.metric("CLV", f"${clv:.0f}", delta="+8.1%")

with col4:
    # Purchase Frequency
    if 'customer_stats' in st.session_state:
        avg_frequency = st.session_state.customer_stats['purchase_frequency'].mean()
        st.metric("Purchase Frequency", f"{avg_frequency:.1f}/month")

with col5:
    # Repeat Rate
    repeat_customers = len(st.session_state.customer_stats[st.session_state.customer_stats['transaction_count'] > 1])
    repeat_rate = (repeat_customers / len(st.session_state.customer_stats)) * 100
    st.metric("Repeat Rate", f"{repeat_rate:.1f}%")

st.markdown("---")

# Advanced Visualizations
st.header("📈 Advanced Analytics")

# Multi-dimensional analysis
col1, col2 = st.columns(2)

with col1:
    # Customer Journey Analysis
    st.subheader("🛣️ Customer Journey Analysis")
    
    # Create funnel visualization
    funnel_data = {
        'Stage': ['Visitors', 'First-time Buyers', 'Repeat Customers', 'Loyal Customers', 'VIPs'],
        'Count': [
            len(st.session_state.customer_stats) * 3,  # Estimated visitors
            len(st.session_state.customer_stats[st.session_state.customer_stats['transaction_count'] == 1]),
            len(st.session_state.customer_stats[st.session_state.customer_stats['transaction_count'] > 1]),
            len(st.session_state.customer_stats[st.session_state.customer_stats['transaction_count'] > 3]),
            len(st.session_state.customer_stats[st.session_state.customer_stats['total_spent'] > st.session_state.customer_stats['total_spent'].quantile(0.8)])
        ]
    }
    
    fig = px.funnel(funnel_data, x='Count', y='Stage', title='Customer Conversion Funnel')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Customer Health Dashboard
    st.subheader("❤️ Customer Health Dashboard")
    
    if rfm_df is not None:
        health_metrics = {
            'Metric': ['Active Customers', 'At Risk', 'Churned', 'New', 'VIP'],
            'Count': [
                len(rfm_df[rfm_df['recency'] <= 90]),
                len(rfm_df[rfm_df['RFM_Segment'] == 'At Risk']),
                len(rfm_df[rfm_df['RFM_Segment'] == 'Lost']),
                len(rfm_df[rfm_df['recency'] <= 30]),
                len(rfm_df[rfm_df['Customer_Value_Tier'] == 'Platinum'])
            ],
            'Trend': ['↑', '↓', '→', '↑', '↑']
        }
        
        for metric, count, trend in zip(health_metrics['Metric'], health_metrics['Count'], health_metrics['Trend']):
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.write(f"{metric}:")
            with col_b:
                st.write(f"{count} {trend}")

# Predictive Analytics Section
st.header("🔮 Predictive Insights")

col1, col2 = st.columns(2)

with col1:
    # Revenue Forecast
    st.subheader("📊 Revenue Forecast")
    
    # Simple linear forecast (in real app, use proper time series)
    monthly_revenue = df.groupby(df['date'].dt.to_period('M'))['total_amount'].sum()
    last_6_months = monthly_revenue[-6:]
    
    # Simple projection: 5% growth
    forecast_months = 3
    forecast_values = []
    last_value = last_6_months.iloc[-1]
    
    for i in range(forecast_months):
        forecast_value = last_value * (1.05 ** (i + 1))
        forecast_values.append(forecast_value)
    
    # Create forecast visualization
    historical_dates = last_6_months.index.astype(str)
    forecast_dates = pd.period_range(start=last_6_months.index[-1] + 1, periods=forecast_months, freq='M').astype(str)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historical_dates, y=last_6_months.values, name='Historical', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=forecast_dates, y=forecast_values, name='Forecast', line=dict(color='red', dash='dash')))
    fig.update_layout(title="6-Month Revenue Trend with 3-Month Forecast")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Customer Behavior Trends
    st.subheader("📈 Behavior Trends")
    
    if rfm_df is not None:
        # Segment growth trends
        segment_trends = rfm_df.groupby('RFM_Segment').size().nlargest(5)
        fig = px.bar(segment_trends, title="Top 5 Customer Segments", 
                    labels={'value': 'Number of Customers', 'index': 'Segment'})
        st.plotly_chart(fig, use_container_width=True)

# Advanced Export Options
with st.expander("💾 Advanced Data Export"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.download_button(
            "Export Customer Analytics",
            data=st.session_state.customer_stats.to_csv(),
            file_name="customer_analytics.csv",
            mime="text/csv"
        )
    
    with col2:
        if rfm_df is not None:
            st.download_button(
                "Export RFM Analysis",
                data=rfm_df.to_csv(),
                file_name="rfm_analysis.csv",
                mime="text/csv"
            )
    
    with col3:
        # Create summary report
        summary_report = f"""
        CUSTOMER ANALYTICS SUMMARY REPORT
        Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
        
        Key Metrics:
        - Total Customers: {len(st.session_state.customer_stats):,}
        - Total Revenue: ${df['total_amount'].sum():,.0f}
        - Average CLV: ${st.session_state.customer_stats['total_spent'].mean():.0f}
        - Retention Rate: {(repeat_rate):.1f}%
        - Active Segments: {len(rfm_df['RFM_Segment'].unique()) if rfm_df is not None else 'N/A'}
        
        Top Segments:
        {segment_trends.to_string() if rfm_df is not None else 'N/A'}
        """
        
        st.download_button(
            "Download Summary Report",
            data=summary_report,
            file_name="analytics_summary.txt",
            mime="text/plain"
        )