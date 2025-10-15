import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Advanced Analytics - Customer Intelligence Platform",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for Advanced Analytics page
st.markdown("""
<style>
    .analytics-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    .prediction-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-left: 5px solid;
    }
    .metric-highlight {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    .insight-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class AdvancedAnalytics:
    def __init__(self):
        self.models = {}
        self.scalers = {}
    
    def calculate_clv(self, rfm_df, months=12):
        """Calculate Customer Lifetime Value using advanced methods"""
        try:
            # Simple CLV calculation: (Average Purchase Value * Purchase Frequency) * Customer Lifespan
            avg_purchase_value = rfm_df['monetary'].mean()
            purchase_frequency = rfm_df['frequency'].mean()
            customer_lifespan = months  # Assuming 12 months average lifespan
            
            clv = (avg_purchase_value * purchase_frequency) * customer_lifespan
            
            # Enhanced CLV with RFM factors
            rfm_df['clv_basic'] = (rfm_df['monetary'] / rfm_df['frequency']) * months
            rfm_df['clv_enhanced'] = (
                rfm_df['clv_basic'] * 
                (rfm_df['R_Score'] / 5) *  # Recency factor
                (rfm_df['F_Score'] / 5) *  # Frequency factor
                (rfm_df['M_Score'] / 5)    # Monetary factor
            )
            
            return rfm_df, clv
            
        except Exception as e:
            st.error(f"Error calculating CLV: {str(e)}")
            return rfm_df, 0
    
    def predict_future_clv(self, rfm_df):
        """Predict future CLV using machine learning"""
        try:
            # Prepare features for prediction
            features = ['recency', 'frequency', 'monetary', 'R_Score', 'F_Score', 'M_Score']
            X = rfm_df[features].copy()
            y = rfm_df['monetary']  # Using current monetary as target for prediction
            
            # Handle missing values
            X = X.fillna(X.mean())
            y = y.fillna(y.mean())
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            # Predict
            X_scaled = scaler.transform(X)
            predictions = model.predict(X_scaled)
            
            # Add predictions to dataframe
            rfm_df['predicted_clv'] = predictions
            rfm_df['clv_growth_potential'] = rfm_df['predicted_clv'] - rfm_df['monetary']
            
            # Store model and scaler
            self.models['clv_predictor'] = model
            self.scalers['clv_predictor'] = scaler
            
            return rfm_df, model.score(X_test_scaled, y_test)
            
        except Exception as e:
            st.error(f"Error in CLV prediction: {str(e)}")
            return rfm_df, 0
    
    def churn_risk_analysis(self, rfm_df):
        """Analyze customer churn risk"""
        try:
            # Churn risk based on RFM scores
            # Higher recency = higher churn risk, Lower frequency/monetary = higher churn risk
            rfm_df['churn_risk'] = (
                (6 - rfm_df['R_Score']) * 0.5 +  # Recency contributes 50%
                (6 - rfm_df['F_Score']) * 0.3 +  # Frequency contributes 30%
                (6 - rfm_df['M_Score']) * 0.2    # Monetary contributes 20%
            ) / 3
            
            # Categorize churn risk
            conditions = [
                rfm_df['churn_risk'] >= 4,
                rfm_df['churn_risk'] >= 3,
                rfm_df['churn_risk'] >= 2,
                rfm_df['churn_risk'] < 2
            ]
            
            risk_levels = ['Very High', 'High', 'Medium', 'Low']
            rfm_df['churn_risk_level'] = np.select(conditions, risk_levels, default='Low')
            
            return rfm_df
            
        except Exception as e:
            st.error(f"Error in churn risk analysis: {str(e)}")
            return rfm_df
    
    def customer_health_score(self, rfm_df):
        """Calculate comprehensive customer health score"""
        try:
            # Health score based on multiple factors
            rfm_df['health_score'] = (
                rfm_df['R_Score'] * 0.4 +      # Recency: 40% weight
                rfm_df['F_Score'] * 0.3 +      # Frequency: 30% weight
                rfm_df['M_Score'] * 0.3        # Monetary: 30% weight
            )
            
            # Categorize health
            conditions = [
                rfm_df['health_score'] >= 4,
                rfm_df['health_score'] >= 3,
                rfm_df['health_score'] >= 2,
                rfm_df['health_score'] < 2
            ]
            
            health_levels = ['Excellent', 'Good', 'Fair', 'Poor']
            rfm_df['health_level'] = np.select(conditions, health_levels, default='Fair')
            
            return rfm_df
            
        except Exception as e:
            st.error(f"Error calculating health score: {str(e)}")
            return rfm_df
    
    def revenue_forecasting(self, df, periods=6):
        """Forecast future revenue using time series analysis"""
        try:
            # Aggregate revenue by month
            monthly_revenue = df.groupby(df['date'].dt.to_period('M'))['total_amount'].sum()
            monthly_revenue.index = monthly_revenue.index.astype(str)
            
            # Simple forecasting using moving average
            if len(monthly_revenue) >= 3:
                last_3_months = monthly_revenue[-3:].values
                forecast_values = []
                
                for i in range(periods):
                    # Simple weighted average (more weight to recent months)
                    forecast = (last_3_months[-1] * 0.5 + last_3_months[-2] * 0.3 + last_3_months[-3] * 0.2) * 1.05  # 5% growth
                    forecast_values.append(forecast)
                    last_3_months = np.append(last_3_months[1:], forecast)
                
                return monthly_revenue, forecast_values
            else:
                return monthly_revenue, []
                
        except Exception as e:
            st.error(f"Error in revenue forecasting: {str(e)}")
            return pd.Series(), []

def show_advanced_analytics():
    """Main function for advanced analytics"""
    st.markdown("""
        <div class='analytics-header'>
            <h1>📈 Advanced Analytics & Predictive Insights</h1>
            <p>AI-powered predictions, customer lifetime value analysis, and advanced business intelligence</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Check if data is available
    if 'rfm_df' not in st.session_state:
        st.error("❌ No RFM data available. Please complete RFM analysis first.")
        st.info("💡 Go to the RFM Analysis page to generate customer segments before using advanced analytics.")
        return
    
    rfm_df = st.session_state.rfm_df
    df_processed = st.session_state.df_processed
    
    analyzer = AdvancedAnalytics()
    
    # Advanced Analytics Dashboard
    st.header("🚀 Advanced Analytics Dashboard")
    
    # Calculate advanced metrics
    with st.spinner("🔄 Calculating advanced metrics..."):
        # CLV Analysis
        rfm_df_with_clv, avg_clv = analyzer.calculate_clv(rfm_df)
        
        # Churn Risk Analysis
        rfm_df_with_risk = analyzer.churn_risk_analysis(rfm_df_with_clv)
        
        # Health Score Analysis
        rfm_df_final = analyzer.customer_health_score(rfm_df_with_risk)
        
        # CLV Prediction
        rfm_df_final, prediction_accuracy = analyzer.predict_future_clv(rfm_df_final)
        
        # Revenue Forecasting
        historical_revenue, revenue_forecast = analyzer.revenue_forecasting(df_processed)
    
    st.success("✅ Advanced analytics completed!")
    
    # Key Metrics Overview
    st.header("📊 Advanced Business Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        high_value_customers = len(rfm_df_final[rfm_df_final['M_Score'] >= 4])
        st.metric("High-Value Customers", f"{high_value_customers:,}")
    
    with col2:
        avg_clv_display = rfm_df_final['clv_enhanced'].mean()
        st.metric("Avg Customer Lifetime Value", f"${avg_clv_display:,.0f}")
    
    with col3:
        high_risk_customers = len(rfm_df_final[rfm_df_final['churn_risk_level'].isin(['High', 'Very High'])])
        st.metric("High Churn Risk Customers", f"{high_risk_customers:,}")
    
    with col4:
        prediction_accuracy_pct = prediction_accuracy * 100
        st.metric("CLV Prediction Accuracy", f"{prediction_accuracy_pct:.1f}%")
    
    # CLV Analysis Section
    st.header("💰 Customer Lifetime Value Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CLV Distribution
        fig = px.histogram(
            rfm_df_final, 
            x='clv_enhanced',
            title='Customer Lifetime Value Distribution',
            labels={'clv_enhanced': 'CLV ($)', 'count': 'Number of Customers'},
            color_discrete_sequence=['#667eea']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # CLV by Segment
        clv_by_segment = rfm_df_final.groupby('Segment')['clv_enhanced'].mean().sort_values(ascending=False)
        fig = px.bar(
            x=clv_by_segment.index,
            y=clv_by_segment.values,
            title='Average CLV by Customer Segment',
            labels={'x': 'Segment', 'y': 'Average CLV ($)'},
            color=clv_by_segment.values,
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Churn Risk Analysis
    st.header("⚠️ Churn Risk Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Churn Risk Distribution
        risk_counts = rfm_df_final['churn_risk_level'].value_counts()
        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title='Customer Distribution by Churn Risk Level',
            color_discrete_sequence=px.colors.sequential.Reds_r
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Revenue at Risk
        risk_revenue = rfm_df_final.groupby('churn_risk_level')['monetary'].sum()
        fig = px.bar(
            x=risk_revenue.index,
            y=risk_revenue.values,
            title='Revenue at Risk by Churn Level',
            labels={'x': 'Churn Risk Level', 'y': 'Total Revenue at Risk ($)'},
            color=risk_revenue.values,
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Customer Health Analysis
    st.header("❤️ Customer Health Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Health Score Distribution
        health_counts = rfm_df_final['health_level'].value_counts()
        fig = px.bar(
            x=health_counts.index,
            y=health_counts.values,
            title='Customer Distribution by Health Level',
            labels={'x': 'Health Level', 'y': 'Number of Customers'},
            color=health_counts.values,
            color_continuous_scale='Greens'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Health vs CLV
        fig = px.scatter(
            rfm_df_final,
            x='health_score',
            y='clv_enhanced',
            color='churn_risk_level',
            title='Customer Health Score vs Lifetime Value',
            labels={'health_score': 'Health Score', 'clv_enhanced': 'CLV ($)'},
            size='monetary',
            hover_data=['Segment']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Revenue Forecasting
    st.header("📈 Revenue Forecasting")
    
    if len(historical_revenue) > 0 and len(revenue_forecast) > 0:
        # Create forecast timeline
        historical_dates = historical_revenue.index.tolist()
        last_date = pd.Period(historical_dates[-1])
        
        forecast_dates = []
        for i in range(1, len(revenue_forecast) + 1):
            forecast_dates.append(str(last_date + i))
        
        # Combine historical and forecast data
        all_dates = historical_dates + forecast_dates
        all_values = historical_revenue.values.tolist() + revenue_forecast
        
        # Create plot
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical_dates,
            y=historical_revenue.values,
            name='Historical Revenue',
            line=dict(color='#667eea', width=3)
        ))
        
        # Forecast data
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=revenue_forecast,
            name='Revenue Forecast',
            line=dict(color='#ff6b6b', width=3, dash='dash')
        ))
        
        fig.update_layout(
            title='6-Month Revenue Forecast',
            xaxis_title='Month',
            yaxis_title='Revenue ($)',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Forecast metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            last_historical = historical_revenue.values[-1]
            avg_forecast = np.mean(revenue_forecast)
            growth_pct = ((avg_forecast - last_historical) / last_historical) * 100
            st.metric("Forecasted Growth", f"{growth_pct:.1f}%")
        
        with col2:
            total_forecast_revenue = sum(revenue_forecast)
            st.metric("Total Forecast Revenue", f"${total_forecast_revenue:,.0f}")
        
        with col3:
            st.metric("Forecast Period", "6 months")
    
    else:
        st.info("Not enough historical data for revenue forecasting. Need at least 3 months of data.")
    
    # Predictive Insights
    st.header("🔮 Predictive Insights & Recommendations")
    
    # Generate insights
    insights = []
    
    # Insight 1: High-value customer retention
    high_value_at_risk = len(rfm_df_final[
        (rfm_df_final['M_Score'] >= 4) & 
        (rfm_df_final['churn_risk_level'].isin(['High', 'Very High']))
    ])
    
    if high_value_at_risk > 0:
        insights.append(f"🚨 **{high_value_at_risk} high-value customers** are at risk of churning. Immediate retention actions recommended.")
    
    # Insight 2: CLV growth opportunities
    avg_clv_growth = rfm_df_final['clv_growth_potential'].mean()
    if avg_clv_growth > 0:
        insights.append(f"📈 **Average CLV growth potential**: ${avg_clv_growth:.0f} per customer. Focus on upselling and cross-selling.")
    
    # Insight 3: Customer health trends
    healthy_customers_pct = (len(rfm_df_final[rfm_df_final['health_level'].isin(['Excellent', 'Good'])]) / len(rfm_df_final)) * 100
    insights.append(f"❤️ **{healthy_customers_pct:.1f}% of customers** are in good or excellent health.")
    
    # Insight 4: Revenue opportunities
    low_frequency_high_value = len(rfm_df_final[
        (rfm_df_final['M_Score'] >= 4) & 
        (rfm_df_final['F_Score'] <= 2)
    ])
    
    if low_frequency_high_value > 0:
        insights.append(f"💎 **{low_frequency_high_value} high-value customers** have low purchase frequency. Great opportunity for frequency increase.")
    
    # Display insights
    for i, insight in enumerate(insights, 1):
        st.markdown(f"<div class='insight-box'>{insight}</div>", unsafe_allow_html=True)
    
    # Actionable Recommendations
    st.header("🎯 Actionable Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🛡️ Retention Strategies")
        st.write("1. **Implement VIP program** for high-value at-risk customers")
        st.write("2. **Personalized win-back campaigns** for high churn risk segments")
        st.write("3. **Proactive customer success** outreach for medium-risk customers")
        st.write("4. **Loyalty rewards** for customers with good health scores")
    
    with col2:
        st.subheader("🚀 Growth Opportunities")
        st.write("1. **Upsell campaigns** targeting high CLV growth potential customers")
        st.write("2. **Frequency optimization** for high-value, low-frequency buyers")
        st.write("3. **Cross-selling** based on customer segment preferences")
        st.write("4. **Referral programs** leveraging champion customers")
    
    # Detailed Data View
    st.header("📋 Advanced Analytics Data")
    
    with st.expander("View Detailed Analytics Data", expanded=False):
        # Add filters
        col1, col2 = st.columns(2)
        
        with col1:
            risk_filter = st.multiselect(
                "Filter by Churn Risk",
                options=rfm_df_final['churn_risk_level'].unique(),
                default=rfm_df_final['churn_risk_level'].unique()
            )
        
        with col2:
            health_filter = st.multiselect(
                "Filter by Health Level",
                options=rfm_df_final['health_level'].unique(),
                default=rfm_df_final['health_level'].unique()
            )
        
        filtered_data = rfm_df_final[
            rfm_df_final['churn_risk_level'].isin(risk_filter) &
            rfm_df_final['health_level'].isin(health_filter)
        ]
        
        st.dataframe(filtered_data, use_container_width=True)
    
    # Export Options
    st.header("💾 Export Advanced Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            "📥 Download Analytics Data (CSV)",
            data=rfm_df_final.to_csv(index=False),
            file_name="advanced_analytics.csv",
            mime="text/csv"
        )
    
    with col2:
        # Generate comprehensive report
        if 'clv_data' in locals():
            growth_value = rfm_df_final['predicted_clv'].sum() - rfm_df_final['monetary'].sum()
            growth_str = f"{growth_value:.0f}"
        else:
            growth_str = 'N/A'
        
        report = f"""
        ADVANCED ANALYTICS REPORT
        Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
        
        EXECUTIVE SUMMARY:
        - Total Customers Analyzed: {len(rfm_df_final):,}
        - Average CLV: ${rfm_df_final['clv_enhanced'].mean():,.0f}
        - High Churn Risk Customers: {high_risk_customers:,}
        - CLV Prediction Accuracy: {prediction_accuracy_pct:.1f}%
        - Predicted CLV Growth: {growth_str}
        
        CUSTOMER HEALTH OVERVIEW:
        - Excellent Health: {len(rfm_df_final[rfm_df_final['health_level'] == 'Excellent']):,}
        - Good Health: {len(rfm_df_final[rfm_df_final['health_level'] == 'Good']):,}
        - Fair Health: {len(rfm_df_final[rfm_df_final['health_level'] == 'Fair']):,}
        - Poor Health: {len(rfm_df_final[rfm_df_final['health_level'] == 'Poor']):,}
        
        KEY INSIGHTS:
        {chr(10).join(['- ' + insight.replace('**', '') for insight in insights])}
        
        RECOMMENDATIONS:
        1. Focus retention efforts on high-value at-risk customers
        2. Develop personalized marketing for different health segments
        3. Implement predictive analytics for proactive customer service
        4. Monitor CLV trends and adjust strategies accordingly
        """
        
        st.download_button(
            "📋 Download Analysis Report",
            data=report,
            file_name="advanced_analytics_report.txt",
            mime="text/plain"
        )

# Main execution
if __name__ == "__main__":
    show_advanced_analytics()