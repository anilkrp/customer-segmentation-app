import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(
    page_title="RFM Analysis - Customer Intelligence Platform",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS for RFM Analysis page
st.markdown("""
<style>
    .rfm-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    .segment-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-left: 5px solid;
    }
    .metric-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

class RFMAnalyzer:
    def __init__(self):
        self.segment_definitions = {
            'Champions': {'min_score': 13, 'color': '#2E8B57', 'icon': '🏆', 'description': 'Your best customers'},
            'Loyal Customers': {'min_score': 11, 'color': '#4169E1', 'icon': '🤝', 'description': 'Regular buyers'},
            'Potential Loyalists': {'min_score': 9, 'color': '#FFD700', 'icon': '⭐', 'description': 'Recent customers with potential'},
            'Recent Customers': {'min_score': 8, 'color': '#32CD32', 'icon': '🆕', 'description': 'New customers'},
            'Promising': {'min_score': 7, 'color': '#FFA500', 'icon': '📈', 'description': 'Showing good potential'},
            'Need Attention': {'min_score': 6, 'color': '#FF6347', 'icon': '⚠️', 'description': 'At risk of churning'},
            'About To Sleep': {'min_score': 5, 'color': '#A9A9A9', 'icon': '😴', 'description': 'Inactive recently'},
            'At Risk': {'min_score': 4, 'color': '#DC143C', 'icon': '🎯', 'description': 'High risk of churn'},
            "Can't Lose Them": {'min_score': 3, 'color': '#8B0000', 'icon': '💔', 'description': 'High value but inactive'},
            'Lost': {'min_score': 0, 'color': '#696969', 'icon': '👋', 'description': 'Churned customers'}
        }

    def calculate_rfm_scores(self, df):
        """Calculate RFM scores from transaction data"""
        snapshot_date = df['date'].max() + timedelta(days=1)
        
        # Calculate RFM metrics
        rfm_df = df.groupby('customer_id').agg({
            'date': lambda x: (snapshot_date - x.max()).days,  # Recency
            'transaction_id': 'count',  # Frequency
            'total_amount': 'sum'  # Monetary
        }).reset_index()
        
        rfm_df.columns = ['customer_id', 'recency', 'frequency', 'monetary']
        
        # Create RFM scores with error handling for small datasets
        try:
            # Recency: lower is better (5 being most recent)
            rfm_df['R_Score'] = pd.qcut(rfm_df['recency'], 5, labels=[5, 4, 3, 2, 1], duplicates='drop')
        except ValueError:
            # Fallback: use fewer bins if not enough unique values
            unique_recency = rfm_df['recency'].nunique()
            n_bins_recency = min(5, unique_recency)
            rfm_df['R_Score'] = pd.qcut(rfm_df['recency'], n_bins_recency, labels=range(n_bins_recency, 0, -1), duplicates='drop')
        
        try:
            # Frequency: higher is better
            rfm_df['F_Score'] = pd.qcut(rfm_df['frequency'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        except ValueError:
            # Fallback: use fewer bins if not enough unique values
            unique_frequency = rfm_df['frequency'].nunique()
            n_bins_frequency = min(5, unique_frequency)
            rfm_df['F_Score'] = pd.qcut(rfm_df['frequency'], n_bins_frequency, labels=range(1, n_bins_frequency + 1), duplicates='drop')
        
        try:
            # Monetary: higher is better
            rfm_df['M_Score'] = pd.qcut(rfm_df['monetary'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        except ValueError:
            # Fallback: use fewer bins if not enough unique values
            unique_monetary = rfm_df['monetary'].nunique()
            n_bins_monetary = min(5, unique_monetary)
            rfm_df['M_Score'] = pd.qcut(rfm_df['monetary'], n_bins_monetary, labels=range(1, n_bins_monetary + 1), duplicates='drop')
        
        # Convert to numeric (handle categorical conversion)
        for col in ['R_Score', 'F_Score', 'M_Score']:
            rfm_df[col] = pd.to_numeric(rfm_df[col], errors='coerce')
        
        # Fill any NaN values with median
        rfm_df['R_Score'] = rfm_df['R_Score'].fillna(rfm_df['R_Score'].median())
        rfm_df['F_Score'] = rfm_df['F_Score'].fillna(rfm_df['F_Score'].median())
        rfm_df['M_Score'] = rfm_df['M_Score'].fillna(rfm_df['M_Score'].median())
        
        # Convert to integers
        rfm_df['R_Score'] = rfm_df['R_Score'].astype(int)
        rfm_df['F_Score'] = rfm_df['F_Score'].astype(int)
        rfm_df['M_Score'] = rfm_df['M_Score'].astype(int)
        
        # Total RFM score
        rfm_df['RFM_Score'] = rfm_df['R_Score'] + rfm_df['F_Score'] + rfm_df['M_Score']
        
        return rfm_df

    def segment_customers(self, rfm_df):
        """Segment customers based on RFM scores"""
        # Define segmentation rules
        conditions = [
            rfm_df['RFM_Score'] >= 13,
            rfm_df['RFM_Score'] >= 11,
            (rfm_df['R_Score'] >= 4) & (rfm_df['F_Score'] >= 3),
            rfm_df['R_Score'] >= 4,
            (rfm_df['F_Score'] >= 3) & (rfm_df['M_Score'] <= 2),
            rfm_df['RFM_Score'] >= 6,
            (rfm_df['R_Score'] <= 3) & (rfm_df['F_Score'] >= 3),
            (rfm_df['R_Score'] <= 2) & (rfm_df['F_Score'] >= 2),
            (rfm_df['R_Score'] <= 2) & (rfm_df['F_Score'] <= 2) & (rfm_df['M_Score'] >= 3),
            rfm_df['RFM_Score'] <= 3
        ]
        
        segments = [
            'Champions', 'Loyal Customers', 'Potential Loyalists',
            'Recent Customers', 'Promising', 'Need Attention',
            'About To Sleep', 'At Risk', "Can't Lose Them", 'Lost'
        ]
        
        rfm_df['Segment'] = np.select(conditions, segments, default='Neutral')
        
        # Add segment colors and icons
        rfm_df['Segment_Color'] = rfm_df['Segment'].map(
            {k: v['color'] for k, v in self.segment_definitions.items()}
        )
        rfm_df['Segment_Icon'] = rfm_df['Segment'].map(
            {k: v['icon'] for k, v in self.segment_definitions.items()}
        )
        
        return rfm_df

def show_rfm_analysis():
    """Main RFM analysis function"""
    st.markdown("""
        <div class='rfm-header'>
            <h1>🎯 RFM Customer Segmentation Analysis</h1>
            <p>Advanced customer segmentation using Recency, Frequency, and Monetary analysis</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Check if data is available
    if 'df_processed' not in st.session_state:
        st.error("❌ No data available. Please upload data or use sample data on the main page first.")
        st.info("💡 Go to the Home page and upload your customer data to get started with RFM analysis.")
        return
    
    df = st.session_state.df_processed
    rfm_analyzer = RFMAnalyzer()
    
    # Calculate RFM scores
    with st.spinner("🔄 Calculating RFM scores..."):
        rfm_df = rfm_analyzer.calculate_rfm_scores(df)
        rfm_df = rfm_analyzer.segment_customers(rfm_df)
    
    # Store RFM data in session state for other pages
    st.session_state.rfm_df = rfm_df
    
    st.success(f"✅ RFM analysis completed! Analyzed {len(rfm_df)} customers.")
    
    # RFM Overview
    st.header("📊 RFM Analysis Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_recency = rfm_df['recency'].mean()
        st.metric("Average Recency", f"{avg_recency:.1f} days")
    
    with col2:
        avg_frequency = rfm_df['frequency'].mean()
        st.metric("Average Frequency", f"{avg_frequency:.1f} transactions")
    
    with col3:
        avg_monetary = rfm_df['monetary'].mean()
        st.metric("Average Monetary", f"${avg_monetary:.0f}")
    
    with col4:
        segments_count = rfm_df['Segment'].nunique()
        st.metric("Customer Segments", segments_count)
    
    # RFM Distribution
    st.header("📈 RFM Score Distribution")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig = px.histogram(rfm_df, x='R_Score', title='Recency Score Distribution',
                          color_discrete_sequence=['#667eea'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(rfm_df, x='F_Score', title='Frequency Score Distribution',
                          color_discrete_sequence=['#764ba2'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        fig = px.histogram(rfm_df, x='M_Score', title='Monetary Score Distribution',
                          color_discrete_sequence=['#f093fb'])
        st.plotly_chart(fig, use_container_width=True)
    
    # Customer Segments
    st.header("👥 Customer Segments Analysis")
    
    # Segment distribution
    col1, col2 = st.columns(2)
    
    with col1:
        segment_counts = rfm_df['Segment'].value_counts()
        fig = px.pie(
            values=segment_counts.values,
            names=segment_counts.index,
            title="Customer Distribution by Segment",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        segment_revenue = rfm_df.groupby('Segment')['monetary'].sum().sort_values(ascending=False)
        fig = px.bar(
            x=segment_revenue.index,
            y=segment_revenue.values,
            title="Revenue by Customer Segment",
            labels={'x': 'Segment', 'y': 'Total Revenue ($)'},
            color=segment_revenue.values,
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Segment Details
    st.header("🔍 Segment Details")
    
    # Segment statistics
    segment_stats = rfm_df.groupby('Segment').agg({
        'customer_id': 'count',
        'recency': 'mean',
        'frequency': 'mean',
        'monetary': 'mean',
        'RFM_Score': 'mean'
    }).round(2)
    
    segment_stats.columns = ['Customers', 'Avg Recency (days)', 'Avg Frequency', 'Avg Monetary ($)', 'Avg RFM Score']
    segment_stats = segment_stats.sort_values('Avg Monetary ($)', ascending=False)
    
    st.dataframe(segment_stats, use_container_width=True)
    
    # Detailed segment information
    st.subheader("🎯 Segment Characteristics & Recommendations")
    
    selected_segment = st.selectbox(
        "Select Segment to View Details",
        options=rfm_df['Segment'].unique(),
        help="Choose a customer segment to see detailed analysis and recommendations"
    )
    
    if selected_segment:
        segment_data = rfm_df[rfm_df['Segment'] == selected_segment]
        segment_info = rfm_analyzer.segment_definitions.get(selected_segment, {})
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"""
                <div class='segment-card' style='border-left-color: {segment_info.get("color", "#666")}'>
                    <h3>{segment_info.get('icon', '📊')} {selected_segment}</h3>
                    <p><strong>{segment_info.get('description', 'No description available')}</strong></p>
                    <p><strong>Customers:</strong> {len(segment_data):,}</p>
                    <p><strong>Avg Revenue:</strong> ${segment_data['monetary'].mean():,.0f}</p>
                    <p><strong>Avg Recency:</strong> {segment_data['recency'].mean():.1f} days</p>
                    <p><strong>Avg Frequency:</strong> {segment_data['frequency'].mean():.1f}</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("💡 Marketing Recommendations")
            
            recommendations = {
                'Champions': [
                    "Reward these customers with exclusive offers and early access",
                    "Implement VIP/loyalty programs",
                    "Request testimonials and referrals",
                    "Offer premium services and personalized experiences"
                ],
                'Loyal Customers': [
                    "Offer loyalty rewards and membership benefits",
                    "Provide special discounts on their favorite categories",
                    "Engage with personalized communication",
                    "Introduce referral programs"
                ],
                'Potential Loyalists': [
                    "Offer membership/subscription options",
                    "Provide bundle deals and cross-selling opportunities",
                    "Send personalized recommendations",
                    "Engage through targeted email campaigns"
                ],
                'Recent Customers': [
                    "Send welcome series and onboarding emails",
                    "Offer first-time buyer discounts",
                    "Provide educational content about your products",
                    "Request initial feedback and reviews"
                ],
                'Promising': [
                    "Offer targeted promotions to increase frequency",
                    "Provide product recommendations based on browsing history",
                    "Send reminder emails for abandoned carts",
                    "Engage with social media campaigns"
                ],
                'Need Attention': [
                    "Re-engage with special offers and discounts",
                    "Send personalized win-back campaigns",
                    "Request feedback to understand their needs",
                    "Offer limited-time promotions"
                ],
                'About To Sleep': [
                    "Send reactivation campaigns with special offers",
                    "Provide personalized recommendations",
                    "Offer loyalty points or credits",
                    "Share success stories and product updates"
                ],
                'At Risk': [
                    "Implement aggressive win-back campaigns",
                    "Offer significant discounts or free shipping",
                    "Conduct customer satisfaction surveys",
                    "Provide exceptional customer service"
                ],
                "Can't Lose Them": [
                    "Reach out personally to understand their needs",
                    "Offer exclusive deals and personalized solutions",
                    "Provide premium customer support",
                    "Create customized offers based on their purchase history"
                ],
                'Lost': [
                    "Focus on new customer acquisition",
                    "Analyze reasons for churn",
                    "Consider win-back campaigns with strong incentives",
                    "Learn from this segment to prevent future churn"
                ]
            }
            
            segment_recs = recommendations.get(selected_segment, [
                "Monitor customer behavior closely",
                "Gather more data to understand segment characteristics",
                "Test different engagement strategies"
            ])
            
            for i, rec in enumerate(segment_recs, 1):
                st.write(f"{i}. {rec}")
    
    # RFM Matrix Visualization
    st.header("🎲 RFM Matrix Analysis")
    
    # Create RFM matrix
    rfm_matrix = rfm_df.groupby(['R_Score', 'F_Score']).agg({
        'monetary': 'mean',
        'customer_id': 'count'
    }).reset_index()
    
    rfm_matrix.columns = ['R_Score', 'F_Score', 'Avg_Revenue', 'Customer_Count']
    
    fig = px.scatter(
        rfm_matrix,
        x='R_Score',
        y='F_Score',
        size='Customer_Count',
        color='Avg_Revenue',
        title='RFM Matrix: Recency vs Frequency (Size = Customer Count, Color = Avg Revenue)',
        labels={'R_Score': 'Recency Score (5 = Most Recent)', 'F_Score': 'Frequency Score (5 = Most Frequent)'},
        color_continuous_scale='Viridis',
        size_max=60
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Customer Details Table
    st.header("📋 Customer Details")
    
    with st.expander("View Detailed Customer RFM Data", expanded=False):
        # Add segment filter
        segments_to_show = st.multiselect(
            "Filter by Segments",
            options=rfm_df['Segment'].unique(),
            default=rfm_df['Segment'].unique()[:3],
            help="Select segments to display in the table"
        )
        
        if segments_to_show:
            filtered_rfm = rfm_df[rfm_df['Segment'].isin(segments_to_show)]
            st.dataframe(filtered_rfm, use_container_width=True)
        else:
            st.dataframe(rfm_df, use_container_width=True)
    
    # Export Options
    st.header("💾 Export RFM Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.download_button(
            "📥 Download RFM Data (CSV)",
            data=rfm_df.to_csv(index=False),
            file_name="rfm_analysis.csv",
            mime="text/csv",
            help="Download complete RFM analysis data"
        )
    
    with col2:
        # Create segment summary
        segment_summary = rfm_df.groupby('Segment').agg({
            'customer_id': 'count',
            'monetary': ['mean', 'sum'],
            'recency': 'mean',
            'frequency': 'mean'
        }).round(2)
        
        segment_summary.columns = ['Customer_Count', 'Avg_Revenue', 'Total_Revenue', 'Avg_Recency', 'Avg_Frequency']
        segment_summary_csv = segment_summary.to_csv()
        
        st.download_button(
            "📊 Download Segment Summary (CSV)",
            data=segment_summary_csv,
            file_name="segment_summary.csv",
            mime="text/csv",
            help="Download segment-level summary statistics"
        )
    
    with col3:
        # Generate comprehensive report
        report = f"""
        RFM ANALYSIS REPORT
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
        
        EXECUTIVE SUMMARY:
        - Total Customers Analyzed: {len(rfm_df):,}
        - Total Revenue: ${rfm_df['monetary'].sum():,.0f}
        - Average RFM Score: {rfm_df['RFM_Score'].mean():.1f}
        - Customer Segments Identified: {rfm_df['Segment'].nunique()}
        
        SEGMENT DISTRIBUTION:
        {segment_stats.to_string()}
        
        KEY INSIGHTS:
        1. Top performing segment: {segment_stats.index[0]} with ${segment_stats.iloc[0]['Avg Monetary ($)']:,.0f} avg revenue
        2. Largest segment: {segment_counts.index[0]} with {segment_counts.iloc[0]:,} customers
        3. Highest potential: Focus on {segment_stats.index[1]} and {segment_stats.index[2]} segments
        
        RECOMMENDATIONS:
        - Implement targeted marketing campaigns for each segment
        - Develop loyalty programs for high-value segments
        - Create win-back campaigns for at-risk customers
        - Monitor segment migration over time
        """
        
        st.download_button(
            "📋 Download Analysis Report",
            data=report,
            file_name="rfm_analysis_report.txt",
            mime="text/plain",
            help="Download comprehensive RFM analysis report"
        )

# Main execution
if __name__ == "__main__":
    show_rfm_analysis()