import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Strategic Recommendations", layout="wide")

st.title("🚀 Strategic Recommendations & Action Plan")

# Check if RFM data is available
if 'rfm_df' not in st.session_state or 'segment_analysis' not in st.session_state:
    st.warning("⚠️ Please run RFM analysis first!")
    st.stop()

rfm_df = st.session_state.rfm_df
segment_analysis = st.session_state.segment_analysis

# Executive Summary
st.header("🎯 Executive Summary")

total_customers = len(rfm_df)
total_revenue = rfm_df['monetary'].sum()
champion_customers = len(rfm_df[rfm_df['RFM_Segment'] == 'Champions'])
at_risk_customers = len(rfm_df[rfm_df['RFM_Segment'] == 'At Risk'])

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Customers", f"{total_customers:,}")

with col2:
    st.metric("Total Revenue", f"${total_revenue:,.0f}")

with col3:
    st.metric("Champion Customers", f"{champion_customers}")

with col4:
    st.metric("At-Risk Customers", f"{at_risk_customers}")

# Revenue concentration
top_20_pct = int(total_customers * 0.2)
top_customers_revenue = rfm_df.nlargest(top_20_pct, 'monetary')['monetary'].sum()
revenue_concentration = (top_customers_revenue / total_revenue * 100)

st.info(f"💡 **Insight**: Top 20% of customers generate {revenue_concentration:.1f}% of total revenue")

st.markdown("---")

# Segment-specific Recommendations
st.header("👥 Segment-Specific Strategies")

# Define strategies for each segment
strategies = {
    'Champions': {
        'description': 'Your most valuable customers - recent, frequent, high spenders',
        'goals': ['Retain loyalty', 'Increase lifetime value', 'Leverage for referrals'],
        'strategies': [
            '💎 **VIP Loyalty Program**: Exclusive benefits, early access, premium support',
            '🎁 **Personalized Offers**: Tailored recommendations based on purchase history',
            '👑 **Dedicated Account Management**: Assign relationship managers for top customers',
            '📊 **Proactive Communication**: Regular check-ins and personalized updates',
            '🤝 **Referral Program**: Incentivize referrals with premium rewards'
        ],
        'budget_allocation': 'High',
        'expected_roi': '25-40%'
    },
    'VIP Customers': {
        'description': 'High-value customers with strong spending patterns',
        'goals': ['Maintain loyalty', 'Cross-sell premium products', 'Increase engagement'],
        'strategies': [
            '🌟 **Tiered Loyalty Benefits**: Exclusive discounts and early sale access',
            '📧 **Personalized Communication**: Targeted emails with product recommendations',
            '🎯 **Upselling Opportunities**: Premium product showcases and bundles',
            '📱 **Mobile App Features**: Special features for loyal customers',
            '💬 **Community Building**: Exclusive forums or user groups'
        ],
        'budget_allocation': 'High',
        'expected_roi': '20-35%'
    },
    'Loyal Customers': {
        'description': 'Regular customers with good engagement and spending',
        'goals': ['Increase frequency', 'Boost average order value', 'Enhance loyalty'],
        'strategies': [
            '🔄 **Frequency Programs**: Rewards for repeated purchases',
            '💰 **Volume Discounts**: Incentives for larger orders',
            '📦 **Bundle Offers**: Curated product bundles at discounted rates',
            '🎫 **Member-Only Events**: Exclusive sales and promotions',
            '📚 **Educational Content**: How-to guides and best practices'
        ],
        'budget_allocation': 'Medium-High',
        'expected_roi': '15-30%'
    },
    'New Customers': {
        'description': 'Recently acquired customers with potential for growth',
        'goals': ['Build relationship', 'Increase second purchase rate', 'Gather feedback'],
        'strategies': [
            '👋 **Welcome Series**: Onboarding emails and tutorials',
            '🎁 **First Purchase Incentives**: Discount on second purchase',
            '📞 **Personal Welcome**: Welcome call or personalized message',
            '📝 **Feedback Collection**: Surveys to understand their needs',
            '🔍 **Behavior Tracking**: Monitor early engagement patterns'
        ],
        'budget_allocation': 'Medium',
        'expected_roi': '20-40%'
    },
    'At Risk': {
        'description': 'Previously active customers showing signs of churn',
        'goals': ['Re-engage', 'Address concerns', 'Restore relationship'],
        'strategies': [
            '📞 **Win-back Campaigns**: Special offers to encourage return',
            '❓ **Exit Surveys**: Understand reasons for decreased engagement',
            '🎁 **We Miss You Offers**: Personalized incentives to return',
            '🔧 **Problem Resolution**: Address any service issues',
            '📊 **Re-engagement Sequence**: Automated email sequence'
        ],
        'budget_allocation': 'Medium',
        'expected_roi': '15-25%'
    },
    'Frequent Low Spenders': {
        'description': 'Regular purchasers with low average order value',
        'goals': ['Increase basket size', 'Upsell higher-value products', 'Improve margins'],
        'strategies': [
            '🔄 **Upselling Campaigns**: Recommendations for premium products',
            '📦 **Bundle Strategies**: Create value-based bundles',
            '💰 **Minimum Spend Benefits**: Free shipping or gifts at thresholds',
            '🎯 **Cross-selling**: Complementary product recommendations',
            '📊 **Price Tier Education**: Show value of premium options'
        ],
        'budget_allocation': 'Low-Medium',
        'expected_roi': '10-20%'
    },
    'Regular Customers': {
        'description': 'Steady customers with average engagement',
        'goals': ['Increase loyalty', 'Boost frequency', 'Enhance value'],
        'strategies': [
            '📧 **Regular Communication**: Newsletters and updates',
            '🎯 **Targeted Promotions**: Segment-specific offers',
            '🔄 **Loyalty Program**: Points-based reward system',
            '📱 **Engagement Tools**: Mobile notifications and reminders',
            '💡 **Educational Content**: Usage tips and best practices'
        ],
        'budget_allocation': 'Medium',
        'expected_roi': '10-25%'
    }
}

# Display strategies for each segment
for segment in rfm_df['RFM_Segment'].unique():
    if segment in strategies:
        segment_data = strategies[segment]
        segment_customers = len(rfm_df[rfm_df['RFM_Segment'] == segment])
        segment_revenue = rfm_df[rfm_df['RFM_Segment'] == segment]['monetary'].sum()
        
        with st.expander(f"🎯 {segment} ({segment_customers} customers, ${segment_revenue:,.0f} revenue)", expanded=True):
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Description**: {segment_data['description']}")
                st.write("**Primary Goals**:")
                for goal in segment_data['goals']:
                    st.write(f"- {goal}")
                
                st.write("**Recommended Strategies**:")
                for strategy in segment_data['strategies']:
                    st.write(strategy)
            
            with col2:
                st.metric("Budget Priority", segment_data['budget_allocation'])
                st.metric("Expected ROI", segment_data['expected_roi'])
                st.metric("Customer Count", segment_customers)
                st.metric("Segment Revenue", f"${segment_revenue:,.0f}")

st.markdown("---")

# Implementation Roadmap
st.header("🗓️ Implementation Roadmap")

timeline_data = {
    "Phase 1: Foundation (Weeks 1-2)": [
        "Set up customer segmentation in CRM",
        "Develop segment-specific communication templates",
        "Train sales and marketing teams on segments",
        "Establish tracking and measurement framework"
    ],
    "Phase 2: Pilot (Weeks 3-6)": [
        "Launch campaigns for Champion and VIP segments",
        "Implement win-back campaigns for At-Risk customers",
        "Set up automated onboarding for New Customers",
        "Monitor initial results and gather feedback"
    ],
    "Phase 3: Scale (Weeks 7-12)": [
        "Expand campaigns to all customer segments",
        "Implement advanced personalization features",
        "Develop loyalty program for high-value segments",
        "Optimize based on performance data"
    ],
    "Phase 4: Optimize (Months 4-6)": [
        "Advanced A/B testing across segments",
        "Implement machine learning recommendations",
        "Develop predictive churn models",
        "Continuous improvement and refinement"
    ]
}

for phase, tasks in timeline_data.items():
    with st.expander(f"📅 {phase}", expanded=True):
        for task in tasks:
            st.write(f"✅ {task}")

st.markdown("---")

# ROI Projections
st.header("💰 ROI Projections & Business Impact")

# Calculate projected impact
projection_data = []
for segment in rfm_df['RFM_Segment'].unique():
    if segment in strategies:
        segment_customers = len(rfm_df[rfm_df['RFM_Segment'] == segment])
        current_revenue = rfm_df[rfm_df['RFM_Segment'] == segment]['monetary'].sum()
        
        # Simple projection based on segment type
        if segment in ['Champions', 'VIP Customers']:
            growth_rate = 0.15  # 15% growth
        elif segment in ['Loyal Customers', 'Regular Customers']:
            growth_rate = 0.10  # 10% growth
        elif segment == 'New Customers':
            growth_rate = 0.25  # 25% growth
        elif segment == 'At Risk':
            growth_rate = 0.20  # 20% recovery
        else:
            growth_rate = 0.08  # 8% growth
            
        projected_revenue = current_revenue * (1 + growth_rate)
        revenue_increase = projected_revenue - current_revenue
        
        projection_data.append({
            'Segment': segment,
            'Current Customers': segment_customers,
            'Current Revenue': current_revenue,
            'Projected Revenue': projected_revenue,
            'Revenue Increase': revenue_increase,
            'Growth Rate': f"{growth_rate*100:.0f}%"
        })

projection_df = pd.DataFrame(projection_data)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Revenue Projections")
    st.dataframe(projection_df, use_container_width=True)

with col2:
    total_increase = projection_df['Revenue Increase'].sum()
    total_projected = projection_df['Projected Revenue'].sum()
    
    st.metric("Total Projected Revenue Increase", f"${total_increase:,.0f}")
    st.metric("Total Projected Revenue", f"${total_projected:,.0f}")
    st.metric("Overall Growth Rate", f"{(total_increase/total_revenue*100):.1f}%")

# Key Performance Indicators
st.subheader("📊 Key Performance Indicators to Track")

kpis = {
    "Customer Retention Rate": "Measure segment-specific retention quarterly",
    "Customer Lifetime Value": "Track CLV changes by segment monthly",
    "Segment Revenue Contribution": "Monitor revenue share by segment monthly",
    "Campaign Conversion Rates": "Track conversion rates for segment-specific campaigns",
    "Customer Satisfaction Scores": "Segment-level CSAT and NPS tracking",
    "Churn Rate": "Monitor churn rates by segment monthly"
}

for kpi, description in kpis.items():
    st.write(f"**{kpi}**: {description}")

# Export Recommendations
with st.expander("💾 Export Strategic Plan"):
    st.download_button(
        label="Download Strategic Recommendations (PDF)",
        data="\n".join([f"{k}: {v}" for k, v in strategies.items()]),
        file_name="customer_segmentation_strategy.txt",
        mime="text/plain"
    )