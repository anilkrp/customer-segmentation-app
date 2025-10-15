import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Customer Segments - Customer Intelligence Platform",
    page_icon="👥",
    layout="wide"
)

# Custom CSS for Customer Segments page
st.markdown("""
<style>
    .segments-header {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    .cluster-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-left: 5px solid;
    }
    .metric-highlight {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    .segment-indicator {
        display: inline-block;
        width: 15px;
        height: 15px;
        border-radius: 50%;
        margin-right: 8px;
    }
</style>
""", unsafe_allow_html=True)

class CustomerSegmentAnalyzer:
    def __init__(self):
        self.cluster_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
        self.cluster_names = {
            0: "Budget Shoppers",
            1: "Premium Customers", 
            2: "Occasional Buyers",
            3: "Loyal Regulars",
            4: "High-Value VIPs",
            5: "Seasonal Shoppers",
            6: "New Explorers",
            7: "At-Risk Customers"
        }
        self.cluster_descriptions = {
            0: "Price-sensitive customers who make smaller, frequent purchases",
            1: "High-spending customers who prefer premium products",
            2: "Customers who purchase infrequently but in moderate amounts",
            3: "Regular customers with consistent spending patterns",
            4: "Top-spending customers with high loyalty and frequency",
            5: "Customers who shop during specific seasons or promotions",
            6: "New customers who are exploring your product range",
            7: "Previously active customers showing declining engagement"
        }

    def prepare_clustering_data(self, rfm_df):
        """Prepare RFM data for clustering"""
        # Select features for clustering
        features = ['recency', 'frequency', 'monetary']
        
        # Handle missing values
        clustering_data = rfm_df[features].copy()
        clustering_data = clustering_data.fillna(clustering_data.mean())
        
        # Remove outliers using IQR method
        Q1 = clustering_data.quantile(0.25)
        Q3 = clustering_data.quantile(0.75)
        IQR = Q3 - Q1
        clustering_data = clustering_data[~((clustering_data < (Q1 - 1.5 * IQR)) | 
                                          (clustering_data > (Q3 + 1.5 * IQR))).any(axis=1)]
        
        return clustering_data

    def find_optimal_clusters(self, data, max_k=10):
        """Find optimal number of clusters using elbow method"""
        wcss = []
        for i in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
            kmeans.fit(data)
            wcss.append(kmeans.inertia_)
        return wcss

    def perform_clustering(self, rfm_df, n_clusters=5):
        """Perform K-means clustering on RFM data"""
        # Prepare data
        clustering_data = self.prepare_clustering_data(rfm_df)
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(clustering_data)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_data)
        
        # Add cluster labels to the original RFM data
        clustered_rfm = rfm_df.loc[clustering_data.index].copy()
        clustered_rfm['Cluster'] = cluster_labels
        clustered_rfm['Cluster_Name'] = clustered_rfm['Cluster'].map(
            lambda x: self.cluster_names.get(x, f"Cluster {x}")
        )
        
        # Calculate cluster centers in original scale
        cluster_centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
        cluster_centers_df = pd.DataFrame(
            cluster_centers_original, 
            columns=['Recency_Center', 'Frequency_Center', 'Monetary_Center']
        )
        
        return clustered_rfm, kmeans, scaler, cluster_centers_df

    def calculate_cluster_metrics(self, clustered_rfm):
        """Calculate key metrics for each cluster"""
        cluster_metrics = clustered_rfm.groupby('Cluster').agg({
            'customer_id': 'count',
            'recency': 'mean',
            'frequency': 'mean', 
            'monetary': ['mean', 'sum'],
            'R_Score': 'mean',
            'F_Score': 'mean',
            'M_Score': 'mean'
        }).round(2)
        
        # Flatten column names
        cluster_metrics.columns = [
            'Customer_Count', 'Avg_Recency', 'Avg_Frequency', 
            'Avg_Monetary', 'Total_Revenue', 'Avg_R_Score', 
            'Avg_F_Score', 'Avg_M_Score'
        ]
        
        # Calculate additional metrics
        cluster_metrics['Avg_RFM_Score'] = (
            cluster_metrics['Avg_R_Score'] + 
            cluster_metrics['Avg_F_Score'] + 
            cluster_metrics['Avg_M_Score']
        )
        cluster_metrics['Revenue_Per_Customer'] = (
            cluster_metrics['Total_Revenue'] / cluster_metrics['Customer_Count']
        )
        cluster_metrics['Customer_Percentage'] = (
            cluster_metrics['Customer_Count'] / len(clustered_rfm) * 100
        )
        
        return cluster_metrics.sort_values('Total_Revenue', ascending=False)

def show_customer_segments():
    """Main function for customer segments analysis"""
    st.markdown("""
        <div class='segments-header'>
            <h1>👥 Advanced Customer Segmentation</h1>
            <p>AI-powered customer clustering using machine learning algorithms</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Check if data is available
    if 'rfm_df' not in st.session_state:
        st.error("❌ No RFM data available. Please complete RFM analysis first.")
        st.info("💡 Go to the RFM Analysis page to generate customer segments before using this page.")
        return
    
    rfm_df = st.session_state.rfm_df
    analyzer = CustomerSegmentAnalyzer()
    
    # Clustering Configuration
    st.header("⚙️ Clustering Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_clusters = st.slider(
            "Number of Clusters",
            min_value=3,
            max_value=8,
            value=5,
            help="Select the number of customer segments to create"
        )
    
    with col2:
        clustering_method = st.selectbox(
            "Clustering Method",
            ["K-Means", "RFM-Based"],
            help="Choose clustering algorithm"
        )
    
    with col3:
        use_pca = st.checkbox(
            "Use PCA for Visualization",
            value=True,
            help="Use Principal Component Analysis for better cluster visualization"
        )
    
    # Perform clustering
    with st.spinner("🔄 Performing customer segmentation..."):
        if clustering_method == "K-Means":
            clustered_rfm, kmeans, scaler, cluster_centers = analyzer.perform_clustering(rfm_df, n_clusters)
        else:
            # Fallback to RFM-based segmentation
            clustered_rfm = rfm_df.copy()
            clustered_rfm['Cluster'] = clustered_rfm['Segment'].astype('category').cat.codes
            clustered_rfm['Cluster_Name'] = clustered_rfm['Segment']
            kmeans, scaler, cluster_centers = None, None, None
    
    st.success(f"✅ Customer segmentation completed! Created {n_clusters} distinct segments.")
    
    # Cluster Overview
    st.header("📊 Cluster Overview")
    
    # Calculate cluster metrics
    cluster_metrics = analyzer.calculate_cluster_metrics(clustered_rfm)
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_customers = len(clustered_rfm)
        st.metric("Total Customers", f"{total_customers:,}")
    
    with col2:
        total_revenue = clustered_rfm['monetary'].sum()
        st.metric("Total Revenue", f"${total_revenue:,.0f}")
    
    with col3:
        avg_cluster_size = cluster_metrics['Customer_Count'].mean()
        st.metric("Avg Cluster Size", f"{avg_cluster_size:.0f}")
    
    with col4:
        clusters_created = clustered_rfm['Cluster'].nunique()
        st.metric("Segments Created", clusters_created)
    
    # Cluster Distribution
    st.header("📈 Segment Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Customer distribution by cluster
        cluster_distribution = clustered_rfm['Cluster_Name'].value_counts()
        
        fig = px.pie(
            values=cluster_distribution.values,
            names=cluster_distribution.index,
            title="Customer Distribution by Segment",
            color_discrete_sequence=analyzer.cluster_colors
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Revenue distribution by cluster
        revenue_by_cluster = clustered_rfm.groupby('Cluster_Name')['monetary'].sum().sort_values(ascending=False)
        
        fig = px.bar(
            x=revenue_by_cluster.index,
            y=revenue_by_cluster.values,
            title="Revenue Distribution by Segment",
            labels={'x': 'Customer Segment', 'y': 'Total Revenue ($)'},
            color=revenue_by_cluster.values,
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Cluster Characteristics
    st.header("🔍 Segment Characteristics")
    
    # Display cluster metrics table
    st.subheader("📋 Segment Performance Metrics")
    
    display_metrics = cluster_metrics[[
        'Customer_Count', 'Customer_Percentage', 'Avg_Recency', 
        'Avg_Frequency', 'Avg_Monetary', 'Total_Revenue', 'Avg_RFM_Score'
    ]].copy()
    
    display_metrics.columns = [
        'Customers', 'Percentage', 'Avg Recency (days)', 
        'Avg Frequency', 'Avg Revenue', 'Total Revenue', 'Avg RFM Score'
    ]
    
    st.dataframe(display_metrics.style.format({
        'Customers': '{:,}',
        'Percentage': '{:.1f}%',
        'Avg Recency (days)': '{:.1f}',
        'Avg Frequency': '{:.2f}',
        'Avg Revenue': '${:,.0f}',
        'Total Revenue': '${:,.0f}',
        'Avg RFM Score': '{:.1f}'
    }), use_container_width=True)
    
    # Detailed Cluster Analysis
    st.header("🎯 Detailed Segment Analysis")
    
    selected_cluster = st.selectbox(
        "Select Segment for Detailed Analysis",
        options=sorted(clustered_rfm['Cluster_Name'].unique()),
        help="Choose a customer segment to view detailed characteristics and recommendations"
    )
    
    if selected_cluster:
        cluster_data = clustered_rfm[clustered_rfm['Cluster_Name'] == selected_cluster]
        cluster_id = cluster_data['Cluster'].iloc[0]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
                <div class='metric-highlight'>
                    <h3>👥 Segment Size</h3>
                    <p style='font-size: 2rem; margin: 0;'>{len(cluster_data):,}</p>
                    <p>Customers ({len(cluster_data)/len(clustered_rfm)*100:.1f}%)</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class='metric-highlight'>
                    <h3>💰 Segment Revenue</h3>
                    <p style='font-size: 2rem; margin: 0;'>${cluster_data['monetary'].sum():,.0f}</p>
                    <p>Total Revenue</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            avg_rfm_score = (cluster_data['R_Score'].mean() + 
                           cluster_data['F_Score'].mean() + 
                           cluster_data['M_Score'].mean())
            st.markdown(f"""
                <div class='metric-highlight'>
                    <h3>📊 Avg RFM Score</h3>
                    <p style='font-size: 2rem; margin: 0;'>{avg_rfm_score:.1f}</p>
                    <p>Customer Value Score</p>
                </div>
            """, unsafe_allow_html=True)
        
        # Cluster statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Behavioral Metrics")
            
            metrics_data = {
                'Metric': ['Average Recency', 'Average Frequency', 'Average Revenue', 
                          'Average R Score', 'Average F Score', 'Average M Score'],
                'Value': [
                    f"{cluster_data['recency'].mean():.1f} days",
                    f"{cluster_data['frequency'].mean():.2f}",
                    f"${cluster_data['monetary'].mean():,.0f}",
                    f"{cluster_data['R_Score'].mean():.1f}",
                    f"{cluster_data['F_Score'].mean():.1f}",
                    f"{cluster_data['M_Score'].mean():.1f}"
                ]
            }
            
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.subheader("🎯 Segment Description")
            
            cluster_desc = analyzer.cluster_descriptions.get(cluster_id, 
                "This segment shows unique behavioral patterns based on their purchasing history, "
                "recency of purchases, frequency of transactions, and monetary value."
            )
            
            st.markdown(f"""
                <div class='cluster-card' style='border-left-color: {analyzer.cluster_colors[cluster_id % len(analyzer.cluster_colors)]}'>
                    <h4>💡 {selected_cluster}</h4>
                    <p>{cluster_desc}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Recommendations based on cluster type
            st.subheader("💡 Marketing Recommendations")
            
            recommendations = {
                0: [  # Budget Shoppers
                    "Offer price-sensitive promotions and discounts",
                    "Create bundle deals for cost-effectiveness",
                    "Highlight value-for-money products",
                    "Implement loyalty points system"
                ],
                1: [  # Premium Customers
                    "Offer exclusive premium products and early access",
                    "Provide personalized shopping experiences",
                    "Create VIP membership programs",
                    "Offer premium customer support"
                ],
                2: [  # Occasional Buyers
                    "Send targeted reactivation campaigns",
                    "Offer special discounts on their next purchase",
                    "Share product updates and new arrivals",
                    "Implement reminder emails for abandoned carts"
                ],
                3: [  # Loyal Regulars
                    "Reward with loyalty programs and exclusive offers",
                    "Provide early access to sales and new products",
                    "Request testimonials and referrals",
                    "Offer personalized recommendations"
                ],
                4: [  # High-Value VIPs
                    "Assign dedicated account managers",
                    "Offer custom solutions and personalized service",
                    "Provide exclusive event invitations",
                    "Create bespoke product offerings"
                ]
            }
            
            cluster_recs = recommendations.get(cluster_id, [
                "Monitor customer behavior for pattern changes",
                "Test different engagement strategies",
                "Gather customer feedback for insights",
                "Compare with other segments for opportunities"
            ])
            
            for i, rec in enumerate(cluster_recs, 1):
                st.write(f"{i}. {rec}")
    
    # Advanced Visualizations
    st.header("🎨 Advanced Cluster Visualizations")
    
    # PCA Visualization if enabled
    if use_pca and clustering_method == "K-Means":
        try:
            # Prepare data for PCA
            features = ['recency', 'frequency', 'monetary']
            X = clustered_rfm[features].fillna(clustered_rfm[features].mean())
            
            # Apply PCA
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(StandardScaler().fit_transform(X))
            
            # Create PCA dataframe
            pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
            pca_df['Cluster'] = clustered_rfm['Cluster']
            pca_df['Cluster_Name'] = clustered_rfm['Cluster_Name']
            pca_df['Revenue'] = clustered_rfm['monetary']
            
            # Create scatter plot
            fig = px.scatter(
                pca_df,
                x='PC1',
                y='PC2',
                color='Cluster_Name',
                size='Revenue',
                title='Customer Segments Visualization (PCA)',
                hover_data=['Revenue'],
                color_discrete_sequence=analyzer.cluster_colors
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Explain PCA variance
            col1, col2 = st.columns(2)
            with col1:
                st.metric("PCA Component 1 Variance", f"{pca.explained_variance_ratio_[0]*100:.1f}%")
            with col2:
                st.metric("PCA Component 2 Variance", f"{pca.explained_variance_ratio_[1]*100:.1f}%")
                
        except Exception as e:
            st.warning(f"PCA visualization not available: {str(e)}")
    
    # RFM 3D Visualization
    with st.expander("🔮 3D Cluster Visualization", expanded=False):
        try:
            fig = px.scatter_3d(
                clustered_rfm,
                x='recency',
                y='frequency', 
                z='monetary',
                color='Cluster_Name',
                size='monetary',
                title='3D RFM Cluster Visualization',
                hover_data=['customer_id'],
                color_discrete_sequence=analyzer.cluster_colors
            )
            
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"3D visualization error: {str(e)}")
    
    # Customer Details
    st.header("📋 Segment Customer Details")
    
    with st.expander("View Detailed Customer Data by Segment", expanded=False):
        segment_filter = st.multiselect(
            "Filter by Segments",
            options=clustered_rfm['Cluster_Name'].unique(),
            default=clustered_rfm['Cluster_Name'].unique()[:2]
        )
        
        if segment_filter:
            filtered_data = clustered_rfm[clustered_rfm['Cluster_Name'].isin(segment_filter)]
            st.dataframe(filtered_data, use_container_width=True)
        else:
            st.dataframe(clustered_rfm, use_container_width=True)
    
    # Export Options
    st.header("💾 Export Segment Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            "📥 Download Segmented Data (CSV)",
            data=clustered_rfm.to_csv(index=False),
            file_name="customer_segments.csv",
            mime="text/csv"
        )
    
    with col2:
        # Create segment summary report
        report = f"""
        CUSTOMER SEGMENTATION ANALYSIS REPORT
        Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
        
        SEGMENT OVERVIEW:
        - Total Customers: {len(clustered_rfm):,}
        - Total Revenue: ${clustered_rfm['monetary'].sum():,.0f}
        - Number of Segments: {clustered_rfm['Cluster'].nunique()}
        - Clustering Method: {clustering_method}
        
        SEGMENT SUMMARY:
        {display_metrics.to_string()}
        
        KEY INSIGHTS:
        1. Most valuable segment: {cluster_metrics.index[0]} with ${cluster_metrics.iloc[0]['Total_Revenue']:,.0f} revenue
        2. Largest segment: {cluster_metrics.index[0]} with {cluster_metrics.iloc[0]['Customer_Count']:,} customers
        3. Highest average revenue: {cluster_metrics.index[0]} with ${cluster_metrics.iloc[0]['Avg_Monetary']:,.0f}
        
        RECOMMENDATIONS:
        - Develop targeted marketing strategies for each segment
        - Allocate resources based on segment value and size
        - Monitor segment migration and behavior changes
        - Personalize customer experiences based on segment characteristics
        """
        
        st.download_button(
            "📋 Download Analysis Report",
            data=report,
            file_name="customer_segmentation_report.txt",
            mime="text/plain"
        )

# Main execution
if __name__ == "__main__":
    show_customer_segments()