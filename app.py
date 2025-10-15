import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Advanced Customer Intelligence Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .homepage-header {
        text-align: center;
        padding: 3rem 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 2rem 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin: 0.5rem;
        border-left: 5px solid #667eea;
        height: 160px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    .cta-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 15px;
        color: white;
        margin-top: 2rem;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1.5rem;
        margin: 2.5rem 0;
    }
    .segment-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .upload-section {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
    }
    @media (max-width: 768px) {
        .feature-grid {
            grid-template-columns: 1fr;
        }
    }
</style>
""", unsafe_allow_html=True)

class AdvancedDataProcessor:
    def __init__(self):
        self.scaling_factors = {}
    
    def load_data(self, uploaded_file):
        """Advanced data loading with validation"""
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
            return df
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            return None
    
    def preprocess_data(self, df):
        """Advanced preprocessing with feature engineering"""
        df_clean = df.copy()
        
        # Standardize column names
        df_clean.columns = df_clean.columns.str.lower().str.replace(' ', '_')
        
        # Convert date and create advanced time features
        df_clean['date'] = pd.to_datetime(df_clean['date'])
        df_clean['year'] = df_clean['date'].dt.year
        df_clean['month'] = df_clean['date'].dt.month
        df_clean['quarter'] = df_clean['date'].dt.quarter
        df_clean['day_of_week'] = df_clean['date'].dt.day_name()
        df_clean['week_of_year'] = df_clean['date'].dt.isocalendar().week
        df_clean['is_weekend'] = df_clean['date'].dt.dayofweek.isin([5, 6]).astype(int)
        
        # Advanced feature engineering
        if 'price_per_unit' in df_clean.columns:
            df_clean['price_tier'] = pd.cut(df_clean['price_per_unit'], 
                                          bins=[0, 50, 200, 500, float('inf')],
                                          labels=['Budget', 'Standard', 'Premium', 'Luxury'])
        
        return df_clean

    def create_sample_data(self):
        """Create comprehensive sample data for demonstration - FIXED VERSION"""
        np.random.seed(42)
        n_customers = 200
        n_transactions = 2000
        
        # Generate customer base
        customers = [f'CUST_{i:04d}' for i in range(1, n_customers + 1)]
        
        # Generate transaction dates across 18 months
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2024, 6, 30)
        date_range = pd.date_range(start_date, end_date, freq='D')
        
        sample_data = []
        for i in range(n_transactions):
            customer = np.random.choice(customers)
            date = np.random.choice(date_range)
            
            # FIXED: Convert to pandas Timestamp to access weekday
            date_ts = pd.Timestamp(date)
            
            # Create realistic spending patterns
            base_amount = np.random.normal(75, 25)
            weekend_multiplier = 1.2 if date_ts.weekday() >= 5 else 1.0
            amount = max(base_amount * weekend_multiplier, 10)
            
            product_categories = ['Electronics', 'Clothing', 'Home & Garden', 'Books', 'Sports', 'Beauty']
            category = np.random.choice(product_categories, p=[0.3, 0.25, 0.2, 0.1, 0.1, 0.05])
            
            sample_data.append({
                'customer_id': customer,
                'date': date,
                'total_amount': round(amount, 2),
                'product_category': category,
                'transaction_id': f'TXN_{i:06d}',
                'price_per_unit': round(np.random.normal(60, 20), 2)
            })
        
        df = pd.DataFrame(sample_data)
        return df

class AdvancedRFMAnalyzer:
    def __init__(self):
        self.segment_definitions = {
            'Champions': {'min_score': 13, 'color': '#2E8B57', 'icon': '🏆'},
            'Loyal Customers': {'min_score': 11, 'color': '#4169E1', 'icon': '🤝'},
            'Potential Loyalists': {'min_score': 9, 'color': '#FFD700', 'icon': '⭐'},
            'Recent Customers': {'min_score': 8, 'color': '#32CD32', 'icon': '🆕'},
            'Promising': {'min_score': 7, 'color': '#FFA500', 'icon': '📈'},
            'Need Attention': {'min_score': 6, 'color': '#FF6347', 'icon': '⚠️'},
            'About To Sleep': {'min_score': 5, 'color': '#A9A9A9', 'icon': '😴'},
            'At Risk': {'min_score': 4, 'color': '#DC143C', 'icon': '🎯'},
            'Can\'t Lose Them': {'min_score': 3, 'color': '#8B0000', 'icon': '💔'},
            'Lost': {'min_score': 0, 'color': '#696969', 'icon': '👋'}
        }
    
    def calculate_advanced_rfm(self, df):
        """Advanced RFM calculation with dynamic scoring"""
        snapshot_date = df['date'].max() + timedelta(days=1)
        
        # Aggregate customer data
        rfm_df = df.groupby('customer_id').agg({
            'date': lambda x: (snapshot_date - x.max()).days,
            'transaction_id': 'count',
            'total_amount': 'sum',
            'product_category': 'nunique'
        }).reset_index()
        
        rfm_df.columns = ['customer_id', 'recency', 'frequency', 'monetary', 'product_variety']
        
        # Advanced scoring with error handling
        try:
            # Recency: lower is better (5 being most recent)
            rfm_df['R_Score'] = pd.qcut(rfm_df['recency'], 5, labels=[5, 4, 3, 2, 1], duplicates='drop')
            
            # Frequency: higher is better
            rfm_df['F_Score'] = pd.qcut(rfm_df['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
            
            # Monetary: higher is better
            rfm_df['M_Score'] = pd.qcut(rfm_df['monetary'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
            
            # Variety score with flexible binning
            unique_variety = rfm_df['product_variety'].nunique()
            if unique_variety >= 3:
                rfm_df['V_Score'] = pd.qcut(rfm_df['product_variety'], 3, labels=[1, 2, 3], duplicates='drop')
            else:
                rfm_df['V_Score'] = pd.cut(rfm_df['product_variety'], 
                                         bins=[0, 1, 2, float('inf')], 
                                         labels=[1, 2, 3], 
                                         right=False)
            
        except Exception as e:
            st.error(f"❌ Error in RFM scoring: {str(e)}")
            # Fallback scoring
            rfm_df['R_Score'] = pd.cut(rfm_df['recency'], 5, labels=[5, 4, 3, 2, 1])
            rfm_df['F_Score'] = pd.cut(rfm_df['frequency'], 5, labels=[1, 2, 3, 4, 5])
            rfm_df['M_Score'] = pd.cut(rfm_df['monetary'], 5, labels=[1, 2, 3, 4, 5])
            rfm_df['V_Score'] = pd.cut(rfm_df['product_variety'], 3, labels=[1, 2, 3])
        
        # Convert to numeric with error handling
        for col in ['R_Score', 'F_Score', 'M_Score', 'V_Score']:
            try:
                rfm_df[col] = rfm_df[col].astype(int)
            except:
                rfm_df[col] = rfm_df[col].astype('category').cat.codes + 1
        
        # Combined scores
        rfm_df['RFM_Score'] = rfm_df['R_Score'] + rfm_df['F_Score'] + rfm_df['M_Score']
        rfm_df['RFMV_Score'] = rfm_df['R_Score'] + rfm_df['F_Score'] + rfm_df['M_Score'] + rfm_df['V_Score']
        
        # Advanced segmentation
        rfm_df['RFM_Segment'] = self.assign_advanced_segments(rfm_df)
        rfm_df['Customer_Value_Tier'] = self.assign_value_tiers(rfm_df)
        
        return rfm_df
    
    def assign_advanced_segments(self, rfm_df):
        """Assign customers to advanced segments"""
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
            'About To Sleep', 'At Risk', 'Can\'t Lose Them', 'Lost'
        ]
        
        return np.select(conditions, segments, default='Neutral')
    
    def assign_value_tiers(self, rfm_df):
        """Assign customer value tiers"""
        try:
            monetary_quantiles = rfm_df['monetary'].quantile([0.2, 0.5, 0.8])
            
            conditions = [
                rfm_df['monetary'] >= monetary_quantiles[0.8],
                rfm_df['monetary'] >= monetary_quantiles[0.5],
                rfm_df['monetary'] >= monetary_quantiles[0.2]
            ]
            
            tiers = ['Platinum', 'Gold', 'Silver']
            return np.select(conditions, tiers, default='Bronze')
        except:
            return 'Standard'

def calculate_customer_stats(df):
    """Calculate comprehensive customer statistics"""
    try:
        customer_stats = df.groupby('customer_id').agg({
            'transaction_id': 'count',
            'total_amount': 'sum',
            'date': ['min', 'max']
        }).reset_index()
        
        # Flatten column names
        customer_stats.columns = ['customer_id', 'transaction_count', 'total_spent', 'first_purchase', 'last_purchase']
        
        # Calculate additional metrics
        customer_stats['days_active'] = (customer_stats['last_purchase'] - customer_stats['first_purchase']).dt.days
        customer_stats['days_active'] = customer_stats['days_active'].replace(0, 1)  # Avoid division by zero
        
        # Purchase frequency (transactions per month)
        customer_stats['purchase_frequency'] = customer_stats['transaction_count'] / (customer_stats['days_active'] / 30.44)
        
        # Average transaction value
        customer_stats['avg_transaction_value'] = customer_stats['total_spent'] / customer_stats['transaction_count']
        
        return customer_stats
        
    except Exception as e:
        st.error(f"Error calculating customer stats: {str(e)}")
        return None

def show_advanced_welcome():
    """Show advanced welcome page with professional design"""
    # Homepage Header
    st.markdown("""
        <div class='homepage-header'>
            <h1 style='color: white; margin-bottom: 1rem; font-size: 2.5rem;'>Advanced Customer Intelligence Platform</h1>
            <p style='font-size: 1.3rem; color: white; margin: 0; opacity: 0.9;'>
                Transform your customer data into actionable intelligence with advanced analytics and AI-powered insights.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Feature Cards Grid
    st.markdown("<div class='feature-grid'>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class='metric-card'>
                <h3 style='color: #667eea; margin-bottom: 1rem;'>📊 Advanced Analytics</h3>
                <p style='color: #666; margin: 0;'>Multi-dimensional customer segmentation and behavioral analysis</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class='metric-card'>
                <h3 style='color: #667eea; margin-bottom: 1rem;'>🔮 Predictive Analytics</h3>
                <p style='color: #666; margin: 0;'>CLV Forecasting & Churn prediction with machine learning</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='metric-card'>
                <h3 style='color: #667eea; margin-bottom: 1rem;'>🤖 AI Insights</h3>
                <p style='color: #666; margin: 0;'>Machine learning recommendations and automated insights</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class='metric-card'>
                <h3 style='color: #667eea; margin-bottom: 1rem;'>📈 Real-Time Dashboards</h3>
                <p style='color: #666; margin: 0;'>Interactive visualization and real-time monitoring</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # CTA Section
    st.markdown("""
        <div class='cta-section'>
            <h3 style='color: white; margin-top: 0; margin-bottom: 1.5rem;'>🚀 Get Started with Customer Intelligence</h3>
            <p style='color: white; font-size: 1.1rem; margin-bottom: 1.5rem;'>Unlock your customer data potential with our advanced platform:</p>
            <div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;'>
                <div style='color: white;'>✅ Advanced Customer Segmentation</div>
                <div style='color: white;'>✅ Customer Lifetime Value Prediction</div>
                <div style='color: white;'>✅ Churn Risk Analysis</div>
                <div style='color: white;'>✅ Personalized Marketing Strategies</div>
            </div>
            <p style='color: white; font-size: 1.1rem; margin-top: 2rem;'>
                <strong>👈 Upload your dataset in the sidebar</strong> or use sample data to explore the platform!
            </p>
        </div>
    """, unsafe_allow_html=True)

def show_data_upload_section():
    """Show data upload and processing section"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("📁 Data Management")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload Customer Data (CSV or Excel)",
        type=['csv', 'xlsx'],
        help="Upload file with columns: customer_id, date, total_amount, product_category"
    )
    
    use_sample_data = st.sidebar.checkbox("Use Sample Data", value=False, 
                                        help="Generate realistic sample data for demonstration")
    
    return uploaded_file, use_sample_data

def show_quick_insights(df, customer_stats, rfm_df):
    """Show quick insights on the main page"""
    if df is None:
        return
    
    st.header("🚀 Quick Insights")
    
    # Basic metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_revenue = df['total_amount'].sum()
        st.metric("Total Revenue", f"${total_revenue:,.0f}")
    
    with col2:
        total_customers = customer_stats['customer_id'].nunique()
        st.metric("Total Customers", f"{total_customers:,}")
    
    with col3:
        avg_transaction = df['total_amount'].mean()
        st.metric("Avg Transaction", f"${avg_transaction:.0f}")
    
    with col4:
        repeat_rate = (len(customer_stats[customer_stats['transaction_count'] > 1]) / len(customer_stats)) * 100
        st.metric("Repeat Rate", f"{repeat_rate:.1f}%")
    
    # Quick charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Revenue trend
        monthly_revenue = df.groupby(df['date'].dt.to_period('M'))['total_amount'].sum()
        monthly_revenue.index = monthly_revenue.index.astype(str)
        
        fig = px.line(
            monthly_revenue, 
            title="Monthly Revenue Trend",
            labels={'value': 'Revenue ($)', 'index': 'Month'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Customer segments
        if rfm_df is not None and 'RFM_Segment' in rfm_df.columns:
            segment_counts = rfm_df['RFM_Segment'].value_counts().head(5)
            fig = px.pie(
                values=segment_counts.values,
                names=segment_counts.index,
                title="Top 5 Customer Segments"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Category distribution as fallback
            if 'product_category' in df.columns:
                category_sales = df.groupby('product_category')['total_amount'].sum().nlargest(5)
                fig = px.bar(
                    x=category_sales.index,
                    y=category_sales.values,
                    title="Top 5 Categories by Revenue"
                )
                st.plotly_chart(fig, use_container_width=True)

def main():
    # Initialize session state
    if 'processor' not in st.session_state:
        st.session_state.processor = AdvancedDataProcessor()
    if 'rfm_analyzer' not in st.session_state:
        st.session_state.rfm_analyzer = AdvancedRFMAnalyzer()
    
    # Sidebar
    st.sidebar.title("🎯 Customer Intelligence Platform")
    
    # Navigation
    st.sidebar.subheader("📊 Navigation")
    page_options = ["Home", "Dashboard", "RFM Analysis", "Customer Segments", "Recommendations", "Advanced Analytics"]
    selected_page = st.sidebar.radio("Go to", page_options)
    
    # Data upload section
    uploaded_file, use_sample_data = show_data_upload_section()
    
    # Initialize data variables
    df = None
    customer_stats = None
    rfm_df = None
    
    # Process uploaded file
    if uploaded_file is not None:
        # Show spinner in main area for data processing
        with st.spinner("🔄 Processing uploaded data..."):
            df = st.session_state.processor.load_data(uploaded_file)
            if df is not None:
                df = st.session_state.processor.preprocess_data(df)
                customer_stats = calculate_customer_stats(df)
                
                # Calculate RFM
                try:
                    rfm_df = st.session_state.rfm_analyzer.calculate_advanced_rfm(df)
                    st.sidebar.success("✅ RFM analysis completed!")
                except Exception as e:
                    st.sidebar.error(f"❌ RFM analysis failed: {str(e)}")
    
    # Process sample data
    elif use_sample_data and df is None:
        # Show spinner in main area for data processing
        with st.spinner("🔄 Generating sample data..."):
            df = st.session_state.processor.create_sample_data()
            customer_stats = calculate_customer_stats(df)
            
            # Calculate RFM for sample data
            try:
                rfm_df = st.session_state.rfm_analyzer.calculate_advanced_rfm(df)
                st.sidebar.success("✅ Sample data ready!")
            except Exception as e:
                st.sidebar.error(f"❌ RFM analysis failed: {str(e)}")
    
    # Store in session state for other pages
    if df is not None:
        st.session_state.df_processed = df
        st.session_state.customer_stats = customer_stats
        if rfm_df is not None:
            st.session_state.rfm_df = rfm_df
    
    # Show selected page
    if selected_page == "Home":
        show_advanced_welcome()
        
        # Show quick insights if data is available
        if df is not None:
            show_quick_insights(df, customer_stats, rfm_df)
        else:
            st.info("""
            ## 📊 Ready to Get Started?
            
            To begin your customer intelligence journey:
            
            1. **Upload your customer data** using the file uploader in the sidebar
            2. **Or use sample data** by checking "Use Sample Data"
            3. **Navigate to different sections** using the sidebar menu
            
            **Required data columns:**
            - `customer_id`: Unique customer identifier
            - `date`: Transaction date
            - `total_amount`: Transaction amount
            - `product_category`: Product category (optional but recommended)
            """)
    
    elif selected_page == "Dashboard":
        if 'df_processed' not in st.session_state:
            st.warning("⚠️ Please upload data first or use sample data!")
            return
        
        st.markdown('<h1 class="main-header">📊 Advanced Dashboard</h1>', unsafe_allow_html=True)
        show_quick_insights(st.session_state.df_processed, 
                          st.session_state.customer_stats, 
                          st.session_state.rfm_df if 'rfm_df' in st.session_state else None)
    
    elif selected_page == "RFM Analysis":
        if 'df_processed' not in st.session_state:
            st.warning("⚠️ Please upload data first or use sample data!")
            return
        
        st.markdown('<h1 class="main-header">🔍 RFM Analysis</h1>', unsafe_allow_html=True)
        st.info("Navigate to the RFM Analysis page for detailed segmentation insights")
        # Add RFM-specific content here
    
    elif selected_page == "Customer Segments":
        if 'df_processed' not in st.session_state:
            st.warning("⚠️ Please upload data first or use sample data!")
            return
        
        st.markdown('<h1 class="main-header">👥 Customer Segments</h1>', unsafe_allow_html=True)
        st.info("Navigate to the Customer Segments page for detailed customer grouping")
        # Add segment-specific content here
    
    elif selected_page == "Recommendations":
        if 'df_processed' not in st.session_state:
            st.warning("⚠️ Please upload data first or use sample data!")
            return
        
        st.markdown('<h1 class="main-header">💡 Recommendations</h1>', unsafe_allow_html=True)
        st.info("Navigate to the Recommendations page for AI-powered insights")
        # Add recommendations content here
    
    elif selected_page == "Advanced Analytics":
        if 'df_processed' not in st.session_state:
            st.warning("⚠️ Please upload data first or use sample data!")
            return
        
        st.markdown('<h1 class="main-header">📈 Advanced Analytics</h1>', unsafe_allow_html=True)
        st.info("Navigate to the Advanced Analytics page for deep customer insights")
        # Add advanced analytics content here
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **💡 Tips:**
    - Use sample data to explore features
    - Upload your own CSV/Excel for real insights
    - All pages share the same processed data
    """)
    
    # Main area footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; margin-top: 2rem;'>"
        "Advanced Customer Intelligence Platform • Powered by Streamlit • "
        f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()