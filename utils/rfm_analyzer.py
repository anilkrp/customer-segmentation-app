import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import streamlit as st

class RFMAnalyzer:
    def __init__(self, df):
        self.df = df
        self.rfm_df = None
        self.scaler = StandardScaler()
    
    def calculate_rfm(self):
        """Calculate RFM metrics"""
        snapshot_date = self.df['date'].max() + timedelta(days=1)
        
        self.rfm_df = self.df.groupby('customer_id').agg({
            'date': lambda x: (snapshot_date - x.max()).days,  # Recency
            'transaction_id': 'count',                         # Frequency
            'total_amount': 'sum'                             # Monetary
        }).reset_index()
        
        self.rfm_df.columns = ['customer_id', 'recency', 'frequency', 'monetary']
        
        return self.rfm_df
    
    def calculate_rfm_scores(self, method='quantile'):
        """Calculate RFM scores"""
        if self.rfm_df is None:
            self.calculate_rfm()
        
        # Recency scoring (lower is better)
        if method == 'quantile':
            self.rfm_df['R_Score'] = pd.qcut(self.rfm_df['recency'], 5, labels=[5, 4, 3, 2, 1])
        else:
            self.rfm_df['R_Score'] = pd.cut(self.rfm_df['recency'], 5, labels=[5, 4, 3, 2, 1])
        
        # Frequency scoring (higher is better)
        if method == 'quantile':
            self.rfm_df['F_Score'] = pd.qcut(self.rfm_df['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
        else:
            self.rfm_df['F_Score'] = pd.cut(self.rfm_df['frequency'], 5, labels=[1, 2, 3, 4, 5])
        
        # Monetary scoring (higher is better)
        if method == 'quantile':
            self.rfm_df['M_Score'] = pd.qcut(self.rfm_df['monetary'], 5, labels=[1, 2, 3, 4, 5])
        else:
            self.rfm_df['M_Score'] = pd.cut(self.rfm_df['monetary'], 5, labels=[1, 2, 3, 4, 5])
        
        # Convert to numeric
        self.rfm_df['R_Score'] = self.rfm_df['R_Score'].astype(int)
        self.rfm_df['F_Score'] = self.rfm_df['F_Score'].astype(int)
        self.rfm_df['M_Score'] = self.rfm_df['M_Score'].astype(int)
        
        # Combined scores
        self.rfm_df['RFM_Score'] = self.rfm_df['R_Score'].astype(str) + self.rfm_df['F_Score'].astype(str) + self.rfm_df['M_Score'].astype(str)
        self.rfm_df['RFM_Sum'] = self.rfm_df['R_Score'] + self.rfm_df['F_Score'] + self.rfm_df['M_Score']
        
        return self.rfm_df
    
    def segment_customers(self):
        """Segment customers using traditional RFM approach"""
        conditions = [
            (self.rfm_df['RFM_Sum'] >= 13),
            (self.rfm_df['RFM_Sum'] >= 11) & (self.rfm_df['RFM_Sum'] < 13),
            (self.rfm_df['R_Score'] >= 4) & (self.rfm_df['F_Score'] <= 2),
            (self.rfm_df['F_Score'] >= 4) & (self.rfm_df['M_Score'] <= 2),
            (self.rfm_df['R_Score'] <= 2) & (self.rfm_df['F_Score'] >= 3),
            (self.rfm_df['R_Score'] <= 2) & (self.rfm_df['F_Score'] <= 2),
            (self.rfm_df['M_Score'] >= 4) & (self.rfm_df['F_Score'] >= 4),
        ]
        
        segments = [
            'Champions', 'Loyal Customers', 'New Customers',
            'Frequent Low Spenders', 'At Risk', 'Lost Customers', 'VIP Customers'
        ]
        
        self.rfm_df['RFM_Segment'] = np.select(conditions, segments, default='Regular Customers')
        return self.rfm_df
    
    def perform_clustering(self, n_clusters=4):
        """Perform K-means clustering"""
        features = ['R_Score', 'F_Score', 'M_Score']
        X = self.rfm_df[features]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.rfm_df['Cluster'] = kmeans.fit_predict(X_scaled)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(X_scaled, self.rfm_df['Cluster'])
        
        return self.rfm_df, silhouette_avg
    
    def get_segment_analysis(self):
        """Get comprehensive segment analysis"""
        if 'RFM_Segment' not in self.rfm_df.columns:
            self.segment_customers()
        
        segment_analysis = self.rfm_df.groupby('RFM_Segment').agg({
            'recency': ['mean', 'std'],
            'frequency': ['mean', 'std'],
            'monetary': ['mean', 'std', 'sum'],
            'customer_id': 'count'
        }).round(2)
        
        segment_analysis.columns = ['Avg_Recency', 'Std_Recency', 'Avg_Frequency', 'Std_Frequency',
                                 'Avg_Monetary', 'Std_Monetary', 'Total_Revenue', 'Customer_Count']
        
        segment_analysis['Revenue_Share_Pct'] = (segment_analysis['Total_Revenue'] / segment_analysis['Total_Revenue'].sum() * 100).round(2)
        segment_analysis['Customer_Share_Pct'] = (segment_analysis['Customer_Count'] / segment_analysis['Customer_Count'].sum() * 100).round(2)
        
        return segment_analysis.sort_values('Total_Revenue', ascending=False)