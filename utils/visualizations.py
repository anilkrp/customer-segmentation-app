import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.colors import qualitative
import pandas as pd
import streamlit as st

class RFMVisualizations:
    def __init__(self, rfm_df):
        self.rfm_df = rfm_df
    
    def create_rfm_distribution(self):
        """Create RFM distribution plots"""
        fig = sp.make_subplots(
            rows=2, cols=2,
            subplot_titles=('Recency Distribution', 'Frequency Distribution', 
                          'Monetary Distribution', 'RFM Score Distribution'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Recency distribution
        fig.add_trace(
            go.Histogram(x=self.rfm_df['recency'], name='Recency', nbinsx=20),
            row=1, col=1
        )
        
        # Frequency distribution
        fig.add_trace(
            go.Histogram(x=self.rfm_df['frequency'], name='Frequency', nbinsx=20),
            row=1, col=2
        )
        
        # Monetary distribution
        fig.add_trace(
            go.Histogram(x=self.rfm_df['monetary'], name='Monetary', nbinsx=20),
            row=2, col=1
        )
        
        # RFM Score distribution
        fig.add_trace(
            go.Histogram(x=self.rfm_df['RFM_Sum'], name='RFM Score', nbinsx=15),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False, title_text="RFM Distributions")
        return fig
    
    def create_segment_analysis(self):
        """Create segment analysis visualization"""
        segment_summary = self.rfm_df.groupby('RFM_Segment').agg({
            'recency': 'mean',
            'frequency': 'mean',
            'monetary': 'mean',
            'customer_id': 'count'
        }).reset_index()
        
        fig = sp.make_subplots(
            rows=2, cols=2,
            subplot_titles=('Segment Size', 'Average Recency by Segment',
                          'Average Frequency by Segment', 'Average Monetary by Segment'),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Segment size pie chart
        fig.add_trace(
            go.Pie(labels=segment_summary['RFM_Segment'], 
                   values=segment_summary['customer_id'],
                   name="Segment Size"),
            row=1, col=1
        )
        
        # Average recency by segment
        fig.add_trace(
            go.Bar(x=segment_summary['RFM_Segment'], 
                   y=segment_summary['recency'],
                   name="Avg Recency"),
            row=1, col=2
        )
        
        # Average frequency by segment
        fig.add_trace(
            go.Bar(x=segment_summary['RFM_Segment'], 
                   y=segment_summary['frequency'],
                   name="Avg Frequency"),
            row=2, col=1
        )
        
        # Average monetary by segment
        fig.add_trace(
            go.Bar(x=segment_summary['RFM_Segment'], 
                   y=segment_summary['monetary'],
                   name="Avg Monetary"),
            row=2, col=2
        )
        
        fig.update_layout(height=700, showlegend=False)
        return fig
    
    def create_3d_scatter(self):
        """Create 3D scatter plot of RFM segments"""
        fig = px.scatter_3d(
            self.rfm_df,
            x='R_Score',
            y='F_Score',
            z='M_Score',
            color='RFM_Segment',
            size='monetary',
            hover_data=['customer_id', 'recency', 'frequency', 'monetary'],
            title="3D RFM Segmentation",
            color_discrete_sequence=qualitative.Light24
        )
        
        fig.update_layout(
            scene=dict(
                xaxis_title='Recency Score (Higher = Better)',
                yaxis_title='Frequency Score (Higher = Better)',
                zaxis_title='Monetary Score (Higher = Better)'
            )
        )
        
        return fig
    
    def create_customer_journey(self, customer_id):
        """Create customer journey visualization for specific customer"""
        # This would require the original transaction data
        pass

def create_dashboard_metrics(df):
    """Create key metrics dashboard"""
    # Calculate key metrics
    total_revenue = df['total_amount'].sum()
    total_customers = df['customer_id'].nunique()
    avg_transaction = df['total_amount'].mean()
    transactions_per_customer = len(df) / total_customers
    
    metrics = {
        "Total Revenue": f"${total_revenue:,.0f}",
        "Total Customers": f"{total_customers:,}",
        "Avg Transaction": f"${avg_transaction:.0f}",
        "Transactions/Customer": f"{transactions_per_customer:.1f}"
    }
    
    return metrics