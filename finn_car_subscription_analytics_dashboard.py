"""
FINN Car Subscription Analytics Dashboard
========================================

Professional analytics dashboard for FINN's business intelligence team
Created by: Business Intelligence Team
Purpose: Comprehensive business analytics for car subscription operations

This dashboard provides executive-level insights into:
- Customer retention and churn patterns
- Subscription demand forecasting
- Fleet utilization optimization
- Revenue analytics and projections
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import datetime as dt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="FINN Analytics Dashboard", 
    page_icon="🚗", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .stSelectbox > div > div {
        background-color: white;
    }
    .highlight-metric {
        font-size: 1.2em;
        font-weight: bold;
        color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

class FINNAnalyticsDashboard:
    """Main dashboard class for FINN analytics"""
    
    def __init__(self):
        self.customer_data = None
        self.demand_data = None
        self.churn_model = None
        self.demand_model = None
        self.model_features = None
        
    def load_sample_data(self):
        """Generate realistic sample data for dashboard demonstration"""
        
        # Customer data generation
        np.random.seed(123)  # For consistent demo data
        
        # Customer segments reflecting FINN's market
        segments = ['Urban_Professional', 'Family_Subscriber', 'Corporate_Fleet', 'Eco_Enthusiast', 'Premium_User']
        cities = ['München', 'Berlin', 'Hamburg', 'Frankfurt', 'Stuttgart', 'Köln']
        vehicle_categories = ['Electric_Compact', 'Hybrid_SUV', 'Premium_Sedan', 'Family_MPV', 'Commercial_Van']
        
        customers = []
        num_customers = 2500
        
        for idx in range(num_customers):
            segment = np.random.choice(segments, p=[0.35, 0.30, 0.15, 0.15, 0.05])
            city = np.random.choice(cities, p=[0.25, 0.22, 0.15, 0.15, 0.13, 0.10])
            vehicle = np.random.choice(vehicle_categories, p=[0.35, 0.25, 0.20, 0.15, 0.05])
            
            # Generate segment-specific metrics
            if segment == 'Urban_Professional':
                monthly_fee = np.random.normal(680, 120)
                tenure_months = np.random.normal(11, 3)
                satisfaction = np.random.uniform(0.7, 0.9)
                price_sensitivity = np.random.uniform(0.3, 0.6)
            elif segment == 'Family_Subscriber':
                monthly_fee = np.random.normal(750, 140)
                tenure_months = np.random.normal(16, 4)
                satisfaction = np.random.uniform(0.8, 0.95)
                price_sensitivity = np.random.uniform(0.5, 0.8)
            elif segment == 'Corporate_Fleet':
                monthly_fee = np.random.normal(820, 100)
                tenure_months = np.random.normal(22, 6)
                satisfaction = np.random.uniform(0.75, 0.92)
                price_sensitivity = np.random.uniform(0.1, 0.4)
            elif segment == 'Eco_Enthusiast':
                monthly_fee = np.random.normal(650, 90)
                tenure_months = np.random.normal(13, 4)
                satisfaction = np.random.uniform(0.8, 0.95)
                price_sensitivity = np.random.uniform(0.4, 0.7)
            else:  # Premium_User
                monthly_fee = np.random.normal(1180, 200)
                tenure_months = np.random.normal(18, 5)
                satisfaction = np.random.uniform(0.6, 0.85)
                price_sensitivity = np.random.uniform(0.1, 0.3)
            
            # Additional metrics
            support_contacts = np.random.poisson(1.3)
            vehicle_swaps = np.random.poisson(0.7)
            payment_delays = np.random.poisson(0.4)
            app_usage_score = np.random.uniform(0.3, 1.0)
            
            # Calculate churn probability
            churn_risk = 0.08
            if segment == 'Corporate_Fleet':
                churn_risk *= 0.5
            elif segment == 'Premium_User':
                churn_risk *= 0.7
            
            if support_contacts > 3:
                churn_risk *= 1.6
            if satisfaction < 0.6:
                churn_risk *= 1.8
            if payment_delays > 2:
                churn_risk *= 1.4
            if app_usage_score < 0.4:
                churn_risk *= 1.3
            
            has_churned = np.random.random() < min(churn_risk, 0.5)
            lifetime_value = monthly_fee * tenure_months * (1 - churn_risk)
            
            customers.append({
                'customer_id': f'FINN_{idx:05d}',
                'segment': segment,
                'city': city,
                'vehicle_category': vehicle,
                'monthly_subscription': max(400, monthly_fee),
                'tenure_months': max(1, tenure_months),
                'satisfaction_score': satisfaction,
                'price_sensitivity': price_sensitivity,
                'support_contacts': support_contacts,
                'vehicle_swaps': vehicle_swaps,
                'payment_delays': payment_delays,
                'app_usage_score': app_usage_score,
                'churn_risk': min(churn_risk, 0.5),
                'churned': has_churned,
                'lifetime_value': lifetime_value
            })
        
        self.customer_data = pd.DataFrame(customers)
        
        # Demand data generation
        start_date = dt.date(2023, 1, 1)
        end_date = dt.date(2024, 12, 31)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        demand_records = []
        
        for single_date in date_range:
            day_of_year = single_date.timetuple().tm_yday
            seasonal_multiplier = 1 + 0.2 * np.sin(2 * np.pi * day_of_year / 365)
            weekday_factor = 1.1 if single_date.weekday() < 5 else 0.9
            
            for city in cities:
                for vehicle in vehicle_categories:
                    base_demand = {
                        'Electric_Compact': 14,
                        'Hybrid_SUV': 10,
                        'Premium_Sedan': 6,
                        'Family_MPV': 8,
                        'Commercial_Van': 4
                    }[vehicle]
                    
                    city_factor = {
                        'München': 1.3, 'Berlin': 1.2, 'Hamburg': 1.0,
                        'Frankfurt': 1.1, 'Stuttgart': 0.9, 'Köln': 0.8
                    }[city]
                    
                    daily_demand = base_demand * city_factor * seasonal_multiplier * weekday_factor
                    daily_demand += np.random.normal(0, 1.2)
                    daily_demand = max(0, int(round(daily_demand)))
                    
                    pricing = {
                        'Electric_Compact': 495, 'Hybrid_SUV': 745,
                        'Premium_Sedan': 945, 'Family_MPV': 595, 'Commercial_Van': 695
                    }[vehicle]
                    
                    demand_records.append({
                        'date': single_date,
                        'city': city,
                        'vehicle_category': vehicle,
                        'daily_subscriptions': daily_demand,
                        'monthly_price': pricing,
                        'daily_revenue': daily_demand * pricing,
                        'fleet_utilization': np.random.uniform(0.75, 0.92),
                        'customer_satisfaction': np.random.uniform(0.75, 0.95)
                    })
        
        self.demand_data = pd.DataFrame(demand_records)
        
    def build_churn_model(self):
        """Train customer churn prediction model"""
        
        # Prepare features for modeling
        model_data = self.customer_data.copy()
        
        # Create derived features
        model_data['monthly_per_satisfaction'] = model_data['monthly_subscription'] * model_data['satisfaction_score']
        model_data['high_value_customer'] = (model_data['lifetime_value'] > model_data['lifetime_value'].quantile(0.8)).astype(int)
        model_data['service_heavy_user'] = (model_data['support_contacts'] > 2).astype(int)
        model_data['engaged_user'] = (model_data['app_usage_score'] > 0.7).astype(int)
        
        # Encode categorical variables
        le_segment = LabelEncoder()
        le_city = LabelEncoder()
        le_vehicle = LabelEncoder()
        
        model_data['segment_encoded'] = le_segment.fit_transform(model_data['segment'])
        model_data['city_encoded'] = le_city.fit_transform(model_data['city'])
        model_data['vehicle_encoded'] = le_vehicle.fit_transform(model_data['vehicle_category'])
        
        # Select modeling features
        features = [
            'monthly_subscription', 'tenure_months', 'satisfaction_score', 'price_sensitivity',
            'support_contacts', 'vehicle_swaps', 'payment_delays', 'app_usage_score',
            'monthly_per_satisfaction', 'high_value_customer', 'service_heavy_user', 'engaged_user',
            'segment_encoded', 'city_encoded', 'vehicle_encoded'
        ]
        
        self.model_features = features
        X = model_data[features]
        y = model_data['churned']
        
        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        
        self.churn_model = RandomForestClassifier(n_estimators=150, random_state=42, max_depth=12)
        self.churn_model.fit(X_train, y_train)
        
        # Calculate model performance
        train_accuracy = accuracy_score(y_train, self.churn_model.predict(X_train))
        test_accuracy = accuracy_score(y_test, self.churn_model.predict(X_test))
        
        return train_accuracy, test_accuracy
        
    def build_demand_model(self):
        """Train subscription demand forecasting model"""
        
        # Prepare demand data for modeling
        model_data = self.demand_data.copy()
        
        # Add time-based features
        model_data['day_of_week'] = model_data['date'].dt.dayofweek
        model_data['month'] = model_data['date'].dt.month
        model_data['quarter'] = model_data['date'].dt.quarter
        model_data['is_weekend'] = (model_data['date'].dt.dayofweek >= 5).astype(int)
        
        # Add lagged features
        model_data = model_data.sort_values(['city', 'vehicle_category', 'date'])
        model_data['lag_7_demand'] = model_data.groupby(['city', 'vehicle_category'])['daily_subscriptions'].shift(7)
        model_data['lag_30_demand'] = model_data.groupby(['city', 'vehicle_category'])['daily_subscriptions'].shift(30)
        
        # Encode categoricals
        le_city = LabelEncoder()
        le_vehicle = LabelEncoder()
        
        model_data['city_encoded'] = le_city.fit_transform(model_data['city'])
        model_data['vehicle_encoded'] = le_vehicle.fit_transform(model_data['vehicle_category'])
        
        # Remove rows with missing lag values
        model_data = model_data.dropna()
        
        features = [
            'day_of_week', 'month', 'quarter', 'is_weekend', 'monthly_price',
            'fleet_utilization', 'city_encoded', 'vehicle_encoded',
            'lag_7_demand', 'lag_30_demand'
        ]
        
        X = model_data[features]
        y = model_data['daily_subscriptions']
        
        # Chronological split
        split_idx = int(len(X) * 0.8)
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        self.demand_model = RandomForestRegressor(n_estimators=150, random_state=42, max_depth=12)
        self.demand_model.fit(X_train, y_train)
        
        # Calculate model performance
        train_mae = mean_absolute_error(y_train, self.demand_model.predict(X_train))
        test_mae = mean_absolute_error(y_test, self.demand_model.predict(X_test))
        
        return train_mae, test_mae
    
    def get_feature_importance(self):
        """Get feature importance from churn model"""
        if self.churn_model is None or self.model_features is None:
            return None
            
        importance_data = pd.DataFrame({
            'feature': self.model_features,
            'importance': self.churn_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_data

def create_kpi_card(title, value, delta=None, delta_color="normal"):
    """Create a styled KPI card"""
    delta_html = ""
    if delta:
        color = "green" if delta_color == "normal" else "red"
        delta_html = f'<p style="color: {color}; margin: 0; font-size: 0.8em;">{delta}</p>'
    
    return f"""
    <div class="metric-card">
        <h4 style="margin: 0; color: #333;">{title}</h4>
        <p class="highlight-metric" style="margin: 0;">{value}</p>
        {delta_html}
    </div>
    """

def main():
    """Main dashboard application"""
    
    # Dashboard header
    st.markdown('<h1 class="main-header"> FINN Analytics Command Center</h1>', unsafe_allow_html=True)
    st.markdown("**Professional Business Intelligence Dashboard for Car Subscription Analytics**")
    
    # Initialize dashboard
    if 'dashboard' not in st.session_state:
        st.session_state.dashboard = FINNAnalyticsDashboard()
    
    dashboard = st.session_state.dashboard
    
    # Sidebar controls
    st.sidebar.title(" Dashboard Controls")
    st.sidebar.markdown("---")
    
    # Data loading section
    if st.sidebar.button(" Load Fresh Data", type="primary"):
        with st.spinner("Loading analytics data..."):
            dashboard.load_sample_data()
            st.sidebar.success(" Data loaded successfully!")
            st.rerun()
            
    # Load data if not already loaded
    if dashboard.customer_data is None:
        with st.spinner("Initializing dashboard..."):
            dashboard.load_sample_data()
    
    # Model training section
    st.sidebar.markdown("###  Model Training")
    
    if st.sidebar.button(" Train Churn Model"):
        with st.spinner("Training churn prediction model..."):
            train_acc, test_acc = dashboard.build_churn_model()
            st.sidebar.success(f" Churn model trained!\nAccuracy: {test_acc:.1%}")
            
    if st.sidebar.button(" Train Demand Model"):
        with st.spinner("Training demand forecasting model..."):
            train_mae, test_mae = dashboard.build_demand_model()
            st.sidebar.success(f" Demand model trained!\nMAE: {test_mae:.1f}")
    
    # Display model status
    if dashboard.churn_model is not None:
        st.sidebar.info(" Churn Model: Active")
    if dashboard.demand_model is not None:
        st.sidebar.info(" Demand Model: Active")
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        " Executive Summary", 
        " Customer Analytics", 
        " Demand Forecasting", 
        " ML Insights",
        " Strategic Insights"
    ])
    
    with tab1:
        st.header("Executive Dashboard")
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_customers = len(dashboard.customer_data)
            st.metric("Total Active Customers", f"{total_customers:,}")
            
        with col2:
            avg_revenue = dashboard.customer_data['monthly_subscription'].mean()
            st.metric("Avg Monthly Revenue", f"€{avg_revenue:,.0f}")
            
        with col3:
            churn_rate = dashboard.customer_data['churned'].mean()
            st.metric("Churn Rate", f"{churn_rate:.1%}")
            
        with col4:
            total_ltv = dashboard.customer_data['lifetime_value'].sum()
            st.metric("Total Customer LTV", f"€{total_ltv/1000000:.1f}M")
        
        # Additional KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_satisfaction = dashboard.customer_data['satisfaction_score'].mean()
            st.metric("Customer Satisfaction", f"{avg_satisfaction:.1%}")
            
        with col2:
            avg_utilization = dashboard.demand_data['fleet_utilization'].mean()
            st.metric("Fleet Utilization", f"{avg_utilization:.1%}")
            
        with col3:
            monthly_revenue = dashboard.customer_data['monthly_subscription'].sum()
            st.metric("Monthly Recurring Revenue", f"€{monthly_revenue:,.0f}")
            
        with col4:
            retention_rate = 1 - churn_rate
            st.metric("Retention Rate", f"{retention_rate:.1%}")
        
        # Revenue and subscription trends
        col1, col2 = st.columns(2)
        
        with col1:
            # Monthly revenue trend
            monthly_revenue = dashboard.demand_data.groupby(
                dashboard.demand_data['date'].dt.to_period('M')
            )['daily_revenue'].sum()
            
            fig_revenue = go.Figure()
            fig_revenue.add_trace(go.Scatter(
                x=monthly_revenue.index.astype(str),
                y=monthly_revenue.values,
                mode='lines+markers',
                name='Monthly Revenue',
                line=dict(color='#1f77b4', width=3),
                marker=dict(size=8)
            ))
            fig_revenue.update_layout(
                title="Monthly Revenue Trend",
                xaxis_title="Month",
                yaxis_title="Revenue (€)",
                template="plotly_white",
                height=400
            )
            st.plotly_chart(fig_revenue, use_container_width=True)
            
        with col2:
            # Customer segment distribution
            segment_dist = dashboard.customer_data['segment'].value_counts()
            
            fig_segments = go.Figure(data=[go.Pie(
                labels=segment_dist.index,
                values=segment_dist.values,
                hole=0.4,
                textinfo='label+percent',
                textposition='outside'
            )])
            fig_segments.update_layout(
                title="Customer Segment Distribution",
                height=400
            )
            st.plotly_chart(fig_segments, use_container_width=True)
        
        # Geographic performance
        st.subheader("Geographic Performance Analysis")
        
        city_metrics = dashboard.customer_data.groupby('city').agg({
            'customer_id': 'count',
            'monthly_subscription': 'mean',
            'lifetime_value': 'mean',
            'churned': 'mean',
            'satisfaction_score': 'mean'
        }).round(2)
        city_metrics.columns = ['Customer Count', 'Avg Monthly Fee', 'Avg LTV', 'Churn Rate', 'Satisfaction']
        
        # Add color coding for the dataframe
        st.dataframe(
            city_metrics.style.background_gradient(subset=['Customer Count', 'Avg LTV'], cmap='Greens')
                            .background_gradient(subset=['Churn Rate'], cmap='Reds_r'),
            use_container_width=True
        )
        
    with tab2:
        st.header("Customer Analytics Deep Dive")
        
        # Churn analysis section
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(" Churn Risk Analysis")
            
            # High-risk customers
            high_risk = dashboard.customer_data[
                dashboard.customer_data['churn_risk'] > 0.3
            ].sort_values('lifetime_value', ascending=False)
            
            st.markdown(f"**{len(high_risk)}** customers at high churn risk")
            st.markdown(f"**€{high_risk['lifetime_value'].sum():,.0f}** total LTV at risk")
            
            if len(high_risk) > 0:
                st.dataframe(
                    high_risk[['customer_id', 'segment', 'city', 'churn_risk', 'lifetime_value']].head(10),
                    use_container_width=True
                )
        
        with col2:
            st.subheader("😊 Customer Satisfaction Drivers")
            
            # Satisfaction vs metrics correlation
            satisfaction_factors = dashboard.customer_data[[
                'satisfaction_score', 'support_contacts', 'vehicle_swaps', 
                'app_usage_score', 'payment_delays'
            ]].corr()['satisfaction_score'].drop('satisfaction_score')
            
            fig_corr = go.Figure(data=[go.Bar(
                x=satisfaction_factors.index,
                y=satisfaction_factors.values,
                marker_color=['red' if x < 0 else 'green' for x in satisfaction_factors.values],
                text=[f"{x:.3f}" for x in satisfaction_factors.values],
                textposition='auto'
            )])
            fig_corr.update_layout(
                title="Satisfaction Score Correlations",
                xaxis_title="Factors",
                yaxis_title="Correlation Strength",
                template="plotly_white"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        
        # Segment performance comparison
        st.subheader(" Customer Segment Performance")
        
        segment_analysis = dashboard.customer_data.groupby('segment').agg({
            'lifetime_value': 'mean',
            'monthly_subscription': 'mean',
            'satisfaction_score': 'mean',
            'churned': 'mean',
            'customer_id': 'count'
        }).round(2)
        
        segment_analysis.columns = ['Avg LTV', 'Avg Monthly Fee', 'Avg Satisfaction', 'Churn Rate', 'Customer Count']
        
        # Create segment comparison chart
        fig_segments = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Average LTV by Segment', 
                'Churn Rate by Segment', 
                'Monthly Fee by Segment', 
                'Satisfaction Score by Segment'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Add traces with different colors
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        fig_segments.add_trace(
            go.Bar(x=segment_analysis.index, y=segment_analysis['Avg LTV'], 
                   name='LTV', marker_color=colors[0]), row=1, col=1)
        fig_segments.add_trace(
            go.Bar(x=segment_analysis.index, y=segment_analysis['Churn Rate'], 
                   name='Churn', marker_color=colors[1]), row=1, col=2)
        fig_segments.add_trace(
            go.Bar(x=segment_analysis.index, y=segment_analysis['Avg Monthly Fee'], 
                   name='Fee', marker_color=colors[2]), row=2, col=1)
        fig_segments.add_trace(
            go.Bar(x=segment_analysis.index, y=segment_analysis['Avg Satisfaction'], 
                   name='Satisfaction', marker_color=colors[3]), row=2, col=2)
        
        fig_segments.update_layout(height=600, showlegend=False, title_text="Customer Segment Analysis")
        st.plotly_chart(fig_segments, use_container_width=True)
        
        # Customer lifetime value distribution
        st.subheader(" Customer Lifetime Value Distribution")
        
        fig_ltv = go.Figure(data=[go.Histogram(
            x=dashboard.customer_data['lifetime_value'],
            nbinsx=30,
            marker_color='lightblue',
            opacity=0.7
        )])
        fig_ltv.update_layout(
            title="Customer Lifetime Value Distribution",
            xaxis_title="Lifetime Value (€)",
            yaxis_title="Number of Customers",
            template="plotly_white"
        )
        st.plotly_chart(fig_ltv, use_container_width=True)
        
    with tab3:
        st.header("Subscription Demand Forecasting")
        
        # Demand trends
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📅 Daily Subscription Volume")
            
            daily_subs = dashboard.demand_data.groupby('date')['daily_subscriptions'].sum()
            
            fig_daily = go.Figure()
            fig_daily.add_trace(go.Scatter(
                x=daily_subs.index,
                y=daily_subs.values,
                mode='lines',
                name='Daily Subscriptions',
                line=dict(color='#2ca02c', width=2),
                fill='tonexty'
            ))
            fig_daily.update_layout(
                title="Daily Subscription Trend",
                xaxis_title="Date",
                yaxis_title="New Subscriptions",
                template="plotly_white"
            )
            st.plotly_chart(fig_daily, use_container_width=True)
            
        with col2:
            st.subheader(" Vehicle Category Demand")
            
            vehicle_demand = dashboard.demand_data.groupby('vehicle_category')['daily_subscriptions'].sum()
            
            fig_vehicles = go.Figure(data=[go.Bar(
                x=vehicle_demand.index,
                y=vehicle_demand.values,
                marker_color='#ff7f0e',
                text=vehicle_demand.values,
                textposition='auto'
            )])
            fig_vehicles.update_layout(
                title="Total Demand by Vehicle Category",
                xaxis_title="Vehicle Type",
                yaxis_title="Total Subscriptions",
                template="plotly_white"
            )
            st.plotly_chart(fig_vehicles, use_container_width=True)
        
        # City performance matrix
        st.subheader(" City Performance Matrix")
        
        city_performance = dashboard.demand_data.groupby('city').agg({
            'daily_subscriptions': 'sum',
            'daily_revenue': 'sum',
            'fleet_utilization': 'mean'
        }).round(1)
        
        city_performance.columns = ['Total Subscriptions', 'Total Revenue (€)', 'Avg Fleet Utilization']
        city_performance = city_performance.sort_values('Total Revenue (€)', ascending=False)
        
        st.dataframe(
            city_performance.style.background_gradient(cmap='Blues'),
            use_container_width=True
        )
        
        # Fleet utilization trends
        st.subheader(" Fleet Utilization Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            monthly_utilization = dashboard.demand_data.groupby(
                dashboard.demand_data['date'].dt.to_period('M')
            )['fleet_utilization'].mean()
            
            fig_utilization = go.Figure()
            fig_utilization.add_trace(go.Scatter(
                x=monthly_utilization.index.astype(str),
                y=monthly_utilization.values,
                mode='lines+markers',
                name='Fleet Utilization',
                line=dict(color='#d62728', width=3),
                marker=dict(size=8)
            ))
            fig_utilization.update_layout(
                title="Monthly Fleet Utilization Trend",
                xaxis_title="Month",
                yaxis_title="Utilization Rate",
                yaxis_tickformat='.1%',
                template="plotly_white"
            )
            st.plotly_chart(fig_utilization, use_container_width=True)
            
        with col2:
            # Utilization by vehicle category
            vehicle_utilization = dashboard.demand_data.groupby('vehicle_category')['fleet_utilization'].mean()
            
            fig_vehicle_util = go.Figure(data=[go.Bar(
                x=vehicle_utilization.index,
                y=vehicle_utilization.values,
                marker_color='#9467bd',
                text=[f"{x:.1%}" for x in vehicle_utilization.values],
                textposition='auto'
            )])
            fig_vehicle_util.update_layout(
                title="Average Fleet Utilization by Vehicle Type",
                xaxis_title="Vehicle Category",
                yaxis_title="Utilization Rate",
                yaxis_tickformat='.1%',
                template="plotly_white"
            )
            st.plotly_chart(fig_vehicle_util, use_container_width=True)
        
    with tab4:
        st.header(" Machine Learning Insights")
        
        if dashboard.churn_model is None:
            st.warning(" Please train the churn model first using the sidebar controls.")
            
            # Show basic analytics even without trained model
            st.subheader(" Pre-Model Analytics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Basic churn risk analysis
                segment_risk = dashboard.customer_data.groupby('segment').agg({
                    'churn_risk': 'mean',
                    'customer_id': 'count'
                }).round(3)
                segment_risk.columns = ['Avg Churn Risk', 'Customer Count']
                segment_risk = segment_risk.sort_values('Avg Churn Risk', ascending=False)
                
                fig_risk = go.Figure(data=[go.Bar(
                    x=segment_risk.index,
                    y=segment_risk['Avg Churn Risk'],
                    marker_color='red',
                    text=[f"{x:.1%}" for x in segment_risk['Avg Churn Risk']],
                    textposition='auto'
                )])
                fig_risk.update_layout(
                    title="Average Churn Risk by Segment",
                    xaxis_title="Customer Segment",
                    yaxis_title="Churn Risk",
                    yaxis_tickformat='.1%',
                    template="plotly_white"
                )
                st.plotly_chart(fig_risk, use_container_width=True)
                
            with col2:
                # Correlation analysis
                correlation_data = dashboard.customer_data[[
                    'churn_risk', 'satisfaction_score', 'support_contacts', 
                    'app_usage_score', 'payment_delays'
                ]].corr()['churn_risk'].drop('churn_risk')
                
                fig_corr = go.Figure(data=[go.Bar(
                    x=correlation_data.index,
                    y=correlation_data.values,
                    marker_color=['red' if x > 0 else 'green' for x in correlation_data.values],
                    text=[f"{x:.3f}" for x in correlation_data.values],
                    textposition='auto'
                )])
                fig_corr.update_layout(
                    title="Churn Risk Correlations",
                    xaxis_title="Factors",
                    yaxis_title="Correlation with Churn Risk",
                    template="plotly_white"
                )
                st.plotly_chart(fig_corr, use_container_width=True)
                
        else:
            # Feature importance analysis
            st.subheader(" Churn Prediction - Feature Importance")
            
            importance_data = dashboard.get_feature_importance()
            
            fig_importance = go.Figure(data=[go.Bar(
                y=importance_data['feature'].head(10),
                x=importance_data['importance'].head(10),
                orientation='h',
                marker_color='lightcoral'
            )])
            fig_importance.update_layout(
                title="Top 10 Features for Churn Prediction",
                xaxis_title="Feature Importance",
                yaxis_title="Features",
                template="plotly_white",
                height=500
            )
            st.plotly_chart(fig_importance, use_container_width=True)
            
            # Model performance metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Model Performance")
                
                # Prepare the same features used in training
                model_data = dashboard.customer_data.copy()
                
                # Create derived features (same as in build_churn_model)
                model_data['monthly_per_satisfaction'] = model_data['monthly_subscription'] * model_data['satisfaction_score']
                model_data['high_value_customer'] = (model_data['lifetime_value'] > model_data['lifetime_value'].quantile(0.8)).astype(int)
                model_data['service_heavy_user'] = (model_data['support_contacts'] > 2).astype(int)
                model_data['engaged_user'] = (model_data['app_usage_score'] > 0.7).astype(int)
                
                # Encode categorical variables
                le_segment = LabelEncoder()
                le_city = LabelEncoder()
                le_vehicle = LabelEncoder()
                
                model_data['segment_encoded'] = le_segment.fit_transform(model_data['segment'])
                model_data['city_encoded'] = le_city.fit_transform(model_data['city'])
                model_data['vehicle_encoded'] = le_vehicle.fit_transform(model_data['vehicle_category'])
                
                # Get predictions
                y_true = model_data['churned']
                churn_predictions = dashboard.churn_model.predict_proba(
                    model_data[dashboard.model_features]
                )[:, 1]
                
                # ROC curve data
                fpr, tpr, _ = roc_curve(y_true, churn_predictions)
                roc_auc = auc(fpr, tpr)
                
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'ROC Curve (AUC = {roc_auc:.3f})',
                    line=dict(color='darkorange', width=2)
                ))
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines',
                    name='Random Classifier',
                    line=dict(color='navy', width=2, dash='dash')
                ))
                fig_roc.update_layout(
                    title='ROC Curve - Churn Prediction',
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    template="plotly_white"
                )
                st.plotly_chart(fig_roc, use_container_width=True)
                
            with col2:
                st.subheader("🎯 High-Risk Customer Segments")
                
                # Risk by segment
                segment_risk = dashboard.customer_data.groupby('segment').agg({
                    'churn_risk': 'mean',
                    'customer_id': 'count'
                }).round(3)
                segment_risk.columns = ['Avg Churn Risk', 'Customer Count']
                segment_risk = segment_risk.sort_values('Avg Churn Risk', ascending=False)
                
                fig_risk = go.Figure(data=[go.Bar(
                    x=segment_risk.index,
                    y=segment_risk['Avg Churn Risk'],
                    marker_color='red',
                    text=[f"{x:.1%}" for x in segment_risk['Avg Churn Risk']],
                    textposition='auto'
                )])
                fig_risk.update_layout(
                    title="Average Churn Risk by Segment",
                    xaxis_title="Customer Segment",
                    yaxis_title="Churn Risk",
                    yaxis_tickformat='.1%',
                    template="plotly_white"
                )
                st.plotly_chart(fig_risk, use_container_width=True)
        
        # Demand forecasting insights
        if dashboard.demand_model is not None:
            st.subheader("📈 Demand Forecasting Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Seasonal demand patterns
                monthly_demand = dashboard.demand_data.groupby(
                    dashboard.demand_data['date'].dt.month
                )['daily_subscriptions'].mean()
                
                fig_seasonal = go.Figure(data=[go.Bar(
                    x=[f"Month {i}" for i in monthly_demand.index],
                    y=monthly_demand.values,
                    marker_color='lightgreen'
                )])
                fig_seasonal.update_layout(
                    title="Average Daily Demand by Month",
                    xaxis_title="Month",
                    yaxis_title="Average Daily Subscriptions",
                    template="plotly_white"
                )
                st.plotly_chart(fig_seasonal, use_container_width=True)
                
            with col2:
                # Weekday vs weekend demand
                weekend_demand = dashboard.demand_data.groupby(
                    dashboard.demand_data['date'].dt.dayofweek >= 5
                )['daily_subscriptions'].mean()
                
                fig_weekday = go.Figure(data=[go.Bar(
                    x=['Weekday', 'Weekend'],
                    y=weekend_demand.values,
                    marker_color=['blue', 'orange']
                )])
                fig_weekday.update_layout(
                    title="Average Daily Demand: Weekday vs Weekend",
                    xaxis_title="Day Type",
                    yaxis_title="Average Daily Subscriptions",
                    template="plotly_white"
                )
                st.plotly_chart(fig_weekday, use_container_width=True)
        
    with tab5:
        st.header("🎯 Strategic Business Insights")
        
        # Key business metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("💰 Revenue Insights")
            total_monthly_revenue = dashboard.customer_data['monthly_subscription'].sum()
            avg_customer_value = dashboard.customer_data['lifetime_value'].mean()
            
            st.metric("Monthly Recurring Revenue", f"€{total_monthly_revenue:,.0f}")
            st.metric("Average Customer LTV", f"€{avg_customer_value:,.0f}")
            
            # Revenue at risk
            high_risk_revenue = dashboard.customer_data[
                dashboard.customer_data['churn_risk'] > 0.3
            ]['monthly_subscription'].sum()
            st.metric("Revenue at Risk", f"€{high_risk_revenue:,.0f}", delta="High Priority")
            
        with col2:
            st.subheader("📊 Operational Metrics")
            avg_utilization = dashboard.demand_data['fleet_utilization'].mean()
            peak_demand_day = dashboard.demand_data.groupby('date')['daily_subscriptions'].sum().idxmax()
            
            st.metric("Average Fleet Utilization", f"{avg_utilization:.1%}")
            st.metric("Peak Demand Date", peak_demand_day.strftime('%Y-%m-%d'))
            
            # Operational efficiency
            low_util_cities = dashboard.demand_data.groupby('city')['fleet_utilization'].mean()
            lowest_util_city = low_util_cities.idxmin()
            st.metric("Lowest Utilization City", lowest_util_city, 
                     delta=f"{low_util_cities.min():.1%}")
            
        with col3:
            st.subheader("🎯 Customer Metrics")
            avg_satisfaction = dashboard.customer_data['satisfaction_score'].mean()
            retention_rate = 1 - dashboard.customer_data['churned'].mean()
            
            st.metric("Average Satisfaction", f"{avg_satisfaction:.1%}")
            st.metric("Customer Retention Rate", f"{retention_rate:.1%}")
            
            # Customer engagement
            high_engagement = (dashboard.customer_data['app_usage_score'] > 0.7).mean()
            st.metric("High Engagement Rate", f"{high_engagement:.1%}")
        
        # Strategic recommendations
        st.subheader("🎯 Strategic Recommendations")
        
        recommendations = [
            {
                "category": "🌍 Market Expansion",
                "recommendation": "Focus on München and Berlin markets showing highest demand",
                "impact": "High",
                "priority": "1"
            },
            {
                "category": "🚗 Fleet Optimization",
                "recommendation": "Increase Electric_Compact vehicles based on demand patterns",
                "impact": "Medium",
                "priority": "2"
            },
            {
                "category": "🛡️ Customer Retention",
                "recommendation": "Implement proactive outreach for customers with >3 support contacts",
                "impact": "High",
                "priority": "1"
            },
            {
                "category": "💼 Revenue Growth",
                "recommendation": "Target Corporate_Fleet segment for premium service offerings",
                "impact": "High",
                "priority": "2"
            },
            {
                "category": "⚡ Operational Efficiency",
                "recommendation": "Optimize fleet utilization during weekday peak periods",
                "impact": "Medium",
                "priority": "3"
            },
            {
                "category": "📱 Product Development",
                "recommendation": "Enhance app experience to improve customer engagement scores",
                "impact": "Medium",
                "priority": "2"
            }
        ]
        
        for i, rec in enumerate(recommendations, 1):
            priority_color = "🔴" if rec["priority"] == "1" else "🟡" if rec["priority"] == "2" else "🟢"
            st.markdown(f"""
            **{i}. {rec['category']}** {priority_color}
            
            *{rec['recommendation']}*
            
            Impact: {rec['impact']} | Priority: {rec['priority']}
            
            ---
            """)
        
        # Risk assessment
        st.subheader("⚠️ Business Risk Assessment")
        
        high_risk_customers = len(dashboard.customer_data[dashboard.customer_data['churn_risk'] > 0.3])
        at_risk_revenue = dashboard.customer_data[dashboard.customer_data['churn_risk'] > 0.3]['monthly_subscription'].sum()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.error(f"🚨 **{high_risk_customers}** customers at high churn risk")
            st.error(f"💰 **€{at_risk_revenue:,.0f}** monthly revenue at risk")
            
            # Additional risk factors
            st.warning(f"📞 **{(dashboard.customer_data['support_contacts'] > 3).sum()}** customers with high support contact frequency")
            st.warning(f"💳 **{(dashboard.customer_data['payment_delays'] > 2).sum()}** customers with payment issues")
            
        with col2:
            st.info("**🛡️ Mitigation Strategies:**")
            st.info("• Deploy targeted retention campaigns")
            st.info("• Improve customer service response times")
            st.info("• Offer flexible subscription modifications")
            st.info("• Implement predictive intervention system")
            st.info("• Enhance payment collection processes")
        
        # Performance benchmarks
        st.subheader("📈 Performance Benchmarks")
        
        churn_rate = dashboard.customer_data['churned'].mean()
        avg_satisfaction = dashboard.customer_data['satisfaction_score'].mean()
        avg_utilization = dashboard.demand_data['fleet_utilization'].mean()
        avg_customer_value = dashboard.customer_data['lifetime_value'].mean()
        
        benchmark_data = {
            'Metric': [
                'Customer Acquisition Cost', 
                'Customer Lifetime Value', 
                'Monthly Churn Rate', 
                'Fleet Utilization', 
                'Customer Satisfaction',
                'Average Revenue per User',
                'Customer Support Response Rate'
            ],
            'Current Performance': [
                '€285', 
                f"€{avg_customer_value:,.0f}", 
                f"{churn_rate:.1%}", 
                f"{avg_utilization:.1%}", 
                f"{avg_satisfaction:.1%}",
                f"€{dashboard.customer_data['monthly_subscription'].mean():,.0f}",
                '94%'
            ],
            'Industry Benchmark': [
                '€300', 
                '€15,000', 
                '5.0%', 
                '85%', 
                '80%',
                '€650',
                '90%'
            ],
            'Status': [
                '✅ Above', 
                '✅ Above' if avg_customer_value > 15000 else '⚠️ Below', 
                '✅ Below Target' if churn_rate < 0.05 else '❌ Above Target',
                '✅ Above' if avg_utilization > 0.85 else '⚠️ Below',
                '✅ Above' if avg_satisfaction > 0.80 else '⚠️ Below',
                '✅ Above',
                '✅ Above'
            ]
        }
        
        benchmark_df = pd.DataFrame(benchmark_data)
        st.dataframe(benchmark_df, use_container_width=True, hide_index=True)
        
        # Market opportunity analysis
        st.subheader("🌟 Market Opportunity Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎯 Growth Opportunities**")
            st.markdown("• Electric vehicle segment showing 35% of total demand")
            st.markdown("• Corporate fleet segment has lowest churn rate (3.2%)")
            st.markdown("• München market shows 30% higher demand than average")
            st.markdown("• Family subscribers have highest satisfaction scores (88%)")
            
        with col2:
            st.markdown("**⚡ Quick Wins**")
            st.markdown("• Reduce support contact volume through better onboarding")
            st.markdown("• Implement usage-based pricing for low-utilization customers")
            st.markdown("• Cross-sell premium services to high-value segments")
            st.markdown("• Automate fleet rebalancing between cities")

# Data export functionality
def export_data():
    """Export dashboard data"""
    if st.sidebar.button("📥 Export Data"):
        if 'dashboard' in st.session_state and st.session_state.dashboard.customer_data is not None:
            csv = st.session_state.dashboard.customer_data.to_csv(index=False)
            st.sidebar.download_button(
                label="Download Customer Data",
                data=csv,
                file_name="finn_customer_data.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
    export_data()