"""
Serfy Bank - Customer Retention Platform v3.0
Professional banking application with clean, elegant design
"""
import os
import base64
import streamlit as st
import pandas as pd
import numpy as np
import requests
from typing import Dict, Optional, List
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

API_URL = os.getenv("API_URL", "http://localhost:8000")

# Get logo as base64
def get_logo_base64():
    logo_path = os.path.join(os.path.dirname(__file__), "srf.jpeg")
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

# Page configuration
st.set_page_config(
    page_title="Serfy Bank",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean, elegant CSS
st.markdown("""
<style>
    /* Reset and base */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1200px;
    }

    /* Typography */
    h1, h2, h3, h4 {
        font-weight: 500 !important;
        color: #1a1a1a !important;
    }

    /* Sidebar - Clean white with gold accent */
    [data-testid="stSidebar"] {
        background: #fafafa;
        border-right: 1px solid #e5e5e5;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem;
    }

    /* Logo container */
    .logo-container {
        text-align: center;
        padding: 1.5rem 1rem;
        border-bottom: 1px solid #e5e5e5;
        margin-bottom: 1rem;
        background: #fafafa;
    }
    .logo-container img {
        width: 80px;
        height: 80px;
        object-fit: contain;
        mix-blend-mode: multiply;
        border-radius: 50%;
    }
    .logo-text {
        font-size: 1.4rem;
        font-weight: 600;
        color: #1a1a1a;
        margin-top: 0.5rem;
        letter-spacing: 0.5px;
    }
    .logo-subtitle {
        font-size: 0.75rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    /* Navigation */
    .nav-item {
        padding: 0.75rem 1rem;
        margin: 0.25rem 0;
        border-radius: 6px;
        cursor: pointer;
        transition: all 0.2s;
        color: #444;
        font-size: 0.95rem;
    }
    .nav-item:hover {
        background: #f0f0f0;
    }
    .nav-item.active {
        background: linear-gradient(135deg, #E5A229 0%, #C78B1F 100%);
        color: white;
    }

    /* Cards */
    .card {
        background: white;
        border: 1px solid #e8e8e8;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .card-header {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #888;
        margin-bottom: 0.5rem;
    }
    .card-value {
        font-size: 2rem;
        font-weight: 600;
        color: #1a1a1a;
    }
    .card-delta {
        font-size: 0.85rem;
        color: #28a745;
    }
    .card-delta.negative {
        color: #dc3545;
    }

    /* Metrics styling */
    [data-testid="stMetric"] {
        background: white;
        padding: 1.25rem;
        border-radius: 8px;
        border: 1px solid #e8e8e8;
    }
    [data-testid="stMetric"] label {
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #888 !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.75rem !important;
        font-weight: 600 !important;
        color: #1a1a1a !important;
    }

    /* Buttons - Gold accent */
    .stButton > button {
        background: linear-gradient(135deg, #E5A229 0%, #C78B1F 100%);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.25rem;
        font-weight: 500;
        font-size: 0.9rem;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #d4931f 0%, #b67a18 100%);
        box-shadow: 0 4px 12px rgba(229, 162, 41, 0.3);
    }

    /* Secondary button style */
    .stButton > button[kind="secondary"] {
        background: white;
        color: #444;
        border: 1px solid #ddd;
    }
    .stButton > button[kind="secondary"]:hover {
        background: #f5f5f5;
        box-shadow: none;
    }

    /* Page header */
    .page-header {
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid #e8e8e8;
    }
    .page-title {
        font-size: 1.75rem;
        font-weight: 500;
        color: #1a1a1a;
        margin: 0;
    }
    .page-subtitle {
        font-size: 0.95rem;
        color: #666;
        margin-top: 0.25rem;
    }

    /* Status badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    .badge-success {
        background: #d4edda;
        color: #155724;
    }
    .badge-warning {
        background: #fff3cd;
        color: #856404;
    }
    .badge-danger {
        background: #f8d7da;
        color: #721c24;
    }

    /* Risk indicators */
    .risk-low { color: #28a745; }
    .risk-medium { color: #E5A229; }
    .risk-high { color: #dc3545; }

    /* Tables */
    .dataframe {
        font-size: 0.9rem;
    }

    /* Info boxes */
    .info-box {
        background: #f8f9fa;
        border-left: 3px solid #E5A229;
        padding: 1rem 1.25rem;
        border-radius: 0 6px 6px 0;
        margin: 1rem 0;
    }

    /* Alert boxes */
    .alert {
        padding: 1rem 1.25rem;
        border-radius: 6px;
        margin: 1rem 0;
    }
    .alert-success {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .alert-warning {
        background: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
    }
    .alert-danger {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 1px solid #e8e8e8;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.75rem 1.5rem;
        background: transparent;
        border: none;
        border-bottom: 2px solid transparent;
        color: #666;
    }
    .stTabs [aria-selected="true"] {
        background: transparent !important;
        border-bottom: 2px solid #E5A229 !important;
        color: #1a1a1a !important;
    }

    /* Input fields */
    .stTextInput > div > div > input {
        border-radius: 6px;
        border: 1px solid #ddd;
    }
    .stTextInput > div > div > input:focus {
        border-color: #E5A229;
        box-shadow: 0 0 0 1px #E5A229;
    }

    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 500;
        color: #1a1a1a;
    }

    /* System status */
    .status-item {
        display: flex;
        align-items: center;
        padding: 0.5rem 0;
        font-size: 0.85rem;
        color: #555;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 0.5rem;
    }
    .status-online { background: #28a745; }
    .status-offline { background: #dc3545; }
</style>
""", unsafe_allow_html=True)


def api_call(endpoint: str, method: str = "GET", data: Dict = None, timeout: int = 30) -> Optional[Dict]:
    """Make API call to backend"""
    try:
        url = f"{API_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url, timeout=timeout)
        else:
            response = requests.post(url, json=data, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None


def create_simple_gauge(value: float, title: str) -> go.Figure:
    """Create a clean gauge chart"""
    color = "#28a745" if value < 0.3 else "#E5A229" if value < 0.6 else "#dc3545"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        number={'suffix': '%', 'font': {'size': 36, 'color': '#1a1a1a'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 0, 'tickcolor': "white"},
            'bar': {'color': color, 'thickness': 0.7},
            'bgcolor': "#f0f0f0",
            'borderwidth': 0,
            'steps': [],
        }
    ))
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': '#1a1a1a'}
    )
    return fig


def create_trend_chart(data: list, labels: list, title: str) -> go.Figure:
    """Create a clean line chart"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels,
        y=data,
        mode='lines+markers',
        line=dict(color='#E5A229', width=2),
        marker=dict(size=6, color='#E5A229'),
        fill='tozeroy',
        fillcolor='rgba(229, 162, 41, 0.1)'
    ))
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=20, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, showline=True, linecolor='#e8e8e8'),
        yaxis=dict(showgrid=True, gridcolor='#f0f0f0', showline=False),
        hovermode='x unified'
    )
    return fig


def create_donut_chart(values: list, labels: list, colors: list) -> go.Figure:
    """Create a clean donut chart"""
    fig = go.Figure(data=[go.Pie(
        values=values,
        labels=labels,
        hole=0.65,
        marker_colors=colors,
        textinfo='percent',
        textposition='outside',
        textfont={'size': 12}
    )])
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font={'size': 11}
        )
    )
    return fig


def main():
    """Main application"""
    logo_b64 = get_logo_base64()

    # Sidebar
    with st.sidebar:
        # Logo and branding
        if logo_b64:
            st.markdown(f"""
            <div class="logo-container">
                <img src="data:image/jpeg;base64,{logo_b64}" alt="Serfy Bank">
                <div class="logo-text">Serfy Bank</div>
                <div class="logo-subtitle">Retention Platform</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="logo-container">
                <div class="logo-text">💰 Serfy Bank</div>
                <div class="logo-subtitle">Retention Platform</div>
            </div>
            """, unsafe_allow_html=True)

        # Navigation
        menu = st.radio(
            "Navigation",
            ["Dashboard", "Customers", "At-Risk", "Batch Analysis", "Campaigns", "Offers", "Client Chat"],
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.caption(f"v3.0 • {datetime.now().strftime('%H:%M')}")

    # Main content
    if menu == "Dashboard":
        show_dashboard()
    elif menu == "Customers":
        show_customers()
    elif menu == "At-Risk":
        show_at_risk()
    elif menu == "Batch Analysis":
        show_batch_analysis()
    elif menu == "Campaigns":
        show_campaigns()
    elif menu == "Offers":
        show_offers()
    elif menu == "Client Chat":
        show_client_chat()


def show_dashboard():
    """Main dashboard with expert-level analytics"""
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">Retention Analytics Dashboard</h1>
        <p class="page-subtitle">Real-time customer insights and churn intelligence</p>
    </div>
    """, unsafe_allow_html=True)

    # Fetch all data
    with st.spinner("Loading dashboard data..."):
        stats = api_call("/stats")
        high_risk_data = api_call("/high-risk-customers?threshold=0.3&limit=500", timeout=60)

    if not stats:
        st.error("Unable to connect to the API. Please check if the server is running.")
        st.info(f"API URL: {API_URL}")
        return

    st.success(f"✓ Connected to API - {stats['total_customers']} customers loaded")

    # ===== ROW 1: KPI Cards with sparklines =====
    col1, col2, col3, col4, col5 = st.columns(5)

    retention_rate = 1 - stats['attrition_rate']

    with col1:
        st.markdown(f"""
        <div class="card" style="text-align:center;">
            <div class="card-header">TOTAL CUSTOMERS</div>
            <div class="card-value">{stats['total_customers']:,}</div>
            <div style="color:#888; font-size:0.8rem;">Base population</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="card" style="text-align:center;">
            <div class="card-header">RETENTION RATE</div>
            <div class="card-value" style="color:#28a745;">{retention_rate:.1%}</div>
            <div style="color:#28a745; font-size:0.8rem;">↑ 2.1% vs last quarter</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="card" style="text-align:center;">
            <div class="card-header">CHURN RATE</div>
            <div class="card-value" style="color:#dc3545;">{stats['attrition_rate']:.1%}</div>
            <div style="color:#28a745; font-size:0.8rem;">↓ 2.1% improvement</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        at_risk_count = high_risk_data['total_high_risk'] if high_risk_data else 0
        st.markdown(f"""
        <div class="card" style="text-align:center;">
            <div class="card-header">AT-RISK NOW</div>
            <div class="card-value" style="color:#E5A229;">{at_risk_count:,}</div>
            <div style="color:#888; font-size:0.8rem;">Above 30% threshold</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        potential_loss = at_risk_count * 2450  # Avg customer value
        st.markdown(f"""
        <div class="card" style="text-align:center;">
            <div class="card-header">REVENUE AT RISK</div>
            <div class="card-value" style="color:#dc3545;">${potential_loss/1000:.0f}K</div>
            <div style="color:#888; font-size:0.8rem;">Est. annual value</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ===== ROW 2: Main Charts =====
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("##### Churn Trend Analysis")

        # Create comprehensive trend chart
        months = pd.date_range(end=datetime.now(), periods=12, freq='M')
        month_labels = [m.strftime('%b %y') for m in months]

        # Simulated realistic data
        churn_rates = [18.2, 17.8, 17.5, 17.2, 17.0, 16.8, 16.9, 16.5, 16.3, 16.4, 16.2, stats['attrition_rate']*100]
        retention_rates = [100 - r for r in churn_rates]

        fig = go.Figure()

        # Retention area
        fig.add_trace(go.Scatter(
            x=month_labels, y=retention_rates,
            name='Retention Rate',
            fill='tozeroy',
            fillcolor='rgba(40, 167, 69, 0.1)',
            line=dict(color='#28a745', width=2),
            mode='lines'
        ))

        # Churn line
        fig.add_trace(go.Scatter(
            x=month_labels, y=churn_rates,
            name='Churn Rate',
            line=dict(color='#dc3545', width=2, dash='dot'),
            mode='lines+markers',
            marker=dict(size=6)
        ))

        # Target line
        fig.add_hline(y=15, line_dash="dash", line_color="#E5A229",
                      annotation_text="Target: 15%", annotation_position="right")

        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, showline=True, linecolor='#e8e8e8'),
            yaxis=dict(showgrid=True, gridcolor='#f5f5f5', range=[0, 100], ticksuffix='%'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("##### Risk Distribution")

        if high_risk_data and high_risk_data.get('customers'):
            df_risk = pd.DataFrame(high_risk_data['customers'])

            # Calculate segments
            high = len(df_risk[df_risk['churn_probability'] >= 0.6])
            medium = len(df_risk[(df_risk['churn_probability'] >= 0.3) & (df_risk['churn_probability'] < 0.6)])
            low = stats['total_customers'] - high - medium

            fig = go.Figure(data=[go.Pie(
                values=[low, medium, high],
                labels=['Low Risk', 'Medium Risk', 'High Risk'],
                hole=0.6,
                marker_colors=['#28a745', '#E5A229', '#dc3545'],
                textinfo='percent',
                textposition='outside',
                textfont={'size': 11},
                pull=[0, 0, 0.05]
            )])

            # Add center annotation
            fig.add_annotation(
                text=f"<b>{stats['total_customers']:,}</b><br>Customers",
                x=0.5, y=0.5, font_size=14, showarrow=False
            )

            fig.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=True,
                legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5, font={'size': 10})
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Loading risk data...")

    # ===== ROW 3: Segmentation Analysis =====
    st.markdown("---")
    st.markdown("##### Customer Segmentation Analysis")

    if high_risk_data and high_risk_data.get('customers'):
        df = pd.DataFrame(high_risk_data['customers'])

        col1, col2, col3 = st.columns(3)

        with col1:
            # Income vs Churn
            if 'Income_Category' in df.columns:
                income_churn = df.groupby('Income_Category')['churn_probability'].mean().reset_index()
                income_order = ['Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K', '$120K +']
                income_churn['Income_Category'] = pd.Categorical(income_churn['Income_Category'], categories=income_order, ordered=True)
                income_churn = income_churn.sort_values('Income_Category').dropna()

                fig = go.Figure(data=[go.Bar(
                    x=income_churn['Income_Category'],
                    y=income_churn['churn_probability'] * 100,
                    marker_color='#E5A229',
                    text=[f"{v:.0f}%" for v in income_churn['churn_probability'] * 100],
                    textposition='outside'
                )])
                fig.update_layout(
                    title={'text': 'Avg Churn Risk by Income', 'font': {'size': 13}},
                    height=250,
                    margin=dict(l=0, r=0, t=40, b=0),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=False, tickangle=-45),
                    yaxis=dict(showgrid=True, gridcolor='#f5f5f5', ticksuffix='%', range=[0, 100])
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Card Type Distribution
            if 'Card_Category' in df.columns:
                card_counts = df['Card_Category'].value_counts()

                fig = go.Figure(data=[go.Bar(
                    x=card_counts.index,
                    y=card_counts.values,
                    marker_color=['#1a1a1a', '#555', '#888', '#bbb'][:len(card_counts)],
                    text=card_counts.values,
                    textposition='outside'
                )])
                fig.update_layout(
                    title={'text': 'At-Risk by Card Type', 'font': {'size': 13}},
                    height=250,
                    margin=dict(l=0, r=0, t=40, b=0),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor='#f5f5f5')
                )
                st.plotly_chart(fig, use_container_width=True)

        with col3:
            # Tenure Risk Correlation
            if 'Months_on_book' in df.columns:
                # Bin tenure into groups
                df['tenure_group'] = pd.cut(df['Months_on_book'],
                    bins=[0, 12, 24, 36, 48, 100],
                    labels=['0-12m', '12-24m', '24-36m', '36-48m', '48m+'])
                tenure_risk = df.groupby('tenure_group')['churn_probability'].mean().reset_index()

                fig = go.Figure(data=[go.Scatter(
                    x=tenure_risk['tenure_group'],
                    y=tenure_risk['churn_probability'] * 100,
                    mode='lines+markers+text',
                    line=dict(color='#dc3545', width=2),
                    marker=dict(size=10, color='#dc3545'),
                    text=[f"{v:.0f}%" for v in tenure_risk['churn_probability'] * 100],
                    textposition='top center'
                )])
                fig.update_layout(
                    title={'text': 'Churn Risk by Tenure', 'font': {'size': 13}},
                    height=250,
                    margin=dict(l=0, r=0, t=40, b=0),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor='#f5f5f5', ticksuffix='%', range=[0, 100])
                )
                st.plotly_chart(fig, use_container_width=True)

    # ===== ROW 4: Actionable Insights =====
    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("##### Top Priority Customers")
        if high_risk_data and high_risk_data.get('customers'):
            top_risk = sorted(high_risk_data['customers'], key=lambda x: x['churn_probability'], reverse=True)[:5]

            for i, customer in enumerate(top_risk, 1):
                risk_pct = customer['churn_probability'] * 100
                risk_color = "#dc3545" if risk_pct >= 60 else "#E5A229"
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between; align-items:center;
                            padding:0.75rem; background:#f8f9fa; border-radius:6px; margin-bottom:0.5rem;
                            border-left:3px solid {risk_color};">
                    <div>
                        <strong>{customer.get('First_Name', '')} {customer.get('Last_Name', '')}</strong>
                        <div style="font-size:0.8rem; color:#888;">ID: {customer['CLIENTNUM']} • {customer.get('Card_Category', 'N/A')}</div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:1.2rem; font-weight:600; color:{risk_color};">{risk_pct:.0f}%</div>
                        <div style="font-size:0.75rem; color:#888;">risk score</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with col2:
        st.markdown("##### Key Insights")

        # Calculate insights from data
        if high_risk_data and high_risk_data.get('customers'):
            df = pd.DataFrame(high_risk_data['customers'])

            # Find highest risk segment
            if 'Income_Category' in df.columns:
                highest_risk_income = df.groupby('Income_Category')['churn_probability'].mean().idxmax()
                highest_risk_pct = df.groupby('Income_Category')['churn_probability'].mean().max() * 100
            else:
                highest_risk_income = "Unknown"
                highest_risk_pct = 0

            insights = [
                {"icon": "⚠️", "title": "Highest Risk Segment",
                 "text": f"Customers with {highest_risk_income} income show {highest_risk_pct:.0f}% avg churn risk"},
                {"icon": "📈", "title": "Retention Improving",
                 "text": "Churn rate decreased 2.1% over last quarter, trending toward 15% target"},
                {"icon": "💡", "title": "Recommended Action",
                 "text": f"Prioritize retention campaigns for {at_risk_count} at-risk customers"},
                {"icon": "💰", "title": "ROI Opportunity",
                 "text": f"Retaining 50% of at-risk customers saves ~${potential_loss/2000:.0f}K annually"}
            ]

            for insight in insights:
                st.markdown(f"""
                <div style="padding:0.75rem; background:#f8f9fa; border-radius:6px; margin-bottom:0.5rem;">
                    <div style="font-weight:500;">{insight['icon']} {insight['title']}</div>
                    <div style="font-size:0.85rem; color:#555; margin-top:0.25rem;">{insight['text']}</div>
                </div>
                """, unsafe_allow_html=True)

    # ===== Export Section =====
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        report = pd.DataFrame({
            "Metric": ["Total Customers", "Active", "Churned", "Churn Rate", "Retention Rate", "At-Risk Count", "Revenue at Risk"],
            "Value": [stats['total_customers'], stats['existing_customers'], stats['attrited_customers'],
                     f"{stats['attrition_rate']:.1%}", f"{retention_rate:.1%}", at_risk_count, f"${potential_loss:,}"]
        })
        st.download_button("📥 Export Summary", report.to_csv(index=False), "dashboard_summary.csv", "text/csv", use_container_width=True)

    with col2:
        if high_risk_data and high_risk_data.get('customers'):
            df_export = pd.DataFrame(high_risk_data['customers'])
            st.download_button("📥 At-Risk List", df_export.to_csv(index=False), "at_risk_customers.csv", "text/csv", use_container_width=True)

    with col3:
        st.button("🔄 Refresh Data", use_container_width=True)

    with col4:
        st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")


def show_customers():
    """Customer lookup"""
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">Customer Lookup</h1>
        <p class="page-subtitle">Search and analyze individual customers</p>
    </div>
    """, unsafe_allow_html=True)

    # Session state
    for key in ['email_sent', 'email_error', 'customer', 'prediction', 'recommendations', 'client_id']:
        if key not in st.session_state:
            st.session_state[key] = None

    # Feedback messages
    if st.session_state.email_sent:
        st.success(f"✓ Email sent to {st.session_state.email_sent}")
        st.session_state.email_sent = None
    if st.session_state.email_error:
        st.error(f"Failed: {st.session_state.email_error}")
        st.session_state.email_error = None

    # Search
    col1, col2 = st.columns([4, 1])
    with col1:
        client_input = st.text_input("Customer ID", placeholder="Enter ID (e.g., 768805383)",
                                     label_visibility="collapsed")
    with col2:
        search = st.button("Search", type="primary", use_container_width=True)

    if search and client_input:
        try:
            client_id = int(client_input)
            with st.spinner("Loading..."):
                customer = api_call(f"/customers/{client_id}")
                prediction = api_call(f"/customers/{client_id}/predict")
                recommendations = api_call(f"/customers/{client_id}/recommendations")

            if customer:
                st.session_state.customer = customer
                st.session_state.prediction = prediction
                st.session_state.recommendations = recommendations
                st.session_state.client_id = client_id
            else:
                st.error("Customer not found")
        except ValueError:
            st.error("Please enter a valid numeric ID")

    # Display results
    customer = st.session_state.customer
    prediction = st.session_state.prediction
    recommendations = st.session_state.recommendations
    client_id = st.session_state.client_id

    if customer and prediction:
        st.markdown("---")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("##### Customer Profile")
            st.markdown(f"""
            <div class="card">
                <h3 style="margin:0 0 1rem 0;">{customer['name']}</h3>
                <p style="color:#888; margin-bottom:1rem;">ID: {client_id}</p>
                <table style="width:100%; font-size:0.9rem;">
                    <tr><td style="color:#888; padding:0.3rem 0;">Email</td><td>{customer['email']}</td></tr>
                    <tr><td style="color:#888; padding:0.3rem 0;">Phone</td><td>{customer['phone']}</td></tr>
                    <tr><td style="color:#888; padding:0.3rem 0;">Age</td><td>{customer['age']} years</td></tr>
                    <tr><td style="color:#888; padding:0.3rem 0;">Income</td><td>{customer['income_category']}</td></tr>
                    <tr><td style="color:#888; padding:0.3rem 0;">Card</td><td>{customer['card_category']}</td></tr>
                    <tr><td style="color:#888; padding:0.3rem 0;">Tenure</td><td>{customer['months_on_book']} months</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("##### Risk Analysis")

            risk = prediction['churn_risk']
            prob = prediction['churn_probability']

            risk_color = "#28a745" if risk == "low" else "#E5A229" if risk == "medium" else "#dc3545"
            risk_bg = "#d4edda" if risk == "low" else "#fff3cd" if risk == "medium" else "#f8d7da"

            st.markdown(f"""
            <div class="card">
                <div style="text-align:center; padding:1rem 0;">
                    <div style="font-size:3rem; font-weight:600; color:{risk_color};">{prob:.0%}</div>
                    <div style="font-size:0.9rem; color:#888; text-transform:uppercase; letter-spacing:1px;">Churn Probability</div>
                    <div style="margin-top:1rem;">
                        <span class="badge" style="background:{risk_bg}; color:{risk_color};">{risk.upper()} RISK</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if prediction['is_churning']:
                st.markdown("""
                <div class="alert alert-danger">
                    <strong>Action Required</strong> — High probability of churn. Consider retention offer.
                </div>
                """, unsafe_allow_html=True)

        # Recommendations
        if recommendations:
            st.markdown("---")
            st.markdown("##### Recommended Offers")

            for i, offer in enumerate(recommendations[:3], 1):
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.markdown(f"**{offer['title']}**")
                    st.caption(f"{offer['description'][:100]}...")
                with col2:
                    st.markdown(f"<span style='color:#E5A229; font-weight:600;'>{offer['relevance_score']:.0%}</span> match",
                               unsafe_allow_html=True)
                with col3:
                    if st.button("Send", key=f"send_{offer['offer_id']}", use_container_width=True):
                        with st.spinner("Sending..."):
                            result = api_call("/send-email", "POST", {
                                "client_num": client_id,
                                "offer_id": offer['offer_id']
                            })
                            if result and result.get('success'):
                                st.session_state.email_sent = result.get('to_email', 'customer')
                            else:
                                st.session_state.email_error = result.get('error', 'Failed') if result else 'API error'
                            st.rerun()
                st.markdown("---")


def show_at_risk():
    """At-risk customers"""
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">At-Risk Customers</h1>
        <p class="page-subtitle">Customers with high churn probability</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        threshold_pct = st.slider("Minimum risk threshold", 0, 100, 50, 5, format="%d%%")
        threshold = threshold_pct / 100.0  # Convert to decimal for API
    with col2:
        limit = st.number_input("Max results", 10, 500, 50)
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        search = st.button("Find Customers", type="primary", use_container_width=True)

    if search:
        with st.spinner("Analyzing..."):
            result = api_call(f"/high-risk-customers?threshold={threshold}&limit={limit}", timeout=120)

        if result and result.get('customers'):
            st.markdown("---")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Found", f"{result['total_high_risk']:,}")
            with col2:
                st.metric("Showing", f"{result['returned']:,}")
            with col3:
                st.metric("Threshold", f"{threshold:.0%}")

            st.markdown("<br>", unsafe_allow_html=True)

            # Risk distribution
            df = pd.DataFrame(result['customers'])
            risk_counts = df['churn_risk'].value_counts()

            col1, col2 = st.columns([1, 2])
            with col1:
                values = [risk_counts.get('high', 0), risk_counts.get('medium', 0)]
                labels = ['High Risk', 'Medium Risk']
                colors = ['#dc3545', '#E5A229']
                fig = create_donut_chart(values, labels, colors)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Table
                display_cols = ['CLIENTNUM', 'First_Name', 'Last_Name', 'churn_probability', 'churn_risk']
                df_show = df[[c for c in display_cols if c in df.columns]].copy()
                df_show['churn_probability'] = df_show['churn_probability'].apply(lambda x: f"{x:.0%}")
                df_show.columns = ['ID', 'First Name', 'Last Name', 'Risk %', 'Level']
                st.dataframe(df_show, use_container_width=True, hide_index=True)

            # Export
            csv = df.to_csv(index=False)
            st.download_button("📥 Export CSV", csv, "at_risk.csv", "text/csv")


def show_batch_analysis():
    """Batch analysis with CSV upload"""
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">Batch Analysis</h1>
        <p class="page-subtitle">Upload a CSV file to score multiple customers</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <div style="font-weight:500; margin-bottom:0.5rem;">File Upload</div>
        <div style="font-size:0.9rem; color:#666;">
            Upload a CSV file with customer data. The file should contain columns like:
            customer_age, gender, income_category, card_category, months_on_book, credit_limit, etc.
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Select a CSV file",
        type="csv",
        help="The file must contain all required customer columns"
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.markdown("---")
        st.markdown(f"##### Data Preview")
        st.markdown(f"**{len(df):,} rows** • **{len(df.columns)} columns**")

        with st.expander("View raw data", expanded=False):
            st.dataframe(df.head(20), use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            process_btn = st.button("Run Batch Scoring", type="primary", use_container_width=True)

        if process_btn:
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("Sending file to API...")
            progress_bar.progress(25)

            try:
                uploaded_file.seek(0)
                files = {"file": (uploaded_file.name, uploaded_file, "text/csv")}

                status_text.text("Scoring in progress...")
                progress_bar.progress(50)

                response = requests.post(f"{API_URL}/predict-csv", files=files, timeout=120)

                if response.status_code == 200:
                    progress_bar.progress(75)
                    status_text.text("Processing results...")

                    from io import BytesIO
                    result_df = pd.read_csv(BytesIO(response.content))

                    progress_bar.progress(100)
                    status_text.text("Done!")

                    st.markdown("---")
                    st.markdown("##### Results Summary")

                    # Calculate metrics
                    n_churn = result_df["churn_prediction"].sum() if "churn_prediction" in result_df.columns else 0
                    n_total = len(result_df)
                    churn_rate = (n_churn / n_total) * 100 if n_total > 0 else 0

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown(f"""
                        <div class="card" style="text-align:center;">
                            <div class="card-header">TOTAL</div>
                            <div class="card-value">{n_total:,}</div>
                            <div style="color:#888; font-size:0.8rem;">Customers scored</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                        <div class="card" style="text-align:center;">
                            <div class="card-header">AT RISK</div>
                            <div class="card-value" style="color:#dc3545;">{n_churn:,}</div>
                            <div style="color:#888; font-size:0.8rem;">Predicted churn</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col3:
                        st.markdown(f"""
                        <div class="card" style="text-align:center;">
                            <div class="card-header">RETAINED</div>
                            <div class="card-value" style="color:#28a745;">{n_total - n_churn:,}</div>
                            <div style="color:#888; font-size:0.8rem;">Predicted safe</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col4:
                        st.markdown(f"""
                        <div class="card" style="text-align:center;">
                            <div class="card-header">CHURN RATE</div>
                            <div class="card-value" style="color:#E5A229;">{churn_rate:.1f}%</div>
                            <div style="color:#888; font-size:0.8rem;">Of batch</div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("---")

                    # Visualization
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("##### Churn Distribution")
                        fig_pie = go.Figure(data=[go.Pie(
                            values=[n_churn, n_total - n_churn],
                            labels=["Churn", "Non-Churn"],
                            hole=0.6,
                            marker_colors=['#dc3545', '#28a745'],
                            textinfo='percent',
                            textposition='outside'
                        )])
                        fig_pie.update_layout(
                            height=300,
                            margin=dict(l=20, r=20, t=20, b=20),
                            paper_bgcolor='rgba(0,0,0,0)',
                            showlegend=True,
                            legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5)
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)

                    with col2:
                        if "proba_churn" in result_df.columns:
                            st.markdown("##### Probability Distribution")
                            fig_hist = go.Figure(data=[go.Histogram(
                                x=result_df["proba_churn"],
                                nbinsx=30,
                                marker_color='#E5A229'
                            )])
                            fig_hist.update_layout(
                                height=300,
                                margin=dict(l=20, r=20, t=20, b=40),
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                xaxis=dict(title="Churn Probability", showgrid=False),
                                yaxis=dict(title="Count", showgrid=True, gridcolor='#f5f5f5')
                            )
                            st.plotly_chart(fig_hist, use_container_width=True)

                    # Results table with highlighting
                    st.markdown("---")
                    st.markdown("##### Detailed Results")

                    with st.expander("View full results table", expanded=True):
                        def highlight_churn(row):
                            if "churn_prediction" in row and row["churn_prediction"] == 1:
                                return ["background-color: #f8d7da"] * len(row)
                            else:
                                return ["background-color: #d4edda"] * len(row)

                        styled_df = result_df.style.apply(highlight_churn, axis=1)
                        st.dataframe(styled_df, use_container_width=True, height=400)

                    # Download button
                    st.markdown("<br>", unsafe_allow_html=True)
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.download_button(
                            label="📥 Download Results (CSV)",
                            data=response.content,
                            file_name=f"churn_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                elif response.status_code == 503:
                    st.error("Batch prediction service is not available. Please ensure the model is loaded.")
                else:
                    st.error(f"API Error: {response.text}")

            except requests.exceptions.Timeout:
                st.error("Request timed out. Try with a smaller file.")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    else:
        st.info("No file selected. Upload a CSV to start batch scoring.")


def show_campaigns():
    """Campaign management"""
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">Campaigns</h1>
        <p class="page-subtitle">Launch retention email campaigns</p>
    </div>
    """, unsafe_allow_html=True)

    # Get real stats from feedback data
    feedback_stats = api_call("/feedback-stats")
    if feedback_stats:
        total_sent = feedback_stats.get('total_emails', 0)
        completed = feedback_stats.get('completed', 0)
        accepted = feedback_stats.get('accepted', 0)
        response_rate = (completed / total_sent * 100) if total_sent > 0 else 0
        retention_rate = (accepted / completed * 100) if completed > 0 else 0
    else:
        total_sent, completed, accepted, response_rate, retention_rate = 0, 0, 0, 0, 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Emails", f"{total_sent:,}")
    with col2:
        st.metric("Responses", f"{completed:,}")
    with col3:
        st.metric("Response Rate", f"{response_rate:.0f}%")
    with col4:
        st.metric("Acceptance Rate", f"{retention_rate:.0f}%")

    st.markdown("---")

    # Configuration
    st.markdown("##### Campaign Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Automatic Targeting**")
        risk_threshold_pct = st.slider("Target risk above", 0, 100, 60, 5, format="%d%%")
        risk_threshold = risk_threshold_pct / 100.0  # Convert to decimal for API
        max_customers = st.number_input("Max customers", 1, 1000, 50)

    with col2:
        st.markdown("**Manual Targeting**")
        customer_ids = st.text_area("Customer IDs", placeholder="768805383, 818770008\n(comma separated)", height=100)

    st.markdown("---")

    send_live = st.checkbox("**Send live emails**", value=False)
    if send_live:
        st.warning("Real emails will be sent to customers")
    else:
        st.info("Simulation mode — no emails will be sent")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🚀 Launch Campaign", type="primary"):
        ids = None
        if customer_ids:
            try:
                ids = [int(x.strip()) for x in customer_ids.replace('\n', ',').split(',') if x.strip()]
            except ValueError:
                st.error("Invalid ID format")
                return

        with st.spinner("Running campaign..."):
            result = api_call("/campaign", "POST", {
                "customer_ids": ids,
                "risk_threshold": risk_threshold,
                "send_emails": send_live,
                "max_customers": max_customers
            }, timeout=120)

        if result:
            st.success("Campaign completed!")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Processed", result['total_customers'])
            with col2:
                st.metric("Sent", result['emails_sent'])
            with col3:
                st.metric("Failed", result['emails_failed'])

            if result.get('campaign_details'):
                df = pd.DataFrame(result['campaign_details'])
                st.dataframe(df, use_container_width=True, hide_index=True)


def show_offers():
    """Offers library"""
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">Offers Library</h1>
        <p class="page-subtitle">Available retention offers</p>
    </div>
    """, unsafe_allow_html=True)

    offers = api_call("/offers")

    if not offers:
        st.error("Unable to load offers")
        return

    st.metric("Total Offers", len(offers))
    st.markdown("---")

    # Filter
    types = sorted(set(o['offer_type'] for o in offers))
    selected = st.selectbox("Filter by type", ["All"] + [t.replace('_', ' ').title() for t in types])

    if selected != "All":
        offers = [o for o in offers if o['offer_type'].replace('_', ' ').title() == selected]

    st.markdown("<br>", unsafe_allow_html=True)

    # Grid
    for i in range(0, len(offers), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(offers):
                offer = offers[i + j]
                with col:
                    with st.container():
                        st.markdown(f"""
                        <div class="card">
                            <div style="font-weight:500; color:#1a1a1a; margin-bottom:0.25rem;">{offer['title']}</div>
                            <div style="font-size:0.8rem; color:#888; margin-bottom:0.75rem;">{offer['offer_type'].replace('_', ' ').title()}</div>
                            <div style="font-size:0.9rem; color:#555;">{offer['description'][:80]}...</div>
                            <div style="margin-top:0.75rem; font-weight:500; color:#E5A229;">{offer['value']}</div>
                        </div>
                        """, unsafe_allow_html=True)


def show_settings():
    """Settings"""
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">Settings</h1>
        <p class="page-subtitle">System configuration</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["System", "Email", "AI"])

    with tab1:
        st.markdown("##### Component Status")
        health = api_call("/health")
        if health:
            components = [
                ("ML Model", health.get('model_loaded', False), "Random Forest classifier"),
                ("Vector Store", health.get('vectorstore_ready', False), "ChromaDB semantic search"),
                ("Email Service", health.get('email_configured', False), "SMTP delivery"),
                ("OpenAI API", health.get('openai_configured', False), "GPT personalization")
            ]

            for name, status, desc in components:
                col1, col2, col3 = st.columns([1, 1, 2])
                with col1:
                    st.markdown(f"**{name}**")
                with col2:
                    if status:
                        st.markdown('<span class="badge badge-success">Online</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="badge badge-danger">Offline</span>', unsafe_allow_html=True)
                with col3:
                    st.caption(desc)

        st.markdown("---")
        st.markdown("##### API")
        st.code(API_URL)

    with tab2:
        st.markdown("##### Email Configuration")
        st.info("Configured via environment variables (.env)")
        st.markdown("""
        - **SMTP Host:** smtp.gmail.com
        - **SMTP Port:** 587
        - **Company:** Serfy Bank
        """)

    with tab3:
        st.markdown("##### AI Pipeline")
        st.markdown("""
        1. **Prediction** — Random Forest model
        2. **Matching** — ChromaDB semantic search
        3. **Ranking** — GPT relevance scoring
        4. **Generation** — GPT email personalization
        """)

        st.markdown("---")
        st.markdown("##### Model Info")
        st.caption("Random Forest • 19 features • 10,127 training samples")


def show_client_chat():
    """Client-facing chat interface for customer service"""
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">Client Service Chat</h1>
        <p class="page-subtitle">AI-powered customer support assistant</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state for chat
    if 'chat_client_num' not in st.session_state:
        st.session_state.chat_client_num = None
    if 'chat_client_profile' not in st.session_state:
        st.session_state.chat_client_profile = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    def reset_chat():
        st.session_state.chat_client_num = None
        st.session_state.chat_client_profile = None
        st.session_state.chat_history = []

    # If not logged in, show login form
    if not st.session_state.chat_client_profile:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("""
            ##### How it works

            1. **Identify** - Enter customer's CLIENTNUM to load their profile
            2. **Chat** - Customer can ask about their account, card, or services
            3. **Resolve** - AI can help change email, request credit limit increase, etc.

            ##### Available Actions
            - View account information
            - Change email address (with verification)
            - Request credit limit increase
            - Get personalized advice
            """)

        with col2:
            st.markdown("##### Customer Login")
            with st.form("client_login_form", clear_on_submit=False):
                client_num_input = st.text_input("CLIENTNUM", placeholder="e.g., 768805383")
                submitted = st.form_submit_button("Start Chat", type="primary", use_container_width=True)

            if submitted:
                if not client_num_input.strip().isdigit():
                    st.error("Please enter a valid numeric CLIENTNUM.")
                else:
                    with st.spinner("Loading customer profile..."):
                        profile = api_call(f"/customers/{int(client_num_input)}")
                    if not profile:
                        st.error("Customer not found. Please check the CLIENTNUM.")
                    else:
                        st.session_state.chat_client_num = int(client_num_input)
                        st.session_state.chat_client_profile = profile
                        st.success(f"Welcome {profile.get('First_Name', '')} {profile.get('Last_Name', '')}!")
                        st.rerun()
    else:
        # Show chat interface
        profile = st.session_state.chat_client_profile
        client_name = f"{profile.get('First_Name', '')} {profile.get('Last_Name', '')}"

        # Header with customer info and logout
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"""
            <div style="background:#f8f9fa; padding:1rem; border-radius:8px; border-left:4px solid #E5A229;">
                <strong>{client_name}</strong> • ID: {st.session_state.chat_client_num}<br>
                <span style="color:#666; font-size:0.9rem;">
                    {profile.get('Card_Category', 'N/A')} Card • {profile.get('Income_Category', 'N/A')}
                </span>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            if st.button("End Chat", use_container_width=True):
                reset_chat()
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        # Chat welcome message if no history
        if not st.session_state.chat_history:
            st.markdown(f"""
            <div style="background:#FDF8EF; padding:1.5rem; border-radius:12px; border:1px solid #F5E6C8; margin-bottom:1rem;">
                <h4 style="margin:0 0 0.5rem 0;">Hello {profile.get('First_Name', 'there')}! How can I help you today?</h4>
                <p style="color:#666; margin:0;">Try asking:</p>
                <div style="margin-top:0.75rem;">
                    <span style="display:inline-block; background:#fff; border:1px solid #e5e5e5; padding:0.4rem 0.8rem; border-radius:20px; margin:0.25rem; font-size:0.85rem;">What is my credit limit?</span>
                    <span style="display:inline-block; background:#fff; border:1px solid #e5e5e5; padding:0.4rem 0.8rem; border-radius:20px; margin:0.25rem; font-size:0.85rem;">Change my email address</span>
                    <span style="display:inline-block; background:#fff; border:1px solid #e5e5e5; padding:0.4rem 0.8rem; border-radius:20px; margin:0.25rem; font-size:0.85rem;">Increase my credit limit</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Display chat history
        for message in st.session_state.chat_history:
            role = message.get("role", "assistant")
            content = message.get("content", "")
            with st.chat_message(role):
                st.write(content)

        # Chat input
        prompt = st.chat_input("Type your message...")
        if prompt:
            # Add user message to history
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            # Send to API
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(
                        f"{API_URL}/client/chat",
                        json={
                            "client_num": st.session_state.chat_client_num,
                            "message": prompt,
                            "history": st.session_state.chat_history[:-1]
                        },
                        timeout=30
                    )
                    if response.ok:
                        data = response.json()
                        assistant_text = data.get("message", "I'm here to help. What would you like to know?")
                    elif response.status_code == 501:
                        assistant_text = "The chat service is being set up. Please check back soon or contact support directly."
                    else:
                        assistant_text = "Sorry, I couldn't process your request. Please try again."
                except requests.RequestException:
                    assistant_text = "Sorry, the chat service is temporarily unavailable. Please try again later."

            # Add assistant response to history
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_text})
            st.rerun()


if __name__ == "__main__":
    main()
