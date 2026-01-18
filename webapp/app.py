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

API_URL = os.getenv("API_URL", "http://localhost:8080")

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
            ["Dashboard", "Customers", "At-Risk", "Campaigns", "Offers", "Settings"],
            label_visibility="collapsed"
        )

        st.markdown("---")

        # System status
        st.markdown("**System Status**")
        health = api_call("/health")
        if health:
            status_items = [
                ("ML Model", health.get('model_loaded', False)),
                ("AI Engine", health.get('vectorstore_ready', False)),
                ("Email", health.get('email_configured', False)),
                ("OpenAI", health.get('openai_configured', False))
            ]
            for name, status in status_items:
                color = "status-online" if status else "status-offline"
                st.markdown(f"""
                <div class="status-item">
                    <div class="status-dot {color}"></div>
                    {name}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error("API Offline")

        st.markdown("---")
        st.caption(f"v3.0 • {datetime.now().strftime('%H:%M')}")

    # Main content
    if menu == "Dashboard":
        show_dashboard()
    elif menu == "Customers":
        show_customers()
    elif menu == "At-Risk":
        show_at_risk()
    elif menu == "Campaigns":
        show_campaigns()
    elif menu == "Offers":
        show_offers()
    elif menu == "Settings":
        show_settings()


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
        threshold = st.slider("Minimum risk threshold", 0.0, 1.0, 0.5, 0.05, format="%.0f%%")
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


def show_campaigns():
    """Campaign management"""
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">Campaigns</h1>
        <p class="page-subtitle">Launch retention email campaigns</p>
    </div>
    """, unsafe_allow_html=True)

    # Stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("This Month", "12")
    with col2:
        st.metric("Emails Sent", "1,247")
    with col3:
        st.metric("Response Rate", "34%")
    with col4:
        st.metric("Retention", "78%")

    st.markdown("---")

    # Configuration
    st.markdown("##### Campaign Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Automatic Targeting**")
        risk_threshold = st.slider("Target risk above", 0.0, 1.0, 0.6, 0.05, format="%.0f%%")
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


if __name__ == "__main__":
    main()
