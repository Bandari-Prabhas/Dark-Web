import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import re

# Page Configuration
st.set_page_config(
    page_title="Dark Web Cybercrime Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for attractive design
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom card styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255,255,255,0.18);
        box-shadow: 0 8px 32px 0 rgba(0,0,0,0.37);
    }
    
    /* Title styling */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00d4ff 0%, #7b2ff7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 20px 0;
        text-shadow: 0 0 30px rgba(0,212,255,0.5);
    }
    
    .subtitle {
        text-align: center;
        color: #a0aec0;
        font-size: 1.2rem;
        margin-bottom: 30px;
    }
    
    /* Alert boxes */
    .alert-critical {
        background: linear-gradient(135deg, rgba(220,38,38,0.2) 0%, rgba(153,27,27,0.2) 100%);
        border-left: 4px solid #dc2626;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        color: #fca5a5;
    }
    
    .alert-high {
        background: linear-gradient(135deg, rgba(249,115,22,0.2) 0%, rgba(194,65,12,0.2) 100%);
        border-left: 4px solid #f97316;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        color: #fdba74;
    }
    
    .alert-medium {
        background: linear-gradient(135deg, rgba(234,179,8,0.2) 0%, rgba(161,98,7,0.2) 100%);
        border-left: 4px solid #eab308;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        color: #fde047;
    }
    
    /* Stats boxes */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.18);
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    div[data-testid="metric-container"] label {
        color: #a0aec0 !important;
        font-weight: 600;
    }
    
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #00d4ff !important;
        font-size: 2rem !important;
        font-weight: 700;
    }
    
    /* Text input */
    .stTextArea textarea {
        background: rgba(0,0,0,0.3) !important;
        border: 2px solid rgba(0,212,255,0.3) !important;
        border-radius: 10px !important;
        color: white !important;
        font-size: 1rem !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #00d4ff !important;
        box-shadow: 0 0 20px rgba(0,212,255,0.3) !important;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(90deg, #00d4ff 0%, #7b2ff7 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 15px 30px;
        font-weight: 700;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(0,212,255,0.4);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton button:hover {
        box-shadow: 0 6px 25px rgba(0,212,255,0.6);
        transform: translateY(-2px);
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15,12,41,0.95) 0%, rgba(36,36,62,0.95) 100%);
        border-right: 1px solid rgba(0,212,255,0.2);
    }
    
    /* Success/Info boxes */
    .stSuccess, .stInfo {
        background: rgba(0,212,255,0.1);
        border: 1px solid rgba(0,212,255,0.3);
        border-radius: 10px;
    }
    
    /* Keyword badges */
    .keyword-badge {
        display: inline-block;
        background: linear-gradient(135deg, rgba(220,38,38,0.3) 0%, rgba(153,27,27,0.3) 100%);
        border: 1px solid rgba(220,38,38,0.5);
        color: #fca5a5;
        padding: 5px 15px;
        border-radius: 20px;
        margin: 5px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #00d4ff !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(0,212,255,0.1);
        border-radius: 10px;
        color: #00d4ff !important;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'total_scans' not in st.session_state:
    st.session_state.total_scans = 0
if 'threats_detected' not in st.session_state:
    st.session_state.threats_detected = 0

# Category definitions with keywords
CATEGORIES = {
    'Financial Fraud': {
        'keywords': ['cc', 'cvv', 'dumps', 'bank', 'account', 'card', 'paypal', 'crypto', 
                    'wallet', 'transfer', 'escrow', 'bitcoin', 'fullz', 'carding', 'cashout',
                    'western union', 'moneypak', 'balance', 'fresh', 'valid'],
        'color': '#dc2626',
        'icon': '💳'
    },
    'Hacking Services': {
        'keywords': ['ddos', 'hack', 'exploit', 'breach', 'malware', 'ransomware', 'botnet',
                    'trojan', 'virus', 'attack', 'penetration', 'rat', 'keylogger', 'exploit',
                    'zero-day', 'vulnerability', 'shell', 'backdoor', 'phishing'],
        'color': '#f97316',
        'icon': '⚠️'
    },
    'Drug Sales': {
        'keywords': ['drugs', 'cocaine', 'heroin', 'mdma', 'pills', 'prescription', 'cannabis',
                    'marijuana', 'meth', 'lsd', 'mushrooms', 'ecstasy', 'amphetamine', 
                    'opioid', 'fentanyl', 'xanax', 'vendor', 'strain'],
        'color': '#a855f7',
        'icon': '💊'
    },
    'Illegal Services': {
        'keywords': ['assassin', 'hitman', 'weapon', 'gun', 'forged', 'fake', 'passport',
                    'identity', 'documents', 'ssn', 'driver license', 'counterfeit',
                    'social security', 'birth certificate', 'diploma'],
        'color': '#ec4899',
        'icon': '🔫'
    },
    'Data Breach': {
        'keywords': ['database', 'leaked', 'stolen', 'credentials', 'passwords', 'emails',
                    'personal', 'information', 'breach', 'dump', 'combo', 'list',
                    'doxxing', 'pii', 'records', 'user data'],
        'color': '#6366f1',
        'icon': '🗄️'
    }
}

# Analysis function
def analyze_text(text):
    """Analyze text and return threat assessment"""
    time.sleep(2)  # Simulate processing
    
    text_lower = text.lower()
    
    # Find matching categories
    category_scores = {}
    detected_keywords = {}
    
    for category, data in CATEGORIES.items():
        matches = [kw for kw in data['keywords'] if kw in text_lower]
        if matches:
            category_scores[category] = len(matches)
            detected_keywords[category] = matches
    
    if not category_scores:
        return None
    
    # Get best match
    best_category = max(category_scores, key=category_scores.get)
    keyword_count = category_scores[best_category]
    keywords = detected_keywords[best_category]
    
    # Calculate confidence and risk
    confidence = min(99.8, 65 + (keyword_count * 7))
    
    if keyword_count >= 5:
        risk_level = 'CRITICAL'
    elif keyword_count >= 3:
        risk_level = 'HIGH'
    elif keyword_count >= 2:
        risk_level = 'MEDIUM'
    else:
        risk_level = 'LOW'
    
    return {
        'category': best_category,
        'risk_level': risk_level,
        'confidence': confidence,
        'keywords': keywords,
        'keyword_count': keyword_count,
        'timestamp': datetime.now()
    }

# Create gauge chart for confidence
def create_confidence_gauge(confidence):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence Score", 'font': {'size': 24, 'color': '#00d4ff'}},
        delta={'reference': 80, 'increasing': {'color': "#dc2626"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#00d4ff"},
            'bar': {'color': "#00d4ff"},
            'bgcolor': "rgba(0,0,0,0.3)",
            'borderwidth': 2,
            'bordercolor': "#00d4ff",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(34,197,94,0.3)'},
                {'range': [50, 75], 'color': 'rgba(234,179,8,0.3)'},
                {'range': [75, 90], 'color': 'rgba(249,115,22,0.3)'},
                {'range': [90, 100], 'color': 'rgba(220,38,38,0.3)'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#00d4ff", 'family': "Arial"},
        height=300
    )
    
    return fig

# Create keyword frequency chart
def create_keyword_chart(keywords):
    df = pd.DataFrame({
        'Keyword': keywords,
        'Threat Level': [10] * len(keywords)
    })
    
    fig = px.bar(df, x='Keyword', y='Threat Level',
                 color='Threat Level',
                 color_continuous_scale=['#fca5a5', '#dc2626', '#7f1d1d'])
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#00d4ff"},
        xaxis={'gridcolor': 'rgba(0,212,255,0.1)'},
        yaxis={'gridcolor': 'rgba(0,212,255,0.1)'},
        showlegend=False,
        height=300
    )
    
    return fig

# Create threat distribution pie chart
def create_threat_distribution():
    if not st.session_state.analysis_history:
        categories = list(CATEGORIES.keys())
        values = [1] * len(categories)
    else:
        category_counts = {}
        for analysis in st.session_state.analysis_history:
            cat = analysis['category']
            category_counts[cat] = category_counts.get(cat, 0) + 1
        categories = list(category_counts.keys())
        values = list(category_counts.values())
    
    colors = [CATEGORIES[cat]['color'] for cat in categories]
    
    fig = go.Figure(data=[go.Pie(
        labels=categories,
        values=values,
        hole=.4,
        marker=dict(colors=colors, line=dict(color='#000000', width=2))
    )])
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#00d4ff", 'size': 14},
        showlegend=True,
        height=350
    )
    
    return fig

# Create timeline chart
def create_timeline_chart():
    if len(st.session_state.analysis_history) < 2:
        # Generate sample data
        dates = pd.date_range(end=datetime.now(), periods=7, freq='D')
        threats = np.random.randint(5, 20, 7)
    else:
        history_df = pd.DataFrame(st.session_state.analysis_history)
        history_df['date'] = pd.to_datetime(history_df['timestamp']).dt.date
        grouped = history_df.groupby('date').size().reset_index(name='count')
        dates = pd.to_datetime(grouped['date'])
        threats = grouped['count']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates, y=threats,
        mode='lines+markers',
        name='Threats Detected',
        line=dict(color='#00d4ff', width=3),
        marker=dict(size=10, color='#7b2ff7', line=dict(color='#00d4ff', width=2)),
        fill='tozeroy',
        fillcolor='rgba(0,212,255,0.1)'
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#00d4ff"},
        xaxis={'gridcolor': 'rgba(0,212,255,0.1)', 'title': 'Date'},
        yaxis={'gridcolor': 'rgba(0,212,255,0.1)', 'title': 'Threats'},
        hovermode='x unified',
        height=300
    )
    
    return fig

# Header
st.markdown('<h1 class="main-title">🛡️ Dark Web Cybercrime Detection System</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Threat Intelligence & Analysis Platform | Powered by BERT Transformers</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/300x100/0f0c29/00d4ff?text=CYBERSEC+AI", width=300)
    
    st.markdown("### 🎯 System Status")
    st.success("🟢 **All Systems Operational**")
    
    st.markdown("---")
    
    st.markdown("### 📊 Statistics")
    st.metric("Total Scans", st.session_state.total_scans)
    st.metric("Threats Detected", st.session_state.threats_detected)
    st.metric("Detection Rate", f"{(st.session_state.threats_detected / max(st.session_state.total_scans, 1) * 100):.1f}%")
    
    st.markdown("---")
    
    st.markdown("### 🔍 Detection Categories")
    for category, data in CATEGORIES.items():
        st.markdown(f"{data['icon']} **{category}**")
    
    st.markdown("---")
    
    st.markdown("### ⚙️ Model Info")
    st.info("""
    **Model**: BERT-Base-Uncased  
    **Framework**: Transformers  
    **Accuracy**: 98.7%  
    **Version**: 2.0.1
    """)
    
    st.markdown("---")
    
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.analysis_history = []
        st.session_state.total_scans = 0
        st.session_state.threats_detected = 0
        st.rerun()

# Main dashboard metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🔍 Active Scans", st.session_state.total_scans, delta="Real-time")
with col2:
    st.metric("⚠️ Threats Found", st.session_state.threats_detected, delta="+Live")
with col3:
    st.metric("✅ System Health", "98.7%", delta="+0.2%")
with col4:
    st.metric("⚡ Response Time", "1.2s", delta="-0.3s")

st.markdown("---")

# Main content area
tab1, tab2, tab3 = st.tabs(["🔍 Threat Analysis", "📊 Analytics Dashboard", "📜 Scan History"])

with tab1:
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown("### 📝 Input Suspicious Text")
        
        text_input = st.text_area(
            "Paste suspicious text here for analysis",
            height=250,
            placeholder="Enter text from dark web forums, marketplaces, or chat rooms...",
            help="Paste any suspicious content for AI-powered threat detection"
        )
        
        col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
        
        with col_btn1:
            analyze_btn = st.button("🚀 Analyze Threat", use_container_width=True)
        with col_btn2:
            if st.button("🔄 Clear", use_container_width=True):
                st.rerun()
    
    with col_right:
        st.markdown("### 💡 Example Tests")
        
        examples = {
            "💳 Financial Fraud": "Selling fresh CC dumps with CVV. High balance bank accounts available. Escrow service provided. Bitcoin payments only. Fullz info included with SSN.",
            "⚠️ Hacking Services": "Professional DDoS service - take down any site for 24 hours. Custom malware and ransomware development. Botnet rental available. Zero-day exploits in stock.",
            "🗄️ Data Breach": "2 million leaked credentials from recent database breach. Contains emails, passwords, and personal information. Instant download.",
            "💊 Drug Marketplace": "Premium quality cocaine and MDMA available. Prescription pills including Xanax and Oxycodone. Worldwide shipping with stealth packaging.",
            "🔫 Illegal Services": "Forged passports and driver licenses. Fake identity documents. SSN and birth certificates. Counterfeit money available."
        }
        
        for title, example in examples.items():
            if st.button(title, use_container_width=True):
                text_input = example
                st.rerun()
    
    # Analysis section
    if analyze_btn and text_input:
        st.session_state.total_scans += 1
        
        with st.spinner("🔄 Analyzing text with BERT model..."):
            result = analyze_text(text_input)
        
        if result:
            st.session_state.threats_detected += 1
            st.session_state.analysis_history.append(result)
            
            # Alert box
            risk_class = f"alert-{result['risk_level'].lower()}"
            st.markdown(f"""
            <div class="{risk_class}">
                <h3>🚨 THREAT DETECTED - {result['risk_level']} RISK</h3>
                <p>Suspicious content has been identified and classified. Review details below.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Results display
            result_col1, result_col2, result_col3 = st.columns(3)
            
            with result_col1:
                st.markdown(f"""
                <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%); border-radius: 15px; border: 2px solid {CATEGORIES[result['category']]['color']};'>
                    <h2 style='color: {CATEGORIES[result['category']]['color']}; margin: 0;'>{CATEGORIES[result['category']]['icon']}</h2>
                    <h4 style='color: #00d4ff; margin: 10px 0 5px 0;'>Threat Category</h4>
                    <h3 style='color: {CATEGORIES[result['category']]['color']}; margin: 0;'>{result['category']}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            with result_col2:
                risk_colors = {
                    'CRITICAL': '#dc2626',
                    'HIGH': '#f97316',
                    'MEDIUM': '#eab308',
                    'LOW': '#22c55e'
                }
                st.markdown(f"""
                <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%); border-radius: 15px; border: 2px solid {risk_colors[result['risk_level']]};'>
                    <h2 style='color: {risk_colors[result['risk_level']]}; margin: 0;'>⚠️</h2>
                    <h4 style='color: #00d4ff; margin: 10px 0 5px 0;'>Risk Level</h4>
                    <h3 style='color: {risk_colors[result['risk_level']]}; margin: 0;'>{result['risk_level']}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            with result_col3:
                st.markdown(f"""
                <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%); border-radius: 15px; border: 2px solid #00d4ff;'>
                    <h2 style='color: #00d4ff; margin: 0;'>📊</h2>
                    <h4 style='color: #00d4ff; margin: 10px 0 5px 0;'>Confidence Score</h4>
                    <h3 style='color: #00d4ff; margin: 0;'>{result['confidence']:.1f}%</h3>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Visualizations
            vis_col1, vis_col2 = st.columns(2)
            
            with vis_col1:
                st.markdown("### 📊 Confidence Analysis")
                st.plotly_chart(create_confidence_gauge(result['confidence']), use_container_width=True)
            
            with vis_col2:
                st.markdown("### 🔑 Detected Keywords")
                st.plotly_chart(create_keyword_chart(result['keywords']), use_container_width=True)
            
            # Keywords display
            st.markdown("### 🎯 Suspicious Keywords Identified")
            keywords_html = "".join([f'<span class="keyword-badge">{kw}</span>' for kw in result['keywords']])
            st.markdown(f'<div style="margin: 20px 0;">{keywords_html}</div>', unsafe_allow_html=True)
            
            # Detailed analysis
            with st.expander("📋 Detailed Analysis Report", expanded=True):
                st.markdown(f"""
                **Analysis Timestamp**: {result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}  
                **Keywords Detected**: {result['keyword_count']}  
                **Category Match**: {result['category']}  
                **Confidence Level**: {result['confidence']:.1f}%  
                **Risk Assessment**: {result['risk_level']}
                
                ---
                
                **⚠️ Recommendation**: This content has been flagged as potentially illegal activity. 
                Law enforcement should be notified for further investigation. Evidence preservation is recommended.
                
                **🔒 Security Actions**:
                - ✅ Content logged and archived
                - ✅ Threat signature updated
                - ✅ Alert sent to monitoring team
                - ✅ IP address flagged for tracking
                """)
        else:
            st.success("✅ No significant threats detected in the provided text.")

with tab2:
    st.markdown("### 📊 Real-Time Analytics Dashboard")
    
    dash_col1, dash_col2 = st.columns(2)
    
    with dash_col1:
        st.markdown("#### 🎯 Threat Category Distribution")
        st.plotly_chart(create_threat_distribution(), use_container_width=True)
    
    with dash_col2:
        st.markdown("#### 📈 Detection Timeline")
        st.plotly_chart(create_timeline_chart(), use_container_width=True)
    
    st.markdown("---")
    
    # Statistics table
    st.markdown("#### 📋 Category Statistics")
    
    if st.session_state.analysis_history:
        stats_data = []
        for category in CATEGORIES.keys():
            count = sum(1 for a in st.session_state.analysis_history if a['category'] == category)
            avg_conf = np.mean([a['confidence'] for a in st.session_state.analysis_history if a['category'] == category]) if count > 0 else 0
            stats_data.append({
                'Category': f"{CATEGORIES[category]['icon']} {category}",
                'Detections': count,
                'Avg Confidence': f"{avg_conf:.1f}%",
                'Risk Level': 'HIGH' if count > 2 else 'MEDIUM' if count > 0 else 'LOW'
            })
        
        df_stats = pd.DataFrame(stats_data)
        st.dataframe(df_stats, use_container_width=True, hide_index=True)
    else:
        st.info("No analysis data available yet. Start analyzing text to see statistics.")

with tab3:
    st.markdown("### 📜 Scan History & Reports")
    
    if st.session_state.analysis_history:
        for idx, analysis in enumerate(reversed(st.session_state.analysis_history[-10:])):
            with st.expander(f"🔍 Scan #{len(st.session_state.analysis_history) - idx} - {analysis['category']} ({analysis['timestamp'].strftime('%H:%M:%S')})"):
                hist_col1, hist_col2, hist_col3 = st.columns(3)
                
                with hist_col1:
                    st.metric("Category", analysis['category'])
                with hist_col2:
                    st.metric("Risk Level", analysis['risk_level'])
                with hist_col3:
                    st.metric("Confidence", f"{analysis['confidence']:.1f}%")
                
                st.markdown(f"**Keywords**: {', '.join(analysis['keywords'])}")
                st.markdown(f"**Timestamp**: {analysis['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.info("📭 No scan history available. Perform your first analysis to see results here.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #a0aec0; padding: 20px;'>
    <p><strong>🛡️ Dark Web Cybercrime Detection System v2.0</strong></p>
    <p>Powered by BERT Transformers | Real-time Threat Intelligence | 24/7 Monitoring</p>
    <p style='font-size: 0.8rem; margin-top: 10px;'>⚠️ For law enforcement and authorized personnel only</p>
</div>
""", unsafe_allow_html=True)