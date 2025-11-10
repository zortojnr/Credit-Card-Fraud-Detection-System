import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)

# Advanced Professional CSS Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Header with Animation */
    .main-header {
        font-size: 3.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 1.5rem;
        padding: 1.5rem 0;
        animation: gradient-shift 3s ease infinite;
        background-size: 200% 200%;
    }
    
    @keyframes gradient-shift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    /* Navigation Cards with Hover Effects */
    .nav-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 2rem;
        border-radius: 16px;
        margin: 1.5rem 0;
        border: 2px solid #e9ecef;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
    }
    
    .nav-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 12px 24px rgba(102, 126, 234, 0.2);
        border-color: #667eea;
    }
    
    .nav-card h3 {
        color: #2c3e50;
        margin-bottom: 0.75rem;
        font-size: 1.5rem;
        font-weight: 700;
    }
    
    .nav-card p {
        color: #5a6c7d;
        margin: 0;
        line-height: 1.8;
        font-size: 1rem;
    }
    
    /* Alert Boxes with Animations */
    .fraud-alert {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        border-left: 6px solid #c92a2a;
        padding: 2rem;
        border-radius: 16px;
        margin: 1.5rem 0;
        color: white;
        box-shadow: 0 8px 16px rgba(238, 90, 111, 0.4);
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.01); }
    }
    
    .fraud-alert h2 {
        color: white;
        margin-bottom: 1rem;
        font-size: 2rem;
        font-weight: 700;
    }
    
    .valid-alert {
        background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
        border-left: 6px solid #2f9e44;
        padding: 2rem;
        border-radius: 16px;
        margin: 1.5rem 0;
        color: white;
        box-shadow: 0 8px 16px rgba(64, 192, 87, 0.4);
    }
    
    .valid-alert h2 {
        color: white;
        margin-bottom: 1rem;
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2c3e50;
        margin: 2.5rem 0 1.5rem 0;
        padding-bottom: 1rem;
        border-bottom: 4px solid;
        border-image: linear-gradient(135deg, #667eea 0%, #764ba2 100%) 1;
    }
    
    /* Info Box */
    .info-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 2rem;
        border-radius: 16px;
        border-left: 6px solid #2196f3;
        margin: 1.5rem 0;
        box-shadow: 0 4px 12px rgba(33, 150, 243, 0.15);
    }
    
    /* Feature List */
    .feature-list {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        margin: 1.5rem 0;
        transition: transform 0.3s;
    }
    
    .feature-list:hover {
        transform: translateY(-4px);
    }
    
    .feature-list li {
        padding: 0.75rem 0;
        color: #2c3e50;
        font-size: 1.05rem;
        line-height: 1.8;
    }
    
    /* Quick Start Section */
    .quick-start {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        margin: 2.5rem 0;
        box-shadow: 0 12px 24px rgba(245, 87, 108, 0.3);
    }
    
    .quick-start h3 {
        color: white;
        margin-bottom: 1.5rem;
        font-size: 2rem;
        font-weight: 700;
    }
    
    .quick-start ol {
        color: white;
        font-size: 1.15rem;
        line-height: 2.2;
        padding-left: 2rem;
    }
    
    .quick-start li {
        margin: 1rem 0;
        font-weight: 500;
    }
    
    /* Loading Animation */
    .loading-spinner {
        display: inline-block;
        width: 40px;
        height: 40px;
        border: 4px solid #f3f3f3;
        border-top: 4px solid #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2.5rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.5);
    }
    
    /* Metric Cards */
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 16px;
        color: white;
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.3);
        margin: 0.5rem 0;
    }
    
    /* Data Table Styling */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* Progress Bar */
    .progress-container {
        background: #f0f0f0;
        border-radius: 10px;
        padding: 0.5rem;
        margin: 1rem 0;
    }
    
    /* Sidebar Enhancements */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
    }
    
    /* Input Field Styling */
    .stNumberInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        transition: all 0.3s;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 1rem 1.5rem;
        font-weight: 600;
    }
    
    /* Success/Error Messages */
    .stSuccess {
        border-radius: 12px;
        padding: 1rem;
    }
    
    .stError {
        border-radius: 12px;
        padding: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []

@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    """Load the credit card dataset with validation"""
    try:
        with st.spinner("🔄 Loading dataset..."):
            data = pd.read_csv("creditcard.csv")
            
            # Validate data structure
            required_cols = ['Time', 'Amount', 'Class'] + [f'V{i+1}' for i in range(28)]
            missing_cols = [col for col in required_cols if col not in data.columns]
            
            if missing_cols:
                st.error(f"❌ Dataset missing required columns: {', '.join(missing_cols)}")
                return None
            
            # Validate data quality
            if data.empty:
                st.error("❌ Dataset is empty!")
                return None
            
            if 'Class' not in data.columns:
                st.error("❌ 'Class' column not found in dataset!")
                return None
            
            # Check for valid Class values
            if not data['Class'].isin([0, 1]).all():
                st.warning("⚠️ 'Class' column contains invalid values. Expected 0 or 1.")
            
            return data
    except FileNotFoundError:
        st.error("❌ creditcard.csv file not found! Please make sure the file is in the project directory.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        return None

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    """Load the trained model with validation"""
    try:
        with st.spinner("🔄 Loading trained model..."):
            model = joblib.load('fraud_detection_model.pkl')
            
            # Validate model
            if model is None:
                st.error("❌ Model file is empty or corrupted!")
                return None
            
            return model
    except FileNotFoundError:
        st.error("❌ Model file not found! Please run main.py first to train and save the model.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

def validate_transaction_input(data_dict):
    """Validate transaction input data"""
    errors = []
    
    # Check required fields
    required_fields = ['Time', 'Amount'] + [f'V{i+1}' for i in range(28)]
    for field in required_fields:
        if field not in data_dict:
            errors.append(f"Missing required field: {field}")
    
    # Validate Time
    if 'Time' in data_dict and (data_dict['Time'] < 0 or data_dict['Time'] > 172792):
        errors.append("Time must be between 0 and 172792")
    
    # Validate Amount
    if 'Amount' in data_dict and data_dict['Amount'] < 0:
        errors.append("Amount cannot be negative")
    
    return errors

def predict_transaction(model, transaction_data):
    """Predict if a transaction is fraudulent with error handling"""
    try:
        prediction = model.predict(transaction_data)
        probability = model.predict_proba(transaction_data)
        return prediction[0], probability[0], None
    except Exception as e:
        return None, None, str(e)

def create_advanced_visualizations(data):
    """Create advanced interactive visualizations"""
    fraud = data[data['Class'] == 1]
    valid = data[data['Class'] == 0]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Transaction Amount Distribution', 'Time vs Amount', 
                       'Fraud by Hour', 'Feature Correlation (Top 10)'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Amount distribution
    fig.add_trace(
        go.Histogram(x=valid['Amount'], name='Valid', opacity=0.7, nbinsx=50, marker_color='green'),
        row=1, col=1
    )
    fig.add_trace(
        go.Histogram(x=fraud['Amount'], name='Fraud', opacity=0.7, nbinsx=50, marker_color='red'),
        row=1, col=1
    )
    
    # Time vs Amount scatter
    fig.add_trace(
        go.Scatter(x=valid['Time'], y=valid['Amount'], mode='markers', 
                  name='Valid', marker=dict(color='green', opacity=0.3, size=3)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=fraud['Time'], y=fraud['Amount'], mode='markers', 
                  name='Fraud', marker=dict(color='red', opacity=0.6, size=5)),
        row=1, col=2
    )
    
    # Fraud by hour
    data['Hour'] = (data['Time'] / 3600) % 24
    fraud_by_hour = data[data['Class'] == 1].groupby('Hour').size()
    fig.add_trace(
        go.Bar(x=fraud_by_hour.index, y=fraud_by_hour.values, name='Fraud Cases', marker_color='red'),
        row=2, col=1
    )
    
    # Top correlations
    corr_matrix = data.corr()['Class'].abs().sort_values(ascending=False).head(11)
    corr_matrix = corr_matrix[corr_matrix.index != 'Class']
    fig.add_trace(
        go.Bar(x=corr_matrix.values, y=corr_matrix.index, orientation='h', 
              name='Correlation', marker_color='purple'),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=True, title_text="Advanced Data Analysis Dashboard")
    fig.update_xaxes(title_text="Amount", row=1, col=1)
    fig.update_xaxes(title_text="Time", row=1, col=2)
    fig.update_xaxes(title_text="Hour of Day", row=2, col=1)
    fig.update_xaxes(title_text="Correlation", row=2, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Amount", row=1, col=2)
    fig.update_yaxes(title_text="Fraud Cases", row=2, col=1)
    
    return fig

def main():
    # Header with animation
    st.markdown('<h1 class="main-header">💳 Credit Card Fraud Detection System</h1>', unsafe_allow_html=True)
    
    # Professional Sidebar Navigation
    st.sidebar.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 2rem;
                border-radius: 16px;
                margin-bottom: 2rem;
                text-align: center;
                box-shadow: 0 8px 16px rgba(102, 126, 234, 0.3);">
        <h2 style="color: white; margin: 0; font-size: 1.8rem; font-weight: 700;">🧭 Navigation</h2>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.radio(
        "Select a page:",
        ["🏠 Home", "📊 Data Analysis", "🔍 Fraud Detection", "📈 Model Performance", "📋 Prediction History"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    
    # System Status
    st.sidebar.markdown("""
    <div style="padding: 1.5rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                border-radius: 12px; margin-top: 2rem;">
        <h4 style="color: #2c3e50; margin-bottom: 1rem;">⚙️ System Status</h4>
        <p style="color: #5a6c7d; font-size: 0.95rem; line-height: 1.8;">
        <strong>Model:</strong> Random Forest<br>
        <strong>Status:</strong> <span style="color: #51cf66;">● Active</span><br>
        <strong>Version:</strong> 2.0
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data and model
    data = load_data()
    model = load_model()
    
    if data is None or model is None:
        st.warning("⚠️ Please ensure both the dataset and model files are available.")
        st.stop()
    
    # Route to appropriate page
    if page == "🏠 Home":
        show_home_page(data)
    elif page == "📊 Data Analysis":
        show_data_analysis(data)
    elif page == "🔍 Fraud Detection":
        show_fraud_detection(model, data)
    elif page == "📈 Model Performance":
        show_model_performance(model, data)
    elif page == "📋 Prediction History":
        show_prediction_history()

def show_home_page(data):
    """Display enhanced home page"""
    st.markdown('<div class="section-header">Welcome to Credit Card Fraud Detection System</div>', unsafe_allow_html=True)
    
    # Real-time Metrics with Progress Indicators
    col1, col2, col3, col4 = st.columns(4)
    
    total_transactions = len(data)
    fraud_cases = len(data[data['Class'] == 1])
    valid_cases = len(data[data['Class'] == 0])
    fraud_percentage = (fraud_cases / total_transactions) * 100
    
    with col1:
        st.metric("Total Transactions", f"{total_transactions:,}", help="Total number of transactions in the dataset")
        st.progress(1.0)
    
    with col2:
        st.metric("Fraud Cases", f"{fraud_cases:,}", delta=f"{fraud_percentage:.2f}%", 
                 delta_color="inverse", help="Number of fraudulent transactions detected")
        st.progress(fraud_percentage / 100)
    
    with col3:
        st.metric("Valid Transactions", f"{valid_cases:,}", help="Number of legitimate transactions")
        st.progress(valid_cases / total_transactions)
    
    with col4:
        st.metric("Fraud Rate", f"{fraud_percentage:.3f}%", help="Percentage of fraudulent transactions")
        st.progress(fraud_percentage / 100)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Quick Stats Visualization
    st.markdown('<div class="section-header">📊 Quick Statistics</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Amount statistics
        fraud_amount = data[data['Class'] == 1]['Amount']
        valid_amount = data[data['Class'] == 0]['Amount']
        
        stats_data = pd.DataFrame({
            'Type': ['Valid', 'Fraud'],
            'Mean Amount': [valid_amount.mean(), fraud_amount.mean()],
            'Median Amount': [valid_amount.median(), fraud_amount.median()],
            'Max Amount': [valid_amount.max(), fraud_amount.max()]
        })
        
        st.dataframe(stats_data, use_container_width=True, hide_index=True)
    
    with col2:
        # Class distribution pie chart
        class_counts = data['Class'].value_counts()
        fig = px.pie(
            values=class_counts.values, 
            names=['Valid', 'Fraud'],
            title="Transaction Class Distribution",
            color_discrete_map={'Valid': '#51cf66', 'Fraud': '#ff6b6b'}
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    # About Section
    st.markdown('<div class="section-header">📋 About This System</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <p style="font-size: 1.15rem; line-height: 2; color: #2c3e50;">
        This <strong>Credit Card Fraud Detection System</strong> uses advanced machine learning techniques 
        with a <strong>Random Forest Classifier</strong> to identify potentially fraudulent transactions in real-time. 
        The system provides comprehensive analysis tools, interactive visualizations, and intuitive interfaces 
        to help detect and prevent fraud with high accuracy.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features Grid
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-list">
            <h4 style="color: #667eea; margin-bottom: 1.5rem; font-size: 1.3rem;">✨ Key Features</h4>
            <ul style="list-style: none; padding: 0;">
                <li>🔍 <strong>Real-time Fraud Detection</strong> - Analyze individual transactions instantly with high accuracy</li>
                <li>📊 <strong>Advanced Data Analysis</strong> - Explore transaction patterns with interactive visualizations</li>
                <li>📈 <strong>Comprehensive Metrics</strong> - View detailed model evaluation and performance metrics</li>
                <li>📉 <strong>Interactive Dashboards</strong> - Dynamic charts, graphs, and correlation heatmaps</li>
                <li>💾 <strong>Batch Processing</strong> - Analyze multiple transactions simultaneously</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-list">
            <h4 style="color: #667eea; margin-bottom: 1.5rem; font-size: 1.3rem;">🛡️ Advanced Capabilities</h4>
            <ul style="list-style: none; padding: 0;">
                <li>⚡ <strong>Fast Processing</strong> - Optimized algorithms for quick analysis</li>
                <li>🎯 <strong>High Accuracy</strong> - State-of-the-art ML algorithms for precise detection</li>
                <li>📱 <strong>User-Friendly Interface</strong> - Intuitive design for seamless navigation</li>
                <li>📋 <strong>Prediction History</strong> - Track and review past predictions</li>
                <li>🔐 <strong>Data Validation</strong> - Comprehensive input validation and error handling</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Quick Start Guide
    st.markdown("""
    <div class="quick-start">
        <h3>🚀 Quick Start Guide</h3>
        <ol>
            <li><strong>Navigate to 📊 Data Analysis</strong> to explore the dataset, view comprehensive statistics, and analyze transaction patterns with interactive visualizations</li>
            <li><strong>Go to 🔍 Fraud Detection</strong> to check if transactions are fraudulent using manual input or CSV file upload with real-time predictions</li>
            <li><strong>Visit 📈 Model Performance</strong> to see detailed model evaluation metrics including accuracy, precision, recall, F1-score, and confusion matrix visualization</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation Cards
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">📍 Navigation</div>', unsafe_allow_html=True)
    
    nav_col1, nav_col2, nav_col3 = st.columns(3)
    
    with nav_col1:
        st.markdown("""
        <div class="nav-card">
            <h3>📊 Data Analysis</h3>
            <p>Explore the dataset with advanced interactive visualizations, comprehensive statistics, 
            correlation analysis, and real-time data exploration tools. Understand transaction patterns 
            and data distribution in detail.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with nav_col2:
        st.markdown("""
        <div class="nav-card">
            <h3>🔍 Fraud Detection</h3>
            <p>Check if transactions are fraudulent using manual input or batch processing. 
            Get instant predictions with detailed probability scores, risk assessment, and 
            comprehensive transaction analysis.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with nav_col3:
        st.markdown("""
        <div class="nav-card">
            <h3>📈 Model Performance</h3>
            <p>View comprehensive model evaluation metrics including accuracy, precision, recall, 
            F1-score, Matthews Correlation Coefficient, and detailed confusion matrix visualization 
            with classification reports.</p>
        </div>
        """, unsafe_allow_html=True)

def show_data_analysis(data):
    """Display advanced data analysis page"""
    st.markdown('<div class="section-header">📊 Advanced Data Analysis</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Overview", "📈 Statistics", "🎨 Visualizations", "🔍 Exploratory Analysis", "📄 Raw Data"])
    
    with tab1:
        st.subheader("Dataset Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", f"{data.shape[0]:,}")
        with col2:
            st.metric("Total Columns", f"{data.shape[1]}")
        with col3:
            st.metric("Memory Usage", f"{data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        st.dataframe(data.head(20), use_container_width=True, height=400)
        
        st.subheader("Data Types & Info")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Data Types:**")
            st.dataframe(data.dtypes.to_frame('Type'), use_container_width=True)
        with col2:
            st.write("**Missing Values:**")
            missing = data.isnull().sum()
            if missing.sum() == 0:
                st.success("✅ No missing values in the dataset!")
            else:
                st.dataframe(missing[missing > 0].to_frame('Missing Count'))
        
        st.subheader("Class Distribution")
        fraud = data[data['Class'] == 1]
        valid = data[data['Class'] == 0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Fraudulent Transactions", len(fraud), delta=f"{(len(fraud)/len(data)*100):.3f}%")
            st.dataframe(fraud.Amount.describe().to_frame('Fraud Amount'), use_container_width=True)
        with col2:
            st.metric("Valid Transactions", len(valid), delta=f"{(len(valid)/len(data)*100):.3f}%")
            st.dataframe(valid.Amount.describe().to_frame('Valid Amount'), use_container_width=True)
    
    with tab2:
        st.subheader("Comprehensive Statistical Summary")
        st.dataframe(data.describe(), use_container_width=True)
        
        st.subheader("Correlation with Target")
        corr_with_class = data.corr()['Class'].abs().sort_values(ascending=False)
        corr_with_class = corr_with_class[corr_with_class.index != 'Class']
        st.dataframe(corr_with_class.head(15).to_frame('Correlation'), use_container_width=True)
    
    with tab3:
        st.subheader("Interactive Visualizations")
        
        # Advanced dashboard
        st.plotly_chart(create_advanced_visualizations(data), use_container_width=True)
        
        # Additional visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Amount Distribution")
            fig = px.histogram(data, x='Amount', color='Class', nbins=50, 
                             color_discrete_map={0: 'green', 1: 'red'},
                             labels={'Class': 'Transaction Type'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Time Distribution")
            data['Hour'] = (data['Time'] / 3600) % 24
            fig = px.histogram(data, x='Hour', color='Class', nbins=24,
                             color_discrete_map={0: 'green', 1: 'red'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        if st.checkbox("Show Full Correlation Heatmap", value=False):
            st.subheader("Feature Correlation Matrix")
            corrmat = data.corr()
            fig = px.imshow(corrmat, color_continuous_scale='RdBu', aspect='auto')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Exploratory Data Analysis")
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            min_amount = st.number_input("Min Amount", min_value=0.0, value=0.0, step=1.0)
        with col2:
            max_amount = st.number_input("Max Amount", min_value=0.0, value=float(data['Amount'].max()), step=1.0)
        
        filtered_data = data[(data['Amount'] >= min_amount) & (data['Amount'] <= max_amount)]
        
        st.metric("Filtered Transactions", len(filtered_data))
        
        # Analysis by class
        if len(filtered_data) > 0:
            fraud_filtered = filtered_data[filtered_data['Class'] == 1]
            valid_filtered = filtered_data[filtered_data['Class'] == 0]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Fraud Cases", len(fraud_filtered))
            with col2:
                st.metric("Valid Cases", len(valid_filtered))
            with col3:
                if len(filtered_data) > 0:
                    st.metric("Fraud Rate", f"{(len(fraud_filtered)/len(filtered_data)*100):.2f}%")
            with col4:
                st.metric("Avg Amount", f"${filtered_data['Amount'].mean():.2f}")
    
    with tab5:
        st.subheader("Raw Data Explorer")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            rows_to_show = st.slider("Rows to display", 10, min(5000, len(data)), 100)
        with col2:
            show_fraud_only = st.checkbox("Show fraud only", value=False)
        with col3:
            sort_by = st.selectbox("Sort by", ['Time', 'Amount', 'Class'])
        
        display_data = data.copy()
        if show_fraud_only:
            display_data = display_data[display_data['Class'] == 1]
        
        display_data = display_data.sort_values(by=sort_by)
        st.dataframe(display_data.head(rows_to_show), use_container_width=True, height=600)
        
        # Download option
        csv = display_data.head(rows_to_show).to_csv(index=False)
        st.download_button("📥 Download Filtered Data", csv, "filtered_data.csv", "text/csv")

def show_fraud_detection(model, data):
    """Display advanced fraud detection interface"""
    st.markdown('<div class="section-header">🔍 Advanced Fraud Detection</div>', unsafe_allow_html=True)
    
    detection_method = st.radio(
        "Choose detection method:",
        ["📝 Manual Input", "📁 Upload CSV File", "🎲 Sample Transaction"],
        horizontal=True
    )
    
    if detection_method == "📝 Manual Input":
        st.subheader("Enter Transaction Details")
        
        with st.form("transaction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                time = st.number_input("Time", min_value=0.0, value=float(data['Time'].mean()), 
                                      step=1.0, help="Time in seconds since first transaction")
                amount = st.number_input("Amount", min_value=0.0, value=0.0, 
                                        step=0.01, format="%.2f", help="Transaction amount")
            
            with col2:
                st.write("**PCA Features (V1-V28)**")
                st.caption("Enter values for principal component analysis features")
            
            # Organized V features input
            v_features = {}
            st.write("**Feature Values:**")
            cols = st.columns(4)
            for i in range(28):
                col_idx = i % 4
                with cols[col_idx]:
                    v_features[f'V{i+1}'] = st.number_input(
                        f"V{i+1}", 
                        value=0.0, 
                        step=0.001,
                        format="%.6f",
                        key=f"v{i+1}",
                        help=f"Principal component {i+1}"
                    )
            
            submitted = st.form_submit_button("🔍 Detect Fraud", type="primary", use_container_width=True)
            
            if submitted:
                with st.spinner("🔄 Analyzing transaction..."):
                    time.sleep(0.5)  # Simulate processing
                    
                    # Validate input
                    transaction_data = {'Time': time, 'Amount': amount, **v_features}
                    errors = validate_transaction_input(transaction_data)
                    
                    if errors:
                        for error in errors:
                            st.error(f"❌ {error}")
                    else:
                        # Prepare input data
                        feature_names = ['Time'] + [f'V{i+1}' for i in range(28)] + ['Amount']
                        input_values = [time] + [v_features[f'V{i+1}'] for i in range(28)] + [amount]
                        
                        transaction_df = pd.DataFrame([input_values], columns=feature_names)
                        
                        # Make prediction
                        prediction, probability, error = predict_transaction(model, transaction_df)
                        
                        if error:
                            st.error(f"❌ Prediction error: {error}")
                        else:
                            # Store in history
                            st.session_state.predictions_history.append({
                                'time': time,
                                'amount': amount,
                                'prediction': prediction,
                                'fraud_prob': probability[1],
                                'valid_prob': probability[0],
                                'timestamp': pd.Timestamp.now()
                            })
                            
                            # Display result
                            st.markdown("---")
                            if prediction == 1:
                                st.markdown(f"""
                                <div class="fraud-alert">
                                    <h2>⚠️ FRAUD DETECTED!</h2>
                                    <p style="font-size: 1.2rem;">This transaction has been flagged as potentially fraudulent.</p>
                                    <p style="font-size: 1.1rem;"><strong>Fraud Probability:</strong> {probability[1]:.2%}</p>
                                    <p style="font-size: 1.1rem;"><strong>Risk Level:</strong> {'HIGH' if probability[1] > 0.8 else 'MEDIUM'}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class="valid-alert">
                                    <h2>✅ VALID TRANSACTION</h2>
                                    <p style="font-size: 1.2rem;">This transaction appears to be legitimate.</p>
                                    <p style="font-size: 1.1rem;"><strong>Valid Probability:</strong> {probability[0]:.2%}</p>
                                    <p style="font-size: 1.1rem;"><strong>Confidence:</strong> {'HIGH' if probability[0] > 0.9 else 'MEDIUM'}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Probability visualization
                            col1, col2 = st.columns(2)
                            with col1:
                                fig = go.Figure(data=[
                                    go.Bar(x=['Valid', 'Fraud'], 
                                          y=[probability[0], probability[1]],
                                          marker_color=['#51cf66', '#ff6b6b'])
                                ])
                                fig.update_layout(title="Prediction Probabilities", height=300)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                st.metric("Valid Probability", f"{probability[0]:.2%}")
                                st.metric("Fraud Probability", f"{probability[1]:.2%}")
                                st.metric("Transaction Amount", f"${amount:,.2f}")
    
    elif detection_method == "📁 Upload CSV File":
        st.subheader("Batch Transaction Analysis")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv", help="Upload a CSV file with transaction data")
        
        if uploaded_file is not None:
            try:
                with st.spinner("🔄 Loading file..."):
                    df = pd.read_csv(uploaded_file)
                    st.success(f"✅ File uploaded successfully! ({len(df)} rows)")
                
                # Check if required columns exist
                required_cols = ['Time'] + [f'V{i+1}' for i in range(28)] + ['Amount']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                    st.info("💡 Required columns: Time, V1-V28, Amount")
                else:
                    if st.button("🔍 Analyze All Transactions", type="primary", use_container_width=True):
                        with st.spinner("🔄 Processing transactions..."):
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            # Make predictions
                            X = df[required_cols]
                            predictions = model.predict(X)
                            probabilities = model.predict_proba(X)
                            
                            progress_bar.progress(100)
                            status_text.success("✅ Analysis complete!")
                            
                            # Add predictions to dataframe
                            df['Prediction'] = predictions
                            df['Fraud_Probability'] = probabilities[:, 1]
                            df['Valid_Probability'] = probabilities[:, 0]
                            df['Risk_Level'] = df['Fraud_Probability'].apply(
                                lambda x: 'HIGH' if x > 0.8 else 'MEDIUM' if x > 0.5 else 'LOW'
                            )
                            
                            # Display results
                            st.markdown("---")
                            st.subheader("📊 Detection Results Summary")
                            
                            fraud_count = sum(predictions == 1)
                            valid_count = sum(predictions == 0)
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Transactions", len(df))
                            with col2:
                                st.metric("Fraudulent", fraud_count, delta=f"{(fraud_count/len(df)*100):.2f}%", delta_color="inverse")
                            with col3:
                                st.metric("Valid", valid_count)
                            with col4:
                                st.metric("Avg Fraud Prob", f"{df['Fraud_Probability'].mean():.2%}")
                            
                            # Visualization
                            col1, col2 = st.columns(2)
                            with col1:
                                fig = px.pie(values=[valid_count, fraud_count], 
                                           names=['Valid', 'Fraud'],
                                           title="Prediction Distribution",
                                           color_discrete_map={'Valid': '#51cf66', 'Fraud': '#ff6b6b'})
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                fig = px.histogram(df, x='Fraud_Probability', nbins=30,
                                                 title="Fraud Probability Distribution",
                                                 color_discrete_sequence=['#ff6b6b'])
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Results table
                            st.subheader("📋 Detailed Results")
                            st.dataframe(df, use_container_width=True, height=400)
                            
                            # Download results
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv,
                                file_name=f"fraud_detection_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
                st.info("💡 Please ensure the file is a valid CSV with the required columns.")
    
    else:  # Sample Transaction
        st.subheader("Test with Sample Transaction")
        
        if st.button("🎲 Generate Random Sample", type="primary"):
            # Get a random sample from the dataset
            sample = data.sample(1).iloc[0]
            
            st.info("💡 Using a random transaction from the dataset")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Actual Class", "Fraud" if sample['Class'] == 1 else "Valid")
            with col2:
                st.metric("Amount", f"${sample['Amount']:,.2f}")
            
            # Prepare for prediction
            feature_names = ['Time'] + [f'V{i+1}' for i in range(28)] + ['Amount']
            sample_df = sample[feature_names].to_frame().T
            
            with st.spinner("🔄 Analyzing sample transaction..."):
                prediction, probability, error = predict_transaction(model, sample_df)
                
                if error:
                    st.error(f"❌ Error: {error}")
                else:
                    # Display result
                    actual = "Fraud" if sample['Class'] == 1 else "Valid"
                    predicted = "Fraud" if prediction == 1 else "Valid"
                    correct = (sample['Class'] == prediction)
                    
                    if correct:
                        st.success(f"✅ Correct! Predicted: {predicted}, Actual: {actual}")
                    else:
                        st.error(f"❌ Incorrect! Predicted: {predicted}, Actual: {actual}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Valid Probability", f"{probability[0]:.2%}")
                    with col2:
                        st.metric("Fraud Probability", f"{probability[1]:.2%}")

def show_model_performance(model, data):
    """Display comprehensive model performance metrics"""
    st.markdown('<div class="section-header">📈 Model Performance Analysis</div>', unsafe_allow_html=True)
    
    with st.spinner("🔄 Calculating performance metrics..."):
        # Prepare data
        X = data.drop(['Class'], axis=1)
        Y = data['Class']
        xData = X.values
        yData = Y.values
        
        from sklearn.model_selection import train_test_split
        xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size=0.2, random_state=42)
        
        # Make predictions
        yPred = model.predict(xTest)
        yPred_proba = model.predict_proba(xTest)
        
        # Calculate metrics
        accuracy = accuracy_score(yTest, yPred)
        precision = precision_score(yTest, yPred)
        recall = recall_score(yTest, yPred)
        f1 = f1_score(yTest, yPred)
        mcc = matthews_corrcoef(yTest, yPred)
        
        # Display metrics in cards
        st.subheader("📊 Evaluation Metrics")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        metrics_data = [
            ("Accuracy", accuracy, "#667eea"),
            ("Precision", precision, "#51cf66"),
            ("Recall", recall, "#f093fb"),
            ("F1-Score", f1, "#f5576c"),
            ("MCC", mcc, "#764ba2")
        ]
        
        for i, (name, value, color) in enumerate(metrics_data):
            with [col1, col2, col3, col4, col5][i]:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {color} 0%, {color}dd 100%);
                            padding: 1.5rem;
                            border-radius: 12px;
                            color: white;
                            text-align: center;
                            box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
                    <h3 style="margin: 0; font-size: 0.9rem; opacity: 0.9;">{name}</h3>
                    <h2 style="margin: 0.5rem 0 0 0; font-size: 2rem; font-weight: 700;">{value:.4f}</h2>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Confusion Matrix
        st.subheader("📉 Confusion Matrix")
        conf_matrix = confusion_matrix(yTest, yPred)
        
        fig = px.imshow(
            conf_matrix,
            labels=dict(x="Predicted", y="Actual"),
            x=['Normal', 'Fraud'],
            y=['Normal', 'Fraud'],
            text_auto=True,
            aspect="auto",
            color_continuous_scale='Blues',
            title="Confusion Matrix Visualization"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Classification Report
        st.subheader("📋 Detailed Classification Report")
        report = classification_report(yTest, yPred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)
        
        # Model Information
        st.markdown("---")
        st.subheader("ℹ️ Model Information")
        
        info_col1, info_col2, info_col3 = st.columns(3)
        with info_col1:
            st.info(f"**Algorithm:** Random Forest Classifier\n\n**Training Samples:** {len(xTrain):,}")
        with info_col2:
            st.info(f"**Test Samples:** {len(xTest):,}\n\n**Features:** {X.shape[1]}")
        with info_col3:
            st.info(f"**Model Type:** Supervised Learning\n\n**Task:** Binary Classification")
        
        # Performance Visualization
        st.subheader("📊 Performance Metrics Visualization")
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'MCC'],
            'Score': [accuracy, precision, recall, f1, mcc]
        })
        
        fig = px.bar(metrics_df, x='Metric', y='Score', 
                    title="Model Performance Metrics",
                    color='Score',
                    color_continuous_scale='Viridis',
                    text='Score')
        fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def show_prediction_history():
    """Display prediction history"""
    st.markdown('<div class="section-header">📋 Prediction History</div>', unsafe_allow_html=True)
    
    if not st.session_state.predictions_history:
        st.info("📝 No predictions yet. Start detecting fraud to see your history here!")
    else:
        history_df = pd.DataFrame(st.session_state.predictions_history)
        
        st.metric("Total Predictions", len(history_df))
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            fraud_count = sum(history_df['prediction'] == 1)
            st.metric("Fraud Detected", fraud_count)
        with col2:
            valid_count = sum(history_df['prediction'] == 0)
            st.metric("Valid Transactions", valid_count)
        with col3:
            st.metric("Avg Fraud Prob", f"{history_df['fraud_prob'].mean():.2%}")
        with col4:
            st.metric("Total Amount", f"${history_df['amount'].sum():,.2f}")
        
        # Display history
        st.subheader("📊 Recent Predictions")
        display_df = history_df[['timestamp', 'amount', 'prediction', 'fraud_prob', 'valid_prob']].copy()
        display_df['prediction'] = display_df['prediction'].map({0: 'Valid', 1: 'Fraud'})
        display_df.columns = ['Timestamp', 'Amount', 'Prediction', 'Fraud Probability', 'Valid Probability']
        st.dataframe(display_df.sort_values('Timestamp', ascending=False), use_container_width=True, height=400)
        
        # Clear history
        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.predictions_history = []
            st.success("✅ History cleared!")
            st.rerun()

if __name__ == "__main__":
    main()
