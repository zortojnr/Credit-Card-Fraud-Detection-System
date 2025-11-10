import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    /* Main Header */
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem 0;
    }
    
    /* Navigation Cards */
    .nav-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .nav-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .nav-card h3 {
        color: #2c3e50;
        margin-bottom: 0.5rem;
        font-size: 1.3rem;
    }
    
    .nav-card p {
        color: #5a6c7d;
        margin: 0;
        line-height: 1.6;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Alert Boxes */
    .fraud-alert {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        border-left: 5px solid #c92a2a;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 4px 6px rgba(238, 90, 111, 0.3);
    }
    
    .fraud-alert h2 {
        color: white;
        margin-bottom: 0.5rem;
    }
    
    .valid-alert {
        background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
        border-left: 5px solid #2f9e44;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 4px 6px rgba(64, 192, 87, 0.3);
    }
    
    .valid-alert h2 {
        color: white;
        margin-bottom: 0.5rem;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 2rem;
        font-weight: 600;
        color: #2c3e50;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    
    /* Info Box */
    .info-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #2196f3;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Feature List */
    .feature-list {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .feature-list li {
        padding: 0.5rem 0;
        color: #2c3e50;
    }
    
    /* Quick Start Section */
    .quick-start {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        margin: 2rem 0;
        box-shadow: 0 8px 16px rgba(245, 87, 108, 0.3);
    }
    
    .quick-start h3 {
        color: white;
        margin-bottom: 1rem;
        font-size: 1.8rem;
    }
    
    .quick-start ol {
        color: white;
        font-size: 1.1rem;
        line-height: 2;
    }
    
    .quick-start li {
        margin: 0.8rem 0;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        padding-top: 3rem;
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
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the credit card dataset"""
    try:
        data = pd.read_csv("creditcard.csv")
        return data
    except FileNotFoundError:
        st.error("❌ creditcard.csv file not found! Please make sure the file is in the project directory.")
        return None

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model = joblib.load('fraud_detection_model.pkl')
        return model
    except FileNotFoundError:
        st.error("❌ Model file not found! Please run main.py first to train and save the model.")
        return None

def predict_transaction(model, transaction_data):
    """Predict if a transaction is fraudulent"""
    prediction = model.predict(transaction_data)
    probability = model.predict_proba(transaction_data)
    return prediction[0], probability[0]

def main():
    # Header
    st.markdown('<h1 class="main-header">💳 Credit Card Fraud Detection System</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation with professional styling
    st.sidebar.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 1.5rem;
                border-radius: 12px;
                margin-bottom: 2rem;
                text-align: center;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <h2 style="color: white; margin: 0; font-size: 1.5rem;">🧭 Navigation</h2>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.radio(
        "Select a page:",
        ["🏠 Home", "📊 Data Analysis", "🔍 Fraud Detection", "📈 Model Performance"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="padding: 1rem; background: #f8f9fa; border-radius: 8px; margin-top: 2rem;">
        <h4 style="color: #2c3e50; margin-bottom: 0.5rem;">💡 Quick Tips</h4>
        <p style="color: #5a6c7d; font-size: 0.9rem; line-height: 1.6;">
        • Use <strong>Data Analysis</strong> to explore patterns<br>
        • <strong>Fraud Detection</strong> for real-time checks<br>
        • <strong>Model Performance</strong> for metrics
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data and model
    data = load_data()
    model = load_model()
    
    if data is None or model is None:
        st.stop()
    
    if page == "🏠 Home":
        show_home_page(data)
    elif page == "📊 Data Analysis":
        show_data_analysis(data)
    elif page == "🔍 Fraud Detection":
        show_fraud_detection(model, data)
    elif page == "📈 Model Performance":
        show_model_performance(model, data)

def show_home_page(data):
    """Display home page with overview"""
    # Welcome Section
    st.markdown('<div class="section-header">Welcome to Credit Card Fraud Detection System</div>', unsafe_allow_html=True)
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_transactions = len(data)
    fraud_cases = len(data[data['Class'] == 1])
    valid_cases = len(data[data['Class'] == 0])
    fraud_percentage = (fraud_cases / total_transactions) * 100
    
    with col1:
        st.metric("Total Transactions", f"{total_transactions:,}", help="Total number of transactions in the dataset")
    with col2:
        st.metric("Fraud Cases", f"{fraud_cases:,}", delta=f"{fraud_percentage:.2f}%", delta_color="inverse", help="Number of fraudulent transactions detected")
    with col3:
        st.metric("Valid Transactions", f"{valid_cases:,}", help="Number of legitimate transactions")
    with col4:
        st.metric("Fraud Rate", f"{fraud_percentage:.3f}%", help="Percentage of fraudulent transactions")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # About Section
    st.markdown('<div class="section-header">📋 About This System</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <p style="font-size: 1.1rem; line-height: 1.8; color: #2c3e50;">
        This <strong>Credit Card Fraud Detection System</strong> uses advanced machine learning techniques 
        with a <strong>Random Forest Classifier</strong> to identify potentially fraudulent transactions in real-time. 
        The system provides comprehensive analysis tools and intuitive visualizations to help detect and prevent fraud.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-list">
            <h4 style="color: #667eea; margin-bottom: 1rem;">✨ Key Features</h4>
            <ul style="list-style: none; padding: 0;">
                <li>🔍 <strong>Real-time Fraud Detection</strong> - Analyze individual transactions instantly</li>
                <li>📊 <strong>Data Analysis</strong> - Explore transaction patterns and statistics</li>
                <li>📈 <strong>Model Performance Metrics</strong> - View detailed evaluation metrics</li>
                <li>📉 <strong>Interactive Visualizations</strong> - Charts, graphs, and heatmaps</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-list">
            <h4 style="color: #667eea; margin-bottom: 1rem;">🛡️ Security Features</h4>
            <ul style="list-style: none; padding: 0;">
                <li>⚡ <strong>Fast Processing</strong> - Quick analysis of transaction data</li>
                <li>🎯 <strong>High Accuracy</strong> - Advanced ML algorithms for precise detection</li>
                <li>📱 <strong>User-Friendly Interface</strong> - Easy to navigate and use</li>
                <li>💾 <strong>Batch Processing</strong> - Analyze multiple transactions at once</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Quick Start Guide
    st.markdown("""
    <div class="quick-start">
        <h3>🚀 Quick Start Guide</h3>
        <ol style="padding-left: 1.5rem;">
            <li><strong>Navigate to 📊 Data Analysis</strong> to explore the dataset, view statistics, and analyze transaction patterns</li>
            <li><strong>Go to 🔍 Fraud Detection</strong> to check if a transaction is fraudulent using manual input or CSV file upload</li>
            <li><strong>Visit 📈 Model Performance</strong> to see how well the model performs with detailed metrics and confusion matrix</li>
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
            <p>Explore the dataset with interactive visualizations, statistics, and correlation analysis. 
            Understand transaction patterns and data distribution.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with nav_col2:
        st.markdown("""
        <div class="nav-card">
            <h3>🔍 Fraud Detection</h3>
            <p>Check if transactions are fraudulent using manual input or batch processing. 
            Get instant predictions with probability scores.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with nav_col3:
        st.markdown("""
        <div class="nav-card">
            <h3>📈 Model Performance</h3>
            <p>View comprehensive model evaluation metrics including accuracy, precision, recall, 
            F1-score, and confusion matrix visualization.</p>
        </div>
        """, unsafe_allow_html=True)

def show_data_analysis(data):
    """Display data analysis page"""
    st.header("📊 Data Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Statistics", "Visualizations", "Raw Data"])
    
    with tab1:
        st.subheader("Dataset Overview")
        st.dataframe(data.head(10), use_container_width=True)
        st.write(f"**Dataset Shape:** {data.shape[0]} rows × {data.shape[1]} columns")
        
        st.subheader("Class Distribution")
        fraud = data[data['Class'] == 1]
        valid = data[data['Class'] == 0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Fraudulent Transactions", len(fraud))
            st.write(fraud.Amount.describe())
        with col2:
            st.metric("Valid Transactions", len(valid))
            st.write(valid.Amount.describe())
    
    with tab2:
        st.subheader("Statistical Summary")
        st.dataframe(data.describe(), use_container_width=True)
        
        st.subheader("Missing Values")
        missing = data.isnull().sum()
        if missing.sum() == 0:
            st.success("✅ No missing values in the dataset!")
        else:
            st.dataframe(missing[missing > 0])
    
    with tab3:
        st.subheader("Visualizations")
        
        # Class distribution pie chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        class_counts = data['Class'].value_counts()
        ax1.pie(class_counts, labels=['Valid', 'Fraud'], autopct='%1.2f%%', 
                colors=['#4caf50', '#f44336'], startangle=90)
        ax1.set_title('Transaction Class Distribution')
        
        # Amount distribution
        data[data['Class'] == 0]['Amount'].hist(bins=50, ax=ax2, alpha=0.7, label='Valid', color='green')
        data[data['Class'] == 1]['Amount'].hist(bins=50, ax=ax2, alpha=0.7, label='Fraud', color='red')
        ax2.set_xlabel('Amount')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Transaction Amount Distribution')
        ax2.legend()
        ax2.set_yscale('log')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Correlation heatmap
        if st.checkbox("Show Correlation Heatmap"):
            st.subheader("Feature Correlation Matrix")
            corrmat = data.corr()
            fig, ax = plt.subplots(figsize=(12, 9))
            sns.heatmap(corrmat, vmax=0.8, square=True, ax=ax, cmap='coolwarm')
            st.pyplot(fig)
    
    with tab4:
        st.subheader("Raw Data")
        rows_to_show = st.slider("Number of rows to display", 10, 1000, 100)
        st.dataframe(data.head(rows_to_show), use_container_width=True)

def show_fraud_detection(model, data):
    """Display fraud detection interface"""
    st.header("🔍 Fraud Detection")
    
    st.subheader("Check a Transaction")
    
    detection_method = st.radio(
        "Choose detection method:",
        ["📝 Manual Input", "📁 Upload CSV File"]
    )
    
    if detection_method == "📝 Manual Input":
        st.write("Enter transaction details below:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            time = st.number_input("Time", min_value=0.0, value=0.0, step=0.1)
            amount = st.number_input("Amount", min_value=0.0, value=0.0, step=0.01)
        
        with col2:
            st.write("**PCA Features (V1-V28)**")
            st.caption("Enter values for principal component features")
        
        # Create input fields for V1-V28
        v_features = {}
        cols = st.columns(4)
        for i in range(28):
            col_idx = i % 4
            v_features[f'V{i+1}'] = cols[col_idx].number_input(
                f"V{i+1}", 
                value=0.0, 
                step=0.001,
                key=f"v{i+1}"
            )
        
        if st.button("🔍 Detect Fraud", type="primary"):
            # Prepare input data
            feature_names = ['Time'] + [f'V{i+1}' for i in range(28)] + ['Amount']
            input_values = [time] + [v_features[f'V{i+1}'] for i in range(28)] + [amount]
            
            transaction_df = pd.DataFrame([input_values], columns=feature_names)
            
            # Make prediction
            prediction, probability = predict_transaction(model, transaction_df)
            
            # Display result
            st.markdown("---")
            if prediction == 1:
                st.markdown("""
                <div class="fraud-alert">
                    <h2>⚠️ FRAUD DETECTED!</h2>
                    <p>This transaction has been flagged as potentially fraudulent.</p>
                    <p><strong>Fraud Probability:</strong> {:.2%}</p>
                </div>
                """.format(probability[1]), unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="valid-alert">
                    <h2>✅ VALID TRANSACTION</h2>
                    <p>This transaction appears to be legitimate.</p>
                    <p><strong>Valid Probability:</strong> {:.2%}</p>
                </div>
                """.format(probability[0]), unsafe_allow_html=True)
            
            # Show probabilities
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Valid Probability", f"{probability[0]:.2%}")
            with col2:
                st.metric("Fraud Probability", f"{probability[1]:.2%}")
    
    else:  # Upload CSV File
        st.write("Upload a CSV file with transaction data:")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ File uploaded successfully! ({len(df)} rows)")
                
                # Check if required columns exist
                required_cols = ['Time'] + [f'V{i+1}' for i in range(28)] + ['Amount']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                else:
                    if st.button("🔍 Analyze Transactions", type="primary"):
                        # Make predictions
                        X = df[required_cols]
                        predictions = model.predict(X)
                        probabilities = model.predict_proba(X)
                        
                        # Add predictions to dataframe
                        df['Prediction'] = predictions
                        df['Fraud_Probability'] = probabilities[:, 1]
                        df['Valid_Probability'] = probabilities[:, 0]
                        
                        # Display results
                        st.subheader("Detection Results")
                        
                        fraud_count = sum(predictions == 1)
                        valid_count = sum(predictions == 0)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Fraudulent Transactions", fraud_count)
                        with col2:
                            st.metric("Valid Transactions", valid_count)
                        
                        st.dataframe(df, use_container_width=True)
                        
                        # Download results
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name="fraud_detection_results.csv",
                            mime="text/csv"
                        )
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")

def show_model_performance(model, data):
    """Display model performance metrics"""
    st.header("📈 Model Performance")
    
    # Prepare data
    X = data.drop(['Class'], axis=1)
    Y = data['Class']
    xData = X.values
    yData = Y.values
    
    from sklearn.model_selection import train_test_split
    xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size=0.2, random_state=42)
    
    # Make predictions
    yPred = model.predict(xTest)
    
    # Calculate metrics
    accuracy = accuracy_score(yTest, yPred)
    precision = precision_score(yTest, yPred)
    recall = recall_score(yTest, yPred)
    f1 = f1_score(yTest, yPred)
    mcc = matthews_corrcoef(yTest, yPred)
    
    # Display metrics
    st.subheader("Evaluation Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Accuracy", f"{accuracy:.4f}")
    with col2:
        st.metric("Precision", f"{precision:.4f}")
    with col3:
        st.metric("Recall", f"{recall:.4f}")
    with col4:
        st.metric("F1-Score", f"{f1:.4f}")
    with col5:
        st.metric("MCC", f"{mcc:.4f}")
    
    st.markdown("---")
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    conf_matrix = confusion_matrix(yTest, yPred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Fraud'], 
                yticklabels=['Normal', 'Fraud'],
                ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('True Class')
    st.pyplot(fig)
    
    # Model Info
    st.subheader("Model Information")
    st.write(f"**Algorithm:** Random Forest Classifier")
    st.write(f"**Training Set Size:** {len(xTrain):,} samples")
    st.write(f"**Test Set Size:** {len(xTest):,} samples")
    st.write(f"**Features:** {X.shape[1]} features")

if __name__ == "__main__":
    main()

