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

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .fraud-alert {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .valid-alert {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
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
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a page", ["🏠 Home", "📊 Data Analysis", "🔍 Fraud Detection", "📈 Model Performance"])
    
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
    st.header("Welcome to Credit Card Fraud Detection System")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_transactions = len(data)
    fraud_cases = len(data[data['Class'] == 1])
    valid_cases = len(data[data['Class'] == 0])
    fraud_percentage = (fraud_cases / total_transactions) * 100
    
    with col1:
        st.metric("Total Transactions", f"{total_transactions:,}")
    with col2:
        st.metric("Fraud Cases", f"{fraud_cases:,}", delta=f"{fraud_percentage:.2f}%")
    with col3:
        st.metric("Valid Transactions", f"{valid_cases:,}")
    with col4:
        st.metric("Fraud Rate", f"{fraud_percentage:.3f}%")
    
    st.markdown("---")
    
    st.subheader("📋 About This System")
    st.write("""
    This Credit Card Fraud Detection System uses a **Random Forest Classifier** to identify 
    potentially fraudulent transactions. The system provides:
    
    - **Real-time Fraud Detection**: Analyze individual transactions instantly
    - **Data Analysis**: Explore transaction patterns and statistics
    - **Model Performance Metrics**: View detailed evaluation metrics
    - **Visualizations**: Interactive charts and graphs
    """)
    
    st.subheader("🚀 Quick Start")
    st.write("""
    1. Navigate to **📊 Data Analysis** to explore the dataset
    2. Go to **🔍 Fraud Detection** to check if a transaction is fraudulent
    3. Visit **📈 Model Performance** to see how well the model performs
    """)

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

