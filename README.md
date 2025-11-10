# Credit Card Fraud Detection

A machine learning project that detects fraudulent credit card transactions using a Random Forest Classifier. This project analyzes transaction data, performs exploratory data analysis, and builds a predictive model to identify fraudulent activities.

## Features

- **🌐 Interactive Web Interface**: User-friendly Streamlit web app for easy interaction
- **Data Analysis**: Exploratory data analysis with statistical summaries and visualizations
- **Correlation Analysis**: Heatmap visualization of feature correlations
- **Machine Learning Model**: Random Forest Classifier for fraud detection
- **Real-time Fraud Detection**: Check individual transactions or batch process CSV files
- **Model Evaluation**: Comprehensive metrics including accuracy, precision, recall, F1-score, and Matthews Correlation Coefficient
- **Visualization**: Confusion matrix heatmap for model performance visualization

## Requirements

- Python 3.7 or higher
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone or download this repository

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

The project uses a `creditcard.csv` file containing credit card transaction data. The dataset should include:
- Transaction features (V1-V28, Time, Amount)
- Target variable: `Class` (0 = Normal transaction, 1 = Fraudulent transaction)

**Note**: 
- The dataset file is not included in this repository due to size limitations
- You need to provide your own `creditcard.csv` file in the project directory
- Popular datasets can be found on Kaggle (e.g., "Credit Card Fraud Detection" dataset)
- Make sure the `creditcard.csv` file is in the same directory as `main.py` before running the scripts

## Usage

### Step 1: Train the Model

First, train and save the model by running:
```bash
python main.py
```

This will:
- Load and analyze the dataset
- Train the Random Forest Classifier
- Save the model as `fraud_detection_model.pkl`
- Display evaluation metrics

### Step 2: Launch the Web Interface

After training the model, launch the interactive web interface:
```bash
streamlit run app.py
```

The interface will open in your default web browser at `http://localhost:8501`

### Using the Web Interface

The interface includes four main pages:

1. **🏠 Home**: Overview dashboard with key statistics
2. **📊 Data Analysis**: 
   - Dataset overview and statistics
   - Interactive visualizations
   - Correlation heatmaps
   - Raw data exploration
3. **🔍 Fraud Detection**:
   - **Manual Input**: Enter transaction details manually to check for fraud
   - **CSV Upload**: Upload a CSV file with multiple transactions for batch processing
4. **📈 Model Performance**: View detailed model evaluation metrics and confusion matrix

## What the Script Does

1. **Data Loading**: Loads the credit card transaction dataset
2. **Data Exploration**: 
   - Displays first few rows and statistical summary
   - Analyzes fraud vs valid transaction distribution
   - Calculates outlier fraction
   - Shows amount statistics for fraudulent and valid transactions

3. **Data Visualization**: 
   - Creates a correlation heatmap of all features

4. **Model Training**:
   - Splits data into training (80%) and testing (20%) sets
   - Trains a Random Forest Classifier

5. **Model Evaluation**:
   - Predicts on test set
   - Calculates and displays evaluation metrics:
     - Accuracy
     - Precision
     - Recall
     - F1-Score
     - Matthews Correlation Coefficient
   - Displays confusion matrix visualization

## Project Structure

```
Credit card fraud detection/
│
├── main.py                    # Main script with data analysis and model training
├── app.py                     # Streamlit web interface application
├── creditcard.csv             # Dataset file (required)
├── fraud_detection_model.pkl  # Trained model (generated after running main.py)
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## Dependencies

- **numpy**: Numerical computing
- **pandas**: Data manipulation and analysis
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization
- **scikit-learn**: Machine learning library
- **streamlit**: Web interface framework
- **joblib**: Model serialization

## Model Details

- **Algorithm**: Random Forest Classifier
- **Train-Test Split**: 80% training, 20% testing
- **Random State**: 42 (for reproducibility)

## Output

The script will display:
- Dataset head and summary statistics
- Fraud case statistics
- Outlier fraction
- Transaction amount details
- Feature correlation heatmap
- Model evaluation metrics
- Confusion matrix visualization

## Notes

- The dataset is highly imbalanced (fraudulent transactions are rare)
- The Random Forest Classifier handles imbalanced data relatively well
- For production use, consider additional techniques like:
  - SMOTE for handling class imbalance
  - Feature engineering
  - Hyperparameter tuning
  - Cross-validation

## License

This project is open source and available for educational purposes.

