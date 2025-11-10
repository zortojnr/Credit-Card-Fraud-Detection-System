import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("creditcard.csv")
print(data.head())


print(data.describe())


fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]
outlier_fraction = len(fraud)/float(len(valid))
print(outlier_fraction)
print('Fraud Cases: {}'.format(len(data[data['Class'] == 1])))
print(f'Valid Transactions: {len(data[data["Class"] == 0])}')

print('Amount details of the fraudulent transactions')
fraud.Amount.describe()

print('Details of valid transactions')
valid.Amount.describe()


corrmat = data.corr()
fig = plt.figure(figsize = (12, 9))
sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()


X = data.drop(['Class'], axis = 1)
Y = data['Class']
print(X.shape)
print(Y.shape)

xData = X.values
yData = Y.values

from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size = 0.2, random_state = 42)


from sklearn.ensemble import RandomForestClassifier
import joblib

rfc = RandomForestClassifier()
rfc.fit(xTrain, yTrain)

# Save the trained model
joblib.dump(rfc, 'fraud_detection_model.pkl')
print('Model saved successfully!')

yPred = rfc.predict(xTest)


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
accuracy = accuracy_score(yTest, yPred)
precision = precision_score(yTest, yPred)
recall = recall_score(yTest, yPred)
f1 = f1_score(yTest, yPred)
mcc = matthews_corrcoef(yTest, yPred)

print('Model Evaluation Metrics:')
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1:.4f}')
print(f'Matthews Correlation Coefficient: {mcc:.4f}')

conf_matrix = confusion_matrix(yTest, yPred)
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.show()