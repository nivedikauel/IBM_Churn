# Apply SMOTE to balance the classes
from imblearn.over_sampling import SMOTE
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import joblib


data = pd.read_csv('Telco-Customer-Churn.csv')
# Handle non-numeric or missing values by replacing them with NaN
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

data['TotalCharges'].fillna(data['MonthlyCharges'] * data['tenure'], inplace=True)
data['TC'] = data['tenure'] * data['MonthlyCharges']
data['Difference'] = data['TotalCharges'] - data['TC']
#Lets create new categorical variable
data['Plan'] = np.where(
    data['Difference'] < 0, 'discount plan',
    np.where(
        data['Difference'] > 0, 'add-ons plan',
        'Normal Plan'  # This handles the case where Difference == 0
    )
)
data = data.drop(['customerID','gender','PhoneService','TC','Difference'], axis=1)
for column in data.columns:
    if data[column].isin(['Yes', 'No']).all():
      data[column] = data[column].replace({'No': 0, 'Yes': 1})
# Columns to one-hot encode
one_hot_encode_columns = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                          'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                          'Contract', 'PaymentMethod','Plan']
data = pd.get_dummies(data, columns = one_hot_encode_columns,dtype=int,drop_first=True)
scaler = RobustScaler()
X = data.drop(columns = ['Churn'])
y = data['Churn'].values
x= scaler.fit_transform(X)
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(x, y)
X_smote.shape, y_smote.shape
# Split the data
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_smote,y_smote,test_size = 0.30, random_state = 42, stratify=y_smote)

# Train the model
model = XGBClassifier()
model.fit(X_train1, y_train1)

# Save the scaler and the model
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(model, 'best_xgb_model.pkl')