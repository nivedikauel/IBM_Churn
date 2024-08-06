from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('best_xgb_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Convert input values to appropriate types
            input_features = {
                'SeniorCitizen': int(request.form.get('SeniorCitizen', 0)),
                'Partner': int(request.form.get('Partner', 0)),
                'Dependents': int(request.form.get('Dependents', 0)),
                'tenure': int(request.form.get('tenure', 0)),
                'PaperlessBilling': int(request.form.get('PaperlessBilling', 0)),
                'MonthlyCharges': float(request.form.get('MonthlyCharges', 0.0)),
                'TotalCharges': float(request.form.get('TotalCharges', 0.0)),
                'MultipleLines_No phone service': 0,
                'MultipleLines_Yes': 0,
                'InternetService_Fiber optic': 0,
                'InternetService_No': 0,
                'OnlineSecurity_No internet service': 0,
                'OnlineSecurity_Yes': 0,
                'OnlineBackup_No internet service': 0,
                'OnlineBackup_Yes': 0,
                'DeviceProtection_No internet service': 0,
                'DeviceProtection_Yes': 0,
                'TechSupport_No internet service': 0,
                'TechSupport_Yes': 0,
                'StreamingTV_No internet service': 0,
                'StreamingTV_Yes': 0,
                'StreamingMovies_No internet service': 0,
                'StreamingMovies_Yes': 0,
                'Contract_One year': 0,
                'Contract_Two year': 0,
                'PaymentMethod_Credit card (automatic)': 0,
                'PaymentMethod_Electronic check': 0,
                'PaymentMethod_Mailed check': 0,
                'Plan_add-ons plan': 0,
                'Plan_discount plan': 0
            }

            # Update categorical variables
            multiple_lines = request.form.get('MultipleLines', '')
            if multiple_lines == 'No phone service':
                input_features['MultipleLines_No phone service'] = 1
            elif multiple_lines == 'Yes':
                input_features['MultipleLines_Yes'] = 1

            internet_service = request.form.get('InternetService', '')
            if internet_service == 'Fiber optic':
                input_features['InternetService_Fiber optic'] = 1
            elif internet_service == 'No':
                input_features['InternetService_No'] = 1

            online_security = request.form.get('OnlineSecurity', '')
            if online_security == 'No internet service':
                input_features['OnlineSecurity_No internet service'] = 1
            elif online_security == 'Yes':
                input_features['OnlineSecurity_Yes'] = 1

            online_backup = request.form.get('OnlineBackup', '')
            if online_backup == 'No internet service':
                input_features['OnlineBackup_No internet service'] = 1
            elif online_backup == 'Yes':
                input_features['OnlineBackup_Yes'] = 1

            device_protection = request.form.get('DeviceProtection', '')
            if device_protection == 'No internet service':
                input_features['DeviceProtection_No internet service'] = 1
            elif device_protection == 'Yes':
                input_features['DeviceProtection_Yes'] = 1

            tech_support = request.form.get('TechSupport', '')
            if tech_support == 'No internet service':
                input_features['TechSupport_No internet service'] = 1
            elif tech_support == 'Yes':
                input_features['TechSupport_Yes'] = 1

            streaming_tv = request.form.get('StreamingTV', '')
            if streaming_tv == 'No internet service':
                input_features['StreamingTV_No internet service'] = 1
            elif streaming_tv == 'Yes':
                input_features['StreamingTV_Yes'] = 1

            streaming_movies = request.form.get('StreamingMovies', '')
            if streaming_movies == 'No internet service':
                input_features['StreamingMovies_No internet service'] = 1
            elif streaming_movies == 'Yes':
                input_features['StreamingMovies_Yes'] = 1

            contract = request.form.get('Contract', '')
            if contract == 'One year':
                input_features['Contract_One year'] = 1
            elif contract == 'Two year':
                input_features['Contract_Two year'] = 1

            payment_method = request.form.get('PaymentMethod', '')
            if payment_method == 'Credit card (automatic)':
                input_features['PaymentMethod_Credit card (automatic)'] = 1
            elif payment_method == 'Electronic check':
                input_features['PaymentMethod_Electronic check'] = 1
            elif payment_method == 'Mailed check':
                input_features['PaymentMethod_Mailed check'] = 1

            plan = request.form.get('Plan', '')
            if plan == 'add-ons':
                input_features['Plan_add-ons plan'] = 1
            elif plan == 'discount plan':
                input_features['Plan_discount plan'] = 1

            # Create a dataframe with the input data
            input_data = pd.DataFrame([input_features])

            # Scale the input data
            scaled_input_data = scaler.transform(input_data)

            # Make prediction
            prediction = model.predict(scaled_input_data)[0]
            probability = model.predict_proba(scaled_input_data)[0][1]

            return render_template('result.html', prediction=prediction, probability=probability)
        
        except Exception as e:
            return str(e)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
