from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from io import BytesIO
import base64

# Load the trained model and scaler
with open(f'models/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open(f'models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

app = Flask(__name__,
            static_folder='static',
            template_folder='templates')

@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get the input values from the form
        gender = request.form['gender']
        age = int(request.form['age'])
        marital_status = request.form['marital_status']
        percent_salary_hike = float(request.form['percent_salary_hike'])
        num_companies_worked = int(request.form['number_of_companies_worked'])
        total_working_years = int(request.form['total_working_years'])
        education = request.form['education']
        education_field = request.form['education_field']
        daily_rate = int(request.form['daily_rate'])
        hourly_rate = int(request.form['hourly_rate'])
        monthly_rate = int(request.form['monthly_rate'])
        monthly_income = int(request.form['monthly_income'])
        environment_satisfaction = int(request.form['environment_satisfaction'])
        job_involvement = int(request.form['job_involvement'])
        job_level = int(request.form['job_level'])
        job_satisfaction = int(request.form['job_satisfaction'])
        performance_rating = int(request.form['performance_rating'])
        relationship_satisfaction = int(request.form['relationship_satisfaction'])
        work_life_balance = int(request.form['work_life_balance'])
        stock_option_level = int(request.form['stock_option_level'])
        training_times_last_year = int(request.form['training_times_last_year'])
        years_at_company = int(request.form['years_at_company'])
        years_in_current_role = int(request.form['years_in_current_role'])
        years_since_last_promotion = int(request.form['years_since_last_promotion'])
        years_with_current_manager = int(request.form['years_with_current_manager'])
        business_travel = request.form['business_travel']
        department = request.form['department']
        job_role = request.form['job_role']
        overtime = request.form['overtime']
        distance_from_home = int(request.form['distance_from_home'])

        # Create a DataFrame from the input values
        data = pd.DataFrame({
            'Gender': [gender],
            'Age': [age],
            'MaritalStatus': [marital_status],
            'PercentSalaryHike': [percent_salary_hike],
            'NumCompaniesWorked': [num_companies_worked],
            'TotalWorkingYears': [total_working_years],
            'Education': [education],
            'EducationField': [education_field],
            'DailyRate': [daily_rate],
            'HourlyRate': [hourly_rate],
            'MonthlyRate': [monthly_rate],
            'MonthlyIncome': [monthly_income],
            'EnvironmentSatisfaction': [environment_satisfaction],
            'JobInvolvement': [job_involvement],
            'JobLevel': [job_level],
            'JobSatisfaction': [job_satisfaction],
            'PerformanceRating': [performance_rating],
            'RelationshipSatisfaction': [relationship_satisfaction],
            'WorkLifeBalance': [work_life_balance],
            'StockOptionLevel': [stock_option_level],
            'TrainingTimesLastYear': [training_times_last_year],
            'YearsAtCompany': [years_at_company],
            'YearsInCurrentRole': [years_in_current_role],
            'YearsSinceLastPromotion': [years_since_last_promotion],
            'YearsWithCurrManager': [years_with_current_manager],
            'BusinessTravel': [business_travel],
            'Department': [department],
            'JobRole': [job_role],
            'OverTime': [overtime],
            'DistanceFromHome': [distance_from_home]
        })


        # Encode categorical variables

        label_encoders = {}
        categorical_cols = categorical_cols = ['Gender', 'MaritalStatus', 'Education', 'EducationField', 'BusinessTravel', 'Department', 'JobRole', 'OverTime']
        for col in categorical_cols:
            data[col] = label_encoders[col].transform(data[col])



        # Scale the input data using the loaded scaler
        # scaled_data = scaler.transform(data.values)
        scaled_data = scaler.transform(data)

        # Make predictions using the loaded model
        predictions = model.predict(scaled_data)

        # Process the predictions and display the result
        if predictions[0] == 1:
            result = "Attrition: Yes"
        else:
            result = "Attrition: No"

        return render_template("main.html", result=result)
    else:
        return render_template("main.html")

if __name__ == "__main__":
    app.run(debug=True)
