
from flask import Flask,render_template,request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


application = Flask(__name__)
app = application

model = pickle.load(open(r'C:\Users\UNIQUE\Desktop\Jupyter\Project4Reg_dynamic_pricing\model\rf_model.pkl','rb'))

## Route for Homepage

@app.route("/")
def index():
    return render_template('index.html')

## Route for  singel data point prediction
@app.route('/predict',methods = ['GET','POST'])
def predict_datapoint():
    result = ""

    if request.method == 'POST':
        Vehicle_Type_encode = int(request.form.get('Vehicle_Type'))
        Expected_Ride_Duration = float(request.form.get('Expected_Ride_Duration'))
        Number_of_Riders = float(request.form.get('Number_of_Riders'))
        Number_of_Drivers = float(request.form.get('Number_of_Drivers'))
        Number_of_Past_Rides = float(request.form.get('Number_of_Past_Rides'))
        Average_Ratings = float(request.form.get('Average_Ratings'))

        # Retrieve encoded values
        Location_Category_encode = int(request.form.get('Location_Category'))  # Ordinal encoded
        Customer_Loyalty_Status_encode = int(request.form.get('Customer_Loyalty_Status'))  # Ordinal encoded

        # One hot encoded values for Time_of_Booking
        Time_of_Booking = request.form.get('Time_of_Booking')
        Time_of_Booking_Morning = 1 if Time_of_Booking == 'morning' else 0
        Time_of_Booking_Afternoon = 1 if Time_of_Booking == 'afternoon' else 0
        Time_of_Booking_Evening = 1 if Time_of_Booking == 'evening' else 0
        Time_of_Booking_Night = 1 if Time_of_Booking == 'night' else 0

        new_data = [[Number_of_Riders, Number_of_Drivers, Number_of_Past_Rides,
                Average_Ratings, Expected_Ride_Duration, Location_Category_encode,
                Customer_Loyalty_Status_encode, Time_of_Booking_Afternoon,
                Time_of_Booking_Evening, Time_of_Booking_Morning,
                Time_of_Booking_Night, Vehicle_Type_encode]]

        predict = model.predict(new_data)

        return render_template('single_prediction.html', result=predict)

    else:
            return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
