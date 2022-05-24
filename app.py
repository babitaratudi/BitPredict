from flask import Flask, render_template, request, redirect, url_for, session
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re
import os
import numpy as np
import pandas as pd
from keras.models import load_model
import plotly
import plotly.graph_objs as go
from datetime import date
from sklearn.preprocessing import MinMaxScaler
import json
from markupsafe import escape
from datetime import date,datetime


app = Flask(__name__)
  
app.secret_key = os.urandom(32)
  
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'root'
app.config['MYSQL_DB'] = 'geeklogin'
  
mysql = MySQL(app)
  

def create_lookback(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

@app.route('/')
@app.route('/login', methods =['GET', 'POST'])
def login():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = % s AND password = % s', (username, password, ))
        account = cursor.fetchone()
        if account:
            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']
            msg = 'Logged in successfully !'
            data=pd.read_csv("./bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv")
            data['date'] = pd.to_datetime(data['Timestamp'],unit='s').dt.date
            group = data.groupby('date')
            Daily_Price = group['Weighted_Price'].mean()

            d0 = date(2014, 1, 1)
            d1 = date(2021, 1, 1)
            delta = d1 - d0
            days_look = delta.days + 1
            #print(days_look)

            d2 = date(2020, 8, 21)
            d3 = date(2021, 3, 31)
            delta = d3 - d2
            days_from_train = delta.days + 1
            #print(days_from_train)

            d4 = date(2021, 1, 1)
            d5 = date(2021, 3, 31)
            delta = d5 - d4
            days_from_end = delta.days + 1
            #print(days_from_end)

            df_train= Daily_Price[len(Daily_Price)-days_look-days_from_end:len(Daily_Price)-days_from_train]
            df_test= Daily_Price[len(Daily_Price)-days_from_train:]
            #print(len(df_train), len(df_test))

            working_data = [df_train, df_test]
            working_data = pd.concat(working_data)

            working_data = working_data.reset_index()
            working_data['date'] = pd.to_datetime(working_data['date'])
            working_data = working_data.set_index('date')

            df_train = working_data[:-60]
            df_test = working_data[-60:]

            training_set = df_train.values
            training_set = np.reshape(training_set, (len(training_set), 1))
            test_set = df_test.values
            test_set = np.reshape(test_set, (len(test_set), 1))

            #scale datasets
            scaler = MinMaxScaler()
            training_set = scaler.fit_transform(training_set)
            test_set = scaler.transform(test_set)

            # create datasets which are suitable for time series forecasting
            look_back = 1
            X_train, Y_train = create_lookback(training_set, look_back)
            X_test, Y_test = create_lookback(test_set, look_back)

            # reshape datasets so that they will be ok for the requirements of the LSTM model in Keras
            X_train = np.reshape(X_train, (len(X_train), 1, X_train.shape[1]))
            X_test = np.reshape(X_test, (len(X_test), 1, X_test.shape[1]))

            model = load_model('model_bitcoin.h5')

            # add one additional data point to align shapes of the predictions and true labels
            X_test = np.reshape(X_test, (len(X_test), -1, 1))

            # get predictions and then make some transformations to be able to calculate RMSE properly in USD
            prediction = model.predict(X_test)
            prediction_inverse = scaler.inverse_transform(prediction.reshape(-1, 1))
            Y_test_inverse = scaler.inverse_transform(Y_test.reshape(-1, 1))
            prediction2_inverse = np.array(prediction_inverse[:,0][1:])
            Y_test2_inverse = np.array(Y_test_inverse[:,0])

            trace1 = go.Scatter(
                x = np.arange(0, len(prediction2_inverse), 1),
                y = prediction2_inverse,
                mode = 'lines',
                name = 'Predicted labels',
                line = dict(color=('rgb(244, 146, 65)'), width=2)
            )
            trace2 = go.Scatter(
                x = np.arange(0, len(Y_test2_inverse), 1),
                y = Y_test2_inverse,
                mode = 'lines',
                name = 'True labels',
                line = dict(color=('rgb(66, 244, 155)'), width=2)
            )

            data = [trace1, trace2]
            layout = dict(title = 'Comparison of true prices (on the test dataset) with prices our model predicted',
                    xaxis = dict(title = 'Day number'), yaxis = dict(title = 'Price, USD'))
            fig = dict(data=data, layout=layout)

            graph1JSON = json.dumps(fig ,cls =plotly.utils.PlotlyJSONEncoder)

            Test_Dates = Daily_Price[len(Daily_Price)-days_from_end:].index

            trace3 = go.Scatter(x=Test_Dates, y=Y_test2_inverse, name= 'Actual Price',line = dict(color = ('rgb(66, 244, 155)'),width = 2))
            trace4 = go.Scatter(x=Test_Dates, y=prediction2_inverse, name= 'Predicted Price',line = dict(color = ('rgb(244, 146, 65)'),width = 2))
            data1 = [trace3, trace4]
            layout1 = dict(title = 'Comparison of true prices (on the test dataset) with prices our model predicted, by dates',
                    xaxis = dict(title = 'Date'), yaxis = dict(title = 'Price, USD'))
            fig1 = dict(data=data1, layout=layout1)

            graph2JSON = json.dumps(fig1 ,cls =plotly.utils.PlotlyJSONEncoder)

            Test_Dates = [datetime(2021, 4, 30),datetime(2021, 5, 30),datetime(2021, 6, 30),datetime(2021, 7, 30),datetime(2021, 8, 30),datetime(2021, 9, 30), datetime(2021, 10, 31), datetime(2021, 11, 30), datetime(2021, 12, 31), 
                    datetime(2022, 1,31 ), datetime(2022, 2, 28), datetime(2022, 3, 31),datetime(2022, 4, 19)]

            trace5 = go.Scatter(x=Test_Dates, y=Y_test2_inverse, name= 'Actual Price',
                        line = dict(color = ('rgb(66, 244, 155)'),width = 2))
            trace6 = go.Scatter(x=Test_Dates, y=prediction2_inverse, name= 'Predicted Price',
                        line = dict(color = ('rgb(244, 146, 65)'),width = 2))
            data2 = [trace5, trace6]
            layout2 = dict(title = 'Comparison of true prices (on the test dataset) with prices our model predicted, by dates',
                    xaxis = dict(title = 'Date'), yaxis = dict(title = 'Price, USD'))
            fig2 = dict(data=data2, layout=layout2)

            graph3JSON = json.dumps(fig2 ,cls =plotly.utils.PlotlyJSONEncoder)

            return render_template('index.html',msg=msg, graph1JSON=graph1JSON,graph2JSON=graph2JSON,graph3JSON=graph3JSON)
            # index()
            # return render_template('index.html', msg = msg)
        else:
            msg = 'Incorrect username / password !'
    return render_template('login.html', msg = msg)
  
@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('username', None)
    return redirect(url_for('login'))
  
@app.route('/register', methods =['GET', 'POST'])
def register():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form :
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = % s', (username, ))
        account = cursor.fetchone()
        if account:
            msg = 'Account already exists !'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address !'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only characters and numbers !'
        elif not username or not password or not email:
            msg = 'Please fill out the form !'
        else:
            cursor.execute('INSERT INTO accounts VALUES (NULL, % s, % s, % s)', (username, password, email, ))
            mysql.connection.commit()
            msg = 'You have successfully registered !'
    elif request.method == 'POST':
        msg = 'Please fill out the form !'
    return render_template('register.html', msg = msg)

if __name__=="__main__":
    app.run(debug=True)