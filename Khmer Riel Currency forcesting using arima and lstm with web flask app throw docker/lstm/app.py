from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from keras.models import load_model
import matplotlib.pyplot as plt
import io
import base64
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from datetime import timedelta

class StockPricePredictionApp:
    def __init__(self):
        self.app = Flask(__name__)
        self.app.route('/')(self.index)
        self.app.route('/predict', methods=['POST'])(self.predict)

        # Load models
        self.arima_model = ARIMAModel()
        self.ets_model = ETSModel()

    def run(self):
        self.app.run(debug=True, port=8181)

    def index(self):
        return render_template('arima_ets.html')

    def predict(self):
        start_date = pd.to_datetime(request.form['start_date'])
        end_date = pd.to_datetime(request.form['end_date'])
        model_name = request.form['model']

        df = pd.read_csv('data.csv')  # Load and preprocess dataset
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        if model_name == 'ARIMA':
            predictions = self.arima_model.predict(df['Price'], start_date, end_date)
        elif model_name == 'ETS':
            predictions = self.ets_model.predict(df['Price'], start_date, end_date)
        else:
            return render_template('error.html', message='Invalid model selected.')

        result_df = pd.DataFrame({'Date': predictions.index, 'Predicted Price': predictions})

        # Save the DataFrame to a HTML table
        result_table = result_df.to_html(index=False)

        plot_data = df.iloc[-100:]  # Get latest 100 days of actual data for plotting

        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.plot(plot_data.index, plot_data['Price'], label='Actual')
        plt.plot(predictions.index, predictions, label='Predicted')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(f'{model_name} Price Prediction')
        plt.legend()

        # Save the plot to a BytesIO object
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)

        # Convert the plot image to base64 encoding
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return render_template('result.html', image_base64=image_base64, result_table=result_table)


class ARIMAModel:
    def __init__(self):
        self.order = (4, 2, 0)  # ARIMA order

    def predict(self, prices, start_date, end_date):
        model = ARIMA(prices, order=self.order)
        model_fit = model.fit()
        predictions = model_fit.predict(start=start_date, end=end_date)
        return predictions


class ETSModel:
    def __init__(self):
        self.trend = 'add'  # Trend component type
        self.seasonal = 'add'  # Seasonal component type

    def predict(self, prices, start_date, end_date):
        model = ETSModel(prices, error='add', trend=self.trend, seasonal=self.seasonal)
        model_fit = model.fit()
        predictions = model_fit.predict(start=start_date, end=end_date)
        return predictions
    
if __name__ == '__main__':
    app = StockPricePredictionApp()
    app.run()

